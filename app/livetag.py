"""Who is talking, while the meeting is still going on.

Each new line of a live recording gets a voice print; lines whose voice is close to one heard before get the same
label ("Speaker 2"), a new voice gets a new label. Once a voice has a few seconds of speech it is compared with the
people you've named before, and gets their name. The full speaker separation at the end still decides; this is the
best guess while it streams in (and what stays if that separation fails or is switched off)."""
import logging
from typing import Optional

import numpy as np

from . import transcribe, voices
from .config import settings

log = logging.getLogger("whisper.livetag")

MIN_S = 1.2          # shorter lines are too short for a voice print: they get the voice around them
MAX_CLIP_S = 15      # enough of a long line for its voice print
MAX_VOICES = 12


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else a


def _closest(e: np.ndarray, vs: list[dict]) -> tuple[Optional[dict], float]:
    best, sim = None, -1.0
    for v in vs:
        s = float(np.dot(e, _unit(v["c"])))
        if s > sim:
            best, sim = v, s
    return best, sim


def tag(path: str, state: dict, first: int) -> int:
    """Label state["segments"][first:] (dicts from partial()). Updates state["voices"] and state["names"].
    Returns how many lines got a voice."""
    segs = state.get("segments") or []
    vs = state.setdefault("voices", [])        # [{"label": "Speaker 1", "c": [...], "n": lines, "secs": s}]
    done = 0
    for i in range(first, len(segs)):
        s = segs[i]
        start, end = float(s.get("start", 0)), float(s.get("end", 0))
        if end - start < MIN_S:
            continue
        try:
            e = transcribe.embed_clip(transcribe.load_range(path, start, min(end, start + MAX_CLIP_S)), start)
        except Exception as x:                       # noqa: BLE001 - a voice print failing only loses the label
            log.warning("live voice print failed: %s", x)
            e = None
        if e is None:
            if transcribe.embed_error:                # no voice model here (no token, not installed): stop trying
                break
            continue
        u = _unit(e)
        v, sim = _closest(u, vs)
        if v is None or (sim < settings.live_same_voice and len(vs) < MAX_VOICES):
            v = {"label": f"Speaker {len(vs) + 1}", "c": [round(float(x), 5) for x in u], "n": 0, "secs": 0.0}
            vs.append(v)
        else:                                        # running average of the voice: steadier with every line
            n = min(v["n"], 30)                      # but still follows them (a cold, a different mic distance)
            c = _unit((_unit(v["c"]) * n + u) / (n + 1))
            v["c"] = [round(float(x), 5) for x in c]
        v["n"] += 1
        v["secs"] = round(v["secs"] + (end - start), 1)
        s["speaker"] = v["label"]
        done += 1
    # short lines ("yes", "ok") take the voice of the line just before (or after) when it's close in time
    for i in range(first, len(segs)):
        s = segs[i]
        if s.get("speaker"):
            continue
        prev = segs[i - 1] if i > 0 else None
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if prev and prev.get("speaker") and float(s["start"]) - float(prev["end"]) < 1.5:
            s["speaker"] = prev["speaker"]
        elif nxt and nxt.get("speaker") and float(nxt["start"]) - float(s["end"]) < 1.5:
            s["speaker"] = nxt["speaker"]
    if vs:
        try:
            state["names"] = voices.match({v["label"]: {"embedding": v["c"], "seconds": v["secs"]} for v in vs})
        except Exception as x:                       # noqa: BLE001
            log.warning("live naming failed: %s", x)
    return done


def speakers(state: dict) -> dict:
    """The voices heard so far, in the shape of Transcript.speakers (for when there's no separation at the end)."""
    return {v["label"]: {"seconds": v["secs"], "embedding": v["c"]} for v in state.get("voices") or []}
