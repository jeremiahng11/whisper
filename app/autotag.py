"""Tag one line, and every other line in the meeting that sounds like the same person follows.

When you say who spoke a line (POST /api/jobs/{id}/segments/{i}), a background pass compares the voice of every line
of that meeting with the people you have tagged so far and gives a line to a person when it clearly sounds like them:
  * similarity >= AUTOTAG_MATCH (cosine of voice prints, default 0.60), and
  * clearly closer to them than to anyone else (margin AUTOTAG_MARGIN, default 0.05) - other tagged people and the
    diarizer's other speakers that do NOT sound like them.
Lines you tagged yourself, and lines where two people talk at once, are never changed. Voice prints of lines are worked out once per meeting and kept in
seg_emb.json, so the next tag is quick. GET /api/jobs/{id}/autotag reports progress.
"""
import json
import logging
import math
import os
import threading
import time

from . import db, transcribe, voices, worker
from .config import settings

log = logging.getLogger("autotag")
_state: dict[str, dict] = {}
_lock = threading.Lock()
_running: set[str] = set()
_job_locks: dict[str, threading.Lock] = {}


def job_lock(jid: str) -> threading.Lock:
    """One change at a time per meeting (your tags and auto-tagging both read and rewrite the transcript)."""
    with _lock:
        return _job_locks.setdefault(jid, threading.Lock())


def status(jid: str) -> dict:
    with _lock:
        st = dict(_state.get(jid) or {"state": "idle"})
    j = db.get(jid) or {}
    opts = j.get("options") or {}
    st["manual"] = sorted(int(k) for k in (opts.get("manual_lines") or {}))
    st["auto"] = sorted(opts.get("auto_lines") or [])
    return st


def _set(jid: str, **kw) -> None:
    with _lock:
        _state.setdefault(jid, {}).update(kw)


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cos(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    return sum(x * y for x, y in zip(_norm(a), _norm(b)))


def _mean(vs):
    vs = [_norm(v) for v in vs if v]
    if not vs:
        return None
    return _norm([sum(c) / len(vs) for c in zip(*vs)])


def segment_embeddings(jid: str, job: dict, segs: list) -> dict:
    """{"12": [..], "13": None, ...} for every line (None = too short to tell)."""
    path = worker.result_dir(jid) / "seg_emb.json"
    try:
        cache = json.loads(path.read_text()) if path.exists() else {}
    except ValueError:
        cache = {}
    todo = [i for i in range(len(segs)) if str(i) not in cache]
    audio = job.get("audio_path")
    if todo and (not audio or not os.path.exists(audio)):
        raise RuntimeError("the audio of this recording is no longer kept")
    last = time.time()
    for n, i in enumerate(todo):
        s = segs[i]
        start, end = float(s["start"]), float(s["end"])
        emb = None
        if end - start >= 1.2:
            clip = transcribe.load_range(audio, start, min(end, start + 20))
            emb = transcribe.embed_clip(clip, t=start)
        cache[str(i)] = emb
        _set(jid, state="running", step="listening", done=n + 1, total=len(todo))
        if time.time() - last > 5:                         # keep progress if it's interrupted
            worker._write(jid, "seg_emb.json", json.dumps(cache))
            last = time.time()
    if todo:
        worker._write(jid, "seg_emb.json", json.dumps(cache))
    return cache


def apply(jid: str) -> int:
    """Re-assign lines that sound like a tagged person. Returns how many lines changed."""
    job = db.get(jid)
    tr = worker.load_transcript(jid)
    if not job or not tr:
        return 0
    segs = tr.get("segments", [])
    opts = dict(job.get("options") or {})
    manual = {int(k): v for k, v in (opts.get("manual_lines") or {}).items()}
    names = dict(opts.get("speaker_names") or {})
    embs = segment_embeddings(jid, job, segs)
    _set(jid, state="running", step="matching")
    # each tagged person: the lines you tagged (+ their remembered voice print, if any)
    groups: dict[str, list] = {}
    for i, lab in manual.items():
        e = embs.get(str(i))
        if e and lab in names and not (segs[i].get("also") if i < len(segs) else None):   # mixed voices teach nothing
            groups.setdefault(lab, []).append(e)
    known = {n: v for n, v, _ in voices._all()}
    spk = worker.load_json(jid, "speakers.json", {}) or {}
    merged = opts.get("merged") or {}
    for g, lab in merged.items():                          # a whole group you named: its voice print is theirs
        e = (spk.get(g) or {}).get("embedding")
        if e and lab in names:
            groups.setdefault(lab, []).append(e)
    for lab in names:
        e = (spk.get(lab) or {}).get("embedding")
        if e and (lab in groups or lab not in (opts.get("speaker_auto") or [])):   # named by you ("Save names")
            groups.setdefault(lab, []).append(e)
    people = {}
    for lab, vs in groups.items():
        if names.get(lab) in known:
            vs = vs + [known[names[lab]]]
        c = _mean(vs)
        if c:
            people[lab] = c
    if not people:
        if embs and not any(embs.values()) and transcribe.embed_error:
            raise RuntimeError("voice comparison isn't working on the server: " + transcribe.embed_error)
        return 0
    thr, margin = settings.autotag_match, settings.autotag_margin
    # 1. whole groups: the speaker separation often splits one person into two or three groups (noisy room,
    #    long talker). A group whose voice clearly matches a tagged person becomes theirs, all its lines at once.
    group_to: dict[str, str] = {}
    for g, info in spk.items():
        e = (info or {}).get("embedding")
        if not e or g in people or g in merged or (names.get(g) and g not in (opts.get("speaker_auto") or [])):
            continue
        sims = sorted(((_cos(e, c), lab) for lab, c in people.items()), reverse=True)
        second = sims[1][0] if len(sims) > 1 else -1.0
        if sims[0][0] >= settings.autotag_group and sims[0][0] - second >= 0.05:
            group_to[g] = sims[0][1]
    # the diarizer's speakers count as "someone else" unless they sound like a tagged person (then they're them)
    others = {}
    for lab, info in spk.items():
        e = (info or {}).get("embedding")
        if not e or lab in people or lab in merged or names.get(lab) or lab in group_to:
            continue
        if max(_cos(e, c) for c in people.values()) >= thr:
            continue
        others[lab] = e
    decide: dict[int, str] = {}                            # line -> person, from this snapshot
    unsure: set[int] = set()
    for i, s in enumerate(segs):
        if i in manual or s.get("also"):                   # yours, or two people at once: leave it
            continue
        g = s.get("orig") or s.get("speaker")
        if s.get("speaker") == g and g in group_to:        # its whole group sounds like them
            decide[i] = group_to[g]
            continue
        e = embs.get(str(i))
        if not e:
            continue
        sims = sorted(((_cos(e, c), lab) for lab, c in people.items()), reverse=True)
        best, lab = sims[0]
        rival = max([x for x, _ in sims[1:]] + [_cos(e, v) for v in others.values()] + [-1.0])
        if best >= thr and best - rival >= margin:
            decide[i] = lab
        else:
            unsure.add(i)
    # write onto the LATEST transcript and settings: you may have tagged more lines while this was running
    changed = 0
    with job_lock(jid):
        job = db.get(jid)
        tr = worker.load_transcript(jid)
        if not job or not tr:
            return 0
        segs = tr.get("segments", [])
        opts = dict(job.get("options") or {})
        manual = {int(k) for k in (opts.get("manual_lines") or {})}
        auto = set(opts.get("auto_lines") or [])
        for i, lab in decide.items():
            if i >= len(segs) or i in manual or segs[i].get("also"):
                continue
            if segs[i].get("speaker") != lab:
                segs[i].setdefault("orig", segs[i].get("speaker", ""))
                segs[i]["speaker"] = lab
                changed += 1
            auto.add(i)
        for i in unsure:
            if i in auto and i < len(segs) and segs[i].get("speaker") in people:
                auto.discard(i)                            # no longer sure: leave it, but not marked as auto
        worker._write(jid, "transcript.json", json.dumps(tr, ensure_ascii=False, indent=1))
        if group_to:
            m = dict(opts.get("merged") or {})
            m.update(group_to)
            opts["merged"] = m
        opts["auto_lines"] = sorted(auto - manual)
        db.update(jid, options=opts)
        worker.build_outputs(jid)
    return changed


def start(jid: str) -> None:
    """Run apply() in the background (one at a time per meeting; a new tag during a run re-runs it after)."""
    with _lock:
        if jid in _running:
            _state.setdefault(jid, {})["again"] = True
            return
        _running.add(jid)
        _state[jid] = {"state": "running", "step": "starting"}

    def run():
        try:
            while True:
                n = apply(jid)
                with _lock:
                    again = _state.get(jid, {}).pop("again", False)
                if not again:
                    break
            _set(jid, state="done", changed=n, at=int(time.time()))
        except Exception as e:                             # noqa: BLE001
            log.warning("auto-tag %s failed: %s", jid, e)
            _set(jid, state="error", error=str(e)[:200])
        finally:
            with _lock:
                _running.discard(jid)

    threading.Thread(target=run, daemon=True).start()
