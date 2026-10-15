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
import re
import logging
import math
import os
import threading
import time
from typing import Optional

import numpy as np

from . import db, transcribe, voices, worker
from .config import settings

log = logging.getLogger("autotag")
_state: dict[str, dict] = {}
_lock = threading.Lock()
_running: set[str] = set()
_job_locks: dict[str, threading.Lock] = {}


_STARTED = time.time()                                     # a failure before a redeploy is retried


def job_lock(jid: str) -> threading.Lock:
    """One change at a time per meeting (your tags and auto-tagging both read and rewrite the transcript)."""
    with _lock:
        return _job_locks.setdefault(jid, threading.Lock())


RULES = "r2"          # bump when the matching rules change: every meeting is checked again


# ---------------------------------------------------------------- pitch: a cheap, strong check against mix-ups
# Voice prints can call a woman's voice a man's (or the reverse) when the room is noisy or one of them has few
# samples. Pitch can't: typical speaking pitch differs by ~1.7x between low and high voices. A line or a whole
# speaker group is never given to a person whose usual pitch is far from theirs.
PITCH_RATIO = 1.35


def _line_pitch(jid: str, job: dict, segs: list, idxs: list[int]) -> dict[int, float]:
    """Median pitch (Hz) of these lines, cached in seg_pitch.json. 0 = not enough voiced audio to tell."""
    from . import people as _people
    cache = worker.load_json(jid, "seg_pitch.json", {}) or {}
    path = job.get("audio_path")
    changed = False
    out = {}
    for i in idxs:
        k = str(i)
        if k not in cache:
            p = 0.0
            if path and os.path.exists(path) and i < len(segs):
                s = segs[i]
                try:
                    f0 = _people.pitch_track(transcribe.load_range(path, s["start"], min(s["end"], s["start"] + 12)))
                    if len(f0) >= 12:
                        p = float(np.median(f0))
                except Exception:                          # noqa: BLE001
                    p = 0.0
            cache[k] = round(p, 1)
            changed = True
        out[i] = cache[k]
    if changed:
        worker._write(jid, "seg_pitch.json", json.dumps(cache))
    return out


def _person_pitch(name: str) -> float:
    from . import people as _people
    p = (_people.get(name) or {}).get("voice") or {}
    return float(p.get("pitch_hz") or 0)


def pitch_ok(person_hz: float, hz: float) -> bool:
    if not person_hz or not hz:
        return True                                        # can't tell: no veto
    r = hz / person_hz
    return 1 / PITCH_RATIO <= r <= PITCH_RATIO


def voices_signature() -> str:
    """Changes whenever a voice is added, learned more or forgotten."""
    with db._lock:
        r = db._conn.execute("SELECT COUNT(*) AS n, COALESCE(MAX(updated_at), 0) AS t FROM voices").fetchone()
    return f"{r['n']}:{round(float(r['t']), 3)}:{RULES}"


def stale(job: dict) -> bool:
    """Not yet checked against the voices you have now (and it still has its audio)."""
    if not job or job.get("status") != "done" or not job.get("audio_path") or not os.path.exists(job["audio_path"]):
        return False
    opts, sig = job.get("options") or {}, voices_signature()
    if opts.get("recognized_voices") == sig:
        return False
    # no remembered voices: still worth one look if you tagged people in this meeting
    return not sig.startswith("0:") or bool(opts.get("manual_lines"))


def status(jid: str) -> dict:
    with _lock:
        st = dict(_state.get(jid) or {"state": "idle"})
    j = db.get(jid) or {}
    opts = j.get("options") or {}
    st["manual"] = sorted(int(k) for k in (opts.get("manual_lines") or {}))
    st["auto"] = sorted(opts.get("auto_lines") or [])
    st["stale"] = stale(j)
    st["no_speakers"] = needs_speakers(j) if j else False
    st["diarize_error"] = opts.get("diarize_error", "")
    st["report"] = opts.get("autotag_report")
    st["loose"] = bool(opts.get("autotag_loose"))
    return st


def active(jid: str | None = None):
    """What the voice work is doing right now: {"step", "done", "total", "minutes"} per meeting (or one meeting)."""
    with _lock:
        rows = {k: dict(_state.get(k) or {}) for k in _running}
    out = {}
    for k, st in rows.items():
        out[k] = {"step": st.get("step", "starting"), "done": st.get("done"), "total": st.get("total"), "pct": st.get("pct"),
                  "minutes": int((time.time() - st.get("since", time.time())) // 60)}
    return out.get(jid) if jid is not None else out


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
    # not yet done, or failed before (voice prints weren't working then): long lines get another try
    todo = [i for i in range(len(segs)) if str(i) not in cache
            or (cache[str(i)] is None and float(segs[i]["end"]) - float(segs[i]["start"]) >= 1.2)]
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
    # people the server already knows from other meetings: they count too, even if nobody was tagged here
    tagged = dict(people)
    in_meeting = {names[lab] for lab in people if lab in names}
    for lab, n in names.items():                           # named here (e.g. recognised), no sample of its own
        if lab not in people and n in known:
            if any(_cos(known[n], c) >= settings.autotag_group for c in tagged.values()):
                continue                                   # you tagged that voice as someone else here
            people[lab] = _norm(known[n])
            in_meeting.add(n)
    for n, v in known.items():
        if n in in_meeting or n in names.values():
            continue
        if any(_cos(v, c) >= settings.autotag_group for c in tagged.values()):
            continue                                       # sounds like someone you tagged here: yours wins
        people["voice:" + n] = _norm(v)               # becomes a speaker of this meeting if they're heard
    if not people:
        if embs and not any(embs.values()) and transcribe.embed_error:
            raise RuntimeError("voice comparison isn't working on the server: " + transcribe.embed_error)
        with job_lock(jid):                                # nothing to compare with: checked, don't repeat
            j2 = db.get(jid)
            if j2:
                o2 = dict(j2.get("options") or {})
                o2["recognized_voices"] = voices_signature()
                o2["autotag_report"] = {"at": int(time.time()), "lines": len(segs), "named_now": 0, "people": [],
                                        "unnamed": sum(1 for s in segs if s.get("speaker") not in names),
                                        "short": 0, "close": 0, "far": 0, "close_to": {}}
                db.update(jid, options=o2)
        return 0
    thr, margin = settings.autotag_match, settings.autotag_margin
    if opts.get("autotag_loose"):                          # "Match more" on the web: closer calls are accepted
        thr, margin = thr - 0.08, 0.0
    why = {"short": 0, "close": 0, "far": 0}               # why lines were left alone (shown on the web)
    close_to: dict[str, int] = {}
    # 1. whole groups: the speaker separation often splits one person into two or three groups (noisy room,
    #    long talker). A group whose voice clearly matches a tagged person becomes theirs, all its lines at once.
    who = lambda lab: lab[6:] if lab.startswith("voice:") else names.get(lab, "")   # noqa: E731
    pp = {lab: _person_pitch(who(lab)) for lab in people if who(lab)}
    has_lines = {s.get("speaker") for s in segs}
    group_to: dict[str, str] = {}
    for g, info in spk.items():
        e = (info or {}).get("embedding")
        if not e or g in people or g in merged or (names.get(g) and g not in (opts.get("speaker_auto") or [])):
            continue
        glines = sorted((i for i, s in enumerate(segs) if (s.get("orig") or s.get("speaker")) == g and not s.get("also")),
                        key=lambda i: -(segs[i]["end"] - segs[i]["start"]))[:6]
        gp = [p for p in _line_pitch(jid, job, segs, glines).values() if p] if glines else []
        ghz = float(np.median(gp)) if len(gp) >= 2 else 0.0
        sims = sorted(((_cos(e, c), lab) for lab, c in people.items() if pitch_ok(pp.get(lab, 0), ghz)), reverse=True)
        if not sims:
            continue
        second = sims[1][0] if len(sims) > 1 else -1.0
        # the separation already heard this group as a different person from someone talking under their own
        # label here: only a very close voice overrides that
        need = max(settings.autotag_group, 0.65) if sims[0][1] in has_lines else settings.autotag_group
        if sims[0][0] >= need and sims[0][0] - second >= 0.05:
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
            why["short"] += 1
            continue
        sims = sorted(((_cos(e, c), lab) for lab, c in people.items()), reverse=True)
        best, lab = sims[0]
        rival = max([x for x, _ in sims[1:]] + [_cos(e, v) for v in others.values()] + [-1.0])
        if best >= thr and best - rival >= margin:
            decide[i] = lab
        else:
            unsure.add(i)
            if best >= thr:
                why["close"] += 1
                close_to[lab] = close_to.get(lab, 0) + 1
            else:
                why["far"] += 1
    # earlier runs may have joined a whole group to someone it can't be (before the pitch check): undo that join
    unmerge = set()
    for g, lab in (opts.get("merged") or {}).items():
        if lab in pp:
            glines = sorted((i for i, s in enumerate(segs) if s.get("orig") == g and not s.get("also")),
                            key=lambda i: -(segs[i]["end"] - segs[i]["start"]))[:6]
            gp = [p for p in _line_pitch(jid, job, segs, glines).values() if p] if glines else []
            if len(gp) >= 2 and not pitch_ok(pp[lab], float(np.median(gp))):
                unmerge.add(g)
                for i, s in enumerate(segs):
                    if s.get("orig") == g and i not in manual:
                        unsure.add(i)
                        decide.pop(i, None)
    # pitch veto: a line never goes to someone whose usual pitch is far from the line's (a woman's voice to a man)
    check = [i for i, lab in decide.items() if pp.get(lab)]
    if check:
        lp = _line_pitch(jid, job, segs, check)
        for i in check:
            if not pitch_ok(pp[decide[i]], lp.get(i, 0)):
                unsure.add(i)
                why["far"] += 1
                del decide[i]
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
        names2 = dict(opts.get("speaker_names") or {})
        recog = set(opts.get("speaker_auto") or [])

        def real(lab: str) -> str:
            if not lab.startswith("voice:"):
                return lab
            who = lab[6:]
            hit = next((k for k, v in names2.items() if v == who), "")
            if hit:
                return hit
            used = {s.get("speaker") for s in segs} | set(names2)
            nums = [int(x.split()[-1]) for x in used if x and x.startswith("Speaker ") and x.split()[-1].isdigit()]
            new = f"Speaker {max(nums, default=0) + 1}"
            names2[new] = who
            recog.add(new)                                 # shown as "recognised"
            return new

        decide = {i: real(lab) for i, lab in decide.items()}
        group_to = {g: real(lab) for g, lab in group_to.items()}
        for i, lab in decide.items():
            if i >= len(segs) or i in manual or segs[i].get("also"):
                continue
            if segs[i].get("speaker") != lab:
                segs[i].setdefault("orig", segs[i].get("speaker", ""))
                segs[i]["speaker"] = lab
                changed += 1
            auto.add(i)
        merged_now = dict(opts.get("merged") or {})
        merged_now.update(group_to)
        for g in unmerge:
            merged_now.pop(g, None)
        for i in unsure:
            if i in auto and i < len(segs) and segs[i].get("speaker") in people:
                auto.discard(i)                            # no longer sure: not marked as auto
                o, cur = segs[i].get("orig"), segs[i].get("speaker")
                # it was given to someone by voice earlier, and after your correction it no longer clearly sounds
                # like them: back to the speaker separation's guess (unless you named that whole group as them)
                if o and o != cur and merged_now.get(o) != cur and not segs[i].get("also"):
                    segs[i]["speaker"] = o
                    changed += 1
        worker._write(jid, "transcript.json", json.dumps(tr, ensure_ascii=False, indent=1))
        if group_to or unmerge:
            m = dict(opts.get("merged") or {})
            m.update(group_to)
            for g in unmerge:
                m.pop(g, None)
            opts["merged"] = m
        opts["speaker_names"] = names2
        opts["speaker_auto"] = sorted(recog)
        opts["recognized_voices"] = voices_signature()
        unnamed = sum(1 for s in segs if s.get("speaker") not in names2 and not s.get("also"))
        opts["autotag_report"] = {
            "at": int(time.time()), "lines": len(segs), "named_now": len(decide), "unnamed": unnamed,
            "short": why["short"], "close": why["close"], "far": why["far"],
            "close_to": {(lab[6:] if lab.startswith("voice:") else names2.get(lab, lab)): n for lab, n in close_to.items()},
            "people": sorted({(lab[6:] if lab.startswith("voice:") else names2.get(lab, lab)) for lab in people}),
            "loose": bool(opts.get("autotag_loose")),
        }
        opts["auto_lines"] = sorted(auto - manual)
        db.update(jid, options=opts)
        worker.build_outputs(jid)
    return changed


def start(jid: str, speakers: bool = False, fill: Optional[tuple] = None) -> None:
    """Run apply() in the background (one at a time per meeting; a new tag during a run re-runs it after)."""
    with _lock:
        if jid in _running:
            _state.setdefault(jid, {})["again"] = True
            return
        _running.add(jid)
        _state[jid] = {"state": "running", "step": "starting", "since": int(time.time())}

    def run():
        try:
            if fill:
                n = fill_gap(jid, fill[0], fill[1])
                _set(jid, filled=n)
            if speakers:
                redo_speakers(jid)
                relearn()                                  # tags in this meeting: their voices are learned now
            while True:
                n = apply(jid)
                with _lock:
                    again = _state.get(jid, {}).pop("again", False)
                if not again:
                    break
            if speakers and renumber(jid):                 # speakers separated again: 1, 2, 3 ... (no tagging yet)
                worker.build_outputs(jid)
            _set(jid, state="done", changed=n, at=int(time.time()))
        except Exception as e:                             # noqa: BLE001
            log.warning("auto-tag %s failed: %s", jid, e)
            _set(jid, state="error", error=str(e)[:200])
        finally:
            with _lock:
                _running.discard(jid)

    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------- automatic: new meetings + older ones when idle
def after_transcription(jid: str) -> None:
    """A new meeting is done: name lines of people the server already knows (if there are any)."""
    if not settings.autotag_new or voices_signature().startswith("0:"):
        return
    start(jid)


def fill_gap(jid: str, a: float, b: float) -> int:
    """Nothing was written down between a and b seconds: transcribe that stretch again with a much more sensitive
    speech detector and a louder copy of the audio, and add the lines found (speaker unknown until the voice check).
    Line numbers after the gap move up, so your tags, auto marks and voice prints move with their lines."""
    from . import terms
    job = db.get(jid)
    if not job or not job.get("audio_path") or not os.path.exists(job["audio_path"]):
        raise RuntimeError("the audio of this recording is no longer kept")
    a, b = max(0.0, float(a) - 1.0), float(b) + 1.0
    _set(jid, state="running", step="writing down a part")
    clip = transcribe.load_range(job["audio_path"], a, b)
    opts = job.get("options") or {}
    found, _, _ = transcribe._run(transcribe.level_audio(clip), a, job.get("language") or opts.get("language"),
                                  voices.whisper_prompt(opts.get("attendees")), lambda p, s: None, 0,
                                  vad_threshold=min(settings.vad_threshold, 0.15))
    if not found:
        return 0
    known = terms.known_terms()
    with job_lock(jid):
        job = db.get(jid)
        tr = worker.load_transcript(jid) or {}
        old = tr.get("segments", [])
        busy = [(float(s["start"]), float(s["end"])) for s in old]
        new = []
        for s in found:
            st, en = float(s.start), float(s.end)
            overlap = sum(max(0.0, min(en, y) - max(st, x)) for x, y in busy)
            if overlap > 0.5 * max(0.1, en - st):          # already written down
                continue
            new.append({"start": st, "end": en, "text": terms.fix_text(terms.apply_corrections(s.text), known), "speaker": "", "also": [],
                        "words": s.words, "added": True})
        if not new:
            return 0
        rows = [(float(s["start"]), 0, i, s) for i, s in enumerate(old)] + [(x["start"], 1, -1, x) for x in new]
        rows.sort(key=lambda r: (r[0], r[1]))
        move = {}
        segs = []
        for k, (_, _, i, s) in enumerate(rows):
            if i >= 0:
                move[i] = k
            segs.append(s)
        tr["segments"] = segs
        opts = dict(job.get("options") or {})
        opts["manual_lines"] = {str(move[int(i)]): v for i, v in (opts.get("manual_lines") or {}).items() if int(i) in move}
        opts["auto_lines"] = sorted(move[i] for i in (opts.get("auto_lines") or []) if i in move)
        cache = worker.load_json(jid, "seg_emb.json", {}) or {}
        worker._write(jid, "seg_emb.json", json.dumps({str(move[int(i)]): v for i, v in cache.items() if int(i) in move}))
        worker._write(jid, "transcript.json", json.dumps(tr, ensure_ascii=False, indent=1))
        opts.pop("recognized_voices", None)                # the new lines get a voice check
        db.update(jid, options=opts)
    worker.build_outputs(jid)
    log.info("filled %d lines in %s between %.0f and %.0f s", len(new), jid, a, b)
    return len(new)


_LAB = re.compile(r"^Speaker (\d+)$")


def renumber(jid: str) -> dict:
    """Speakers numbered 1, 2, 3 ... without gaps (joined or emptied speakers leave holes): every place that holds a
    label is renamed together - lines, names, tags, voice groups, and "Speaker N" written in the minutes. The order
    stays the same. Returns {old: new} for the labels that changed."""
    with job_lock(jid):
        job = db.get(jid)
        tr = worker.load_transcript(jid)
        if not job or not tr:
            return {}
        segs = tr.get("segments", [])
        opts = dict(job.get("options") or {})
        names = opts.get("speaker_names") or {}
        merged = opts.get("merged") or {}
        spk = worker.load_json(jid, "speakers.json", {}) or {}
        num = lambda l: int(_LAB.match(l).group(1)) if l and _LAB.match(l) else None   # noqa: E731
        used = {s.get("speaker") for s in segs} | {a for s in segs for a in (s.get("also") or [])}
        active = sorted({l for l in used | set(names) if num(l) and (l in used or l not in merged)}, key=num)
        others = sorted({l for l in set(spk) | set(merged) | {s.get("orig") for s in segs} if num(l)} - set(active), key=num)
        order = active + others
        m = {old: f"Speaker {i + 1}" for i, old in enumerate(order)}
        m = {k: v for k, v in m.items() if k != v}
        if not m:
            return {}
        r = lambda l: m.get(l, l)                          # noqa: E731
        for s in segs:
            if s.get("speaker"):
                s["speaker"] = r(s["speaker"])
            if s.get("orig"):
                s["orig"] = r(s["orig"])
            if s.get("also"):
                s["also"] = [r(a) for a in s["also"]]
        opts["speaker_names"] = {r(k): v for k, v in names.items()}
        opts["speaker_auto"] = sorted({r(k) for k in opts.get("speaker_auto") or []})
        opts["manual_lines"] = {k: r(v) for k, v in (opts.get("manual_lines") or {}).items()}
        opts["merged"] = {r(k): r(v) for k, v in merged.items()}
        if opts.get("names_at_summary"):
            opts["names_at_summary"] = {r(k): v for k, v in opts["names_at_summary"].items()}
        worker._write(jid, "speakers.json", json.dumps({r(k): v for k, v in spk.items()}))
        worker._write(jid, "transcript.json", json.dumps(tr, ensure_ascii=False, indent=1))
        sub = lambda t: re.sub(r"\bSpeaker (\d+)\b", lambda x: m.get(x.group(0), x.group(0)), t)   # noqa: E731
        p = worker.result_dir(jid) / "summary.md"
        if p.exists():                                     # "Speaker 15 will send ..." written in the minutes
            worker._write(jid, "summary.md", sub(p.read_text(encoding="utf-8")))
        raw = worker.load_json(jid, "actions_raw.json")
        if raw is not None:
            worker._write(jid, "actions_raw.json", sub(json.dumps(raw, ensure_ascii=False)))
        db.update(jid, options=opts)
    log.info("renumbered speakers in %s: %s", jid, m)
    return m


def merge_same_names(jid: str) -> list[str]:
    """Two speakers with the same name are one person: their lines, tags and voice prints become one speaker
    (the one with the most lines). Returns the names that were joined."""
    with job_lock(jid):
        job = db.get(jid)
        tr = worker.load_transcript(jid) or {}
        if not job:
            return []
        segs = tr.get("segments", [])
        opts = dict(job.get("options") or {})
        names = dict(opts.get("speaker_names") or {})
        by: dict[str, list[str]] = {}
        for lab, n in names.items():
            if n and n.strip():
                by.setdefault(n.strip().lower(), []).append(lab)
        dup = {k: v for k, v in by.items() if len(v) > 1}
        if not dup:
            return []
        lines = {}
        for s in segs:
            lines[s.get("speaker")] = lines.get(s.get("speaker"), 0) + 1
        auto = set(opts.get("speaker_auto") or [])
        manual = dict(opts.get("manual_lines") or {})
        merged = dict(opts.get("merged") or {})
        joined = []
        for labs in dup.values():
            keep = min(labs, key=lambda l: int(l.split()[-1]) if l.split()[-1].isdigit() else 10 ** 6)   # Speaker 5 stays, 12 goes
            gone = [l for l in labs if l != keep]
            for s in segs:
                if s.get("speaker") in gone:
                    s["speaker"] = keep
                if s.get("also"):
                    a = [keep if x in gone else x for x in s["also"]]
                    s["also"] = [x for i, x in enumerate(a) if x != s.get("speaker") and x not in a[:i]]
            for k, v in list(manual.items()):
                if v in gone:
                    manual[k] = keep
            for g, v in list(merged.items()):
                if v in gone:
                    merged[g] = keep
            for g in gone:
                merged[g] = keep                           # its voice print counts for the kept speaker
                names.pop(g, None)
            if not all(l in auto for l in labs):
                auto.discard(keep)
            auto -= set(gone)
            joined.append(names[keep])
        worker._write(jid, "transcript.json", json.dumps(tr, ensure_ascii=False, indent=1))
        opts.update(speaker_names=names, speaker_auto=sorted(auto), manual_lines=manual, merged=merged)
        db.update(jid, options=opts)
    worker.build_outputs(jid)
    return joined


def accept(jid: str, name: str = "") -> dict:
    """You checked the "auto" lines: they become yours (✓, never changed automatically again) and teach the server
    those voices a little more. name = only that person's lines."""
    learn: dict[str, list] = {}
    with job_lock(jid):
        job = db.get(jid)
        tr = worker.load_transcript(jid) or {}
        if not job:
            return {"accepted": 0}
        segs = tr.get("segments", [])
        opts = dict(job.get("options") or {})
        names = opts.get("speaker_names") or {}
        manual = dict(opts.get("manual_lines") or {})
        auto = set(opts.get("auto_lines") or [])
        cache = worker.load_json(jid, "seg_emb.json", {}) or {}
        done = []
        for i in sorted(auto):
            if i >= len(segs):
                continue
            lab = segs[i].get("speaker", "")
            who = names.get(lab)
            if not who or (name and who.lower() != name.lower()):
                continue                                   # only lines with a real name
            manual[str(i)] = lab
            done.append(i)
            if cache.get(str(i)) and not segs[i].get("also"):
                learn.setdefault(who, []).append((cache[str(i)], i))
        opts["manual_lines"] = manual
        opts["auto_lines"] = sorted(auto - set(done))
        db.update(jid, options=opts)
    for who, es in learn.items():
        for e, i in es[:15]:
            voices.remember(who, e, jid, i)
    return {"accepted": len(done), "learned": sorted(learn)}


def relearn() -> dict:
    """Voices lost (e.g. /data wasn't kept on a redeploy) but your tags are still in the meetings: learn each
    person's voice again from the lines you tagged and the speakers you named. Only for names with no voice now."""
    have = {n.lower() for n, _, _ in voices._all()}
    found: dict[str, list] = {}
    for j in db.list_jobs(100000, "done"):
        jid, opts = j["id"], j.get("options") or {}
        names = opts.get("speaker_names") or {}
        manual = {int(k): v for k, v in (opts.get("manual_lines") or {}).items()}
        auto = set(opts.get("speaker_auto") or [])
        if not names:
            continue
        tr = worker.load_transcript(jid) or {}
        segs = tr.get("segments", [])
        spk = worker.load_json(jid, "speakers.json", {}) or {}
        cache = worker.load_json(jid, "seg_emb.json", {}) or {}
        audio = j.get("audio_path")
        for i, lab in manual.items():
            who = names.get(lab)
            if not who or who.lower() in have or i >= len(segs) or segs[i].get("also"):
                continue
            e = cache.get(str(i))
            if e is None and audio and os.path.exists(audio) and segs[i]["end"] - segs[i]["start"] >= 1.2:
                s = segs[i]
                e = transcribe.embed_clip(transcribe.load_range(audio, float(s["start"]), min(float(s["end"]), float(s["start"]) + 20)),
                                          t=float(s["start"]))
            if e:
                found.setdefault(who, []).append((e, jid, i))
        for lab, who in names.items():                      # whole speakers you named ("Save names")
            e = (spk.get(lab) or {}).get("embedding")
            if e and who and who.lower() not in have and lab not in auto:
                found.setdefault(who, []).append((e, jid, -1))
    for who, es in found.items():
        for e, src, seg in es[:40]:
            voices.remember(who, e, src, seg)
    if found:
        log.info("re-learned %d voices from your tags: %s", len(found), ", ".join(found))
    return {"learned": sorted(found), "samples": sum(min(len(v), 40) for v in found.values())}


def needs_speakers(job: dict) -> bool:
    """Speaker separation was on but failed (no line has a speaker): worth doing again."""
    opts = job.get("options") or {}
    if job.get("status") != "done" or not opts.get("diarize", settings.diarize) or opts.get("speakers_redone"):
        return False
    failed = float(opts.get("diarize_failed_at") or 0)
    if failed > _STARTED and time.time() - failed < 6 * 3600:   # failed since this server started: not every minute
        return False
    if not job.get("audio_path") or not os.path.exists(job["audio_path"]):
        return False
    if worker.load_json(job["id"], "speakers.json", {}):   # it has the separation's speaker groups
        return False
    tr = worker.load_transcript(job["id"]) or {}
    return bool(tr.get("segments"))


def redo_speakers(jid: str) -> int:
    """Run speaker separation again on a finished meeting. Your tagged lines keep their names; the other lines
    get speakers, people with remembered voices are named, and your tags then spread to the same voices."""
    job = db.get(jid)
    if not job:
        return 0
    _set(jid, state="running", step="separating speakers")
    audio = transcribe.load_audio(job["audio_path"])
    tr = worker.load_transcript(jid) or {}
    opts = job.get("options") or {}
    names = opts.get("speaker_names") or {}
    used = {s.get("speaker") for s in tr.get("segments", [])} | set(names)
    nums = [int(x.split()[-1]) for x in used if x and x.startswith("Speaker ") and x.split()[-1].isdigit()]
    try:
        labels, spk = transcribe.label_lines(audio, tr.get("segments", []), opts.get("speakers"), max(nums, default=0) + 1,
                                             lambda f: _set(jid, pct=round(100 * f)))
    except Exception as e:                                 # noqa: BLE001
        with job_lock(jid):
            o = dict((db.get(jid) or {}).get("options") or {})
            o["diarize_error"] = transcribe.diarize_error or str(e)[:200]
            o["diarize_failed_at"] = int(time.time())
            db.update(jid, options=o)
        raise RuntimeError("speaker separation failed: " + (transcribe.diarize_error or str(e)[:200]))
    n = 0
    with job_lock(jid):
        job = db.get(jid)
        tr = worker.load_transcript(jid) or {}
        segs = tr.get("segments", [])
        opts = dict(job.get("options") or {})
        manual = {int(k) for k in (opts.get("manual_lines") or {})}
        for i, s in enumerate(segs):
            if i in manual or i >= len(labels) or not labels[i][0]:
                continue
            s["speaker"], s["also"] = labels[i][0], labels[i][1]
            s.pop("orig", None)
            n += 1
        old = worker.load_json(jid, "speakers.json", {}) or {}
        worker._write(jid, "transcript.json", json.dumps(tr, ensure_ascii=False, indent=1))
        worker._write(jid, "speakers.json", json.dumps({**old, **spk}))
        names2 = dict(opts.get("speaker_names") or {})
        auto = {k: v for k, v in voices.match(spk).items() if k not in names2}
        names2.update(auto)
        opts["speaker_names"] = names2
        opts["speaker_auto"] = sorted(set(opts.get("speaker_auto") or []) | set(auto))
        opts["speakers_redone"] = int(time.time())
        opts.pop("diarize_error", None)
        opts.pop("diarize_failed_at", None)
        opts.pop("recognized_voices", None)                # check again with the new groups
        db.update(jid, options=opts)
    worker.build_outputs(jid)
    return n


def _sweep() -> None:
    from . import worker as w
    try:
        relearn()                                          # voices lost on an old redeploy: back from your tags
    except Exception as e:                                 # noqa: BLE001
        log.warning("re-learning voices: %s", e)
    while True:
        time.sleep(60)
        try:
            if w.current_job is not None or db.next_queued() is not None or _running:
                continue
            redo = next((j for j in db.list_jobs(200, "done") if needs_speakers(j)), None)
            if redo:                                        # speaker separation failed when it was transcribed
                log.info("separating speakers again in %s", redo["id"])
                start(redo["id"], speakers=True)
                continue
            for j in db.list_jobs(200, "done"):            # newest first, one meeting per minute
                if stale(j):
                    log.info("recognising known voices in %s", j["id"])
                    start(j["id"])
                    break
        except Exception as e:                             # noqa: BLE001
            log.warning("voice sweep: %s", e)


def start_sweeper() -> None:
    if settings.autotag_sweep:
        threading.Thread(target=_sweep, daemon=True).start()
