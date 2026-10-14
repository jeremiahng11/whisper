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


_STARTED = time.time()                                     # a failure before a redeploy is retried


def job_lock(jid: str) -> threading.Lock:
    """One change at a time per meeting (your tags and auto-tagging both read and rewrite the transcript)."""
    with _lock:
        return _job_locks.setdefault(jid, threading.Lock())


def voices_signature() -> str:
    """Changes whenever a voice is added, learned more or forgotten."""
    with db._lock:
        r = db._conn.execute("SELECT COUNT(*) AS n, COALESCE(MAX(updated_at), 0) AS t FROM voices").fetchone()
    return f"{r['n']}:{round(float(r['t']), 3)}"


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
        out[k] = {"step": st.get("step", "starting"), "done": st.get("done"), "total": st.get("total"),
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
        for i in unsure:
            if i in auto and i < len(segs) and segs[i].get("speaker") in people:
                auto.discard(i)                            # no longer sure: leave it, but not marked as auto
        worker._write(jid, "transcript.json", json.dumps(tr, ensure_ascii=False, indent=1))
        if group_to:
            m = dict(opts.get("merged") or {})
            m.update(group_to)
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


def start(jid: str, speakers: bool = False) -> None:
    """Run apply() in the background (one at a time per meeting; a new tag during a run re-runs it after)."""
    with _lock:
        if jid in _running:
            _state.setdefault(jid, {})["again"] = True
            return
        _running.add(jid)
        _state[jid] = {"state": "running", "step": "starting", "since": int(time.time())}

    def run():
        try:
            if speakers:
                redo_speakers(jid)
                relearn()                                  # tags in this meeting: their voices are learned now
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


# ---------------------------------------------------------------- automatic: new meetings + older ones when idle
def after_transcription(jid: str) -> None:
    """A new meeting is done: name lines of people the server already knows (if there are any)."""
    if not settings.autotag_new or voices_signature().startswith("0:"):
        return
    start(jid)


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
            keep = max(labs, key=lambda l: (lines.get(l, 0), l not in auto, -int(l.split()[-1]) if l.split()[-1].isdigit() else 0))
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
                learn.setdefault(who, []).append(cache[str(i)])
        opts["manual_lines"] = manual
        opts["auto_lines"] = sorted(auto - set(done))
        db.update(jid, options=opts)
    for who, es in learn.items():
        for e in es[:15]:
            voices.remember(who, e)
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
                found.setdefault(who, []).append(e)
        for lab, who in names.items():                      # whole speakers you named ("Save names")
            e = (spk.get(lab) or {}).get("embedding")
            if e and who and who.lower() not in have and lab not in auto:
                found.setdefault(who, []).append(e)
    for who, es in found.items():
        for e in es[:40]:
            voices.remember(who, e)
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
        labels, spk = transcribe.label_lines(audio, tr.get("segments", []), opts.get("speakers"), max(nums, default=0) + 1)
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
