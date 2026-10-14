"""People: what Oracle Minutes learns about each person you named, across all your meetings.

  * voice      - many separate voice samples (not one average), so recognition improves with every meeting;
                 typical pitch, pitch range, loudness and speaking pace, measured from their own lines
  * language   - the words, phrases and Names & terms they use most (compared with everyone else), and how they
                 mix English with Singlish / Malay / Chinese
  * AI note    - written by your own LLM from their lines: what they usually raise and own, how they communicate,
                 how best to work with them. Only work and communication - no guesses about personality, health,
                 mood, age or background.
The profiles are given to the minutes writer and to Ask, so they know who's who. Everything stays on this server;
forgetting a person removes their samples and profile.
"""
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter

import numpy as np

from . import db, summarize, transcribe, voices, worker
from .config import settings

log = logging.getLogger("people")
_lock = threading.Lock()
SR = transcribe.SR

STOP = set("""a about above after again against all am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from
further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off
on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some
such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves yeah yes okay ok um uh like just so actually really also then well right know think mean
going get got one two thing things something lot kind sort maybe will can would need want see say said go come""".split())
PARTICLES = {"lah", "leh", "lor", "meh", "sia", "hor", "ah", "mah", "liao", "walao", "alamak", "shiok", "paiseh",
             "can", "cannot"}
MALAY = {"boleh", "tak", "sudah", "belum", "makan", "jalan", "kita", "saya", "tidak", "macam", "ini", "itu", "apa",
         "sikit", "banyak", "terima", "kasih", "baik", "tapi", "pun", "dah", "nanti"}
HOKKIEN = {"kiasu", "kiasi", "sian", "buay", "jialat", "ang", "moh", "kopi", "bo", "liao", "tio", "paiseh", "chope"}
CJK = re.compile(r"[一-鿿]")
WORD = re.compile(r"[A-Za-z][A-Za-z'&-]*|[一-鿿]+")


# ---------------------------------------------------------------- voice samples (many per person)
MAX_SAMPLES = 100          # per person; at least 20 come from different meetings when there are that many


def add_sample(name: str, emb: list, job_id: str = "", seg: int = -1) -> None:
    """Keep this clip's voice print as one of the person's samples. Over MAX_SAMPLES, the oldest sample of the
    meeting that has the most samples goes first, so their voice is known from many meetings (rooms, calls, moods)
    rather than from one long one."""
    if not name or not emb:
        return
    n = math.sqrt(sum(x * x for x in emb)) or 1.0
    with db._lock:
        if seg >= 0 and job_id and db._conn.execute("SELECT 1 FROM voice_samples WHERE name=? AND job_id=? AND seg=?",
                                                    (name, job_id, seg)).fetchone():
            return                                          # this very line is already a sample
        db._conn.execute("INSERT INTO voice_samples (name, embedding, job_id, seg, created) VALUES (?,?,?,?,?)",
                         (name, json.dumps([round(x / n, 5) for x in emb]), job_id, seg, time.time()))
        while (db._conn.execute("SELECT COUNT(*) FROM voice_samples WHERE name=?", (name,)).fetchone()[0] > MAX_SAMPLES):
            big = db._conn.execute("""SELECT job_id FROM voice_samples WHERE name=? GROUP BY job_id
                                      ORDER BY COUNT(*) DESC LIMIT 1""", (name,)).fetchone()[0]
            db._conn.execute("""DELETE FROM voice_samples WHERE rowid = (SELECT rowid FROM voice_samples
                                WHERE name=? AND job_id IS ? ORDER BY created LIMIT 1)""", (name, big))
        db._conn.commit()


def top_up(jid: str, per_person: int = 8) -> int:
    """After a meeting's names are settled: take a few more samples of each named person from lines you confirmed
    (✓) or whole speakers you named - their longest, clearest lines, never two people talking at once."""
    from . import autotag
    job = db.get(jid)
    if not job or not job.get("audio_path"):
        return 0
    opts = job.get("options") or {}
    names = opts.get("speaker_names") or {}
    manual = {int(k) for k in (opts.get("manual_lines") or {})}
    auto_named = set(opts.get("speaker_auto") or [])
    tr = worker.load_transcript(jid) or {}
    segs = tr.get("segments", [])
    cache = worker.load_json(jid, "seg_emb.json", {}) or {}
    have = {}
    with db._lock:
        for r in db._conn.execute("SELECT name, seg FROM voice_samples WHERE job_id=?", (jid,)).fetchall():
            have.setdefault(r["name"], set()).add(r["seg"])
    added = 0
    for lab, who in names.items():
        if not who:
            continue
        sure = [i for i, s in enumerate(segs) if s.get("speaker") == lab and not s.get("also")
                and (i in manual or lab not in auto_named) and s["end"] - s["start"] >= 2.5]
        sure.sort(key=lambda i: -(segs[i]["end"] - segs[i]["start"]))
        for i in sure:
            if len(have.get(who, set())) >= per_person:
                break
            if i in have.get(who, set()):
                continue
            e = cache.get(str(i))
            if not e and os.path.exists(job["audio_path"]):
                s = segs[i]
                e = transcribe.embed_clip(transcribe.load_range(job["audio_path"], s["start"], min(s["end"], s["start"] + 20)),
                                          t=s["start"])
            if e:
                add_sample(who, e, jid, i)
                have.setdefault(who, set()).add(i)
                added += 1
    return added


def samples(name: str = "") -> dict[str, list]:
    with db._lock:
        rows = db._conn.execute("SELECT name, embedding FROM voice_samples" + (" WHERE name=?" if name else ""),
                                (name,) if name else ()).fetchall()
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r["name"], []).append(json.loads(r["embedding"]))
    return out


def forget(name: str) -> None:
    with db._lock:
        db._conn.execute("DELETE FROM voice_samples WHERE name=?", (name,))
        db._conn.execute("DELETE FROM people WHERE name=?", (name,))
        db._conn.commit()


def similarity(emb: list, centroid: list, samp: list) -> float:
    """How much a voice sounds like a person: their average voice, or (better) the 3 closest of their samples -
    a person sounds different on a call, across a room or when tired; the samples cover that."""
    best = voices._cos(emb, centroid) if centroid else -1.0
    if samp:
        sims = sorted((voices._cos(emb, s) for s in samp), reverse=True)[:3]
        best = max(best, sum(sims) / len(sims))
    return best


# ---------------------------------------------------------------- how they sound
def pitch_track(a: np.ndarray) -> np.ndarray:
    """F0 (Hz) of voiced 40 ms frames, by autocorrelation (70-400 Hz)."""
    win, hop = int(SR * 0.04), int(SR * 0.02)
    if len(a) < win * 3:
        return np.array([])
    rms_all = np.sqrt(np.mean(a ** 2) + 1e-12)
    lo, hi = int(SR / 400), int(SR / 70)
    out = []
    w = np.hanning(win)
    for i in range(0, len(a) - win, hop):
        x = a[i:i + win]
        if np.sqrt(np.mean(x ** 2)) < rms_all * 0.6:        # quiet / pause
            continue
        x = (x - x.mean()) * w
        f = np.fft.rfft(x, 2 * win)
        r = np.fft.irfft(f * np.conj(f))[:win]
        if r[0] <= 0:
            continue
        r = r / r[0]
        k = lo + int(np.argmax(r[lo:hi]))
        if r[k] > 0.45:
            out.append(SR / k)
    return np.array(out)


def voice_stats(clips: list[np.ndarray]) -> dict:
    f0 = np.concatenate([pitch_track(c) for c in clips if len(c)]) if clips else np.array([])
    loud = [20 * math.log10(float(np.sqrt(np.mean(c ** 2))) + 1e-9) for c in clips if len(c)]
    if len(f0) < 20:
        return {}
    p10, p50, p90 = np.percentile(f0, [10, 50, 90])
    semis = 12 * math.log2(p90 / p10) if p10 > 0 else 0
    return {"pitch_hz": round(float(p50)), "pitch_low_hz": round(float(p10)), "pitch_high_hz": round(float(p90)),
            "pitch_range": "wide" if semis > 9 else "narrow" if semis < 5 else "medium",
            "loudness_db": round(float(np.median(loud)), 1) if loud else None}


# ---------------------------------------------------------------- what they say
def _tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text or "")]


def their_lines(name: str) -> list[dict]:
    """Every line this person said (named or tagged as them), newest meetings first."""
    out = []
    low = name.lower()
    for j in db.list_jobs(100000, "done"):
        names = (j.get("options") or {}).get("speaker_names") or {}
        labs = {lab for lab, n in names.items() if (n or "").lower() == low}
        if not labs:
            continue
        tr = worker.load_transcript(j["id"]) or {}
        for i, s in enumerate(tr.get("segments", [])):
            if s.get("speaker") in labs and not s.get("also"):
                out.append({"job": j["id"], "title": j.get("title") or j["filename"], "when": j["created_at"],
                            "external": bool((j.get("options") or {}).get("external",
                                             (j.get("options") or {}).get("template") == "client")),
                            "i": i, "start": s["start"], "end": s["end"], "text": s.get("text", ""),
                            "words": len(s.get("words") or []) or len((s.get("text") or "").split()),
                            "audio": j.get("audio_path")})
    return out


_bg_cache: dict = {"at": 0, "counts": Counter(), "total": 1}


def _background() -> tuple[Counter, int]:
    """Word counts over all meetings (to find what is typical of one person, not of everyone)."""
    if time.time() - _bg_cache["at"] < 600:
        return _bg_cache["counts"], _bg_cache["total"]
    c = Counter()
    for j in db.list_jobs(100000, "done"):
        tr = worker.load_transcript(j["id"]) or {}
        for s in tr.get("segments", []):
            c.update(_tokens(s.get("text", "")))
    _bg_cache.update(at=time.time(), counts=c, total=sum(c.values()) or 1)
    return c, _bg_cache["total"]


def language_stats(lines: list[dict]) -> dict:
    from . import terms
    toks = [t for ln in lines for t in _tokens(ln["text"])]
    if not toks:
        return {}
    mine = Counter(toks)
    total = len(toks)
    bg, bg_total = _background()
    scored = []
    for w, k in mine.items():
        if k < 3 or w in STOP or len(w) < 3:
            continue
        others = max(bg[w] - k, 0) + 1
        scored.append((math.log((k / total) / (others / bg_total)) * math.log(k + 1), w))
    words = [w for _, w in sorted(scored, reverse=True)[:15]]
    bi = Counter()
    for ln in lines:
        t = [w for w in _tokens(ln["text"])]
        bi.update(f"{a} {b}" for a, b in zip(t, t[1:]) if a not in STOP and b not in STOP and len(a) > 2 and len(b) > 2)
    phrases = [p for p, k in bi.most_common(10) if k >= 2][:8]
    known = {t.lower(): t for t in terms.known_terms()}
    text_all = " ".join(ln["text"] for ln in lines).lower()
    used_terms = sorted(((text_all.count(k), v) for k, v in known.items() if len(k) > 2 and text_all.count(k)),
                        reverse=True)[:12]
    mix = {}
    for label, vocab in (("Singlish", PARTICLES), ("Malay", MALAY), ("Hokkien", HOKKIEN)):
        hits = Counter(t for t in toks if t in vocab)
        if label == "Singlish":
            hits.pop("can", None), hits.pop("cannot", None)
        if sum(hits.values()) >= 3:
            mix[label] = [w for w, _ in hits.most_common(5)]
    cjk = sum(1 for ln in lines if CJK.search(ln["text"]))
    if cjk >= 2:
        mix["Chinese"] = [f"{cjk} lines"]
    secs = sum(max(0.0, ln["end"] - ln["start"]) for ln in lines if ln["end"] - ln["start"] >= 2)
    wcount = sum(ln["words"] for ln in lines if ln["end"] - ln["start"] >= 2)
    wpm = round(wcount / (secs / 60)) if secs > 30 else None
    return {"words": words, "phrases": phrases, "terms": [t for _, t in used_terms], "mix": mix,
            "pace_wpm": wpm, "pace": None if not wpm else "fast" if wpm > 170 else "slow" if wpm < 120 else "steady",
            "avg_line_words": round(sum(ln["words"] for ln in lines) / len(lines), 1),
            "questions": round(100 * sum(1 for ln in lines if ln["text"].rstrip().endswith("?")) / len(lines))}


NOTE_PROMPT = """These are lines one person ({name}) said in work meetings, with a few numbers about how they speak.
Write a short profile in Markdown for colleagues, using only what the lines show:

### Usually raises and owns
- <2-4 bullets: topics, systems, clients, decisions they deal with>

### How they communicate
- <2-4 bullets: e.g. direct or detailed, asks questions, gives updates with numbers, uses certain terms or mixes languages>

### Working with them
- <1-3 bullets: what they need from others, how to bring things to them>

### Internal vs external meetings
- <only if lines are tagged both [internal] and [external]: how their role and way of speaking differs with
  vendors / partners / clients compared with the team; otherwise leave this section out>

Rules: only work and communication style. Never guess personality traits, emotions, health, age, gender, ethnicity,
religion or private life. Never invent facts. Be specific and short."""


def ai_note(name: str, lines: list[dict], lang: dict, voice: dict) -> str:
    if not summarize.enabled() or len(lines) < 8:
        return ""
    picked = sorted(lines, key=lambda ln: -ln["words"])[:80]
    picked.sort(key=lambda ln: (ln["when"], ln["start"]))
    text, size = [], 0
    for ln in picked:
        t = f"[{'external' if ln.get('external') else 'internal'}: {ln['title'][:40]}] {ln['text']}"
        if size + len(t) > 6000:
            break
        text.append(t)
        size += len(t)
    facts = {k: v for k, v in {**lang, **voice}.items() if v not in (None, [], {}, "")}
    msgs = [{"role": "system", "content": NOTE_PROMPT.format(name=name)},
            {"role": "user", "content": "Numbers: " + json.dumps(facts, ensure_ascii=False) + "\n\nLines:\n" + "\n".join(text)}]
    try:
        return summarize.strip_thinking(summarize.chat(msgs)).strip()
    except Exception as e:                                  # noqa: BLE001
        log.warning("profile note for %s: %s", name, e)
        return ""


# ---------------------------------------------------------------- profiles
def build(name: str, with_note: bool = True) -> dict:
    lines = their_lines(name)
    meetings = {}
    for ln in lines:
        m = meetings.setdefault(ln["job"], {"id": ln["job"], "title": ln["title"], "when": ln["when"], "seconds": 0})
        m["seconds"] += max(0.0, ln["end"] - ln["start"])
    clips = []
    for ln in sorted(lines, key=lambda x: -(x["end"] - x["start"]))[:24]:   # their longest lines
        if ln["audio"] and ln["end"] - ln["start"] >= 2:
            try:
                clips.append(transcribe.load_range(ln["audio"], ln["start"], min(ln["end"], ln["start"] + 15)))
            except Exception:                               # noqa: BLE001
                pass
    voice = voice_stats(clips)
    lang = language_stats(lines)
    old = get(name) or {}
    sig = f"{len(lines)}:{len(meetings)}"
    note, note_sig = old.get("note", ""), old.get("note_sig", "")
    grew = not note_sig or len(lines) >= 1.2 * int(note_sig.split(":")[0] or 0) or note_sig.split(":")[1] != str(len(meetings))
    if with_note and grew:
        note = ai_note(name, lines, lang, voice) or note
        note_sig = sig if note else note_sig
    longest = max((ln for ln in lines if ln["audio"]), key=lambda x: x["end"] - x["start"], default=None)
    ext = [m for m in meetings if any(ln["external"] for ln in lines if ln["job"] == m)]
    split = {"external_meetings": len(ext), "internal_meetings": len(meetings) - len(ext),
             "external_share": None, "internal_share": None}
    for kind, ids in (("external", set(ext)), ("internal", set(meetings) - set(ext))):
        mine = sum(meetings[i]["seconds"] for i in ids)
        allt = 0.0
        for i in ids:
            tr = worker.load_transcript(i) or {}
            allt += sum(max(0.0, s["end"] - s["start"]) for s in tr.get("segments", []) if s.get("speaker"))
        if allt > 0:
            split[f"{kind}_share"] = round(100 * mine / allt)
    prof = {"name": name, "lines": len(lines), "split": split,
            "sample": {"job": longest["job"], "start": longest["start"], "end": min(longest["end"], longest["start"] + 15),
                       "text": longest["text"][:200]} if longest else None, "meetings": sorted(meetings.values(), key=lambda m: -m["when"]),
            "talk_seconds": round(sum(m["seconds"] for m in meetings.values())),
            "last_seen": max((m["when"] for m in meetings.values()), default=None),
            "voice": voice, "language": lang, "samples": len(samples(name).get(name, [])),
            "note": note, "note_sig": note_sig, "sig": sig, "updated_at": int(time.time())}
    with db._lock:
        db._conn.execute("INSERT OR REPLACE INTO people (name, profile, updated_at) VALUES (?,?,?)",
                         (name, json.dumps(prof, ensure_ascii=False), time.time()))
        db._conn.commit()
    return prof


def get(name: str) -> dict | None:
    with db._lock:
        r = db._conn.execute("SELECT profile FROM people WHERE lower(name)=lower(?)", (name,)).fetchone()
    return json.loads(r["profile"]) if r else None


def all_names() -> list[str]:
    """Everyone with a voice or a name in a meeting."""
    names = {v["name"] for v in voices.list_voices()}
    for j in db.list_jobs(100000, "done"):
        names |= {n for n in ((j.get("options") or {}).get("speaker_names") or {}).values() if n}
    seen, out = set(), []
    for n in sorted(names, key=str.lower):
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def listing() -> list[dict]:
    out = []
    for n in all_names():
        p = get(n) or {}
        out.append({"name": n, "lines": p.get("lines", 0), "meetings": len(p.get("meetings", [])),
                    "talk_seconds": p.get("talk_seconds", 0), "last_seen": p.get("last_seen"),
                    "has_note": bool(p.get("note")), "pace": (p.get("language") or {}).get("pace"),
                    "pitch_hz": (p.get("voice") or {}).get("pitch_hz"), "built": bool(p)})
    return sorted(out, key=lambda x: -(x["talk_seconds"] or 0))


def context_for(names: list[str], limit: int = 1800) -> str:
    """Short profiles of these people, for the minutes writer and Ask."""
    parts = []
    for n in names:
        p = get(n)
        if not p:
            continue
        bits = []
        if p.get("note"):
            bits.append(re.sub(r"\s+", " ", re.sub(r"#+ ", "", p["note"]))[:450])
        lang = p.get("language") or {}
        if lang.get("terms"):
            bits.append("Terms they use: " + ", ".join(lang["terms"][:8]))
        if bits:
            parts.append(f"{n}: " + " ".join(bits))
    out = "\n".join(parts)
    return out[:limit]


def _sig(name: str) -> str:
    lines = their_lines(name)
    return f"{len(lines)}:{len({ln['job'] for ln in lines})}"


def _sweep() -> None:
    from . import autotag
    time.sleep(90)
    while True:
        try:
            if worker.current_job is None and db.next_queued() is None and not autotag._running:
                for j in db.list_jobs(200, "done"):         # meetings whose names changed: more voice samples
                    o = j.get("options") or {}
                    sig = json.dumps([o.get("speaker_names") or {}, len(o.get("manual_lines") or {})], sort_keys=True)
                    if o.get("speaker_names") and o.get("samples_sig") != sig:
                        n = top_up(j["id"])
                        with autotag.job_lock(j["id"]):
                            o2 = dict((db.get(j["id"]) or {}).get("options") or {})
                            o2["samples_sig"] = sig
                            db.update(j["id"], options=o2)
                        if n:
                            log.info("kept %d more voice samples from %s", n, j["id"])
                            break
                for n in all_names():
                    p = get(n)
                    if not p or p.get("sig") != _sig(n):
                        log.info("updating the profile of %s", n)
                        build(n)
                        break                               # one person per pass: keep the server free
        except Exception as e:                              # noqa: BLE001
            log.warning("people sweep: %s", e)
        time.sleep(120)


def start_sweeper() -> None:
    if settings.people_sweep:
        threading.Thread(target=_sweep, daemon=True).start()
