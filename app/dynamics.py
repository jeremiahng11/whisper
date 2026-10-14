"""How a meeting ran: who talked how much, who answered whom, interruptions, questions, who led - and, with your own
LLM, a short read of the dynamics (decisions, who drove them, open tensions, how the team handled an external
party). Observable behaviour only: no judgements of character, mood or ability."""
import json
import logging
import re
import time

from . import db, people, render, summarize, worker

log = logging.getLogger("dynamics")


def is_external(job: dict) -> bool:
    o = job.get("options") or {}
    if o.get("external") is not None:
        return bool(o["external"])
    return o.get("template") == "client"


def stats(job: dict, tr: dict) -> dict:
    names = render.names_of(job)
    segs = [s for s in tr.get("segments", []) if s.get("speaker")]
    who = lambda lab: names.get(lab, lab)                    # noqa: E731
    per: dict[str, dict] = {}
    turns = []                                               # (speaker, start, end, text)
    for s in segs:
        w = who(s["speaker"])
        if turns and turns[-1][0] == w and s["start"] - turns[-1][2] < 2.5:
            turns[-1] = (w, turns[-1][1], s["end"], turns[-1][3] + " " + s.get("text", ""))
        else:
            turns.append((w, s["start"], s["end"], s.get("text", "")))
        p = per.setdefault(w, {"name": w, "seconds": 0.0, "lines": 0, "questions": 0, "interrupts": 0, "overlaps": 0,
                               "turns": 0, "longest": 0.0})
        p["seconds"] += max(0.0, s["end"] - s["start"])
        p["lines"] += 1
        p["questions"] += s.get("text", "").count("?")
        if s.get("also"):
            p["overlaps"] += 1
            for a in s["also"]:
                q = per.setdefault(who(a), {"name": who(a), "seconds": 0.0, "lines": 0, "questions": 0, "interrupts": 0,
                                            "overlaps": 0, "turns": 0, "longest": 0.0})
                q["overlaps"] += 1
    replies: dict[str, dict] = {}
    for (a, a0, a1, _), (b, b0, b1, _) in zip(turns, turns[1:]):
        if a != b:
            replies.setdefault(b, {}).setdefault(a, 0)
            replies[b][a] += 1
            if b0 < a1 - 0.3:                                # started before the other had finished
                per[b]["interrupts"] += 1
    for w, t0, t1, _ in turns:
        per[w]["turns"] += 1
        per[w]["longest"] = max(per[w]["longest"], t1 - t0)
    total = sum(p["seconds"] for p in per.values()) or 1
    dur = float(tr.get("duration") or (segs[-1]["end"] if segs else 0)) or 1
    for p in per.values():
        p["share"] = round(100 * p["seconds"] / total)
        p["seconds"] = round(p["seconds"])
        p["longest"] = round(p["longest"])
        p["avg_turn"] = round(p["seconds"] / p["turns"]) if p["turns"] else 0
    known = {n.lower() for n in people.all_names()}
    ppl = sorted(per.values(), key=lambda p: -p["seconds"])
    for p in ppl:
        p["known"] = p["name"].lower() in known and not re.match(r"^Speaker \d+$", p["name"])
    pairs = sorted(((k, b, a) for b, d in replies.items() for a, k in d.items()), reverse=True)[:8]
    talk = sum(max(0.0, s["end"] - s["start"]) for s in segs)
    return {"people": ppl, "turns": len(turns), "replies": [{"from": b, "to": a, "times": k} for k, b, a in pairs],
            "opened_by": turns[0][0] if turns else "", "closed_by": turns[-1][0] if turns else "",
            "silence_pct": max(0, round(100 * (1 - talk / dur))), "minutes": round(dur / 60),
            "external": is_external(job), "turns_per_min": round(len(turns) / max(1.0, dur / 60), 1)}


DYN_PROMPT = """You get numbers about how a {kind} meeting ran and an excerpt of what was said. Write in Markdown:

### How the meeting ran
- <2-4 bullets: who led and set the agenda, how discussion flowed, how decisions were reached>

### Between the people
- <2-4 bullets: who answered whom, who asked the questions, who drove which topics, where views differed>
{external}
### Worth noting
- <1-3 bullets: open disagreements, things left hanging, someone who had little chance to speak>

Rules: only what the numbers and the text show. Describe behaviour, never character, mood, ability, health, age,
gender or background. Be specific and short; use people's names."""
EXTERNAL = """
### With the other party
- <2-3 bullets: who spoke for our side, what the other party pushed for, how the team responded, what was agreed>
"""


def note(job: dict, tr: dict, st: dict) -> str:
    if not summarize.enabled():
        return ""
    named = render.apply_names(tr, render.names_of(job))
    text = render.transcript_for_llm(named)
    if len(text) > 9000:
        text = text[:4500] + "\n[...]\n" + text[-4500:]
    summ = worker.load_summary(job["id"]) or ""
    msgs = [{"role": "system", "content": DYN_PROMPT.format(kind="external (vendor / partner / client)" if st["external"]
                                                            else "internal team", external=EXTERNAL if st["external"] else "")},
            {"role": "user", "content": "Numbers: " + json.dumps(st, ensure_ascii=False) + "\n\nMinutes:\n" + summ[:3000]
             + "\n\nTranscript excerpt:\n" + text}]
    try:
        return summarize.strip_thinking(summarize.chat(msgs)).strip()
    except Exception as e:                                   # noqa: BLE001
        log.warning("dynamics note for %s: %s", job["id"], e)
        return ""


_writing: set[str] = set()


def _sig(job: dict, tr: dict, st: dict) -> str:
    return json.dumps([render.names_of(job), st["external"], len(tr.get("segments", []))], sort_keys=True)


def write_note(jid: str) -> None:
    """Write (or rewrite) the AI read of the meeting, in the background."""
    import threading
    if jid in _writing:
        return
    _writing.add(jid)

    def run():
        try:
            job = db.get(jid)
            tr = worker.load_transcript(jid) or {}
            st = stats(job, tr)
            n = note(job, tr, st)
            if n:
                worker._write(jid, "dynamics.json", json.dumps({"note": n, "sig": _sig(job, tr, st), "at": int(time.time())},
                                                               ensure_ascii=False))
        finally:
            _writing.discard(jid)
    threading.Thread(target=run, daemon=True).start()


def get(jid: str) -> dict:
    job = db.get(jid)
    tr = worker.load_transcript(jid) or {}
    st = stats(job, tr)
    cached = worker.load_json(jid, "dynamics.json", {}) or {}
    st["note"] = cached.get("note", "")
    st["note_at"] = cached.get("at")
    st["note_stale"] = bool(cached.get("note")) and cached.get("sig") != _sig(job, tr, st)
    st["writing"] = jid in _writing
    st["llm"] = summarize.enabled()
    return st


def _m(sec: float) -> str:
    sec = int(sec or 0)
    return f"{sec // 60} min {sec % 60:02d} s" if sec >= 60 else f"{sec} s"


def markdown(jid: str) -> str:
    """The Oracle AI analysis as a Markdown section, for exports below the minutes."""
    d = get(jid)
    if not d["people"]:
        return ""
    kind = "with a vendor, partner or client" if d["external"] else "with the team only"
    out = ["## Oracle AI Analysis", "",
           f"*{d['minutes']} min meeting {kind}; {len(d['people'])} people talking, {d['turns']} turns "
           f"({d['turns_per_min']} a minute), {d['silence_pct']}% quiet. Opened by {d['opened_by'] or '-'}, "
           f"closed by {d['closed_by'] or '-'}.*", "", "### Who took part", ""]
    for p in d["people"]:
        bits = [f"{p['share']}% of the talking", f"{p['turns']} turns (average {p['avg_turn']} s, longest {_m(p['longest'])})"]
        if p["questions"]:
            bits.append(f"{p['questions']} question{'s' if p['questions'] > 1 else ''}")
        if p["interrupts"]:
            bits.append(f"cut in {p['interrupts']} time{'s' if p['interrupts'] > 1 else ''}")
        out.append(f"- **{p['name']}:** " + ", ".join(bits))
    if d["replies"]:
        out += ["", "### Who answered whom", ""]
        out += [f"- {r['from']} answered {r['to']} {r['times']} time{'s' if r['times'] > 1 else ''}" for r in d["replies"]]
    if d.get("note"):
        note = re.sub(r"^###\s+", "### ", d["note"].strip(), flags=re.M)
        out += ["", note]
    out.append("")
    return "\n".join(out)


def with_analysis(md: str, jid: str) -> str:
    """Minutes + the analysis section, placed below the minutes and above the transcript."""
    a = markdown(jid)
    if not a:
        return md
    i = md.find("\n## Transcript")
    return (md[:i] + "\n" + a + md[i:]) if i >= 0 else (md.rstrip() + "\n\n" + a)
