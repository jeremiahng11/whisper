"""Company knowledge base: short facts (question + answer) learned from all your meetings and people.

Each finished meeting is read once by the AI (again when its minutes change) and turned into up to ~25 facts:
  {"q": "When does the Idea Saver V2 ship?", "a": "End of November, after the e-ink samples arrive.",
   "kind": "date", "topic": "Idea Saver V2", "alts": ["Idea Saver V2 ship date", ...], "job": ..., "date": ...}
People with a profile add "Who is X?". A newer meeting's answer to the same question replaces the older one.

What uses it:
- Ask (scope "meetings"): the facts matching your question go along with the meeting excerpts.
- The watch answer pack (watchpack.py): ~150 short spoken questions the watch can answer offline.
- The watch's own small AI model (watchlm.py): trained on these facts and their other wordings.

Kept in DATA_DIR/knowledge/ (meetings/<job id>.json per meeting, kb.json merged).
"""
import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path

from . import db, summarize, worker
from .config import settings

log = logging.getLogger("knowledge")
KINDS = ("decision", "action", "date", "fact", "person", "term", "number")
MAX_PER_MEETING = 25
_lock = threading.RLock()
status = {"state": "idle", "done": 0, "total": 0, "error": "", "updated": 0.0}

PROMPT = """From these meeting minutes, write the facts someone in the company may later ask about.
Return ONLY a JSON array (at most {n} items) of objects with these keys:
  "q": a short question a person would say out loud (3-8 words, plain words, no symbols), e.g. "When is the V2 launch",
  "a": the answer in one short sentence (at most 25 words), with names, dates and numbers from the minutes,
  "kind": one of decision, action, date, fact, person, term, number,
  "topic": 1-4 words naming what it is about,
  "alts": 2-3 other ways to ask the same question (short).
Only facts that are clearly stated in the minutes; skip small talk. Write dates as words ("15 October 2026").
Meeting: "{title}" on {date}.

Minutes:
\"\"\"
{minutes}
\"\"\""""


def kdir() -> Path:
    d = settings.data_dir / "knowledge" / "meetings"
    d.mkdir(parents=True, exist_ok=True)
    return d.parent


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", q.lower())).strip()


def _clean_text(s, limit: int) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()[:limit]


def parse_facts(text: str) -> list[dict]:
    """The AI's JSON array -> clean facts (tolerates code fences, chatter and small mistakes)."""
    text = summarize.strip_thinking(text or "")
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except ValueError:
        try:                                    # trailing commas, the most common slip
            items = json.loads(re.sub(r",\s*([\]}])", r"\1", m.group(0)))
        except ValueError:
            return []
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        q, a = _clean_text(it.get("q"), 120).rstrip("?").strip(), _clean_text(it.get("a"), 300)
        if len(_norm(q)) < 4 or len(a) < 2:
            continue
        kind = str(it.get("kind", "fact")).lower()
        alts = [_clean_text(x, 120).rstrip("?") for x in (it.get("alts") or []) if isinstance(x, str)]
        out.append({"q": q, "a": a, "kind": kind if kind in KINDS else "fact", "topic": _clean_text(it.get("topic"), 60),
                    "alts": [x for x in alts if len(_norm(x)) >= 4][:4]})
    return out[:MAX_PER_MEETING]


def _meeting_sig(j: dict, minutes: str) -> str:
    return hashlib.sha1((minutes + "|" + (j.get("title") or "")).encode()).hexdigest()[:12]


def _mpath(jid: str) -> Path:
    return kdir() / "meetings" / f"{jid}.json"


def meeting_facts(jid: str) -> list[dict]:
    try:
        return json.loads(_mpath(jid).read_text()).get("facts", [])
    except (OSError, ValueError):
        return []


def needs_update(j: dict) -> bool:
    minutes = worker.load_summary(j["id"]) or ""
    if len(minutes.strip()) < 40:
        return False
    try:
        old = json.loads(_mpath(j["id"]).read_text())
    except (OSError, ValueError):
        return True
    return old.get("sig") != _meeting_sig(j, minutes)


def learn_meeting(jid: str) -> int:
    """Read one meeting's minutes and keep its facts. Returns how many."""
    j = db.get(jid)
    if not j or j["status"] != "done":
        return 0
    minutes = worker.load_summary(jid) or ""
    if len(minutes.strip()) < 40:
        return 0
    date = worker.meeting_date(j)
    prompt = PROMPT.format(n=MAX_PER_MEETING, title=j.get("title") or j["filename"], date=date, minutes=minutes[:12000])
    facts = parse_facts(summarize.chat([{"role": "system", "content": "You extract facts from meeting minutes as JSON."},
                                        {"role": "user", "content": prompt}]))
    for f in facts:
        f.update(job=jid, date=date, meeting=j.get("title") or "")
    tmp = _mpath(jid).with_suffix(".tmp")
    tmp.write_text(json.dumps({"sig": _meeting_sig(j, minutes), "facts": facts}, ensure_ascii=False))
    tmp.replace(_mpath(jid))
    rebuild()
    return len(facts)


def forget_meeting(jid: str) -> None:
    try:
        _mpath(jid).unlink()
    except OSError:
        pass
    rebuild()


def _people_facts() -> list[dict]:
    from . import people
    out = []
    for n in people.all_names():
        p = people.get(n) or {}
        note = re.sub(r"\s+", " ", re.sub(r"[#*_>]+", " ", p.get("note") or "")).strip()
        if not note:
            continue
        first = re.split(r"(?<=[.!?])\s", note)[0][:240]
        last = p.get("last_seen") or ""
        out.append({"q": f"Who is {n}", "a": first, "kind": "person", "topic": n,
                    "alts": [f"Tell me about {n}", f"What does {n} do"], "job": "", "date": str(last)[:10], "meeting": ""})
    return out


def rebuild() -> list[dict]:
    """Merge every meeting's facts (newest answer wins for the same question) + people -> kb.json."""
    with _lock:
        jobs = {j["id"]: j for j in db.list_jobs(100000, "done")}
        merged: dict[str, dict] = {}
        for p in sorted((kdir() / "meetings").glob("*.json")):
            if p.stem not in jobs:                  # meeting deleted
                continue
            for f in meeting_facts(p.stem):
                k = _norm(f["q"])
                if k not in merged or (f.get("date") or "") >= (merged[k].get("date") or ""):
                    merged[k] = f
        for f in _people_facts():
            merged.setdefault(_norm(f["q"]), f)
        ov = overrides()
        out = []
        for f in sorted(merged.values(), key=lambda f: (f.get("date") or ""), reverse=True):
            f["id"] = fact_id(f["q"])
            o = ov.get(f["id"]) or {}
            if o.get("hidden"):
                continue
            if o.get("a"):
                f["a"], f["edited"] = o["a"], True
            out.append(f)
        facts = out
        (kdir() / "kb.json").write_text(json.dumps(facts, ensure_ascii=False))
        status["updated"] = time.time()
        return facts


def fact_id(q: str) -> str:
    return hashlib.sha1(_norm(q).encode()).hexdigest()[:8]


def overrides() -> dict:
    """Your corrections: {fact id: {"a": "better answer"} or {"hidden": true}} - kept when meetings are re-read."""
    try:
        return json.loads((kdir() / "overrides.json").read_text())
    except (OSError, ValueError):
        return {}


def set_override(fid: str, answer: str | None = None, hidden: bool = False) -> None:
    with _lock:
        ov = overrides()
        if hidden:
            ov[fid] = {"hidden": True}
        elif answer:
            ov[fid] = {"a": _clean_text(answer, 300)}
        else:
            ov.pop(fid, None)
        (kdir() / "overrides.json").write_text(json.dumps(ov, ensure_ascii=False))
    rebuild()


def facts() -> list[dict]:
    try:
        return json.loads((kdir() / "kb.json").read_text())
    except (OSError, ValueError):
        return []


_STOP = set("a an the is are was were be to of in on for and or what who when where why how do does did we our i you "
            "my it its this that with at by from about tell me please can could".split())


def _words(s: str) -> set[str]:
    return {w for w in _norm(s).split() if w not in _STOP and len(w) > 1}


def find(q: str, limit: int = 12) -> list[dict]:
    """Facts that share the most words with the question (question, other wordings, topic and answer)."""
    want = _words(q)
    if not want:
        return []
    scored = []
    for f in facts():
        ask = _words(f["q"] + " " + " ".join(f.get("alts") or []) + " " + (f.get("topic") or ""))
        ans = _words(f["a"])
        s = 2 * len(want & ask) + len(want & ans)
        if s:
            scored.append((s, f.get("date") or "", f))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [f for _, _, f in scored[:limit]]


def context_for(q: str, limit: int = 2500) -> str:
    hits = find(q)
    if not hits:
        return ""
    lines = [f"- {f['q']}? {f['a']}" + (f" ({f['meeting']}, {f['date']})" if f.get("meeting") else "") for f in hits]
    return ("Facts from the company knowledge base (learned from all meetings):\n" + "\n".join(lines))[:limit]


def summary() -> dict:
    fs = facts()
    kinds: dict[str, int] = {}
    for f in fs:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    learned = len(list((kdir() / "meetings").glob("*.json")))
    return {"facts": len(fs), "kinds": kinds, "meetings_read": learned, **status}


def learn_all(force: bool = False) -> None:
    """Read every meeting that's new or changed (one after another, in the background)."""
    with _lock:
        if status["state"] == "learning":
            return
        status.update(state="learning", done=0, error="")
    try:
        todo = [j for j in db.list_jobs(100000, "done") if force or needs_update(j)]
        status["total"] = len(todo)
        for j in todo:
            try:
                learn_meeting(j["id"])
            except Exception as e:                  # noqa: BLE001 - one bad meeting must not stop the rest
                status["error"] = summarize.explain_llm_error(e)
                log.warning("knowledge: %s: %s", j["id"], e)
            status["done"] += 1
        rebuild()
    finally:
        status["state"] = "idle"


def start_learning(force: bool = False) -> bool:
    if status["state"] == "learning" or not summarize.enabled():
        return False
    threading.Thread(target=learn_all, args=(force,), daemon=True).start()
    return True


def _sweep() -> None:
    time.sleep(180)
    while True:
        try:
            if summarize.enabled() and worker.current_job is None and db.next_queued() is None and status["state"] == "idle":
                for j in db.list_jobs(200, "done"):  # one new or changed meeting per pass: keep the server free
                    if needs_update(j):
                        log.info("learning facts from %s", j["id"])
                        learn_meeting(j["id"])
                        break
        except Exception as e:                      # noqa: BLE001
            log.warning("knowledge sweep: %s", e)
        time.sleep(150)


def start_sweeper() -> None:
    if settings.knowledge_sweep:
        threading.Thread(target=_sweep, daemon=True).start()
