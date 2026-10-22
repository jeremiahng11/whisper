"""Company knowledge base: short facts (question + answer) learned from all your meetings and people.

Each finished meeting is read once by the AI (again when its minutes change) and turned into up to ~25 facts:
  {"q": "When does the Idea Saver V2 ship?", "a": "End of November, after the e-ink samples arrive.",
   "kind": "date", "topic": "Idea Saver V2", "alts": ["Idea Saver V2 ship date", ...], "job": ..., "date": ...}
People with a profile add "Who is X?". A newer meeting's answer to the same question replaces the older one.

What uses it:
- Ask (scope "meetings"): the facts matching your question go along with the meeting excerpts.
- The watch answer pack (watchpack.py): ~150 short spoken questions the watch can answer offline.
- The watch's own small AI model (watchlm.py): trained on these facts and their other wordings.

Also learned from your own sources (sources.py): web pages (a whole site section if you like), product sheets,
price lists, FAQs - PDF, Word, text or pasted. Meeting facts win over a source's facts for the same question.

Kept in DATA_DIR/knowledge/ (meetings/<job id>.json per meeting, sources/, kb.json merged).
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
KINDS = ("decision", "action", "date", "fact", "person", "term", "number", "product", "price", "policy", "news")
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


DOC_PROMPT = """From this company document, write the facts someone in the company may later ask about:
products and what they do, prices, specifications, policies, warranty, contacts, opening hours, how-tos.
Return ONLY a JSON array (at most {n} items) of objects with these keys:
  "q": a short question a person would say out loud (3-8 words, plain words, no symbols), e.g. "How much is the Round watch",
  "a": the answer in one short sentence (at most 25 words), with the names, numbers and prices from the document,
  "kind": one of product, price, policy, fact, date, person, term, number,
  "topic": 1-4 words naming what it is about (e.g. the product name),
  "alts": 2-3 other ways to ask the same question (short).
Only facts clearly stated in the document; skip menus, cookie notices and legal boilerplate.
Source: "{title}" ({where}).

Document:
\"\"\"
{text}
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


WHO_PROMPT = """Write who {name} is, for a colleague asking "Who is {name}?" - 1 or 2 short sentences: their full
name if given, their title or role, the company, and what they are responsible for. Use ONLY the material below
(company documents, website pages, facts from meetings). If the material doesn't say their role, say what they do or
talk about in meetings. No guessing, no "based on the material". Plain text.

Material:
\"\"\"
{material}
\"\"\""""


def _who_dir() -> Path:
    d = kdir() / "people"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _who_path(name: str) -> Path:
    return _who_dir() / (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".json")


def person_material(name: str, limit: int = 6000) -> str:
    """Everything known about this person: lines of your documents and pages that name them (e.g. a leadership page:
    "Ryan Tan - Founder, Chairman and CEO"), facts that mention them, and their meeting profile."""
    from . import people, sources
    pat = re.compile(r"\b" + re.escape(name) + r"\b", re.I)
    parts = []
    for m in sources.listing():
        try:
            txt = (sources.sdir() / f"{m['id']}.txt").read_text()
        except OSError:
            continue
        lines = txt.splitlines()
        for i, ln in enumerate(lines):
            if pat.search(ln):
                parts.append(" ".join(x.strip() for x in lines[max(0, i - 1):i + 3] if x.strip())[:500])
    for f in facts():
        if f.get("kind") != "person" and (pat.search(f["q"]) or pat.search(f["a"])):
            parts.append(f"{f['q']}? {f['a']}")
    p = people.get(name) or {}
    if p.get("note"):
        parts.append("From meetings: " + re.sub(r"\s+", " ", re.sub(r"[#*_>]+", " ", p["note"]))[:800])
    out, size = [], 0
    for x in dict.fromkeys(parts):
        if size + len(x) > limit:
            break
        out.append(x)
        size += len(x)
    return "\n".join(out)


def who_is(name: str, refresh: bool = False) -> str:
    """The answer to "Who is <name>?" written by the AI from person_material (kept until that material changes)."""
    mat = person_material(name)
    if not mat:
        return ""
    sig = hashlib.sha1(mat.encode()).hexdigest()[:12]
    p = _who_path(name)
    try:
        old = json.loads(p.read_text())
        if old.get("sig") == sig and not refresh:
            return old.get("a", "")
    except (OSError, ValueError):
        if not refresh:
            return ""                               # not written yet: the sweep does it (it calls the AI)
    a = _clean_text(summarize.strip_thinking(summarize.chat([
        {"role": "system", "content": "You write short, factual descriptions of people at a company."},
        {"role": "user", "content": WHO_PROMPT.format(name=name, material=mat)}])), 300)
    p.write_text(json.dumps({"sig": sig, "a": a}, ensure_ascii=False))
    return a


def who_needs_update(name: str) -> bool:
    mat = person_material(name)
    if not mat:
        return False
    try:
        return json.loads(_who_path(name).read_text()).get("sig") != hashlib.sha1(mat.encode()).hexdigest()[:12]
    except (OSError, ValueError):
        return True


def _people_facts() -> list[dict]:
    from . import people
    out = []
    for n in people.all_names():
        p = people.get(n) or {}
        try:
            written = json.loads(_who_path(n).read_text()).get("a", "")
        except (OSError, ValueError):
            written = ""
        note = re.sub(r"\s+", " ", re.sub(r"[#*_>]+", " ", p.get("note") or "")).strip()
        a = written or (re.split(r"(?<=[.!?])\s", note)[0][:240] if note else "")
        if not a:
            continue
        last = p.get("last_seen") or ""
        out.append({"q": f"Who is {n}", "a": a, "kind": "person", "topic": n,
                    "alts": [f"Tell me about {n}", f"What does {n} do", f"What is {n}'s role"], "job": "",
                    "date": str(last)[:10], "meeting": ""})
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
        from . import sources
        for f in sources.all_facts():                # your documents and pages: meetings win for the same question
            merged.setdefault(_norm(f["q"]), f)
        for f in _people_facts():
            merged.setdefault(_norm(f["q"]), f)
        from . import watchfacts                    # more ways to ask: people's words, last meeting, your tasks...
        for f in watchfacts.derived():
            merged.setdefault(_norm(f["q"]), f)
        from . import core                          # the company profile (core.py): wins for the same question
        for f in core.core_facts():
            merged[_norm(f["q"])] = f
        for f in custom_facts():                    # your own questions win over everything
            merged[_norm(f["q"])] = f
        try:
            from . import research
            d = research.load()
            watchfacts.company_alts(list(merged.values()), (d.get("topics") or d.get("found") or [""])[0])
        except Exception as e:                      # noqa: BLE001
            log.warning("company questions: %s", e)
        ov = overrides()
        byq = _asked_by_fact()
        out = []
        for f in sorted(merged.values(), key=lambda f: (f.get("date") or ""), reverse=True):
            f["id"] = fact_id(f["q"])
            o = ov.get(f["id"]) or {}
            if o.get("hidden"):
                continue
            if o.get("a"):
                f["a"], f["edited"] = o["a"], True
            if o.get("alts"):
                have = {_norm(x) for x in [f["q"]] + (f.get("alts") or [])}
                f["alts"] = (f.get("alts") or []) + [x for x in o["alts"] if _norm(x) not in have]
                f["edited"] = True
            if o.get("pinned"):                     # "Pin to watch": treated like the company profile
                f["core"] = f["pinned"] = True
            a = byq.get(f["id"])
            if a:                                   # your own wordings of this question, and how often you ask it
                have = {_norm(x) for x in [f["q"]] + (f.get("alts") or [])}
                mine = [w for w in a["wordings"] if _norm(w) not in have][-4:]
                f["alts"] = (f.get("alts") or []) + mine
                f["asked"] = a["n"]
            out.append(f)
        facts = out
        tmp = kdir() / f"kb.{threading.get_ident()}.tmp"     # readers never see half a file
        tmp.write_text(json.dumps(facts, ensure_ascii=False))
        tmp.replace(kdir() / "kb.json")
        status["updated"] = time.time()
        return facts


def fact_id(q: str) -> str:
    return hashlib.sha1(_norm(q).encode()).hexdigest()[:8]


# ---------------------------------------------------------------- your own questions (Watch AI > Questions on the watch)
def custom_facts() -> list[dict]:
    """Questions you added yourself: always on the watch and answered there, like the company essentials."""
    try:
        rows = json.loads((kdir() / "custom.json").read_text())
    except (OSError, ValueError):
        rows = []
    return [{"q": r["q"], "a": r["a"], "alts": r.get("alts") or [], "kind": "custom", "topic": "", "job": "",
             "date": r.get("date") or "", "meeting": "Added by you", "core": True, "custom": True} for r in rows if r.get("q")]


def add_custom(q: str, a: str, alts: list[str] | None = None) -> str:
    q = _clean_text(q, 160).rstrip("?").strip()
    a = _clean_text(a, 300)
    if not q or not a:
        raise ValueError("a question and an answer are needed")
    alts = [_clean_text(x, 160).rstrip("?").strip() for x in (alts or []) if x and x.strip()][:6]
    with _lock:
        try:
            rows = json.loads((kdir() / "custom.json").read_text())
        except (OSError, ValueError):
            rows = []
        rows = [r for r in rows if _norm(r["q"]) != _norm(q)]
        rows.append({"q": q, "a": a, "alts": alts, "date": time.strftime("%Y-%m-%d")})
        (kdir() / "custom.json").write_text(json.dumps(rows, ensure_ascii=False))
    rebuild()
    return fact_id(q)


def delete_custom(fid: str) -> bool:
    with _lock:
        try:
            rows = json.loads((kdir() / "custom.json").read_text())
        except (OSError, ValueError):
            rows = []
        keep = [r for r in rows if fact_id(r["q"]) != fid]
        if len(keep) == len(rows):
            return False
        (kdir() / "custom.json").write_text(json.dumps(keep, ensure_ascii=False))
    rebuild()
    return True


def overrides() -> dict:
    """Your corrections: {fact id: {"a": "better answer"} or {"hidden": true}} - kept when meetings are re-read."""
    try:
        return json.loads((kdir() / "overrides.json").read_text())
    except (OSError, ValueError):
        return {}


def set_override(fid: str, answer: str | None = None, hidden: bool = False, pinned: bool | None = None,
                 alts: list[str] | None = None) -> None:
    """pinned=True/False changes only the pin; otherwise: hide it, give the right answer, or (neither) back to what
    the AI read (a pin stays)."""
    with _lock:
        ov = overrides()
        o = dict(ov.get(fid) or {})
        if pinned is not None:
            if pinned:
                o["pinned"] = True
                o.pop("hidden", None)
            else:
                o.pop("pinned", None)
        elif hidden:
            o = {"hidden": True}
        elif answer or alts is not None:
            if answer:
                o["a"] = _clean_text(answer, 300)
            if alts is not None:                    # more ways you ask it (the watch hears these too)
                o["alts"] = [_clean_text(x, 160).rstrip("?").strip() for x in alts if x and x.strip()][:6]
                if not o["alts"]:
                    o.pop("alts")
            o.pop("hidden", None)
        else:
            o = {"pinned": True} if o.get("pinned") else {}
        if o:
            ov[fid] = o
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


def _stem(w: str) -> str:
    """products -> product, services -> service (so "what product" finds "What products ...")."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _words(s: str) -> set[str]:
    return {_stem(w) for w in _norm(s).split() if w not in _STOP and len(w) > 1}


def best_answer(q: str) -> dict | None:
    """The fact that clearly answers the question (Find a fact shows it as the answer), or None."""
    hits = find_scored(q, 2)
    if not hits:
        return None
    top, f = hits[0]
    second = hits[1][0] if len(hits) > 1 else 0
    want = _words(q)
    ask = _words(f["q"] + " " + " ".join(f.get("alts") or []) + " " + (f.get("topic") or ""))
    if top >= 2 and top > second and len(want & ask) * 2 >= len(want):
        return f
    return None


def find(q: str, limit: int = 12) -> list[dict]:
    """Facts that share the most words with the question (question, other wordings, topic and answer)."""
    return [f for _, f in find_scored(q, limit)]


_vocab_cache: dict = {"at": None, "words": set(), "sounds": {}}


def _kb_words(fs: list[dict]) -> tuple[set, dict]:
    """Every word the knowledge base uses (and a sound key for each), rebuilt when kb.json changes."""
    from . import terms
    try:
        at = (kdir() / "kb.json").stat().st_mtime
    except OSError:
        at = None
    if _vocab_cache["at"] != at or not _vocab_cache["words"]:
        words = set()
        for f in fs:
            words |= _words(f["q"] + " " + " ".join(f.get("alts") or []) + " " + (f.get("topic") or "") + " " + f["a"])
        sounds: dict[str, str] = {}
        for w in words:
            if len(w) >= 4:
                sounds.setdefault(terms.sound(w), w)
        _vocab_cache.update(at=at, words=words, sounds=sounds)
    return _vocab_cache["words"], _vocab_cache["sounds"]


def _heard_as_known(want: set, fs: list[dict]) -> set:
    """Speech-to-text writes names the way they sound ("Alita Planet", "Prisila"): a question word the knowledge base
    doesn't know is read as the known word that sounds the same or is spelt almost the same."""
    import difflib
    from . import terms
    words, sounds = _kb_words(fs)
    out = set()
    for w in want:
        if w in words or len(w) < 4:
            out.add(w)
            continue
        k = sounds.get(terms.sound(w))
        if not k:
            close = difflib.get_close_matches(w, words, n=1, cutoff=0.8)
            k = close[0] if close else None
        out.add(k or w)
    return out


def find_scored(q: str, limit: int = 12) -> list[tuple[int, dict]]:
    want = _words(q)
    if not want:
        return []
    fs = facts()
    want = _heard_as_known(want, fs)
    scored = []
    for f in fs:
        ask = _words(f["q"] + " " + " ".join(f.get("alts") or []) + " " + (f.get("topic") or ""))
        ans = _words(f["a"])
        s = 2 * len(want & ask) + len(want & ans)
        if s and f.get("kind") == "person" and f.get("topic") and _words(f["topic"]) & want:
            # "ryan gwee" / "who is ryan": that person's own fact first (all of the name, or a first name)
            s += 3 if _words(f["topic"]) <= want else 2
        if s:
            scored.append((s, f.get("date") or "", f))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [(sc, f) for sc, _, f in scored[:limit]]


# ---------------------------------------------------------------- the questions you actually ask
# Every question asked (Ask on the web or the watch, and the watch's offline questions) is kept. When it clearly
# matches one fact (most of its words, and well ahead of the next best), your wording joins that fact's other
# wordings: the watch model learns how you ask, the pack hears it, and the most-asked facts come first.
MAX_ASKED = 3000


def _asked_path() -> Path:
    return kdir() / "asked.json"


def asked() -> list[dict]:
    try:
        return json.loads(_asked_path().read_text())
    except (OSError, ValueError):
        return []


def link_question(q: str) -> str:
    """The fact this question is about, if that's clear ("" if not)."""
    words = _words(q)
    if len(words) < 2:
        return ""
    hits = find_scored(q, 2)
    if not hits:
        return ""
    top, f = hits[0]
    shared = len(words & _words(f["q"] + " " + " ".join(f.get("alts") or []) + " " + (f.get("topic") or "")))
    second = hits[1][0] if len(hits) > 1 else 0
    if shared >= 2 and shared * 2 >= len(words) and top >= second + 2:
        return f["id"]
    return ""


def note_question(q: str, where: str = "ask", fid: str = "") -> str:
    """Remember a question you asked; returns the fact id it was linked to ("" if none)."""
    q = _clean_text(q, 160).rstrip("?").strip()
    if len(q) < 4:
        return ""
    with _lock:
        fid = fid if fid and any(f["id"] == fid for f in facts()) else link_question(q)
        log_ = asked()
        log_.append({"q": q, "at": time.time(), "where": where, "fact": fid})
        tmp = _asked_path().with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(log_[-MAX_ASKED:], ensure_ascii=False))
        tmp.replace(_asked_path())
    if fid:
        rebuild()
    return fid


def _asked_by_fact() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for a in asked():
        if a.get("fact"):
            o = out.setdefault(a["fact"], {"n": 0, "wordings": []})
            o["n"] += 1
            if a.get("where") != "watch" and _norm(a["q"]) not in [_norm(w) for w in o["wordings"]]:
                o["wordings"].append(a["q"])
    return out


def context_for(q: str, limit: int = 2500) -> str:
    hits = find(q)
    if not hits:
        return ""
    lines = [f"- {f['q']}? {f['a']}" + (f" ({f['meeting']}, {f['date']})" if f.get("meeting") else "") for f in hits]
    return ("Facts from the company knowledge base (learned from all meetings):\n" + "\n".join(lines))[:limit]


def _rag_summary() -> dict:
    try:
        from . import rag
        return rag.summary()
    except Exception:                               # noqa: BLE001
        return {}


def summary() -> dict:
    fs = facts()
    kinds: dict[str, int] = {}
    for f in fs:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    from . import sources
    learned = len(list((kdir() / "meetings").glob("*.json")))
    ak = asked()
    from . import core
    cd = core.load()
    return {"facts": len(fs), "kinds": kinds, "meetings_read": learned, "sources": len(sources.listing()),
            "core": sum(1 for f in fs if f.get("core")), "core_company": cd.get("company") or core.company(),
            "core_made": cd.get("made") or 0, "rag": _rag_summary(), "auto_learning": auto_learning(),
            "asked": len(ak), "asked_linked": sum(1 for a in ak if a.get("fact")), **status}


def learn_all(force: bool = False) -> None:
    """Read every meeting that's new or changed (one after another, in the background)."""
    with _lock:
        if status["state"] == "learning":
            return
        status.update(state="learning", done=0, error="")
    try:
        from . import sources
        todo = [j for j in db.list_jobs(100000, "done") if force or needs_update(j)]
        srcs = [x["id"] for x in sources.listing() if force or sources.due(x)]
        status["total"] = len(todo) + len(srcs)
        for j in todo:
            try:
                learn_meeting(j["id"])
            except Exception as e:                  # noqa: BLE001 - one bad meeting must not stop the rest
                status["error"] = summarize.explain_llm_error(e)
                log.warning("knowledge: %s: %s", j["id"], e)
            status["done"] += 1
        for sid in srcs:
            try:
                sources.learn(sid, force)
            except Exception as e:                  # noqa: BLE001
                status["error"] = str(e)[:300]
                log.warning("knowledge source %s: %s", sid, e)
            status["done"] += 1
        from . import people
        for n in people.all_names():                # then who everyone is, from all of that
            try:
                if force or who_needs_update(n):
                    who_is(n, refresh=True)
            except Exception as e:                  # noqa: BLE001
                log.warning("who is %s: %s", n, e)
        rebuild()
        try:
            from . import core
            core.build(force)
        except Exception as e:                      # noqa: BLE001
            log.warning("company profile: %s", e)
    finally:
        status["state"] = "idle"


def start_learning(force: bool = False) -> bool:
    if status["state"] == "learning" or not summarize.enabled():
        return False
    threading.Thread(target=learn_all, args=(force,), daemon=True).start()
    return True


# ---------------------------------------------------------------- learning by itself: on / off (Account > Watch AI)
def _auto_path() -> Path:
    return kdir() / "auto.json"


def auto_learning() -> bool:
    """Off: the server learns nothing by itself - no facts from new meetings, no who-is, no company profile, no
    re-reading of pages, no research, no watch-model training. The buttons (Learn now, Read again, Research now,
    Build watch AI) still work. Minutes are still written and search by meaning stays up to date."""
    try:
        return bool(json.loads(_auto_path().read_text()).get("on", True))
    except (OSError, ValueError):
        return True


def set_auto_learning(on: bool) -> None:
    tmp = _auto_path().with_suffix(".tmp")
    tmp.write_text(json.dumps({"on": bool(on), "at": time.time()}))
    tmp.replace(_auto_path())
    log.info("learning by itself %s", "on" if on else "off")


def _sweep() -> None:
    time.sleep(180)
    last_rag = last_rebuild = 0.0
    while True:
        try:                                        # the watch's questions about people / your last meetings / your
            if time.time() - last_rebuild > 3 * 3600:   # tasks: kept fresh (they're made from the latest meetings)
                last_rebuild = time.time()
                rebuild()
        except Exception as e:                      # noqa: BLE001
            log.info("rebuild: %s", e)
        try:
            from . import rag                       # find by meaning: embed what's new (no AI chat needed)
            if rag.enabled() and time.time() - last_rag > 300:
                last_rag = time.time()
                rag.update(max_new=800)
        except Exception as e:                      # noqa: BLE001 - the embedding model may still be downloading
            log.info("rag: %s", e)
        try:
            worker.retry_failed_summaries()         # minutes that failed while the AI couldn't be reached
        except Exception as e:                      # noqa: BLE001
            log.info("summary retry: %s", e)
        try:
            if auto_learning() and summarize.enabled() and worker.current_job is None and db.next_queued() is None \
                    and status["state"] == "idle":
                busy = False
                for j in db.list_jobs(200, "done"):  # one new or changed meeting per pass: keep the server free
                    if needs_update(j):
                        log.info("learning facts from %s", j["id"])
                        learn_meeting(j["id"])
                        busy = True
                        break
                if not busy:
                    from . import people
                    for n in people.all_names():      # "Who is X?" written from what's known about X
                        if who_needs_update(n):
                            log.info("writing who %s is", n)
                            who_is(n, refresh=True)
                            rebuild()
                            busy = True
                            break
                if not busy:
                    from . import core               # the company profile, when what it's written from changed
                    if core.needs_update():
                        core.build()
                        busy = True
                if not busy:
                    from . import watchlm
                    if watchlm.nightly(facts()) or watchlm.core_changed(facts()):
                        busy = True
                if not busy:
                    from . import sources
                    for x in sources.listing():      # a new source, or a web page due for its daily check
                        if sources.due(x):
                            log.info("learning facts from source %s", x["id"])
                            sources.learn(x["id"])
                            busy = True
                            break
                if not busy:
                    from . import research           # topics it found, questions nobody could answer
                    research.tick()
        except Exception as e:                      # noqa: BLE001
            log.warning("knowledge sweep: %s", e)
        time.sleep(150)


def start_sweeper() -> None:
    from . import sources
    try:
        sources.reset_stale()
    except Exception as e:                          # noqa: BLE001
        log.warning("sources: %s", e)
    if settings.knowledge_sweep:
        threading.Thread(target=_sweep, daemon=True).start()
