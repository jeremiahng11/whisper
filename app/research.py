"""Research on its own: the server keeps reading the web about your company and learns from what it couldn't answer.

Three things run in the background (the knowledge sweep, when the server is free):

1. Research topics. Each topic ("Aleta Planet", "Sanrio Aleta Adventure", a product) is a "research" source: about
   once a week the server searches the web (Bing, else DuckDuckGo) and the news (Bing News, else Google News), reads
   the pages that are really about it (a headless browser for sites that need one), and the AI turns them into facts.
   New articles come in with every run; pages are kept for a year, news for 60 days.

2. Topics it finds by itself. The names your facts keep coming back to (companies, products, brands - not people)
   become research topics too, so a new product or partner mentioned in meetings gets researched without you adding it.
   You can remove a topic; it isn't added again.

3. Questions it couldn't answer. A question asked on the watch or in Ask that matched no fact is searched on the web
   (with your company's name), a few a day; the facts found answer it next time - and the watch model learns them
   at night.

Ask also does a quick live web search when the knowledge base knows nothing about a question (RESEARCH_LIVE).

Settings in DATA_DIR/knowledge/research.json: {"auto": bool, "topics": [...], "removed": [...], "questions": {...}}.
"""
import json
import logging
import re
import threading
import time
from pathlib import Path

from . import knowledge
from .config import settings

log = logging.getLogger("research")
MAX_AUTO_TOPICS = 8
MIN_FACTS_FOR_TOPIC = 3
QUESTION_MAX_AGE = 30 * 86400
_lock = threading.Lock()
status: dict = {"last_questions": 0, "last_discover": 0}


def _path() -> Path:
    return knowledge.kdir() / "research.json"


def load() -> dict:
    try:
        d = json.loads(_path().read_text())
    except (OSError, ValueError):
        d = {}
    d.setdefault("auto", settings.research_auto)
    d.setdefault("topics", [])
    d.setdefault("found", [])
    d.setdefault("removed", [])
    d.setdefault("questions", {})
    return d


def _save(d: dict) -> None:
    tmp = _path().with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False))
    tmp.replace(_path())


def _key(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def _clean_topic(t: str) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip(" .,;:-")[:80]


# ---------------------------------------------------------------- topics
def discover(fs: list[dict] | None = None) -> list[str]:
    """Names the knowledge base keeps coming back to: companies, products, brands (proper names, not people)."""
    from . import people
    fs = knowledge.facts() if fs is None else fs
    who = {_key(n) for n in people.all_names()}
    for n in list(who):
        who |= set(n.split())                       # "Ryan" as well as "Ryan Tan"
    count: dict[str, int] = {}
    spelt: dict[str, str] = {}
    for f in fs:
        if f.get("kind") == "person" or str(f.get("id", "")).startswith("who-"):
            continue
        t = _clean_topic(f.get("topic") or "")
        k = _key(t)
        words = k.split()
        if not t or len(t) < 3 or not (1 <= len(words) <= 4) or k in who or any(w in who for w in words):
            continue
        if not re.search(r"[A-Z]", t):              # proper names only ("Aleta Planet", not "pricing")
            continue
        count[k] = count.get(k, 0) + 1
        spelt.setdefault(k, t)
    best = sorted((k for k, n in count.items() if n >= MIN_FACTS_FOR_TOPIC), key=lambda k: -count[k])
    out: list[str] = []
    for k in best:                                  # "Aleta Planet" covers "Aleta Planet app"? keep both: both are names
        if any(k == _key(o) for o in out):
            continue
        out.append(spelt[k])
        if len(out) >= MAX_AUTO_TOPICS:
            break
    return out


def topics() -> list[dict]:
    """[{"topic", "auto": bool, "source": id or ""}] - yours first, then the ones it found."""
    from . import sources
    d = load()
    by = {_key(s.get("query", "")): s for s in sources.listing() if s.get("kind") == "research" and not s.get("question")}
    out, seen = [], set()
    for t, auto in [(t, False) for t in d["topics"]] + [(t, True) for t in d["found"]]:
        k = _key(t)
        if k in seen or k in {_key(r) for r in d["removed"]}:
            continue
        seen.add(k)
        s = by.get(k) or {}
        out.append({"topic": t, "auto": auto, "source": s.get("id", ""), "state": s.get("state", ""),
                    "facts": s.get("facts") or 0, "articles": s.get("articles", 0), "fetched": s.get("fetched", 0),
                    "error": s.get("error", ""), "progress": s.get("progress", ""),
                    "searched": s.get("searched"), "queue": s.get("queue")})
    return out


def add_topic(t: str) -> dict:
    t = _clean_topic(t)
    if len(t) < 3:
        raise ValueError("say what to research (a company, a product, a brand)")
    with _lock:
        d = load()
        d["removed"] = [r for r in d["removed"] if _key(r) != _key(t)]
        if _key(t) not in {_key(x) for x in d["topics"]}:
            d["topics"].append(t)
        _save(d)
    return ensure_sources()[_key(t)]


def remove_topic(t: str) -> bool:
    from . import sources
    k = _key(t)
    with _lock:
        d = load()
        known = {_key(x) for x in d["topics"] + d["found"]}
        if k not in known:
            return False
        d["topics"] = [x for x in d["topics"] if _key(x) != k]
        d["found"] = [x for x in d["found"] if _key(x) != k]
        if k not in {_key(r) for r in d["removed"]}:
            d["removed"].append(_clean_topic(t))     # a found topic you removed isn't found again
        _save(d)
    for s in sources.listing():
        if s.get("kind") == "research" and not s.get("question") and _key(s.get("query", "")) == k:
            sources.remove(s["id"])
    return True


def set_auto(on: bool) -> None:
    with _lock:
        d = load()
        d["auto"] = bool(on)
        _save(d)


def ensure_sources() -> dict[str, dict]:
    """A research source for every topic (yours, and the found ones while auto is on); found topics that dropped
    out are left alone (what they found stays useful)."""
    from . import sources
    d = load()
    have = {_key(s.get("query", "")): s for s in sources.listing() if s.get("kind") == "research" and not s.get("question")}
    removed = {_key(r) for r in d["removed"]}
    want = list(d["topics"]) + (list(d["found"]) if d["auto"] else [])
    for t in want:
        k = _key(t)
        if k in have or k in removed:
            continue
        have[k] = sources.add("research", title=f"Research: {t}", url=t)
    return have


def refresh_found(fs: list[dict] | None = None) -> list[str]:
    """Update the topics it found by itself (at most once a day)."""
    with _lock:
        d = load()
        removed = {_key(r) for r in d["removed"]}
        mine = {_key(t) for t in d["topics"]}
        found = [t for t in discover(fs) if _key(t) not in removed and _key(t) not in mine]
        keep = [t for t in d["found"] if _key(t) not in removed]   # once found, kept (until you remove it)
        for t in found:
            if _key(t) not in {_key(x) for x in keep}:
                keep.append(t)
        d["found"] = keep[: MAX_AUTO_TOPICS * 2]
        d["last_discover"] = time.time()
        _save(d)
    return d["found"]


# ---------------------------------------------------------------- questions it couldn't answer
def company_name() -> str:
    """The name most facts are about (to search a question with), "" if none yet."""
    d = load()
    if d["topics"]:
        return d["topics"][0]
    found = d["found"] or discover()
    return found[0] if found else ""


def unanswered(limit: int = 20) -> list[str]:
    """Recent questions that matched no fact and weren't researched yet (most asked first)."""
    d = load()
    now = time.time()
    count: dict[str, int] = {}
    said: dict[str, str] = {}
    for a in knowledge.asked():
        if a.get("fact") or now - (a.get("at") or 0) > QUESTION_MAX_AGE:
            continue
        q = a.get("q") or ""
        k = knowledge._norm(q)
        if len(knowledge._words(q)) < 2 or k in d["questions"]:
            continue
        if knowledge.link_question(q):              # answered since (a fact learned later)
            continue
        count[k] = count.get(k, 0) + 1
        said.setdefault(k, q)
    return [said[k] for k in sorted(count, key=lambda k: -count[k])][:limit]


def research_question(q: str) -> dict:
    """A one-off research source for a question: searched with the company's name; the facts found answer it."""
    from . import sources
    name = company_name()
    words = knowledge._words(q)
    search = q if not name or knowledge._words(name) & words else f"{name} {q}"
    m = sources.add("research", title=f"Asked: {q[:70]}", url=search)
    m.update(question=q, once=True, must=" ".join(sorted(words)), need=max(1, len(words) // 3))
    sources._save(m)
    with _lock:
        d = load()
        d["questions"][knowledge._norm(q)] = {"at": time.time(), "source": m["id"]}
        if len(d["questions"]) > 2000:                # forget the oldest
            d["questions"] = dict(sorted(d["questions"].items(), key=lambda x: x[1]["at"])[-1500:])
        _save(d)
    return m


def relink_asked() -> int:
    """Questions that matched nothing before and match a fact now are linked to it (their wording joins the fact)."""
    n = 0
    with knowledge._lock:
        log_ = knowledge.asked()
        for a in log_:
            if not a.get("fact"):
                fid = knowledge.link_question(a.get("q") or "")
                if fid:
                    a["fact"] = fid
                    n += 1
        if n:
            tmp = knowledge._asked_path().with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(log_, ensure_ascii=False))
            tmp.replace(knowledge._asked_path())
    if n:
        knowledge.rebuild()
    return n


def tick() -> bool:
    """One step of background research (called by the knowledge sweep when the server is free). True if it did
    something worth waiting for."""
    from . import sources
    d = load()
    now = time.time()
    if d["auto"] and now - d.get("last_discover", 0) > 86400:
        found = refresh_found()
        log.info("research topics found: %s", ", ".join(found) or "none yet")
    ensure_sources()
    if d["auto"] and now - status["last_questions"] > 86400 / max(1, settings.research_questions):
        qs = unanswered(1)
        status["last_questions"] = now
        if qs:
            log.info("researching a question nobody could answer: %s", qs[0])
            m = research_question(qs[0])
            try:
                sources.learn(m["id"])
            except Exception as e:                  # noqa: BLE001
                log.warning("research %s: %s", qs[0], e)
            relink_asked()
            return True
    return False


def summary() -> dict:
    d = load()
    from . import sources
    qs = [s for s in sources.listing() if s.get("kind") == "research" and s.get("question")]
    return {"auto": d["auto"], "live": settings.research_live, "topics": topics(),
            "questions_researched": len(qs), "questions_answered": sum(1 for s in qs if s.get("facts")),
            "waiting": len(unanswered(50)), "browser": sources.browser_available(), "running": dict(running)}


# ---------------------------------------------------------------- a quick look on the web while you wait (Ask)
def live_context(q: str, limit: int = 2500, plain: bool = False) -> str:
    """Ask, when the knowledge base knows nothing about the question: the top web and news results for it (titles
    and snippets, a few seconds) so the answer isn't from the AI's old general knowledge alone."""
    if not settings.research_live:
        return ""
    from . import sources
    words = knowledge._words(q)
    if len(words) < 1:
        return ""
    name = "" if plain else company_name()
    search = q if plain or not name or knowledge._words(name) & words else f"{name} {q}"   # (plain: about the world)
    t0 = time.time()
    try:
        # a question you're waiting for: the search at most 3 s, news only for the company's own (plain = about the world)
        hits = sources.web_search(search, 3 if plain else 6)[:6]
        if not plain:
            hits += [h for h in sources.news_search(search, 6)[:4] if h["url"] not in {x["url"] for x in hits}]
    except Exception as e:                          # noqa: BLE001 - offline, blocked: answer without it
        log.info("live search: %s", e)
        return ""
    lines, kept = [], []
    for h in hits:
        hay = knowledge._words(h["title"] + " " + h["snippet"])
        if not hay & words:
            continue
        kept.append(h)
        lines.append(f"- {h['title']} ({h.get('source') or ''}, {h.get('date') or ''}): {h['snippet'][:300]} <{h['url']}>")
    # a search result's few words rarely hold the answer ("how many payment companies are there in Singapore"): the top
    # pages are read too (at the same time, a few seconds at most) and their passages closest to the question kept
    read = _read_top(kept[:3], words, deadline=2.5) if plain and kept else []
    try:
        from . import timing
        timing.note("web search", time.time() - t0, f"{len(lines)} results, {len(read)} pages read")
    except Exception:                               # noqa: BLE001
        pass
    if not lines:
        if plain and needs_fresh(q):                 # nothing found, and it's about now: Groq looks it up itself
            g = groq_search(q)
            if g:
                return ("A web search just now (Groq) found (may be incomplete; say so if unsure, and mention the "
                        "source):\n" + g)[:limit + 1500]
        return ""
    out = "A quick web search just now found (may be incomplete; say so if unsure, and mention the source):\n" + "\n".join(lines)
    if read:
        out += "\n\nFrom the top pages:\n" + "\n\n".join(read)
    return out[:limit + (3000 if read else 0)]


FRESH = re.compile(r"\b(news|latest|recent|recently|today|tonight|yesterday|this (week|month|year)|right now|currently|"
                   r"current|nowadays|score|scores|who won|won the|winner|price|prices|stock|shares|bitcoin|crypto|"
                   r"exchange rate|election|release date|released|launch|launched|new|newest|upcoming|20[2-3]\d|"
                   r"open now|still|anymore|who is the (ceo|president|prime minister|minister|head|owner))\b", re.I)


def needs_fresh(q: str) -> bool:
    """A question whose answer changes over time (news, prices, scores, 'right now', 'latest')."""
    return bool(FRESH.search(q or ""))


def groq_search(q: str) -> str:
    """Groq's own web search (compound models, on the LIVE_LLM key): a few lines of current facts with sources, or ""."""
    if not settings.groq_search or not settings.live_llm_api_key or "groq.com" not in settings.live_llm_base_url:
        return ""
    from . import summarize, sources, timing
    t0 = time.time()
    try:
        r = summarize.cloud_client().post(
            f"{settings.live_llm_base_url}/chat/completions", timeout=settings.groq_search_timeout_s,
            headers={"Authorization": f"Bearer {settings.live_llm_api_key}"},
            json={"model": settings.groq_search_model, "temperature": 0.1, "messages": [
                {"role": "system", "content": "Search the web and reply with the current facts that answer the question: "
                                              "at most 6 short lines, each with its source (site name). No preamble."},
                {"role": "user", "content": q}]})
        r.raise_for_status()
        out = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        sources.search_status.update(provider=sources.search_status.get("provider") or "", groq="ok", at=time.time())
    except Exception as e:                              # noqa: BLE001 - answered without it
        log.info("groq search: %s", e)
        sources.search_status.update(groq=f"failed: {str(e)[:100]}")
        out = ""
    timing.note("web search (Groq)", time.time() - t0, f"{len(out)} characters")
    return out[:2000]


def _read_top(hits: list[dict], words: set, deadline: float = 4.0) -> list[str]:
    """The passages of these pages that share most words with the question (and have figures, when it asks how many /
    how much), ~700 characters per page."""
    from concurrent.futures import ThreadPoolExecutor, wait
    from . import sources
    numeric = bool(words & {"many", "much", "number", "count", "percent", "price", "cost", "big", "large", "size"})

    def one(h):
        try:
            r = sources._get(h["url"], deadline)
            if r.status_code != 200 or "html" not in r.headers.get("content-type", "html"):
                return ""
            text = sources.html_text(r.text[:600000])[1]
        except Exception:                           # noqa: BLE001
            return ""
        best = []
        for para in re.split(r"\n\s*\n|(?<=[.!?])\s{2,}", text):
            para = " ".join(para.split())
            if not 40 <= len(para) <= 1200:
                continue
            sc = len(knowledge._words(para) & words) + (2 if numeric and re.search(r"\d", para) else 0)
            if sc >= 2:
                best.append((sc, para))
        best.sort(key=lambda x: -x[0])
        got = " ... ".join(p for _, p in best[:3])[:700]
        return f"[{h.get('source') or h['url']}] {got}" if got else ""
    ex = ThreadPoolExecutor(3)
    futs = [ex.submit(one, h) for h in hits]
    done, _ = wait(futs, timeout=deadline + 0.5)
    ex.shutdown(wait=False, cancel_futures=True)
    return [f.result() for f in futs if f in done and f.result()]


# ---------------------------------------------------------------- "Research now" (Account > Watch AI)
running = {"state": "idle", "done": 0, "total": 0, "error": ""}


def research_now(questions: int = 5) -> bool:
    """Find topics, read every topic again and research the waiting questions - in the background."""
    with _lock:
        if running["state"] == "running":
            return False
        running.update(state="running", done=0, total=0, error="")
    threading.Thread(target=_research_now, args=(questions,), daemon=True).start()
    return True


def _research_now(questions: int) -> None:
    from . import sources
    try:
        if load()["auto"]:
            refresh_found()
        have = ensure_sources()
        qs = unanswered(questions)
        running["total"] = len(have) + len(qs)
        for m in have.values():
            try:
                sources.learn(m["id"])
            except Exception as e:                  # noqa: BLE001
                running["error"] = str(e)[:300]
            running["done"] += 1
        for q in qs:
            try:
                sources.learn(research_question(q)["id"])
            except Exception as e:                  # noqa: BLE001
                running["error"] = str(e)[:300]
            running["done"] += 1
        relink_asked()
    finally:
        running["state"] = "idle"
