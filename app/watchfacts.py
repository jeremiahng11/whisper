"""More questions for the watch's offline answers, made from what Oracle Minutes already knows.

Besides the facts learned from meetings, pages and documents, the knowledge base gets:
  * per person (from their profile: People): their favourite words and phrases, what they usually talk about, when
    you last met them, how much they talk, what they're working on (their open action items);
  * your recent meetings: what the last meeting was about, what it decided, your own action items, this week's
    meetings;
  * the company: "what is <company> about" / "what does <company> do" point at the company profile's answer.
Answers are short (the watch shows them and reads them out), and each has a few ways of asking.
"""
import json
import logging
import re
import time

from . import db, worker
from .config import settings

log = logging.getLogger("watchfacts")
MAX_A = 230


def _me() -> str:
    try:
        p = settings.data_dir / "me.json"
        return (json.loads(p.read_text()).get("name") or "") if p.exists() else ""
    except (OSError, ValueError):
        return ""


def _first(name: str) -> str:
    return (name or "").strip().split(" ")[0].lower()


def _is_me(owner: str, me: str) -> bool:
    o = (owner or "").strip().lower()
    return o in ("", "me", "i", "myself") or (bool(me) and _first(o) == _first(me))


def _clip(s: str, n: int = MAX_A) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "..."


def _date(ts) -> str:
    try:
        return time.strftime("%a %d %b", time.localtime(float(ts)))
    except (TypeError, ValueError):
        return ""


def _section(md: str, *heads: str) -> list[str]:
    """Bullets / sentences under the first heading that matches (### Decisions, ### Summary...)."""
    for h in heads:
        m = re.search(r"^#{2,4}\s*" + re.escape(h) + r"[^\n]*\n(.*?)(?=^#{1,4}\s|\Z)", md, re.M | re.S | re.I)
        if m:
            out = []
            for line in m.group(1).splitlines():
                t = re.sub(r"^\s*[-*]\s*(\[[ x]\]\s*)?", "", line).strip()
                t = re.sub(r"[*_`]", "", t)
                if t and not t.lower().startswith(("none", "no decisions", "n/a")):
                    out.append(t)
            return out
    return []


def _fact(q, a, alts, kind, topic="", date=""):
    return {"q": q, "a": _clip(a), "alts": alts, "kind": kind, "topic": topic, "job": "", "date": date,
            "meeting": "", "source": "derived"}


def people_facts() -> list[dict]:
    from . import people
    out = []
    jobs = db.list_jobs(400, "done")
    for n in people.all_names():
        p = people.get(n) or {}
        if not p:
            continue
        lang = p.get("language") or {}
        first = n.split(" ")[0]
        # favourite words / phrases
        phrases = [x for x in (lang.get("phrases") or []) if x][:3]
        words = [x for x in (lang.get("words") or []) if x][:4]
        if phrases or words:
            a = f"In meetings {first} often says " + ", ".join(f'"{x}"' for x in (phrases or words[:3]))
            if phrases and words:
                a += "; their most-used words are " + ", ".join(words[:3])
            out.append(_fact(f"What is {n}'s favourite word", a + ".",
                             [f"{n} favourite word", f"What does {n} always say", f"What is {n}'s favourite phrase",
                              f"What words does {n} use"], "person", n))
        # what they usually talk about: the profile note's first section, else their terms
        note = p.get("note") or ""
        raises = _section(note, "Usually raises and owns")
        topics = raises[:2] or [", ".join((lang.get("terms") or [])[:5])] if (raises or lang.get("terms")) else []
        if topics and topics[0]:
            out.append(_fact(f"What does {n} usually talk about", f"{first} usually raises: " + "; ".join(topics) + ".",
                             [f"What does {n} talk about", f"What does {n} usually raise", f"What topics does {n} own"],
                             "person", n))
        # when you last met them
        ms = p.get("meetings") or []
        if ms:
            last = ms[0]
            a = f"You last met {first} on {_date(last.get('when'))} in {last.get('title') or 'a meeting'}."
            if len(ms) > 1:
                a += f" {len(ms)} meetings together in all."
            out.append(_fact(f"When did I last meet {n}", a, [f"When did I last see {n}", f"Last meeting with {n}"],
                             "person", n, str(time.strftime('%Y-%m-%d', time.localtime(last.get('when') or 0)))))
        # how much they talk, how they speak
        secs = p.get("talk_seconds") or 0
        if secs >= 60:
            a = f"{first} has spoken for about {round(secs / 60)} minutes across {len(ms)} meeting{'s' if len(ms) != 1 else ''}"
            if lang.get("pace_wpm"):
                a += f", at a {lang.get('pace')} pace of about {lang['pace_wpm']} words a minute"
            if lang.get("questions"):
                a += f"; {lang['questions']} percent of their lines are questions"
            out.append(_fact(f"How much does {n} talk", a + ".",
                             [f"How much does {n} speak in meetings", f"How does {n} speak", f"Does {n} talk a lot"],
                             "person", n))
        # what they're working on: their open action items in recent meetings
        acts = []
        for j in jobs[:60]:
            for x in worker.load_json(j["id"], "actions_raw.json", []) or []:
                if isinstance(x, dict) and _first(x.get("owner", "")) == _first(n) and x.get("task"):
                    acts.append(x["task"] + (f" (due {x['due']})" if x.get("due") else ""))
            if len(acts) >= 4:
                break
        if acts:
            out.append(_fact(f"What is {n} working on", f"{first}'s recent action items: " + "; ".join(acts[:4]) + ".",
                             [f"What are {n}'s action items", f"What does {n} owe me", f"What is {n} doing"],
                             "person", n))
    return out


def meeting_facts() -> list[dict]:
    out = []
    jobs = [j for j in db.list_jobs(200, "done") if not (j.get("options") or {}).get("template") == "memo"
            and not (j.get("filename") or "").startswith(("memo_", "dict_"))]
    if not jobs:
        return out
    me = _me()
    last = jobs[0]
    md = ""
    try:
        md = (worker.result_dir(last["id"]) / "result.md").read_text(encoding="utf-8")
    except OSError:
        pass
    title = last.get("title") or last.get("filename") or "your last meeting"
    when = _date(last.get("created_at"))
    summ = _section(md, "Summary")
    if summ:
        sent = re.split(r"(?<=[.!?])\s", " ".join(summ))
        out.append(_fact("What was the last meeting about", f"{title} ({when}): " + " ".join(sent[:2]),
                         ["What was my last meeting", "Summarise the last meeting", "Tell me about the last meeting"],
                         "meeting", title))
    dec = _section(md, "Decisions", "Decisions and approvals")
    if dec:
        out.append(_fact("What did we decide in the last meeting", f"In {title}: " + "; ".join(dec[:3]) + ".",
                         ["What was decided last meeting", "What were the decisions", "Last meeting decisions"],
                         "meeting", title))
    mine, cutoff = [], time.time() - 21 * 86400
    for j in jobs:
        if (j.get("created_at") or 0) < cutoff:
            break
        for x in worker.load_json(j["id"], "actions_raw.json", []) or []:
            if isinstance(x, dict) and x.get("task") and _is_me(x.get("owner", ""), me):
                mine.append(x["task"] + (f" (due {x['due']})" if x.get("due") else ""))
    if mine:
        out.append(_fact("What are my action items", f"You have {len(mine)}: " + "; ".join(mine[:4]) + ".",
                         ["What are my tasks", "What do I need to do", "What do I owe people", "My action items"],
                         "meeting"))
    week = [j for j in jobs if (j.get("created_at") or 0) > time.time() - 7 * 86400]
    if week:
        names = [(j.get("title") or j.get("filename") or "")[:40] for j in week[:4]]
        out.append(_fact("How many meetings did I have this week",
                         f"{len(week)} in the last 7 days: " + "; ".join(names) + ("..." if len(week) > 4 else "."),
                         ["What meetings did I have this week", "My meetings this week"], "meeting"))
    return out


def company_alts(facts: list[dict], company: str) -> None:
    """'What is Aleta Planet about' / 'what does Aleta Planet do' ask the company profile's own answer."""
    if not company:
        return
    low = company.lower()
    for f in facts:
        q = f["q"].lower()
        if q.startswith(("what is " + low, "who is " + low, "tell me about " + low)) and f.get("core"):
            have = {x.lower() for x in [f["q"]] + (f.get("alts") or [])}
            more = [x for x in (f"What is {company} about", f"What does {company} do", f"Tell me about {company}",
                                f"What is {company}") if x.lower() not in have]
            f["alts"] = (f.get("alts") or []) + more
            return


def derived() -> list[dict]:
    out = []
    for fn in (people_facts, meeting_facts):
        try:
            out += fn()
        except Exception as e:                                   # noqa: BLE001 - extra questions only
            log.warning("watch facts (%s) failed: %s", fn.__name__, e)
    return out
