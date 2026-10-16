"""Company essentials: the questions that matter most, always on the watch and kept up to date.

"Who founded Aleta Planet?", "Who is the CTO?", "What products does Aleta Planet have?", "What services does it
provide?" - the AI writes these as a small set of facts (a company profile) from everything the server knows: your
pages and documents, its own web research, who-is answers and meetings. It writes them again when that material
changes (at most every few hours), so a new product or a new CTO reaches the watch by itself.

These "core" facts:
- win over other facts with the same question;
- come first in the watch's offline pack (and its spoken answers), and the watch answers them on the watch itself,
  without waiting for the server's AI, even when online;
- are learned harder by the watch's own model (more examples), which retrains soon after they change - not only at night.

Meeting questions stay online (the server's AI with all your meetings). You can also pin any fact ("Pin to watch").

Kept in DATA_DIR/knowledge/core.json: {"sig", "made", "company", "facts"}.
"""
import hashlib
import json
import logging
import re
import time
from pathlib import Path

from . import knowledge, summarize

log = logging.getLogger("core")
MAX_CORE = 60
MIN_GAP_S = 4 * 3600                                # not rewritten more often than this (research comes in bits)
ROLE = re.compile(r"\b(found(er|ed)|co-?founder|chair(man|woman|person)?|ceo|cto|cfo|coo|cio|cmo|chief|president|"
                  r"director|head of|managing|vp|vice president|partner(ship)?|licen[cs]e|regulat|headquarter|"
                  r"based in|office|product|service|card|app|platform|solution|launch)\w*", re.I)

CORE_PROMPT = """You keep the company profile of {company} that staff ask their watch about. From the material below,
write the facts every colleague should be able to get in a second:
- what {company} is and does; where it is based; when it was founded; licences / regulation;
- who founded it; the chairman, CEO, CTO, CFO, COO and other leaders - one fact per role, e.g. "Who is the CTO of
  {company}" (alts like "who is the CTO", "who is our CTO") answered with the name and one line about them;
- for each person in a leadership role, "Who is <name>" with their roles and a line of background;
- what products it has (one fact listing them) and one fact per main product (what it is, for whom, key feature or
  price if stated); what services it provides (one fact listing them, then the main ones);
- main partners and brands (e.g. licensed characters), and the latest launch or news if stated.
Use ONLY the material; skip what it doesn't say (no guessing). Prefer the newest information when material disagrees.
Return ONLY a JSON array (at most {n} items) of objects with keys "q" (a short spoken question, 3-9 words), "a" (one
short sentence, at most 35 words), "kind" (person, product, policy, price, date or fact), "topic" (1-4 words),
"alts" (3-4 other ways to ask, including short spoken ones like "who is our CTO").

Material:
\"\"\"
{material}
\"\"\""""


def _path() -> Path:
    return knowledge.kdir() / "core.json"


def load() -> dict:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return {}


def core_facts() -> list[dict]:
    """The profile's facts as the knowledge base takes them."""
    d = load()
    out = []
    for f in d.get("facts") or []:
        out.append({**f, "core": True, "job": "", "source": "core", "meeting": "Company profile",
                    "date": d.get("date") or ""})
    return out


def company() -> str:
    from . import research
    return research.company_name()


def material(limit: int = 14000) -> str:
    """What the profile is written from: facts from your pages, documents and research (not meeting chatter), who-is
    answers, meeting facts about roles and products, and lines of your pages naming roles."""
    from . import sources
    fs = [f for f in knowledge.facts() if not f.get("core")]
    name = company().lower()
    first, second = [], []
    for f in fs:
        line = f"- {f['q']}? {f['a']}" + (f" ({f['date']})" if f.get("date") else "")
        txt = f["q"] + " " + f["a"] + " " + (f.get("topic") or "")
        if f.get("kind") == "person" or (f.get("source") and ROLE.search(txt)) or (name and name in txt.lower() and ROLE.search(txt)):
            first.append(line)
        elif f.get("source") or (not f.get("job") and f.get("kind") in ("product", "price", "policy")):
            second.append(line)
        elif ROLE.search(txt):                      # a meeting fact about a role or product
            second.append(line)
    lines = []
    for m in sources.listing():                     # a leadership page: "Ryan Tan - Founder, Chairman and CEO"
        if m.get("kind") not in ("url", "file", "text", "research"):
            continue
        try:
            txt = (sources.sdir() / f"{m['id']}.txt").read_text()[:200_000]
        except OSError:
            continue
        for ln in txt.splitlines():
            ln = ln.strip()
            if 12 < len(ln) < 300 and re.search(r"\b(founder|chairman|ceo|cto|cfo|coo|chief \w+ officer)\b", ln, re.I):
                lines.append("- " + ln)
    out, size = [], 0
    for x in dict.fromkeys(first + lines[:80] + second):
        if size + len(x) > limit:
            break
        out.append(x)
        size += len(x) + 1
    return "\n".join(out)


def _sig(mat: str) -> str:
    return hashlib.sha1(mat.encode()).hexdigest()[:12]


def needs_update() -> bool:
    mat = material()
    if len(mat) < 200 or not company():
        return False
    d = load()
    if d.get("sig") == _sig(mat):
        return False
    return time.time() - (d.get("made") or 0) > MIN_GAP_S or not d.get("facts")


def build(force: bool = False) -> int:
    """(Re)write the profile when its material changed. Returns how many core facts there are."""
    mat = material()
    name = company()
    d = load()
    if not name or len(mat) < 200:
        return len(d.get("facts") or [])
    sig = _sig(mat)
    if d.get("sig") == sig and not force:
        return len(d.get("facts") or [])
    log.info("writing the company profile of %s", name)
    got = knowledge.parse_facts(summarize.chat([
        {"role": "system", "content": "You write short, factual company profiles as JSON."},
        {"role": "user", "content": CORE_PROMPT.format(company=name, n=MAX_CORE, material=mat)}]))
    facts, seen = [], set()
    for f in got[:MAX_CORE]:
        k = knowledge._norm(f["q"])
        if k not in seen:
            seen.add(k)
            facts.append(f)
    if not facts and d.get("facts"):                # the AI returned nothing usable: keep the old profile
        return len(d["facts"])
    tmp = _path().with_suffix(".tmp")
    tmp.write_text(json.dumps({"sig": sig, "made": time.time(), "date": time.strftime("%Y-%m-%d"), "company": name,
                               "facts": facts}, ensure_ascii=False))
    tmp.replace(_path())
    knowledge.rebuild()
    return len(facts)


def sig() -> str:
    """Changes when a core (or pinned) question or answer does."""
    rows = sorted([f["q"], f["a"]] for f in knowledge.facts() if f.get("core"))
    return hashlib.sha1(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()[:12] if rows else ""
