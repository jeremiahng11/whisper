"""The watch's offline answer pack: the questions it can hear without the internet, with their answers.

The watch has no speech-to-text of its own. Espressif's MultiNet (ESP-SR) recognises a fixed list of spoken English
phrases on the chip, so the pack gives it ~150 short phrases ("who is ryan", "when is the v two launch") made
from the knowledge base, each with a short answer to show. The watch downloads it (GET /api/watch/pack) when the
version changes and builds its phrase list from it.

Phrase rules (what MultiNet can take): lowercase letters a-z and spaces only, 2-7 words, numbers as words.
"""
import hashlib
import json
import re

from . import knowledge

MAX_ITEMS = 150
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10 - 2] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _num(n % 100))
    if n < 10000 and n % 1000 == 0:
        return _ONES[n // 1000] + " thousand"
    return " ".join(_ONES[int(d)] for d in str(n))      # long numbers: digit by digit (codes, years read out)


def phrase(q: str) -> str:
    """'When is the V2 launch?' -> 'when is the v two launch' ('' when it can't be said as a short phrase)."""
    s = q.lower().replace("&", " and ").replace("%", " percent ").replace("+", " plus ")
    s = re.sub(r"(\d+)", lambda m: " " + _num(int(m.group(1))) + " ", s)
    s = re.sub(r"[^a-z ]+", " ", s)
    words = s.split()
    if not 2 <= len(words) <= 7:
        return ""
    return " ".join(words)


def build(limit: int = MAX_ITEMS) -> dict:
    facts = knowledge.facts()
    # people first (they're asked about most, up to a quarter of the pack), then the newest of everything else
    people = [f for f in facts if f.get("kind") == "person"][: limit // 4]
    rest = sorted((f for f in facts if f.get("kind") != "person"), key=lambda f: f.get("date") or "", reverse=True)
    items, seen = [], set()
    for f in people + rest:
        if len(items) >= limit:
            break
        say = [p for p in dict.fromkeys(phrase(x) for x in [f["q"]] + (f.get("alts") or [])) if p and p not in seen]
        if not say:
            continue
        seen.update(say[:2])
        items.append({"id": f["id"], "q": f["q"], "say": say[:2], "a": f["a"][:240], "kind": f.get("kind", "fact"),
                      "date": f.get("date") or ""})
    ver = hashlib.sha1(json.dumps(items, sort_keys=True).encode()).hexdigest()[:12]
    return {"version": ver, "count": len(items), "items": items}
