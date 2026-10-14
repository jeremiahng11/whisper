"""Names & terms in the transcript: words that were mis-heard as one of your names or terms get its spelling.

Whisper writes what it hears, so "CTRIP" comes out as "C trip" or "Sea trip", "Priscilla" as "Pricilla". After each
transcription (and on "Redo minutes") every line is compared with Names & terms + the names of remembered voices:
  * the same letters with other spacing / case ("c trip", "ctrip")           -> "CTRIP"
  * the same sound, nearly the same spelling ("Pricilla", "Prisila")       -> "Priscilla"
Ordinary English words are never changed, and the line keeps what was heard ("heard") so nothing is lost.
"""
import difflib
import json
import os
import re
from typing import Optional

from . import voices
from .config import settings

# common words that must never be "corrected" into a name or term
COMMON = set("""a an and are as at be been but by can could did do does for from had has have he her here him his how
i if in into is it its just like may me more my no not now of off on one or our out over she so some than that the
their them then there these they this those to too two up us was we were what when where which who why will with
would yes you your about after again all also any back because before being between both down each even first get
going good got great just know last let little long look made make many much need new next only other people right
said same say see should still such sure take tell thing things think three time today trip very want way well work
year yeah okay ok price pricing update refund refunds dashboard team plan meeting week day month money pay payment
sea see c cee seat street tree""".split())


# ---------------------------------------------------------------- corrections you made (heard -> right)
def _corr_path():
    return settings.data_dir / "corrections.json"


def corrections() -> dict:
    """{"tapin": "DABIN", ...}: words the speech model keeps mis-hearing, learned from lines you fixed."""
    try:
        return json.loads(_corr_path().read_text(encoding="utf-8")) if _corr_path().exists() else {}
    except (ValueError, OSError):
        return {}


def save_corrections(c: dict) -> None:
    _corr_path().parent.mkdir(parents=True, exist_ok=True)
    _corr_path().write_text(json.dumps(dict(sorted(c.items())), ensure_ascii=False, indent=1), encoding="utf-8")


def changed_words(before: str, after: str) -> list[tuple[str, str]]:
    """The words / short phrases replaced between two versions of a line (see learn)."""
    a = _WORD.findall(before or "")
    b = _WORD.findall(after or "")
    out = []
    sm = difflib.SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace" and 1 <= i2 - i1 <= 4 and 1 <= j2 - j1 <= 4:
            heard, right = " ".join(a[i1:i2]), " ".join(b[j1:j2])
            if heard != right:
                out.append((heard, right))
    return out


def meeting_fixes(tr: dict) -> dict:
    """Words you fixed in this meeting's lines (learned or "only this line"): {"tapin": "DABIN"} - so the minutes
    written from the old wording follow your fixes too."""
    out = {}
    for s in tr.get("segments", []):
        if s.get("edited") and s.get("heard"):
            for heard, right in changed_words(s["heard"], s["text"]):
                if len(heard.split()) > 1 or heard.lower() not in COMMON:
                    out[heard.lower()] = right
    return out


def learn(before: str, after: str) -> list[tuple[str, str]]:
    """You changed a line from `before` to `after`: remember each replaced word or short phrase (1-4 words each side),
    e.g. "TAPIN" -> "DABIN", "B to B" -> "B2B". Pure deletions / additions aren't learned (that's editing, not
    mis-hearing)."""
    a = _WORD.findall(before or "")
    b = _WORD.findall(after or "")
    out = []
    sm = difflib.SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace" or not (1 <= i2 - i1 <= 4 and 1 <= j2 - j1 <= 4):
            continue
        heard, right = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if heard.lower() == right.lower() and heard == right:
            continue
        hl, rl = heard.lower(), right.lower()
        if len(heard.split()) == 1 and hl in COMMON:
            continue                                        # an everyday word: it would change in every sentence
        if (hl.startswith(rl) or rl.startswith(hl)) and abs(len(hl) - len(rl)) <= 3:
            continue                                        # grammar ("rate" -> "rates"), not a mis-hearing
        out.append((heard, right))
    if out:
        c = corrections()
        for heard, right in out:
            for k, v in list(c.items()):                   # you fixed your own earlier fix: that one changes too
                if v.lower() == heard.lower():
                    c[k] = right
            c[heard.lower()] = right
        save_corrections(c)
    return out


def apply_corrections(text: str, c: Optional[dict] = None) -> str:
    """Exact, whole-word replacements of what you corrected before (any capitals)."""
    c = corrections() if c is None else c
    if not text or not c:
        return text
    for heard in sorted(c, key=len, reverse=True):          # longer phrases first
        text = re.sub(r"(?<![\w-])" + r"\s+".join(map(re.escape, heard.split())) + r"(?![\w-])", lambda m, r=c[heard]: r,
                      text, flags=re.I)
    return text


def known_terms(packs: bool = True) -> list[str]:
    """Names & terms, plus everyone with a remembered voice (they're spelled the same way everywhere), plus the right
    words of your corrections - then the built-in vocabularies (payments / tech / finance), after yours."""
    from . import vocab
    out, seen = [], set()
    own = voices.glossary() + [v["name"] for v in voices.list_voices()] + [me_name()] + list(corrections().values())
    for t in own + (vocab.terms(for_text=True) if packs else []):
        t = (t or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def me_name() -> str:
    p = settings.data_dir / "me.json"
    try:
        return (json.loads(p.read_text()).get("name") or "") if p.exists() else os.getenv("ME_NAME", "")
    except (ValueError, OSError):
        return ""


def _letters(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.lower())


def sound(s: str) -> str:
    """Rough sound key: similar-sounding spellings get the same key (Pricilla / Priscilla / Prisila)."""
    s = _letters(s)
    if not s:
        return ""
    for a, b in (("ph", "f"), ("ck", "k"), ("sch", "sk"), ("sh", "s"), ("ch", "k"), ("qu", "k"), ("q", "k"),
                 ("x", "ks"), ("z", "s"), ("wh", "w"), ("gh", "g"), ("dg", "j"), ("v", "f")):
        s = s.replace(a, b)
    s = re.sub(r"c(?=[eiy])", "s", s).replace("c", "k")
    head, rest = s[0], re.sub(r"[aeiouyhw]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", head + rest)


def _score(window: str, term: str) -> float:
    a, b = _letters(window), _letters(term)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    if sound(window) == sound(term) and r >= 0.70:
        return max(r, 0.9)
    return r


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def _prep(terms: list[str]) -> list[tuple]:
    out = []
    for term in terms:
        tl = _letters(term)
        if len(tl) >= 3:
            out.append((term, tl, max(1, len(term.split())), sound(term)))
    return out


def fix_text(text: str, terms, _prepped=None) -> str:
    """Replace mis-heard versions of the terms in one line."""
    if not text or not terms:
        return text
    words = list(_WORD.finditer(text))
    if not words:
        return text
    prepped = _prepped if _prepped is not None else _prep(terms)
    sizes = {sz for _, _, n, _ in prepped for sz in (n - 1, n, n + 1) if sz > 0}
    cands = []                                              # (start, end, term, score)
    for size in sizes:
        for i in range(len(words) - size + 1):
            ws = words[i:i + size]
            win = text[ws[0].start():ws[-1].end()]
            wl = _letters(win)
            if not wl:
                continue
            common = all(w.group().lower() in COMMON for w in ws)
            wsnd = None
            for term, tl, n, tsnd in prepped:
                if abs(size - n) > 1 or win == term or abs(len(wl) - len(tl)) > max(2, len(tl) // 3):
                    continue
                if wl == tl:
                    cands.append((ws[0].start(), ws[-1].end(), term, 1.0))
                    continue
                if common or len(tl) <= 4 or size > n:      # common words, short terms, split words: exact only
                    continue
                if wsnd is None:
                    wsnd = sound(win)
                if wl[0] != tl[0] and wsnd[:1] != tsnd[:1]:
                    continue
                m = difflib.SequenceMatcher(None, wl, tl)
                if m.real_quick_ratio() < 0.70 or m.quick_ratio() < 0.70:
                    continue
                r = m.ratio()
                sc = max(r, 0.9) if (wsnd == tsnd and r >= 0.70) else r
                if sc >= 0.86:
                    cands.append((ws[0].start(), ws[-1].end(), term, sc))
    if not cands:
        return text
    cands.sort(key=lambda c: (-c[3], -(c[1] - c[0])))
    taken, chosen = [], []
    for c in cands:
        if any(not (c[1] <= a or c[0] >= b) for a, b in taken):
            continue
        taken.append((c[0], c[1]))
        chosen.append(c)
    for s, e, term, _ in sorted(chosen, reverse=True):
        text = text[:s] + term + text[e:]
    return text


def fix_transcript(tr: dict, terms: list[str] | None = None) -> int:
    """Fix every line in place (keeps what was heard in "heard"). Returns how many lines changed."""
    terms = known_terms() if terms is None else terms
    prepped = _prep(terms)
    corr = corrections()
    from . import vocab
    acr = [t for t in terms if re.fullmatch(r"[A-Z0-9]{2,6}\+?", t) and not t.isdigit()]
    n = 0
    for s in tr.get("segments", []):
        if s.get("edited"):                                 # you wrote this line yourself: leave it
            continue
        orig = s.get("heard") or s.get("text", "")
        new = fix_text(vocab.spelled_acronyms(apply_corrections(orig, corr), acr), terms, prepped)
        if new != s.get("text", ""):
            n += 1
        if new != orig:
            s["heard"] = orig
        else:
            s.pop("heard", None)
        s["text"] = new
    return n


def spelled(name: str) -> str:
    """A name typed for a speaker gets the spelling from Names & terms / remembered voices ("Pricilla" -> "Priscilla")."""
    name = (name or "").strip()
    if not name or re.match(r"^Speaker \d+$", name):
        return name
    best, sc = "", 0.0
    for t in known_terms():
        if t.lower() == name.lower():
            return t
        s = _score(name, t)
        if s > sc:
            best, sc = t, s
    # only a spelling fix: about the same length (so "Jeremiah" never becomes "Jeremiah Ng")
    ok = sc >= 0.88 and len(_letters(name)) >= 4 and abs(len(_letters(best)) - len(_letters(name))) <= 1
    return best if ok else name
