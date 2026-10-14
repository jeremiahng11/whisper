"""Glossary (names and terms to spell right) and known voices (to name speakers automatically)."""
import json
import math
import threading
import time
from typing import Optional

from . import db
from .config import settings

_lock = threading.Lock()


# ---------------------------------------------------------------- glossary
def glossary() -> list[str]:
    p = settings.glossary_path
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line[:80])
    return out[:300]


def set_glossary(text: str) -> list[str]:
    settings.glossary_path.parent.mkdir(parents=True, exist_ok=True)
    settings.glossary_path.write_text(text.strip()[:20000] + "\n", encoding="utf-8")
    return glossary()


# Region packs: a short vocabulary hint for Whisper (it reads only ~220 tokens) and a rule for the minutes LLM.
REGION_HINTS = {
    "sg": {
        "whisper": ("Singapore meeting, English mixed with Singlish, Mandarin and Malay: lah, leh, lor, meh, can or not, "
                    "Shenzhen, Guangzhou, Shanghai, Johor Bahru, Jurong, Tampines, Changi, MRT, HDB."),
        "llm": ("The speakers are in Singapore and often mix English with Singlish, Mandarin, Malay or Hokkien words. "
                "Speech recognition often mis-hears Chinese and Malay names and places (for example 'Sun Jun' or "
                "'Shen Jun' for Shenzhen, 'Guang Jo' for Guangzhou, 'Johor Baru' for Johor Bahru). Correct such names "
                "when the context makes it clear, and leave Singlish particles (lah, leh, lor) out of the minutes."),
    },
}


def region_rule() -> str:
    hint = REGION_HINTS.get(settings.region)
    return hint["llm"] if hint else ""


def whisper_prompt(attendees: Optional[list[str]] = None) -> str:
    """Vocabulary hint for Whisper (it only reads the last ~220 tokens, so keep it short)."""
    parts = []
    hint = REGION_HINTS.get(settings.region)
    if hint:
        parts.append(hint["whisper"])
    if settings.initial_prompt:
        parts.append(settings.initial_prompt.strip())
    names = [a for a in (attendees or []) if a]
    if names:
        parts.append("People: " + ", ".join(names[:20]) + ".")
    from .terms import known_terms
    from . import vocab
    terms = [t for t in known_terms(packs=False) if t not in names]
    terms += [t for t in vocab.recent() if t not in terms]  # the payments / tech / finance words your meetings use
    if terms:
        parts.append("Terms: " + ", ".join(terms) + ".")
    return " ".join(parts)[:700]


# ---------------------------------------------------------------- voices
def _cos(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else -1.0


def list_voices() -> list[dict]:
    with db._lock:
        rows = db._conn.execute("SELECT name, samples, updated_at FROM voices ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def _all() -> list[tuple[str, list[float], int]]:
    with db._lock:
        rows = db._conn.execute("SELECT name, embedding, samples FROM voices").fetchall()
    return [(r["name"], json.loads(r["embedding"]), r["samples"]) for r in rows]


def remember(name: str, emb: list[float], job_id: str = "", seg: int = -1) -> None:
    """Add a sample of this person's voice: kept as one of their samples, and in their running average."""
    if not name or not emb:
        return
    from . import people
    people.add_sample(name, emb, job_id, seg)
    n = math.sqrt(sum(x * x for x in emb)) or 1.0
    emb = [x / n for x in emb]
    with _lock, db._lock:
        r = db._conn.execute("SELECT embedding, samples FROM voices WHERE name=?", (name,)).fetchone()
        if r:
            old, k = json.loads(r["embedding"]), r["samples"]
            if len(old) == len(emb):
                emb = [(o * k + e) / (k + 1) for o, e in zip(old, emb)]
                k = min(k + 1, 20)                    # newer samples keep some weight
            else:
                k = 1
        else:
            k = 1
        db._conn.execute("INSERT OR REPLACE INTO voices (name, embedding, samples, updated_at) VALUES (?,?,?,?)",
                         (name, json.dumps([round(x, 5) for x in emb]), k, time.time()))
        db._conn.commit()


def forget(name: str) -> bool:
    from . import people
    people.forget(name)
    with db._lock:
        n = db._conn.execute("DELETE FROM voices WHERE name=?", (name,)).rowcount
        db._conn.commit()
    return n > 0


def match(speakers: dict) -> dict[str, str]:
    """{"Speaker 1": {"embedding": [...]}} -> {"Speaker 1": "San"} for voices we know (each name used once)."""
    from . import people
    known = _all()
    samp = people.samples()
    pairs = []
    for label, info in speakers.items():
        e = (info or {}).get("embedding")
        if not e or (info or {}).get("seconds", 99) < 3:      # too little speech to be sure
            continue
        for name, v, _ in known:
            sim = people.similarity(e, v, samp.get(name, []))   # their average voice or their closest samples
            if sim >= settings.voice_match:
                pairs.append((sim, label, name))
    out, used = {}, set()
    for sim, label, name in sorted(pairs, reverse=True):
        if label not in out and name not in used:
            out[label] = name
            used.add(name)
    return out
