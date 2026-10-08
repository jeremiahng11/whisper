"""Meeting summary with a local LLM (Ollama) or any OpenAI-compatible API."""
import logging
import re
import threading
from typing import Callable, Optional

import httpx

from .config import settings

log = logging.getLogger("summarize")

SYSTEM = """You turn transcripts of meetings and voice notes into clear, accurate minutes.
Rules:
- Use only what is in the transcript. Never invent names, numbers, dates or decisions.
- The transcript comes from speech recognition, so fix obvious mis-heard words from context, but keep meaning.
- If speakers are labelled (Speaker 1, Speaker 2...), keep those labels unless a real name is clearly said.
- Be concise. Leave a section out entirely if there is nothing for it."""

FORMAT = """Write the result in Markdown in exactly this layout:

Title: <a short title, max 8 words>

### Summary
<3-6 sentences>

### Key points
- ...

### Decisions
- ...

### Action items
- [ ] <who> - <what> (<due date if mentioned>)

### Open questions
- ..."""

NOTES_PROMPT = """This is part {i} of {n} of a long transcript. Write compact notes of everything important in
this part: topics, facts, numbers, decisions, tasks with owners and due dates, open questions.
Plain bullet points only.

Transcript part {i}/{n}:
\"\"\"
{text}
\"\"\""""

FINAL_PROMPT = """{lang}{fmt}

{kind}:
\"\"\"
{text}
\"\"\""""


def enabled() -> bool:
    return settings.summary_backend.lower() not in ("none", "off", "", "0")


# ---------------------------------------------------------------- LLM backends
def _ollama_chat(messages: list[dict], retry: bool = True) -> str:
    r = httpx.post(
        f"{settings.ollama_url}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": settings.ollama_num_ctx, "temperature": 0.2},
        },
        timeout=settings.llm_timeout_s,
    )
    if retry and r.status_code == 404 and "not found" in r.text.lower() and settings.ollama_auto_pull:
        ensure_ollama_model()                     # model not downloaded yet: pull it once, then retry
        return _ollama_chat(messages, retry=False)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _openai_chat(messages: list[dict]) -> str:
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"} if settings.openai_api_key else {}
    r = httpx.post(
        f"{settings.openai_base_url}/chat/completions",
        headers=headers,
        json={"model": settings.openai_model, "messages": messages, "temperature": 0.2},
        timeout=settings.llm_timeout_s,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def chat(messages: list[dict]) -> str:
    backend = settings.summary_backend.lower()
    if backend == "ollama":
        return _ollama_chat(messages)
    if backend == "openai":
        return _openai_chat(messages)
    raise RuntimeError(f"unknown SUMMARY_BACKEND {settings.summary_backend!r}")


_pull_lock = threading.Lock()


def ensure_ollama_model() -> None:
    """Download the Ollama model if it isn't there yet (first start can take a few minutes)."""
    if settings.summary_backend.lower() != "ollama":
        return
    with _pull_lock:
        try:
            tags = httpx.get(f"{settings.ollama_url}/api/tags", timeout=10).json().get("models", [])
            names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
            want = settings.ollama_model
            if want in names or f"{want}:latest" in names:
                return
            log.info("pulling ollama model %s ...", want)
            r = httpx.post(f"{settings.ollama_url}/api/pull", json={"model": want, "stream": False}, timeout=None)
            r.raise_for_status()
            log.info("ollama model %s ready", want)
        except Exception as e:
            log.warning("could not prepare ollama model: %s", e)


# ---------------------------------------------------------------- summary
def _chunks(text: str, size: int) -> list[str]:
    """Split on line breaks, keeping parts under `size` characters."""
    parts, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > size and cur:
            parts.append(cur)
            cur = ""
        while len(line) > size:                      # a single giant line
            parts.append(line[:size])
            line = line[size:]
        cur += line
    if cur.strip():
        parts.append(cur)
    return parts


def summarize(transcript_text: str, language: str, progress: Optional[Callable[[float, str], None]] = None) -> tuple[str, str]:
    """Returns (title, markdown body without the title line)."""
    progress = progress or (lambda p, s: None)
    lang = settings.summary_language or language
    lang_line = f"Write the minutes in {lang_name(lang)}.\n\n" if lang else ""
    sys = {"role": "system", "content": SYSTEM}

    parts = _chunks(transcript_text, max(2000, settings.summary_chunk_chars))
    if len(parts) <= 1:
        source, kind = transcript_text, "Transcript"
    else:
        notes = []
        for i, part in enumerate(parts, 1):
            progress((i - 1) / (len(parts) + 1), f"summarising part {i}/{len(parts)}")
            notes.append(chat([sys, {"role": "user", "content": NOTES_PROMPT.format(i=i, n=len(parts), text=part)}]))
        source, kind = "\n\n".join(notes), "Notes from the whole transcript, in order"
    progress(len(parts) / (len(parts) + 1) if len(parts) > 1 else 0.5, "writing summary")
    out = chat([sys, {"role": "user", "content": FINAL_PROMPT.format(lang=lang_line, fmt=FORMAT, kind=kind, text=source)}])
    return split_title(out)


def split_title(out: str) -> tuple[str, str]:
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()        # reasoning models (qwen3, deepseek-r1)
    out = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", out).strip()   # some models wrap everything in a fence
    m = re.match(r"^\**\s*title\s*:?\**\s*:?\s*(.+?)\s*$", out.splitlines()[0], re.I) if out else None
    if m:
        title = m.group(1).strip().strip("*#\"' ")
        body = "\n".join(out.splitlines()[1:]).strip()
        return title, body
    return "", out


_LANGS = {
    "en": "English", "zh": "Chinese", "ms": "Malay", "ta": "Tamil", "id": "Indonesian", "ja": "Japanese",
    "ko": "Korean", "th": "Thai", "vi": "Vietnamese", "hi": "Hindi", "fr": "French", "de": "German",
    "es": "Spanish", "pt": "Portuguese", "it": "Italian", "nl": "Dutch", "ru": "Russian", "ar": "Arabic",
    "tl": "Tagalog", "yue": "Cantonese",
}


def lang_name(code: str) -> str:
    return _LANGS.get((code or "").lower(), code)
