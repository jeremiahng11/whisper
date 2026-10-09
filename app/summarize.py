"""Meeting summary with a local LLM (Ollama) or any OpenAI-compatible API."""
import json
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

# meeting types: (label, layout). Picked per recording with ?template=...
TEMPLATES = {
    "general": ("General meeting", FORMAT),
    "standup": ("Stand-up", """Write the result in Markdown in exactly this layout:

Title: <a short title, max 8 words>

### Summary
<1-3 sentences>

### By person
#### <name>
- **Done:** ...
- **Next:** ...
- **Blockers:** ...

### Action items
- [ ] <who> - <what> (<due date if mentioned>)"""),
    "client": ("Client meeting", """Write the result in Markdown in exactly this layout:

Title: <a short title, max 8 words, include the client's name if said>

### Summary
<3-5 sentences>

### What the client needs
- ...

### What we committed to
- ...

### Commercials (pricing, scope, timelines)
- ...

### Risks and concerns raised
- ...

### Action items
- [ ] <who> - <what> (<due date if mentioned>)

### Next meeting / next steps
- ..."""),
    "one_on_one": ("1:1", """Write the result in Markdown in exactly this layout:

Title: <a short title, max 8 words>

### Summary
<2-4 sentences>

### Topics discussed
- ...

### Feedback given
- ...

### Agreed
- ...

### Action items
- [ ] <who> - <what> (<due date if mentioned>)

### To follow up next time
- ..."""),
    "interview": ("Interview", """Write the result in Markdown in exactly this layout:

Title: Interview - <candidate name if said> - <role if said>

### Summary
<3-5 sentences about the candidate's background as they described it>

### Experience and skills mentioned
- ...

### Notable answers
- **<question topic>:** <what they said>

### Strengths shown
- ...

### Concerns or gaps
- ...

### Candidate's questions
- ...

### Action items
- [ ] <who> - <what> (<due date if mentioned>)

Do not give a hire / no-hire verdict - only what was said."""),
    "board": ("Board / formal minutes", """Write formal minutes in Markdown in exactly this layout:

Title: <name of the meeting, max 8 words>

### Attendees
- <names, with roles if said>

### Agenda and discussion
#### 1. <agenda item>
**Discussion:** <neutral, third-person summary, e.g. "The Board noted that ...">
**Resolution:** RESOLVED THAT ... (only if a decision was actually made; otherwise write "No resolution.")

### Matters arising
- ...

### Action items
- [ ] <who> - <what> (<due date if mentioned>)

### Next meeting
<date if mentioned>

Use formal, neutral, past-tense language. Never invent a resolution, vote or approval."""),
    "memo": ("Voice note", """Write the result in Markdown in exactly this layout:

Title: <a short title, max 8 words>

### Summary
<1-4 sentences>

### Ideas and points
- ...

### To do
- [ ] <what> (<due date if mentioned>)"""),
}


def template_format(name: str) -> str:
    return TEMPLATES.get(name or "general", TEMPLATES["general"])[1]


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
def _ollama_chat(messages: list[dict], retry: bool = True, model: str = "") -> str:
    model = model or settings.ollama_model
    r = httpx.post(
        f"{settings.ollama_url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": settings.ollama_num_ctx, "temperature": 0.2},
        },
        timeout=settings.llm_timeout_s,
    )
    if retry and r.status_code == 404 and "not found" in r.text.lower() and settings.ollama_auto_pull:
        ensure_ollama_model(model)                # model not downloaded yet: pull it once, then retry
        return _ollama_chat(messages, retry=False, model=model)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _openai_chat(messages: list[dict], model: str = "") -> str:
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"} if settings.openai_api_key else {}
    r = httpx.post(
        f"{settings.openai_base_url}/chat/completions",
        headers=headers,
        json={"model": model or settings.openai_model, "messages": messages, "temperature": 0.2},
        timeout=settings.llm_timeout_s,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def best_model() -> str:
    """The "best quality" model name, or "" when none is configured."""
    b = settings.summary_backend.lower()
    return settings.ollama_model_best if b == "ollama" else settings.openai_model_best if b == "openai" else ""


def chat(messages: list[dict], best: bool = False) -> str:
    backend = settings.summary_backend.lower()
    model = best_model() if best else ""
    if backend == "ollama":
        return _ollama_chat(messages, model=model)
    if backend == "openai":
        return _openai_chat(messages, model=model)
    raise RuntimeError(f"unknown SUMMARY_BACKEND {settings.summary_backend!r}")


_pull_lock = threading.Lock()


def ensure_ollama_model(want: str = "") -> None:
    """Download the Ollama model if it isn't there yet (first start can take a few minutes)."""
    if settings.summary_backend.lower() != "ollama":
        return
    with _pull_lock:
        try:
            tags = httpx.get(f"{settings.ollama_url}/api/tags", timeout=10).json().get("models", [])
            names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
            want = want or settings.ollama_model
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


NOTES_EXTRA = """

The person who recorded this also typed these notes during the meeting ([m:ss] = time into the recording).
Use them to get names, decisions and action items right, and give them priority where they are clear:
\"\"\"
{notes}
\"\"\""""


def summarize(transcript_text: str, language: str, progress: Optional[Callable[[float, str], None]] = None,
              user_notes: str = "", template: str = "general", best: bool = False,
              attendees: Optional[list[str]] = None, glossary: Optional[list[str]] = None,
              flagged: str = "", meeting_date: str = "") -> tuple[str, str]:
    """Returns (title, markdown body without the title line)."""
    progress = progress or (lambda p, s: None)
    lang = settings.summary_language or language
    lang_line = f"Write the minutes in {lang_name(lang)}.\n\n" if lang else ""
    system = SYSTEM
    if attendees:
        system += "\n- People in this meeting (from the calendar invite): " + ", ".join(attendees[:30]) + "."
    if glossary:
        system += "\n- Correct spellings of names and terms (fix mis-heard versions): " + ", ".join(glossary[:200]) + "."
    sys = {"role": "system", "content": system}

    parts = _chunks(transcript_text, max(2000, settings.summary_chunk_chars))
    if len(parts) <= 1:
        source, kind = transcript_text, "Transcript"
    else:
        notes = []
        for i, part in enumerate(parts, 1):
            progress((i - 1) / (len(parts) + 1), f"summarising part {i}/{len(parts)}")
            notes.append(chat([sys, {"role": "user", "content": NOTES_PROMPT.format(i=i, n=len(parts), text=part)}], best))
        source, kind = "\n\n".join(notes), "Notes from the whole transcript, in order"
    progress(len(parts) / (len(parts) + 1) if len(parts) > 1 else 0.5, "writing summary")
    date_line = f"The meeting was on {meeting_date}.\n\n" if meeting_date else ""
    prompt = FINAL_PROMPT.format(lang=date_line + lang_line, fmt=template_format(template), kind=kind, text=source)
    if user_notes.strip():
        prompt += NOTES_EXTRA.format(notes=user_notes.strip()[:8000])
    if flagged.strip():
        prompt += FLAGGED_EXTRA.format(flagged=flagged.strip()[:8000])
    out = chat([sys, {"role": "user", "content": prompt}], best)
    return split_title(out)


FLAGGED_EXTRA = """

The person pressed the "mark" key at these moments because they matter. Make sure the minutes cover what
was said there (the transcript around each mark is shown):
\"\"\"
{flagged}
\"\"\""""

ACTIONS_PROMPT = """From these meeting minutes, list every action item as JSON - an array of objects:
[{{"task": "<what to do>", "owner": "<person's name, or empty if nobody>", "due": "<YYYY-MM-DD or empty>"}}]
The meeting was on {date} ({weekday}). Turn relative dates ("Friday", "next week", "end of month") into real dates
after the meeting date. Leave "due" empty if no date was said. Reply with ONLY the JSON array.

Minutes:
\"\"\"
{minutes}
\"\"\""""

_ITEM = re.compile(r"^\s*[-*]\s*\[[ xX]?\]\s*(.+)$")


def actions_from_markdown(md: str) -> list[dict]:
    """Fallback: '- [ ] Who - what (due)' lines from the minutes."""
    out = []
    for line in md.splitlines():
        m = _ITEM.match(line)
        if not m:
            continue
        text = m.group(1).strip()
        due = ""
        dm = re.search(r"\((?:by |due )?([^()]*\d[^()]*)\)\s*$", text)
        if dm:
            text = text[:dm.start()].strip()
            due = dm.group(1).strip()
        owner, task = "", text
        if " - " in text:
            owner, task = [x.strip() for x in text.split(" - ", 1)]
            if len(owner.split()) > 4:
                owner, task = "", text
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", due):
            task = f"{task} ({due})" if due else task
            due = ""
        out.append({"task": task.strip(" .") , "owner": owner.strip("*"), "due": due})
    return out


def extract_actions(minutes: str, meeting_date: str = "", best: bool = False) -> list[dict]:
    from datetime import date as _date
    try:
        d = _date.fromisoformat(meeting_date) if meeting_date else _date.today()
    except ValueError:
        d = _date.today()
    if not re.search(r"\[[ xX]?\]", minutes):
        return []
    try:
        out = chat([{"role": "user", "content": ACTIONS_PROMPT.format(date=d.isoformat(), weekday=d.strftime("%A"),
                                                                      minutes=minutes[:12000])}], best)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
        m = re.search(r"\[.*\]", out, re.S)
        items = json.loads(m.group(0)) if m else None
        if isinstance(items, list):
            clean = []
            for it in items:
                if not isinstance(it, dict) or not str(it.get("task", "")).strip():
                    continue
                due = str(it.get("due") or "").strip()
                clean.append({"task": str(it["task"]).strip()[:300], "owner": str(it.get("owner") or "").strip()[:60],
                              "due": due if re.match(r"^\d{4}-\d{2}-\d{2}$", due) else ""})
            return clean
    except Exception as e:
        log.warning("action item extraction failed, using the minutes' list: %s", e)
    return actions_from_markdown(minutes)


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
