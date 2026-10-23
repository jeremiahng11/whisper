"""Meeting summary with a local LLM (Ollama) or any OpenAI-compatible API."""
import json
import logging
import re
import threading
import time
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
    "manco": ("MANCO (management committee)", """Write management committee (MANCO) minutes in Markdown in exactly this layout:

Title: MANCO - <main topic or date, max 8 words>

### Attendees
- <names, with roles or departments if said>

### Summary
<3-5 sentences: the most important outcomes for the management team>

### Updates by department
#### <department or person, e.g. Finance, Compliance, Technology, Operations, Sales>
- **Update:** ...
- **Numbers mentioned:** <revenue, volumes, costs, headcount, KPIs - only if said>
- **Issues:** ...

### Decisions and approvals
- **<topic>:** <what was decided or approved, by whom> (write "Pending" if it was deferred)

### Risks, compliance and regulatory matters
- ...

### Escalations
- <issue> - <escalated to whom / for what>

### Action items
- [ ] <owner> - <what> (<due date if mentioned>)

### Carried forward to next MANCO
- ...

### Next meeting
<date if mentioned>

Keep it factual and neutral; group updates under the department that gave them. Never invent figures, approvals or
deadlines."""),
    "external": ("External meeting", """Write the result in Markdown in exactly this layout:

Title: <a short title, max 8 words, include the other organisation's name if said>

### Attendees
- **Our side:** <names, roles if said>
- **Their side:** <names and organisation, roles if said>

### Summary
<3-5 sentences>

### What they raised or asked for
- ...

### What we committed to
- ...

### Commercials (pricing, scope, timelines, contracts)
- ...

### Risks and concerns
- ...

### Action items
- [ ] <who> - <what> (<due date if mentioned>)

### Next steps / next meeting
- ...

Keep what each side said apart. Never invent commitments, prices or dates."""),
    "partner": ("Partner meeting", """Write the result in Markdown in exactly this layout:

Title: <partner name - topic, max 8 words>

### Attendees
- **Our side:** <names>
- **Partner side:** <names and organisation>

### Summary
<3-5 sentences>

### Partnership status
- <where the partnership stands: live, onboarding, pilot, renewal - only if said>

### Joint initiatives and progress
- **<initiative>:** <status, numbers mentioned>

### Integration and technical points
- <APIs, environments (UAT / production), certifications, incidents - only if discussed>

### Commercials and contract
- <fees, revenue share, volumes, SLAs, renewals>

### Issues and escalations
- <issue> - <owner, which side>

### Action items
- [ ] <owner (our side / partner)> - <what> (<due date if mentioned>)

### Next meeting
<date if mentioned>"""),
    "collaboration": ("Collaboration meeting", """Write the result in Markdown in exactly this layout:

Title: <a short title, max 8 words>

### Summary
<2-4 sentences: what the group set out to work on and where it landed>

### Goals
- ...

### Ideas and options discussed
- **<idea or option>:** <for / against, who raised it>

### Who is working on what
- **<name>:** <their part>

### Dependencies and blockers
- ...

### Decisions
- ...

### Action items
- [ ] <who> - <what> (<due date if mentioned>)

### Open questions / to explore
- ..."""),
    "emergency": ("Emergency meeting", """Write emergency / incident meeting minutes in Markdown in exactly this layout:

Title: Emergency - <what happened, max 8 words>

### Situation
<2-4 sentences: what happened, when it started, what is affected - only as stated>

### Impact
- <customers, transactions, systems, money, partners affected - with numbers if said>

### Immediate actions
- [ ] <owner> - <action> (<by when, if said>) - <status if said: done / in progress>

### Communications
- **Customers:** <what will be told, by whom, when>
- **Partners / vendors:** ...
- **Regulator (e.g. MAS) / management / board:** <notification needed, deadline, who - only if discussed>

### Decisions
- ...

### Root cause (so far)
- <only what is known; write "Not yet known" if not established>

### Follow-up
- <next check-in time, post-incident review, preventive actions>

Use clear, factual, time-ordered language. Never guess causes, numbers or deadlines."""),
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
def strip_thinking(text: str) -> str:
    """Remove a reasoning model's thinking: <think>...</think>, or everything before a lone </think>
    (the chat template opens the tag in the prompt, so the reply often only contains the closing one)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


# ---------------------------------------------------------------- where Ollama is
# OLLAMA_URL first; when that name doesn't resolve or nothing answers there (a renamed container, a different Docker
# network after a redeploy), the usual places are tried - the "ollama" service, the Docker host (where a published
# port 11434 is reachable) - and the first that answers is used until it stops answering.
_where: dict = {"url": "", "at": 0.0}
_where_lock = threading.Lock()


def _docker_host_ips() -> list[str]:
    out = []
    try:                                            # the container's default gateway = the Docker host
        for line in open("/proc/net/route").read().splitlines()[1:]:
            f = line.split()
            if len(f) > 2 and f[1] == "00000000":
                g = int(f[2], 16)
                out.append(".".join(str((g >> (8 * i)) & 255) for i in range(4)))
    except (OSError, ValueError):
        pass
    return out + ["172.17.0.1"]


def _answers(url: str) -> bool:
    try:
        return httpx.get(f"{url}/api/tags", timeout=2).status_code == 200
    except Exception:                               # noqa: BLE001
        return False


def ollama_url() -> str:
    conf = settings.ollama_url.rstrip("/")
    with _where_lock:
        if _where["url"] and time.time() - _where["at"] < 300:
            return _where["url"]
        tried = [conf, "http://ollama:11434", "http://host.docker.internal:11434"] + \
                [f"http://{ip}:11434" for ip in _docker_host_ips()]
        for u in dict.fromkeys(tried):
            if _answers(u):
                if u != conf:
                    log.warning("Ollama isn't at OLLAMA_URL=%s - using %s (set OLLAMA_URL to that)", conf, u)
                _where.update(url=u, at=time.time())
                return u
        _where.update(url="", at=0.0)
        return conf                                 # nothing answered: the error says where it looked


def ollama_found() -> str:
    """Where Ollama was found ("" if not checked or nowhere)."""
    return _where["url"]


def _ollama_post(path: str, **kw) -> httpx.Response:
    url = ollama_url()
    try:
        return httpx.post(f"{url}{path}", **kw)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        with _where_lock:
            _where.update(url="", at=0.0)           # look again (once) - it may have moved
        url2 = ollama_url()
        if url2 == url:
            raise
        return httpx.post(f"{url2}{path}", **kw)


def _ollama_chat(messages: list[dict], retry: bool = True, model: str = "", ctx: int = 0) -> str:
    model = model or settings.ollama_model
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": ctx or settings.ollama_num_ctx, "temperature": 0.2},
    }
    if model == settings.ollama_model_fast and model != settings.ollama_model:
        body["keep_alive"] = "24h"                # the quick model stays loaded: no load time before an answer
    if any(k in model.lower() for k in ("qwen3", "deepseek-r1", "gpt-oss")):
        body["think"] = False                     # reasoning models: answer straight away (much faster)
        if "qwen3" in model.lower() and messages:  # older Ollama ignores "think": qwen3's own switch
            msgs = [dict(m) for m in messages]
            for m in reversed(msgs):
                if m.get("role") == "user":
                    m["content"] = str(m.get("content", "")) + " /no_think"
                    break
            body["messages"] = msgs
    r = _ollama_post("/api/chat", json=body, timeout=settings.llm_timeout_s)
    if retry and r.status_code == 404 and "not found" in r.text.lower() and model == settings.ollama_model_fast \
            and model != settings.ollama_model:
        # the quick model isn't downloaded yet: this answer comes from the main model, the quick one downloads
        if settings.ollama_auto_pull:
            threading.Thread(target=ensure_ollama_model, args=(model,), daemon=True).start()
        return _ollama_chat(messages, retry=False, model=settings.ollama_model)
    if retry and r.status_code == 404 and "not found" in r.text.lower() and settings.ollama_auto_pull:
        ensure_ollama_model(model)                # model not downloaded yet: pull it once, then retry
        return _ollama_chat(messages, retry=False, model=model)
    r.raise_for_status()
    return strip_thinking(r.json()["message"]["content"])


def _openai_chat(messages: list[dict], model: str = "") -> str:
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"} if settings.openai_api_key else {}
    r = httpx.post(
        f"{settings.openai_base_url}/chat/completions",
        headers=headers,
        json={"model": model or settings.openai_model, "messages": messages, "temperature": 0.2},
        timeout=settings.llm_timeout_s,
    )
    r.raise_for_status()
    return strip_thinking(r.json()["choices"][0]["message"]["content"])


def best_model() -> str:
    """The "best quality" model name, or "" when none is configured."""
    b = settings.summary_backend.lower()
    return settings.ollama_model_best if b == "ollama" else settings.openai_model_best if b == "openai" else ""


_llm_cache: dict = {"at": 0.0, "status": ""}


def llm_status() -> str:
    """'ok', 'off', or why the LLM can't be reached (checked at most every 30 s, 3 s timeout)."""
    import time
    if not enabled():
        return "off"
    if time.time() - _llm_cache["at"] < 30:
        return _llm_cache["status"]
    backend = settings.summary_backend.lower()
    try:
        if backend == "ollama":
            httpx.get(f"{ollama_url()}/api/tags", timeout=3).raise_for_status()
        status = "ok"
    except Exception as e:                                   # noqa: BLE001
        status = explain_llm_error(e)
    _llm_cache.update(at=time.time(), status=status)
    return status


def explain_llm_error(e: Exception) -> str:
    """Turn 'Name or service not known' etc. into something the user can act on."""
    msg = str(e)
    if msg.startswith("cloud AI"):
        return msg
    if settings.summary_backend.lower() == "ollama":
        where = settings.ollama_url
        low = msg.lower()
        if "name or service not known" in low or "nodename nor servname" in low or "temporary failure in name resolution" in low:
            return (f"can't find the Ollama server '{where}' (that name doesn't exist on this server's network; also tried "
                    "http://ollama:11434 and the Docker host on port 11434) - set OLLAMA_URL in Oracle Minutes' "
                    "environment to an address it can reach, e.g. http://<GMK LAN IP>:11434 if Ollama publishes port "
                    "11434, or put both containers on the same Coolify network and use http://<ollama container name>:11434")
        if "connection refused" in low or "connect" in low:
            return (f"Ollama at '{where}' refused the connection - is Ollama running, and listening on 0.0.0.0 "
                    "(OLLAMA_HOST=0.0.0.0) if it is outside this container?")
        if "404" in low and "model" in low:
            return f"Ollama has no model {settings.ollama_model} - run: ollama pull {settings.ollama_model}"
    return msg[:300]


# Live questions come first: while one waits for its answer, background work (meetings, sources, research) holds
# its next request back, so the machine's cores go to the answer.
_live = {"n": 0}
_live_lock = threading.Lock()


def live_waiting() -> bool:
    return _live["n"] > 0


def _yield_to_live(max_s: float = 180) -> None:
    t0 = time.time()
    while _live["n"] > 0 and time.time() - t0 < max_s:
        time.sleep(0.5)


def chat(messages: list[dict], best: bool = False, fast: bool = False) -> str:
    """fast: a live question (the watch, Ask) - the quick model, ahead of background work."""
    backend = settings.summary_backend.lower()
    if not fast:
        _yield_to_live()
        model = best_model() if best else ""
        if backend == "ollama":
            return _ollama_chat(messages, model=model)
        if backend == "openai":
            return _openai_chat(messages, model=model)
        raise RuntimeError(f"unknown SUMMARY_BACKEND {settings.summary_backend!r}")
    if settings.live_llm_api_key:
        return "".join(chat_stream(messages))
    with _live_lock:
        _live["n"] += 1
    try:
        if backend == "ollama":
            m = settings.ollama_model_fast or settings.ollama_model
            return _ollama_chat(messages, model=m, ctx=settings.ollama_fast_ctx if m == settings.ollama_model_fast else 0)
        if backend == "openai":
            return _openai_chat(messages, model=settings.openai_model_fast)
        raise RuntimeError(f"unknown SUMMARY_BACKEND {settings.summary_backend!r}")
    finally:
        with _live_lock:
            _live["n"] -= 1


def _no_think(pieces):
    """Text pieces with a reasoning model's <think>...</think> left out (also when the tags are split across pieces)."""
    buf, thinking = "", False
    for p in pieces:
        buf += p
        while True:
            if thinking:
                e = buf.find("</think>")
                if e < 0:
                    break
                buf, thinking = buf[e + 8:], False
                continue
            s = buf.find("<think>")
            if s >= 0:
                if s:
                    yield buf[:s]
                buf, thinking = buf[s + 7:], True
                continue
            break
        if not thinking and buf and "<" not in buf[-7:]:
            yield buf
            buf = ""
    if buf and not thinking:
        yield buf


_HX_STREAM, _HX_POST = httpx.stream, httpx.post
_client = None
_client_lock = threading.Lock()
_warm_at = 0.0


def cloud_client() -> httpx.Client:
    """One kept-open connection pool for the cloud AI and cloud speech to text (Groq): no new TLS handshake
    (~150-300 ms from here) on every question."""
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(timeout=settings.llm_timeout_s,
                                   limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=120))
        return _client


def warm_cloud() -> None:
    """A question is coming (the watch started sending): open the connection to the cloud AI now, in the background,
    if it isn't open already - by the time the words are known it's ready."""
    global _warm_at
    if not settings.live_llm_api_key or time.time() - _warm_at < 60 or httpx.stream is not _HX_STREAM:
        return
    _warm_at = time.time()

    def go():
        try:
            cloud_client().get(f"{settings.live_llm_base_url}/models", timeout=5,
                               headers={"Authorization": f"Bearer {settings.live_llm_api_key}"})
        except Exception:                                # noqa: BLE001 - only a head start
            pass
    threading.Thread(target=go, daemon=True).start()


def _cloud_pieces(messages: list[dict]):
    import json as _json
    base, model = settings.live_llm_base_url, settings.live_llm_model
    extra: dict = {}
    if "groq.com" in base:                            # reasoning models: as little thinking as possible, none shown
        if "gpt-oss" in model:
            extra = {"reasoning_effort": "low", "include_reasoning": False}
        elif "qwen3" in model or "qwen/" in model:
            extra = {"reasoning_effort": "none", "reasoning_format": "hidden"}
    for attempt in (0, 1):
        body = {"model": model, "messages": messages, "temperature": 0.2, "stream": True, **(extra if attempt == 0 else {})}
        opener = cloud_client().stream if httpx.stream is _HX_STREAM else httpx.stream   # (tests stand in for httpx)
        with opener("POST", f"{base}/chat/completions", timeout=settings.llm_timeout_s,
                          headers={"Authorization": f"Bearer {settings.live_llm_api_key}"}, json=body) as r:
            if r.status_code >= 400:                  # say why (a wrong model name, base URL or key)
                raw = r.read().decode(errors="replace")[:300]
                try:                                  # just the provider's message, it's what matters on a watch
                    msg = (_json.loads(raw).get("error") or {}).get("message") or raw
                except (ValueError, AttributeError):
                    msg = raw
                if attempt == 0 and extra and r.status_code == 400:
                    log.info("cloud AI refused the reasoning settings (%s) - without them", msg)
                    continue
                raise RuntimeError(f"cloud AI said {r.status_code}: {msg} (model {model!r})")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    piece = ((_json.loads(d).get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                except ValueError:
                    continue
                if piece:
                    yield piece
            return


def chat_stream(messages: list[dict]):
    """A live question's answer as it's written (pieces of text): the cloud AI when LIVE_LLM_API_KEY is set, else
    the quick Ollama model; other backends give the whole answer at once. Thinking (<think>...) is left out."""
    import json as _json
    if settings.live_llm_api_key:
        yield from _no_think(_cloud_pieces(messages))
        return
    if settings.summary_backend.lower() != "ollama":
        yield chat(messages, fast=True)
        return
    m = settings.ollama_model_fast or settings.ollama_model
    body = {"model": m, "messages": messages, "stream": True,
            "options": {"num_ctx": settings.ollama_fast_ctx if m == settings.ollama_model_fast else settings.ollama_num_ctx,
                        "temperature": 0.2}}
    if m == settings.ollama_model_fast and m != settings.ollama_model:
        body["keep_alive"] = "24h"
    if any(k in m.lower() for k in ("qwen3", "deepseek-r1", "gpt-oss")):
        body["think"] = False
        if "qwen3" in m.lower():
            msgs = [dict(x) for x in messages]
            for x in reversed(msgs):
                if x.get("role") == "user":
                    x["content"] = str(x.get("content", "")) + " /no_think"
                    break
            body["messages"] = msgs
    with _live_lock:
        _live["n"] += 1
    try:
        with httpx.stream("POST", f"{ollama_url()}/api/chat", json=body, timeout=settings.llm_timeout_s) as r:
            if r.status_code == 404:                  # the quick model isn't there: the whole answer, the usual way
                yield chat(messages, fast=True)
                return
            r.raise_for_status()

            def pieces():
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        d = _json.loads(line)
                    except ValueError:
                        continue
                    yield (d.get("message") or {}).get("content") or ""
                    if d.get("done"):
                        break
            yield from _no_think(pieces())
    finally:
        with _live_lock:
            _live["n"] -= 1


_pull_lock = threading.Lock()


def ensure_ollama_model(want: str = "") -> None:
    """Download the Ollama model if it isn't there yet (first start can take a few minutes)."""
    if settings.summary_backend.lower() != "ollama":
        return
    with _pull_lock:
        try:
            tags = httpx.get(f"{ollama_url()}/api/tags", timeout=10).json().get("models", [])
            names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
            want = want or settings.ollama_model
            if want in names or f"{want}:latest" in names:
                return
            log.info("pulling ollama model %s ...", want)
            r = httpx.post(f"{ollama_url()}/api/pull", json={"model": want, "stream": False}, timeout=None)
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
              flagged: str = "", meeting_date: str = "", people: str = "",
              minutes_lang: str = "") -> tuple[str, str]:
    """Returns (title, markdown body without the title line).
    minutes_lang: "" = SUMMARY_LANGUAGE or the speech language; "en", "zh", ... = that language;
    "both" = the meeting's language + English (English meetings: + Chinese); "zh+en" = exactly those two."""
    progress = progress or (lambda p, s: None)
    spoken = re.sub(r"\[\d+:\d+(:\d+)?\]|\[[^\]]*\]:?", " ", transcript_text)
    if len(spoken.split()) < 18 and not user_notes.strip():          # a quick test or a few words: no minutes
        first = " ".join(spoken.split()[:8])
        return (first[:60] or "Short recording"), "_Short recording - too little speech for minutes; see the transcript._\n"
    first, second = minutes_languages(minutes_lang or settings.summary_language, language)
    lang = first
    lang_line = f"Write the minutes in {lang_name(lang)}.\n\n" if lang else ""
    system = SYSTEM
    if attendees:
        system += "\n- People in this meeting (from the calendar invite): " + ", ".join(attendees[:30]) + "."
    if glossary:
        system += ("\n- Correct spellings of names and terms (fix mis-heard versions; keep acronyms like KYC, MAS, PSP "
                   "in capitals): " + ", ".join(glossary[:260]) + ".")
    if people:
        system += ("\n- What earlier meetings show about the people speaking (use it to understand who means what and "
                   "who owns what; never copy it into the minutes):\n" + people)
    from .voices import region_rule
    if region_rule():
        system += "\n- " + region_rule()
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
    title, body = split_title(out)
    if not second:
        return title, body
    progress(0.9, f"translating the minutes into {lang_name(second)}")
    title2, body2 = translate_minutes(title, body, second, best)
    both_title = f"{title} / {title2}" if title and title2 and title2 != title else (title or title2)
    return both_title, f"# {native_name(first)}\n\n{body}\n\n# {native_name(second)}\n\n{body2}"


def minutes_languages(want: str, spoken: str) -> tuple[str, str]:
    """-> (language to write the minutes in, second language to translate them into or "")."""
    want = (want or "").strip().lower()
    spoken = (spoken or "").strip().lower()
    if want in ("both", "bilingual"):
        first = spoken or "en"
        return first, ("zh" if first.startswith("en") else "en")
    if "+" in want:
        a, b = (x.strip() for x in want.split("+", 1))
        a = a or spoken or "en"
        return a, (b if b and b != a else "")
    return (want or spoken), ""


def primary_part(body: str) -> str:
    """Bilingual minutes ('# 中文 ... # English ...'): only the first language - for the action items."""
    m = re.match(r"^# [^\n]+\n(.*?)\n# [^\n]+\n", body, re.S)
    return m.group(1).strip() if m else body


TRANSLATE_PROMPT = """Translate these meeting minutes into {lang}.
- Keep exactly the same Markdown: the "Title:" line first, headings, bullet points, "- [ ]" checkboxes, bold, tables.
- Keep people's names, company and product names, acronyms (KYC, MAS, API ...), numbers, amounts and dates as
  they are.{names}
- Translate everything else fully and naturally, the way a {lang} speaker would write minutes.
- Reply with ONLY the translated minutes, no notes.

Minutes:
\"\"\"
{text}
\"\"\""""


def translate_minutes(title: str, body: str, target: str, best: bool = False) -> tuple[str, str]:
    """Translate finished minutes, section by section when they are long. -> (title, body)."""
    sections, cur = [], ""
    for block in re.split(r"(?m)^(?=## )", body):
        if cur and len(cur) + len(block) > 3500:
            sections.append(cur)
            cur = ""
        cur += block
    if cur.strip():
        sections.append(cur)
    sys = {"role": "system", "content": "You are a careful professional translator of business meeting minutes."}
    out_title, parts = "", []
    for i, sec in enumerate(sections or [""]):
        text = (f"Title: {title}\n\n" if i == 0 and title else "") + sec.strip()
        if not text.strip():
            continue
        done = strip_thinking(chat([sys, {"role": "user", "content": TRANSLATE_PROMPT.format(
            lang=lang_name(target), text=text, names="" if target.startswith("zh") or target == "yue" else
            "\n- Names in Chinese characters: keep them and add the romanised name once, e.g. 陈伟 (Chen Wei).")}], best))
        if i == 0 and title:
            t, b = split_title(done)
            out_title, done = t, b if t else done
        else:
            done = split_title(done)[1]
        parts.append(done.strip())
    return out_title, "\n\n".join(parts)


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


_NATIVE = {"en": "English", "zh": "中文", "ms": "Bahasa Melayu", "ta": "தமிழ்", "id": "Bahasa Indonesia",
           "ja": "日本語", "ko": "한국어", "th": "ภาษาไทย", "vi": "Tiếng Việt", "yue": "粵語"}


def native_name(code: str) -> str:
    n = _NATIVE.get((code or "").lower())
    return f"{n} ({lang_name(code)})" if n and n != lang_name(code) else lang_name(code)


def lang_name(code: str) -> str:
    return _LANGS.get((code or "").lower(), code)
