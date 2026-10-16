"""Ask AI: short chats with the same LLM that writes the summaries. Answers are produced in a background
thread and kept in memory for an hour; the device polls for them."""
import re
import threading
import time
import uuid

from . import summarize

SYSTEM = ("You are the assistant on a small pocket note-taking device with a tiny screen. "
          "Answer clearly and briefly (a few sentences or a short list) unless asked for more. "
          "Plain text only - no tables, no code fences unless asked for code.")

_lock = threading.Lock()
_items: dict[str, dict] = {}
_sem = threading.Semaphore(1)          # one answer at a time - the LLM runs on the same machine


def _run(aid: str, messages: list[dict], system: str, best: bool, late=None) -> None:
    if late:                                         # context that takes a moment (a web search): found here, not
        try:                                         # while the device waits for its request to be accepted
            extra = late()
            if extra:
                system += "\n\n" + extra
        except Exception:                            # noqa: BLE001 - answer without it
            pass
    with _sem:
        try:
            answer = summarize.chat([{"role": "system", "content": system}] + messages, best, fast=not best)
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)     # reasoning models
            res = {"id": aid, "status": "done", "answer": answer.strip()}
        except Exception as e:                       # noqa: BLE001 - report any LLM failure to the device
            res = {"id": aid, "status": "error", "error": summarize.explain_llm_error(e)[:300]}
    with _lock:
        _items[aid].update(res, finished=time.time())


def start(messages: list[dict], context: str = "", best: bool = False, system: str = "", late=None) -> dict:
    aid = uuid.uuid4().hex[:12]
    now = time.time()
    with _lock:
        for k in [k for k, v in _items.items() if now - v["created"] > 3600]:
            del _items[k]
        _items[aid] = {"id": aid, "status": "pending", "created": now}
    sysmsg = (system or SYSTEM) + (("\n\n" + context) if context else "")
    threading.Thread(target=_run, args=(aid, messages, sysmsg, best, late), daemon=True).start()
    return {"id": aid, "status": "pending"}


def instant(answer: str) -> dict:
    """An answer that needs no AI (a fact the knowledge base has exactly): ready at once."""
    aid = uuid.uuid4().hex[:12]
    with _lock:
        _items[aid] = {"id": aid, "status": "done", "answer": answer, "created": time.time(), "finished": time.time()}
    return {"id": aid, "status": "done", "answer": answer}


def get(aid: str) -> dict | None:
    with _lock:
        v = _items.get(aid)
        return {k: v[k] for k in ("id", "status", "answer", "error") if k in v} if v else None


# ---------------------------------------------------------------- the answer's voice, made as soon as it's ready
# The watch reads answers out: it asks /api/tts with the answer right after getting it. With "speak": true the
# voice is made here the moment the answer is done, so that request finds it ready instead of waiting for Piper.
_voices: dict[str, tuple[float, bytes]] = {}


def _vkey(text: str, lang: str, adpcm: bool) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())[:600] + "|" + (lang or "en") + ("|a" if adpcm else "")


def spoken(text: str, lang: str, adpcm: bool):
    with _lock:
        v = _voices.get(_vkey(text, lang, adpcm))
    return v[1] if v else None


def speak_when_done(aid: str) -> None:
    def run():
        from . import speech
        if not speech.tts_available():
            return
        for _ in range(1200):                       # up to 10 minutes
            v = get(aid)
            if not v or v["status"] != "pending":
                break
            time.sleep(0.5)
        if not v or v.get("status") != "done" or not v.get("answer"):
            return
        text = " ".join(v["answer"].split())[:1500]
        try:
            wav = speech.tts(text)
            data = speech.wav_to_ima_adpcm(wav)
        except Exception:                           # noqa: BLE001 - the watch asks for it the usual way
            return
        now = time.time()
        with _lock:
            for k in [k for k, (t, _) in _voices.items() if now - t > 600]:
                del _voices[k]
            _voices[_vkey(text, "en", True)] = (now, data)
            _voices[_vkey(text, "en", False)] = (now, wav)
    threading.Thread(target=run, daemon=True).start()
