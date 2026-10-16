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
            answer = summarize.chat([{"role": "system", "content": system}] + messages, best)
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


def get(aid: str) -> dict | None:
    with _lock:
        v = _items.get(aid)
        return {k: v[k] for k in ("id", "status", "answer", "error") if k in v} if v else None
