"""How long the last questions took, step by step (speech to text, the answer, its voice) - shown in /api/health as
"timing", so it can be read without digging through the server log."""
import threading
import time
from collections import deque

_last: deque = deque(maxlen=15)
_lock = threading.Lock()


def note(step: str, seconds: float, detail: str = "") -> None:
    with _lock:
        _last.append({"at": time.strftime("%H:%M:%S"), "step": step, "s": round(seconds, 1), "detail": detail[:80]})


def recent() -> list[str]:
    with _lock:
        return [f"{x['at']} {x['step']} {x['s']}s" + (f" ({x['detail']})" if x["detail"] else "") for x in _last]
