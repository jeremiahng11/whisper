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
    if step == "voice trip":                       # kept for the daily stability line (Watches)
        _keep_trip(seconds, detail)


def _keep_trip(seconds: float, detail: str) -> None:
    import json
    import os
    from .config import settings
    try:
        p = settings.data_dir / "voice_trips.jsonl"
        with _lock:
            with open(p, "a") as f:
                f.write(json.dumps({"at": round(time.time()), "s": round(seconds, 2), "d": detail[:60]}) + "\n")
            if p.stat().st_size > 600_000:                  # ~10,000 questions: the oldest half go
                rows = p.read_text().splitlines()
                tmp = p.with_suffix(".tmp")
                tmp.write_text("\n".join(rows[len(rows) // 2:]) + "\n")
                os.replace(tmp, p)
    except OSError:
        pass


def trips() -> list[dict]:
    import json
    from .config import settings
    try:
        return [json.loads(x) for x in (settings.data_dir / "voice_trips.jsonl").read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []


def recent() -> list[str]:
    with _lock:
        return [f"{x['at']} {x['step']} {x['s']}s" + (f" ({x['detail']})" if x["detail"] else "") for x in _last]
