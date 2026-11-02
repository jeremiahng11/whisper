"""Background work: on / only at night / off (Account > Background work).

Everything the server does by itself when nobody asked - learning facts from meetings, researching on the web,
training the watch AI, re-recognising voices, updating people's profiles and the AI read of meetings, search by
meaning - can make the fan spin. "night" lets it run only in the quiet hours (01:00-07:00 by default, your time
zone), "off" stops it completely. What you ask for still happens at once either way: meetings you upload or record
are transcribed and summarised, the watch's questions are answered, and the buttons (Learn now, Research now...)
work."""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import settings

log = logging.getLogger("background")
MODES = ("on", "night", "off")
_cache: dict = {"at": 0.0, "d": None}


def _path() -> Path:
    return settings.data_dir / "background.json"


def _env_night() -> tuple[int, int]:
    import os
    try:
        a, b = (int(x) for x in os.getenv("BACKGROUND_NIGHT", "1-7").split("-"))
        return a % 24, b % 24
    except ValueError:
        return 1, 7


def load() -> dict:
    if time.time() - _cache["at"] < 5 and _cache["d"] is not None:
        return _cache["d"]
    import os
    d = {"mode": os.getenv("BACKGROUND_WORK", "on").lower(), "from": _env_night()[0], "to": _env_night()[1]}
    try:
        d.update(json.loads(_path().read_text()))
    except (OSError, ValueError):
        pass
    if d["mode"] not in MODES:
        d["mode"] = "on"
    _cache.update(at=time.time(), d=d)
    return d


def local_hour() -> int:
    """The hour in your time zone (the server's own clock is often UTC in a container)."""
    try:
        return datetime.now(ZoneInfo(settings.calendar_tz)).hour
    except Exception:                                    # noqa: BLE001 - unknown zone name
        return time.localtime().tm_hour


def in_night(d: dict | None = None) -> bool:
    d = d or load()
    a, b, h = int(d["from"]), int(d["to"]), local_hour()
    return a <= h < b if a <= b else (h >= a or h < b)


def allowed() -> bool:
    """May the server do background work right now?"""
    d = load()
    return d["mode"] == "on" or (d["mode"] == "night" and in_night(d))


def set_mode(mode: str, frm: int | None = None, to: int | None = None) -> dict:
    if mode not in MODES:
        raise ValueError("mode: on, night or off")
    d = dict(load())
    d["mode"] = mode
    if frm is not None:
        d["from"] = int(frm) % 24
    if to is not None:
        d["to"] = int(to) % 24
    d["at"] = time.time()
    tmp = _path().with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(_path())
    _cache.update(at=0.0, d=None)
    log.info("background work: %s", mode)
    if not allowed():
        stop_running()
    return summary()


def stop_running() -> None:
    """Off now: what is still running by itself stops (the watch AI training is the long, hot one)."""
    try:
        from . import watchlm
        watchlm.stop()
    except Exception as e:                               # noqa: BLE001
        log.info("stopping watch AI training: %s", e)


def summary() -> dict:
    d = load()
    return {"mode": d["mode"], "from": d["from"], "to": d["to"], "running_now": allowed(), "hour": local_hour(),
            "tz": settings.calendar_tz}


# ---------------------------------------------------------------- what is using the RAM (/api/health "memory")
_mem: dict = {"at": 0.0, "d": None}


def _meminfo() -> dict:
    out = {}
    try:
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return out


def _rss() -> int:
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def memory() -> dict:
    """The machine's RAM, this server's share, and the AI models Ollama holds right now (cached 30 s)."""
    if _mem["d"] is not None and time.time() - _mem["at"] < 30:
        return _mem["d"]
    from . import transcribe
    gb = lambda n: round(n / 2**30, 1)                     # noqa: E731
    mi = _meminfo()
    tot, avail = mi.get("MemTotal", 0), mi.get("MemAvailable", 0)
    d = {"total_gb": gb(tot), "available_gb": gb(avail),
         "used_pct": round(100 * (tot - avail) / tot) if tot else None,
         "oracle_minutes_gb": gb(_rss()), "speech_models": transcribe.models_in_ram(), "ollama": []}
    if settings.summary_backend.lower() == "ollama" or settings.rag:
        try:
            import httpx
            r = httpx.get(f"{settings.ollama_url}/api/ps", timeout=2)
            d["ollama"] = [{"model": m.get("name"), "gb": gb(m.get("size") or 0),
                            "until": m.get("expires_at", "")[:19]} for m in r.json().get("models", [])]
        except Exception:                                  # noqa: BLE001 - Ollama not there / not reachable
            d["ollama"] = None
    _mem.update(at=time.time(), d=d)
    return d
