"""Upcoming meetings from your calendars' private ICS links (Google, Outlook, iCloud...).

Set CALENDAR_ICS_URLS to one or more links (comma, space or newline separated). The device's Agenda app
calls GET /api/agenda?days=7 and gets {"events": [{title, start, end, all_day, location}]} with epoch seconds.
"""
import logging
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

from .config import settings

log = logging.getLogger("agenda")
_lock = threading.Lock()
_cache: dict[str, tuple[float, bytes]] = {}


def urls() -> list[str]:
    return [u for u in re.split(r"[\s,]+", settings.calendar_ics_urls.strip()) if u]


def _tz():
    try:
        return ZoneInfo(settings.calendar_tz) if settings.calendar_tz else timezone.utc
    except Exception:
        return timezone.utc


def _fetch(url: str) -> bytes:
    with _lock:
        hit = _cache.get(url)
        if hit and time.time() - hit[0] < settings.calendar_cache_s:
            return hit[1]
    u = re.sub(r"^webcals?://", "https://", url, flags=re.I)
    try:
        r = httpx.get(u, timeout=20, follow_redirects=True)
        r.raise_for_status()
        data = r.content
    except Exception as e:
        if hit:                                     # keep using the last good copy
            log.warning("calendar fetch failed (%s), using cached copy", e)
            return hit[1]
        raise
    with _lock:
        _cache[url] = (time.time(), data)
    return data


def _epoch(v, tz) -> tuple[int, bool]:
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=tz)
        return int(v.timestamp()), False
    if isinstance(v, date):
        return int(datetime(v.year, v.month, v.day, tzinfo=tz).timestamp()), True
    return 0, False


def events_from_ics(data: bytes, start: datetime, end: datetime) -> list[dict]:
    import icalendar
    import recurring_ical_events

    tz = _tz()
    cal = icalendar.Calendar.from_ical(data)
    out = []
    for ev in recurring_ical_events.of(cal).between(start, end):
        if str(ev.get("STATUS", "")).upper() == "CANCELLED":
            continue
        s, all_day = _epoch(ev.decoded("DTSTART"), tz)
        if "DTEND" in ev:
            e, _ = _epoch(ev.decoded("DTEND"), tz)
        elif "DURATION" in ev:
            e = s + int(ev.decoded("DURATION").total_seconds())
        else:
            e = s + (86400 if all_day else 0)
        out.append({
            "title": str(ev.get("SUMMARY", "")).strip() or "(no title)",
            "start": s, "end": e, "all_day": all_day,
            "location": str(ev.get("LOCATION", "")).strip().splitlines()[0][:120] if ev.get("LOCATION") else "",
        })
    return out


def agenda(days: int = 7) -> dict:
    tz = _tz()
    now = datetime.now(tz)
    start = datetime(now.year, now.month, now.day, tzinfo=tz)
    end = start + timedelta(days=max(1, min(days, 60)))
    events, errors = [], []
    for u in urls():
        try:
            events += events_from_ics(_fetch(u), start, end)
        except Exception as e:
            log.warning("calendar %s failed: %s", u[:40], e)
            errors.append(str(e)[:200])
    seen, uniq = set(), []
    for ev in sorted(events, key=lambda x: (x["start"], x["title"])):
        k = (ev["start"], ev["title"])
        if k not in seen:                           # same meeting in two calendars
            seen.add(k)
            uniq.append(ev)
    return {"events": uniq, "errors": errors, "generated": int(time.time())}
