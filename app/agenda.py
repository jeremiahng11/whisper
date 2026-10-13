"""Upcoming meetings from your calendars: private ICS links (Google, Outlook, iCloud...) and/or a Microsoft 365
(Teams / Outlook) mailbox read through Microsoft Graph.

Set CALENDAR_ICS_URLS to one or more links (comma, space or newline separated), and/or MS_TENANT_ID, MS_CLIENT_ID,
MS_CLIENT_SECRET and MS_CALENDAR_USER (an Entra ID app with the Calendars.Read application permission). The device's Agenda app
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


def graph_users() -> list[str]:
    if not (settings.ms_tenant_id and settings.ms_client_id and settings.ms_client_secret):
        return []
    return [u for u in re.split(r"[\s,;]+", settings.ms_calendar_user.strip()) if u]


def sources() -> int:
    """How many calendars are configured (ICS links + Microsoft 365 mailboxes)."""
    return len(urls()) + len(graph_users())


# ---------------------------------------------------------------- Microsoft Graph (Teams / Outlook calendar)
GRAPH = "https://graph.microsoft.com/v1.0"
_token: tuple[float, str] = (0.0, "")


def _graph_token() -> str:
    global _token
    with _lock:
        if _token[1] and time.time() < _token[0] - 60:
            return _token[1]
    r = httpx.post(f"https://login.microsoftonline.com/{settings.ms_tenant_id}/oauth2/v2.0/token", timeout=20, data={
        "grant_type": "client_credentials", "client_id": settings.ms_client_id,
        "client_secret": settings.ms_client_secret, "scope": "https://graph.microsoft.com/.default"})
    if r.status_code != 200:
        try:
            msg = r.json().get("error_description", "").split("\r")[0]
        except ValueError:
            msg = r.text[:200]
        raise RuntimeError(f"Microsoft sign-in failed ({r.status_code}): {msg[:200]}")
    j = r.json()
    with _lock:
        _token = (time.time() + int(j.get("expires_in", 3600)), j["access_token"])
    return _token[1]


def _graph_time(v: dict) -> int:
    """{"dateTime": "2026-10-13T06:00:00.0000000", "timeZone": "UTC"} -> epoch seconds."""
    s = (v or {}).get("dateTime", "")[:19]
    if not s:
        return 0
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def events_from_graph(user: str, start: datetime, end: datetime) -> list[dict]:
    key = f"graph:{user}:{start.date()}:{end.date()}"
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < settings.calendar_cache_s:
            import json as _json
            return _json.loads(hit[1])
    tok = _graph_token()
    url = f"{GRAPH}/users/{user}/calendarView"
    params = {"startDateTime": start.astimezone(timezone.utc).isoformat(), "endDateTime": end.astimezone(timezone.utc).isoformat(),
              "$select": "subject,start,end,isAllDay,location,attendees,organizer,isCancelled,showAs", "$top": "200",
              "$orderby": "start/dateTime"}
    out = []
    for _ in range(10):                                   # pages
        r = httpx.get(url, params=params, timeout=20, headers={"Authorization": f"Bearer {tok}", "Prefer": 'outlook.timezone="UTC"'})
        if r.status_code != 200:
            try:
                msg = r.json().get("error", {}).get("message", "")
            except ValueError:
                msg = r.text[:200]
            raise RuntimeError(f"Microsoft calendar for {user} failed ({r.status_code}): {msg[:200]}")
        j = r.json()
        for ev in j.get("value", []):
            if ev.get("isCancelled") or ev.get("showAs") == "free" and not ev.get("attendees"):
                continue
            people = []
            org = ((ev.get("organizer") or {}).get("emailAddress") or {}).get("name", "")
            for p in ([{"emailAddress": {"name": org}, "type": "required"}] if org else []) + (ev.get("attendees") or []):
                if p.get("type") == "resource":
                    continue                              # rooms
                n = ((p.get("emailAddress") or {}).get("name") or "").strip()
                if n and "@" not in n and n not in people:
                    people.append(n[:60])
            s, e, all_day = _graph_time(ev.get("start")), _graph_time(ev.get("end")), bool(ev.get("isAllDay"))
            if all_day:                                   # all-day events are dates in the mailbox's own day
                tz = _tz()
                ds = datetime.strptime(ev["start"]["dateTime"][:10], "%Y-%m-%d")
                de = datetime.strptime(ev["end"]["dateTime"][:10], "%Y-%m-%d")
                s, e = int(ds.replace(tzinfo=tz).timestamp()), int(de.replace(tzinfo=tz).timestamp())
            out.append({"title": (ev.get("subject") or "").strip() or "(no title)", "start": s, "end": e, "all_day": all_day,
                        "location": ((ev.get("location") or {}).get("displayName") or "").strip()[:120], "attendees": people[:30]})
        url, params = j.get("@odata.nextLink"), None
        if not url:
            break
    import json as _json
    with _lock:
        _cache[key] = (time.time(), _json.dumps(out).encode())
    return out


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


def _people(ev) -> list[str]:
    out = []
    vals = ev.get("ATTENDEE", [])
    if not isinstance(vals, list):
        vals = [vals]
    org = ev.get("ORGANIZER")
    for v in ([org] if org else []) + vals:
        try:
            if str(getattr(v, "params", {}).get("CUTYPE", "INDIVIDUAL")).upper() not in ("INDIVIDUAL", ""):
                continue                                # rooms, resources
            name = str(v.params.get("CN", "")).strip().strip('"') if hasattr(v, "params") else ""
            if not name or "@" in name:
                name = str(v).split(":", 1)[-1].split("@")[0].replace(".", " ").title()
            if name and name not in out:
                out.append(name[:60])
        except Exception:
            continue
    return out[:30]


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
            "attendees": _people(ev),
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
    for u in graph_users():
        try:
            events += events_from_graph(u, start, end)
        except Exception as e:
            log.warning("microsoft calendar %s failed: %s", u, e)
            errors.append(str(e)[:200])
    seen, uniq = set(), []
    for ev in sorted(events, key=lambda x: (x["start"], x["title"])):
        k = (ev["start"], ev["title"])
        if k not in seen:                           # same meeting in two calendars
            seen.add(k)
            uniq.append(ev)
    return {"events": uniq, "errors": errors, "generated": int(time.time())}
