"""Everyday questions to the watch, like a phone assistant: the date and time, the weather, and anything about the
world (news, prices, scores, people, places) from a quick web search.

  * every question gets "right now it is <weekday, date, time> in <city>" - the AI otherwise doesn't know the date;
  * weather words -> today's weather and tomorrow's forecast from Open-Meteo (free, no key) for the city named in
    the question, else your home city (HOME_CITY, Singapore);
  * a question about the world (not your company) -> a web search for the question as asked (not mixed with your
    company's name), the results given to the AI to answer from.
"""
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from .config import settings

log = logging.getLogger("worldinfo")

WEATHER = re.compile(r"\b(weather|temperature|rain|raining|sunny|hot|cold|humid|humidity|forecast|umbrella|storm|"
                     r"thunder|wind|windy|degrees)\b", re.I)
WORLD = re.compile(r"\b(news|latest|today|tonight|tomorrow|yesterday|this week|weekend|score|won|win|match|game|"
                   r"price|cost|stock|shares|bitcoin|crypto|exchange rate|currency|convert|dollar|ringgit|yen|euro|"
                   r"president|prime minister|minister|ceo of|capital of|population|how far|distance|how long|"
                   r"how tall|how old|born|died|when is|when was|where is|who won|who is the|what is the|define|"
                   r"meaning of|traffic|flight|open now|opening hours|holiday|public holiday|release|launch|election|"
                   r"recipe|calories|translate|time in|time is it in)\b", re.I)
OURS = re.compile(r"\b(we|our|us|my team|my company|the company|meeting|meetings|minutes|colleague)\b", re.I)

_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog", 51: "light drizzle",
        53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
        67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "light showers", 81: "showers",
        82: "heavy showers", 95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail"}


def home_city() -> str:
    return settings.home_city or "Singapore"


def now_line(tz_min: int | None = None, city: str = "") -> str:
    tz = timezone(timedelta(minutes=settings.default_tz_min if tz_min is None else int(tz_min)))
    now = datetime.now(tz)
    off = int(tz.utcoffset(None).total_seconds() // 60)
    sign = "+" if off >= 0 else "-"
    h = now.strftime("%I:%M %p").lstrip("0").lower()
    return (f"Right now it is {now.strftime('%A %d %B %Y')}, {h} in {city or home_city()} "
            f"(UTC{sign}{abs(off) // 60}{':%02d' % (abs(off) % 60) if abs(off) % 60 else ''}); "
            f"UTC time {datetime.now(timezone.utc).strftime('%H:%M')}. Use this for any question about the date, the "
            f"time, days of the week or how long until something - and for the time in another city, work it out "
            f"from UTC.")


def is_world(q: str) -> bool:
    """A question about the world rather than about the user's company / meetings."""
    return bool(WEATHER.search(q) or WORLD.search(q)) and not OURS.search(q)


def _get_json(url: str, timeout: float = 6) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "OracleMinutes/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _city_in(q: str) -> str:
    m = re.search(r"\b(?:in|at|for|around)\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?:\s+(?:today|tomorrow|tonight|now|this|next)\b|[?.!,]|$)",
                  q, re.I)
    if not m:
        return ""
    c = m.group(1).strip()
    return "" if c.lower() in ("the morning", "the evening", "the afternoon", "general") else c


def weather(q: str, city: str = "") -> str:
    place = _city_in(q) or city or home_city()
    try:
        g = _get_json("https://geocoding-api.open-meteo.com/v1/search?count=1&name=" + urllib.parse.quote(place))
        r = (g.get("results") or [None])[0]
        if not r:
            return ""
        f = _get_json(
            f"https://api.open-meteo.com/v1/forecast?latitude={r['latitude']}&longitude={r['longitude']}"
            "&current=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            "&timezone=auto&forecast_days=2")
    except Exception as e:                                       # noqa: BLE001 - no internet: answer without it
        log.info("weather: %s", e)
        return ""
    c, d = f.get("current") or {}, f.get("daily") or {}
    name = ", ".join(x for x in (r.get("name"), r.get("country")) if x)
    out = [f"Weather now in {name} (Open-Meteo): {_WMO.get(c.get('weather_code'), 'unknown')}, "
           f"{c.get('temperature_2m')}°C (feels like {c.get('apparent_temperature')}°C), humidity "
           f"{c.get('relative_humidity_2m')}%, wind {c.get('wind_speed_10m')} km/h, rain {c.get('precipitation')} mm."]
    for i, label in enumerate(("Today", "Tomorrow")):
        try:
            out.append(f"{label}: {_WMO.get(d['weather_code'][i], '')}, {d['temperature_2m_min'][i]}-"
                       f"{d['temperature_2m_max'][i]}°C, {d['precipitation_probability_max'][i]}% chance of rain.")
        except (KeyError, IndexError, TypeError):
            pass
    return " ".join(out)


_TOMORROW = re.compile(r"\b(tomorrow|tmr|tmrw)\b", re.I)
_TODAY_LATER = re.compile(r"\b(later|tonight|this (afternoon|evening))\b", re.I)
_NOT_JUST_WEATHER = re.compile(r"\b(why|explain|compare|history|record|average|climate|last (week|year|month)|"
                               r"next week|weekend|should i|what to wear|wear)\b", re.I)


def weather_answer(q: str, city: str = "") -> str:
    """A spoken answer for a plain weather question, straight from Open-Meteo - no AI, so it comes in about a second
    ("How's the weather?", "Will it rain tomorrow?", "What's the temperature in Tokyo?"). "" = let the AI answer."""
    if not WEATHER.search(q) or OURS.search(q) or _NOT_JUST_WEATHER.search(q) or len(q.split()) > 14:
        return ""
    place = _city_in(q) or city or home_city()
    try:
        g = _get_json("https://geocoding-api.open-meteo.com/v1/search?count=1&name=" + urllib.parse.quote(place), 5)
        r = (g.get("results") or [None])[0]
        if not r:
            return ""
        f = _get_json(
            f"https://api.open-meteo.com/v1/forecast?latitude={r['latitude']}&longitude={r['longitude']}"
            "&current=temperature_2m,apparent_temperature,weather_code"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            "&timezone=auto&forecast_days=2", 5)
    except Exception as e:                                       # noqa: BLE001 - the AI answers instead
        log.info("weather answer: %s", e)
        return ""
    c, d = f.get("current") or {}, f.get("daily") or {}
    name = r.get("name") or place
    try:
        def day(i: int, label: str) -> str:
            rain = d["precipitation_probability_max"][i]
            return (f"{label} in {name}: {_WMO.get(d['weather_code'][i], 'mixed weather')}, "
                    f"{round(d['temperature_2m_min'][i])} to {round(d['temperature_2m_max'][i])} degrees, "
                    f"{rain} percent chance of rain.")
        if _TOMORROW.search(q):
            return day(1, "Tomorrow")
        now = (f"Right now in {name} it's {_WMO.get(c.get('weather_code'), 'mixed weather')} and "
               f"{round(c.get('temperature_2m'))} degrees, feels like {round(c.get('apparent_temperature'))}.")
        rain = d["precipitation_probability_max"][0]
        rest = (f" Today {round(d['temperature_2m_min'][0])} to {round(d['temperature_2m_max'][0])} degrees, "
                f"{rain} percent chance of rain.")
        if re.search(r"\b(rain|umbrella|storm|thunder)\b", q, re.I):
            rest = (f" {'Yes, bring an umbrella' if rain >= 50 else 'Probably not' if rain < 25 else 'Maybe'}: "
                    f"{rain} percent chance of rain today.")
        return now + rest
    except (KeyError, IndexError, TypeError, ValueError):
        return ""


def context(q: str, tz_min: int | None = None, city: str = "", web=None) -> str:
    """What goes along with an everyday question (besides the date and time, which always does). web: a search for
    it that already started (a future) - used instead of searching again."""
    parts = []
    if WEATHER.search(q):
        w = weather(q, city)
        if w:
            parts.append(w)
    if is_world(q) and not (WEATHER.search(q) and parts):
        from . import research
        t0 = time.time()
        w = web.result() if web is not None else research.live_context(q, plain=True)
        log.info("world search %.1fs: %d chars", time.time() - t0, len(w))
        if w:
            parts.append(w)
    return "\n\n".join(parts)
