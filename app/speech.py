"""Voice features for the round Idea Saver: quick speech-to-text, voice commands, and text-to-speech.

- stt(path): transcribe a short clip right away (not through the job queue).
- intent(text): turn "remind me at 3 to call San" / "timer 10 minutes" / "add task ..." into an action.
  Clear patterns are parsed with rules (instant); anything else goes to the LLM.
- tts(text, lang): speech as 16 kHz mono WAV with Piper voices (optional, TTS=piper).
"""
import io
import struct
import json
import logging
import os
import re
import threading
import time
import wave
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from . import redact, summarize, transcribe, voices
from .config import settings

log = logging.getLogger("speech")


# ---------------------------------------------------------------- speech to text
def pick_language(path: str, candidates: list[str]) -> Optional[str]:
    """Detect the spoken language but only choose among `candidates` (e.g. ["en", "zh"] for a two-way
    translator) - stops accented English being heard as Malay, Indonesian, Welsh..."""
    cands = [c.strip().lower() for c in candidates if c.strip()]
    if not cands:
        return None
    if len(cands) == 1 or settings.transcriber == "fake":
        return cands[0]
    try:
        audio = transcribe.load_audio(path)
        _, _, probs = transcribe.get_model().detect_language(audio, vad_filter=True)
        p = dict(probs)
        return max(cands, key=lambda c: p.get(c, 0.0))
    except Exception as e:                                   # noqa: BLE001
        log.warning("language pick failed: %s", e)
        return cands[0]


def stt(path: str, language: Optional[str] = None, candidates: Optional[list[str]] = None) -> dict:
    if not language and candidates:
        language = pick_language(path, candidates)
    transcribe.quick.on = True
    t0 = time.time()
    try:
        tr = transcribe.transcribe(path, language, lambda p, s: None, diarize=False, prompt=voices.whisper_prompt())
    finally:
        transcribe.quick.on = False
    log.info("speech to text: %.1fs for %.1fs of speech", time.time() - t0, tr.duration)
    from . import timing
    timing.note("speech to text", time.time() - t0, f"{tr.duration:.1f}s of speech")
    text = " ".join(s.text.strip() for s in tr.segments).strip()
    if settings.redact:
        text = redact.redact(text)
    return {"text": text, "language": tr.language, "duration": round(tr.duration, 2)}


# ---------------------------------------------------------------- voice commands
NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
       "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "sixty": 60,
       "half an": 0.5, "half a": 0.5}
UNIT = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}


def _num(s: str) -> Optional[float]:
    s = s.strip().lower()
    try:
        return float(s)
    except ValueError:
        return NUM.get(s)


def _duration(text: str) -> Optional[int]:
    total, found = 0.0, False
    for m in re.finditer(r"(\d+(?:\.\d+)?|half an|half a|an|a|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|"
                         r"twenty|thirty|forty-five|forty|sixty)\s*(second|sec|minute|min|hour|hr)s?\b", text, re.I):
        n = _num(m.group(1))
        if n is not None:
            total += n * UNIT[m.group(2).lower()]
            found = True
    return int(total) if found and total > 0 else None


def _clock(text: str, now: datetime) -> Optional[datetime]:
    """'at 3', 'at 3:30 pm', 'at 15:00', 'tomorrow at 9', 'in 20 minutes', 'tonight at 8'."""
    t = text.lower()
    m = re.search(r"\bin\s+((?:\d+(?:\.\d+)?|half an|half a|an|a|one|two|three|four|five|ten|fifteen|twenty|thirty|"
                  r"forty-five|forty|sixty)\s*(?:second|sec|minute|min|hour|hr)s?(?:\s+and\s+\S+\s+\w+)?)", t)
    if m:
        d = _duration(m.group(1))
        if d:
            return now + timedelta(seconds=d)
    m = re.search(r"\b(?:at|by)\s+(\d{1,2})(?::|\.)?(\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?", t)
    if not m:
        m = re.search(r"\b(noon|midday|midnight)\b", t)
        if not m:
            return None
        h, mi, ap = (12, 0, None) if m.group(1) in ("noon", "midday") else (0, 0, None)
    else:
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(".", "")
    if h > 23 or mi > 59:
        return None
    day = now
    if "tomorrow" in t:
        day = now + timedelta(days=1)
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    elif not ap and h < 12 and ("tonight" in t or "evening" in t or "afternoon" in t):
        h += 12
    cand = day.replace(hour=h, minute=mi, second=0, microsecond=0)
    if "tomorrow" not in t and cand <= now:
        if not ap and h < 12 and cand.replace(hour=h + 12) > now:      # "at 3" said at 1 pm means 3 pm
            cand = cand.replace(hour=h + 12)
        else:
            cand += timedelta(days=1)
    return cand


def _strip_lead(s: str, words: str) -> str:
    return re.sub(rf"^\s*(?:{words})\b[\s,:]*", "", s, flags=re.I).strip(" .,!")


def rule_intent(text: str, now: datetime) -> Optional[dict]:
    t = text.strip().strip(".!?")
    low = t.lower()
    # timer
    if re.search(r"\b(timer|countdown)\b", low) or re.match(r"^(set\s+)?(a\s+)?\d+\s*(minute|min|second|hour)s?\s+timer", low):
        d = _duration(low)
        if d:
            lab = re.search(r"\bfor\s+(?:the\s+)?([a-z][a-z ]{2,30})$", low)
            return {"type": "timer", "seconds": d, "label": lab.group(1).strip() if lab and not _duration(lab.group(1)) else ""}
    # reminder
    m = re.match(r"^(?:please\s+)?remind\s+me\s+(.*)$", low)
    if m:
        rest = t[len(t) - len(m.group(1)):]
        when = _clock(rest, now)
        what = re.sub(r"\b(?:tomorrow\s+)?(?:at|by)\s+\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?|\bin\s+\S+\s+(?:second|sec|minute|min|hour|hr)s?\b|"
                      r"\b(?:tomorrow|tonight|noon|midday|midnight)\b", "", rest, flags=re.I)
        what = _strip_lead(re.sub(r"\s{2,}", " ", what).strip(), "to|that|about")
        repeat = "daily" if re.search(r"\bevery\s*day\b|\bdaily\b", low) else "weekdays" if "weekday" in low else ""
        if when:
            return {"type": "reminder", "text": what or "Reminder", "at": int(when.timestamp()), "repeat": repeat}
    # task
    m = re.match(r"^(?:please\s+)?(?:add|put)\s+(?:a\s+)?(?:task|to-?do)?\s*(?:to\s+)?(.*?)(?:\s+to\s+(?:my\s+)?(?:tasks?|to-?do(?:\s+list)?|list))?$", low)
    if m and re.search(r"\b(tasks?|to-?dos?|list)\b", low):
        what = t[low.find(m.group(1)):low.find(m.group(1)) + len(m.group(1))] if m.group(1) else ""
        what = _strip_lead(what, "task|to-?do|to")
        if what:
            return {"type": "task", "text": what}
    # note
    m = re.match(r"^(?:take\s+a\s+note|note\s+(?:that|down)?|make\s+a\s+note|remember\s+that)\b[\s,:]*(.*)$", low)
    if m and m.group(1):
        return {"type": "note", "text": t[len(t) - len(m.group(1)):].strip()}
    # meeting
    if re.match(r"^(?:start|begin|record)\s+(?:a\s+|the\s+|my\s+)?(?:meeting|recording)\b", low):
        title = re.sub(r"^(?:start|begin|record)\s+(?:a\s+|the\s+|my\s+)?(?:meeting|recording)\s*(?:called|named|for|about)?\s*", "", t, flags=re.I)
        return {"type": "meeting", "title": title.strip()}
    # translate
    m = re.match(r"^(?:how do (?:you|i) say|translate)\s+(.+?)\s+(?:in|into|to)\s+([a-z]+(?: [a-z]+)?)$", low)
    if m:
        return {"type": "translate", "text": t[low.find(m.group(1)):low.find(m.group(1)) + len(m.group(1))].strip(" \"'"),
                "lang": m.group(2).title()}
    return None


INTENT_PROMPT = """You turn a spoken command for a small voice assistant into JSON. Now it is {now} ({weekday}), local time.
Reply with ONLY one JSON object, one of:
{{"type":"reminder","text":"<what>","at":"<YYYY-MM-DD HH:MM local>","repeat":""|"daily"|"weekdays"|"weekly"}}
{{"type":"timer","seconds":<number>,"label":"<optional>"}}
{{"type":"task","text":"<task>"}}
{{"type":"note","text":"<note to save>"}}
{{"type":"meeting","title":"<optional title>"}}
{{"type":"translate","text":"<text>","lang":"<language name>"}}
{{"type":"question"}}   (anything else: a question or chat for the AI)

Command: "{text}\""""


_ASK = re.compile(r"^\s*(hey oracle[, ]+)?(who|whos|who's|what|whats|what's|when|where|which|why|how|is|are|was|were|does|do|did|"
                  r"can|could|tell me about|explain|define)\b", re.I)
_ACT = re.compile(r"\b(remind|reminder|timer|alarm|wake me|note|task|to-?do|add|schedule|book|translate|say .* in|"
                  r"record|meeting at|call me|set (a|an|the)?)\b", re.I)


def plain_question(text: str) -> bool:
    """A question, not something to do: answered straight away (the AI isn't asked what kind of request it is)."""
    return bool(_ASK.search(text)) and not _ACT.search(text) and len(text) < 300


def intent(text: str, now_epoch: Optional[int] = None, tz_min: int = 480) -> dict:
    tz = timezone(timedelta(minutes=tz_min))
    now = datetime.fromtimestamp(now_epoch or time.time(), tz)
    text = (text or "").strip()
    if not text:
        return {"type": "none"}
    r = rule_intent(text, now)
    if r:
        r["source"] = "rules"
        return r
    if plain_question(text):                        # "who is Ryan?": no need to ask the AI what you meant first
        return {"type": "question", "source": "rules"}
    if not summarize.enabled():
        return {"type": "question", "source": "default"}
    try:
        out = summarize.chat([{"role": "user", "content": INTENT_PROMPT.format(
            now=now.strftime("%Y-%m-%d %H:%M"), weekday=now.strftime("%A"), text=text.replace('"', "'"))}], fast=True)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
        m = re.search(r"\{.*\}", out, re.S)
        d = json.loads(m.group(0)) if m else {}
    except Exception as e:
        log.warning("intent LLM failed: %s", e)
        d = {}
    typ = d.get("type")
    if typ == "reminder":
        try:
            at = datetime.strptime(str(d.get("at", "")).strip()[:16], "%Y-%m-%d %H:%M").replace(tzinfo=tz)
            return {"type": "reminder", "text": str(d.get("text") or "Reminder")[:120], "at": int(at.timestamp()),
                    "repeat": d.get("repeat") if d.get("repeat") in ("daily", "weekdays", "weekly") else "", "source": "llm"}
        except ValueError:
            return {"type": "question", "source": "llm"}
    if typ == "timer" and float(d.get("seconds") or 0) > 0:
        return {"type": "timer", "seconds": int(float(d["seconds"])), "label": str(d.get("label") or "")[:40], "source": "llm"}
    if typ in ("task", "note") and str(d.get("text", "")).strip():
        return {"type": typ, "text": str(d["text"]).strip()[:300], "source": "llm"}
    if typ == "meeting":
        return {"type": "meeting", "title": str(d.get("title") or "")[:100], "source": "llm"}
    if typ == "translate" and str(d.get("text", "")).strip():
        return {"type": "translate", "text": str(d["text"])[:500], "lang": str(d.get("lang") or "English")[:30], "source": "llm"}
    return {"type": "question", "source": "llm"}


# ---------------------------------------------------------------- text to speech (Piper)
_voices: dict = {}
_vlock = threading.Lock()
LANG_ALIASES = {"english": "en", "chinese": "zh", "mandarin": "zh", "malay": "ms", "indonesian": "id", "japanese": "ja",
                "korean": "ko", "thai": "th", "vietnamese": "vi", "french": "fr", "german": "de", "spanish": "es",
                "tamil": "ta"}


def tts_voices() -> dict:
    """TTS_VOICES="en=en_US-lessac-medium,zh=zh_CN-huayan-medium" -> {"en": "en_US-lessac-medium", ...}"""
    out = {}
    for part in re.split(r"[,\s]+", settings.tts_voices.strip()):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


tts_error = ""          # why the last ElevenLabs call failed ("" = fine) - shown in /api/health


def eleven_on() -> bool:
    return bool(settings.elevenlabs_key and settings.elevenlabs_voice) and settings.tts_backend != "none"


def voice_name() -> str:
    """Which voice speaks (spoken files made earlier are made again when it changes)."""
    if eleven_on():
        sp = _eleven_speed()
        return f"eleven:{settings.elevenlabs_voice}:{settings.elevenlabs_model}" + (f":x{sp:g}" if sp != 1.0 else "")
    return str(tts_voices().get("en", ""))


def _eleven_speed() -> float:
    return round(min(1.2, max(0.7, settings.elevenlabs_speed or 1.0)), 2)


def _eleven(text: str, model: str = "") -> bytes:
    """16 kHz PCM from ElevenLabs, kept on disk (the same sentence is never paid for twice)."""
    import hashlib
    import json as _json
    import urllib.request
    d = settings.data_dir / "tts_cache"
    d.mkdir(parents=True, exist_ok=True)
    model = (model or settings.elevenlabs_model).strip() or "eleven_multilingual_v2"
    key = hashlib.sha1((voice_name() + "|" + (model if model != settings.elevenlabs_model.strip() else "") + "|" + text)
                       .encode()).hexdigest()
    p = d / f"{key}.pcm"
    if p.exists():
        os.utime(p)
        return p.read_bytes()
    import urllib.error
    global tts_error

    def call(model: str) -> bytes:
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice.strip()}?output_format=pcm_16000",
            data=_json.dumps({"text": text, "model_id": model,
                              **({"voice_settings": {"speed": _eleven_speed()}} if _eleven_speed() != 1.0 else {})}).encode(),
            # (a browser-like agent: Python's default one can be turned away by the API's firewall)
            headers={"xi-api-key": settings.elevenlabs_key.strip(), "Content-Type": "application/json",
                     "Accept": "audio/pcm, */*", "User-Agent": "OracleMinutes/1.23 (+https://github.com/jeremiahng11/whisper)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    try:
        pcm = call(model)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode(errors="replace")[:300]
        except Exception:                           # noqa: BLE001
            pass
        tts_error = f"ElevenLabs HTTP {e.code}: {body}"
        log.warning("%s", tts_error)
        # a model this account / voice can't use: try the standard one before giving up
        if e.code in (400, 404, 422) and model != "eleven_multilingual_v2" and "model" in body.lower():
            pcm = call("eleven_multilingual_v2")
        else:
            raise RuntimeError(tts_error) from e
    except Exception as e:                          # noqa: BLE001 - no internet etc.
        tts_error = f"ElevenLabs: {str(e)[:200]}"
        raise
    tts_error = ""
    if len(pcm) < 320:
        raise RuntimeError("ElevenLabs returned no audio")
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(pcm)
    tmp.replace(p)
    files = sorted(d.glob("*.pcm"), key=lambda f: f.stat().st_mtime)   # keep the cache to its size: oldest go
    total = sum(f.stat().st_size for f in files)
    while files and total > settings.elevenlabs_cache_mb * 1024 * 1024:
        total -= files[0].stat().st_size
        files.pop(0).unlink(missing_ok=True)
    return pcm


def tts_available() -> bool:
    if eleven_on():
        return True
    if settings.tts_backend != "piper":
        return False
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


def _voice_path(name: str):
    """Download a Piper voice (onnx + json) from huggingface.co/rhasspy/piper-voices on first use."""
    d = settings.data_dir / "tts"
    d.mkdir(parents=True, exist_ok=True)
    onnx = d / f"{name}.onnx"
    if not onnx.exists():
        import httpx
        lang_region, voice, quality = name.split("-", 2)
        lang = lang_region.split("_")[0]
        base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{lang_region}/{voice}/{quality}/{name}"
        for ext in (".onnx.json", ".onnx"):
            log.info("downloading piper voice %s%s", name, ext)
            r = httpx.get(base + ext, follow_redirects=True, timeout=300)
            r.raise_for_status()
            (d / f"{name}{ext}").write_bytes(r.content)
    return onnx


def _get_voice(lang: str):
    vs = tts_voices()
    code = LANG_ALIASES.get((lang or "en").lower(), (lang or "en").lower()[:2])
    name = vs.get(code) or vs.get("en")
    if not name:
        raise RuntimeError("no TTS voice configured (TTS_VOICES)")
    with _vlock:
        if name not in _voices:
            from piper import PiperVoice
            _voices[name] = PiperVoice.load(str(_voice_path(name)))
        return _voices[name]


def loud(a: np.ndarray) -> np.ndarray:
    """Spoken answers as loud as the watch's small speaker can take: ElevenLabs voices come out quiet (peaks around a
    third of full scale). Raised so the loud parts (not a single click) reach ~90%, at most 4x; the odd peak above
    that is rounded off instead of clipping."""
    if not len(a):
        return a
    x = np.abs(a.astype(np.float32))
    ref = float(np.percentile(x, 99.9))
    if ref < 50:                                   # silence
        return a
    g = min(4.0, 0.9 * 32767 / ref)
    if g <= 1.02:
        return a
    y = a.astype(np.float32) * g / 32767
    y = np.where(np.abs(y) > 0.9, np.sign(y) * (0.9 + 0.1 * np.tanh((np.abs(y) - 0.9) / 0.1)), y)
    return y * 32767


def to_wav16k(pcm: np.ndarray, rate: int, louder: bool = True) -> bytes:
    a = pcm.astype(np.float32)
    if rate != 16000 and len(a):
        n = int(len(a) * 16000 / rate)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)
    if louder:
        a = loud(a)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.clip(a, -32768, 32767).astype("<i2").tobytes())
    return buf.getvalue()


def warm_voices() -> None:
    """Download (first time) and load every configured voice in the background at start-up."""
    for code in tts_voices():
        try:
            _get_voice(code)
            log.info("tts voice for %s ready", code)
        except Exception as e:                           # noqa: BLE001
            log.warning("tts voice for %s: %s", code, e)


def wav_to_ima_adpcm(wav16: bytes) -> bytes:
    """16 kHz mono PCM16 WAV -> IMA ADPCM WAV in 256-byte blocks (the round Idea Saver's recording format):
    4x smaller, so the watch downloads it and starts talking sooner."""
    try:
        import audioop                                    # stdlib up to 3.12; audioop-lts on 3.13+
    except ImportError:
        return wav16
    from . import transcribe
    with wave.open(io.BytesIO(wav16), "rb") as r:
        pcm = np.frombuffer(r.readframes(r.getnframes()), dtype="<i2")
    align, spb = 256, 505
    pad = (-len(pcm)) % spb
    pcm = np.concatenate([pcm, np.zeros(pad, dtype="<i2")]) if pad else pcm
    out, idx = [], 0
    for b in range(len(pcm) // spb):
        blk = pcm[b * spb:(b + 1) * spb]
        pred = int(blk[0])                                # each block starts with one plain sample + the step index
        nib, state = audioop.lin2adpcm(blk[1:].tobytes(), 2, (pred, idx))
        out.append(struct.pack("<hBB", pred, idx, 0) + nib.translate(transcribe._SWAP))
        idx = state[1]
    data = b"".join(out)
    rate = 16000
    hdr = (b"RIFF" + struct.pack("<I", 40 + len(data)) + b"WAVEfmt " +
           struct.pack("<IHHIIHHHH", 20, 0x11, 1, rate, rate * align // spb, align, 4, 2, spb) +
           b"data" + struct.pack("<I", len(data)))
    return hdr + data


def tts(text: str, lang: str = "en", strict: bool = False, live: bool = False) -> bytes:
    """strict: with ElevenLabs set up, only ElevenLabs (an error instead of Piper) - for files kept on the watch, which
    would otherwise keep Piper's voice for good when ElevenLabs failed once (quota, rate limit, no internet)."""
    text = re.sub(r"[*#`_>\[\]]", "", text or "").strip()[:2000]
    if not text:
        return to_wav16k(np.zeros(1600), 16000)
    if eleven_on():
        try:
            pcm = np.frombuffer(_eleven(text, settings.elevenlabs_model_live if live else ""), dtype="<i2")
            return to_wav16k(pcm, 16000)
        except Exception as e:                      # noqa: BLE001 - no internet / quota: Piper says it instead
            log.warning("ElevenLabs failed (%s)%s", e, "" if strict else " - using Piper")
            if strict or settings.tts_backend != "piper":
                raise
    voice = _get_voice(lang)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        if hasattr(voice, "synthesize_wav"):          # piper-tts >= 1.3
            voice.synthesize_wav(text, w)
        else:
            voice.synthesize(text, w)
    buf.seek(0)
    with wave.open(buf, "rb") as r:
        rate = r.getframerate()
        pcm = np.frombuffer(r.readframes(r.getnframes()), dtype="<i2")
    return to_wav16k(pcm, rate)
