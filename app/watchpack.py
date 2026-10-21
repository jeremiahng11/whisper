"""The watch's offline answer pack: the questions it can hear without the internet, with their answers.

The watch has no speech-to-text of its own. Espressif's MultiNet (ESP-SR) recognises a fixed list of spoken English
phrases on the chip, so the pack gives it ~150 short phrases ("who is ryan", "when is the v two launch") made
from the knowledge base, each with a short answer to show. The watch downloads it (GET /api/watch/pack) when the
version changes and builds its phrase list from it.

Phrase rules (what MultiNet can take): lowercase letters a-z and spaces only, 2-7 words, numbers as words.
"""
import hashlib
import json
import re

from . import knowledge

MAX_ITEMS = 150
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10 - 2] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _num(n % 100))
    if n < 10000 and n % 1000 == 0:
        return _ONES[n // 1000] + " thousand"
    return " ".join(_ONES[int(d)] for d in str(n))      # long numbers: digit by digit (codes, years read out)


def phrase(q: str) -> str:
    """'When is the V2 launch?' -> 'when is the v two launch' ('' when it can't be said as a short phrase)."""
    s = q.lower().replace("&", " and ").replace("%", " percent ").replace("+", " plus ")
    s = re.sub(r"(\d+)", lambda m: " " + _num(int(m.group(1))) + " ", s)
    s = re.sub(r"[^a-z ]+", " ", s)
    words = s.split()
    if not 2 <= len(words) <= 7:
        return ""
    return " ".join(words)


MAX_PHRASES = 400                                   # what MultiNet7 takes in all (ESP_MN_MAX_PHRASE_NUM)


# ---------------------------------------------------------------- spoken answers (read out offline)
def voice_tag(answer: str) -> str:
    return hashlib.sha1(answer.encode()).hexdigest()[:8]


def voice_dir():
    from .config import settings
    d = settings.data_dir / "watchai" / "voice"
    d.mkdir(parents=True, exist_ok=True)
    return d


def voice_file(fid: str):
    """The answer of fact `fid` read out (16 kHz IMA ADPCM WAV, ~8 KB a second), made once with the server's
    voice (Piper) and kept until the answer changes. None if there is no such fact or no voice on this server."""
    from . import speech
    if not re.fullmatch(r"[0-9a-f]{8}", fid or "") or not speech.tts_available():
        return None
    f = next((x for x in knowledge.facts() if x["id"] == fid), None)
    if not f:
        return None
    a = f["a"][:240]
    p = voice_dir() / f"{fid}-{voice_tag(a)}.wav"
    if not p.exists():
        for old in voice_dir().glob(f"{fid}-*.wav"):  # the answer changed: the old recording goes
            old.unlink(missing_ok=True)
        data = speech.wav_to_ima_adpcm(speech.tts(a))
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
    return p


def build(limit: int = MAX_ITEMS, extra: bool = False) -> dict:
    """extra: also the next questions without their answers ("extra": [{"q", "say"}]) - the watch's own model
    answers those, so the watch can hear up to 300 questions while storing only the pack's answers."""
    from . import speech
    facts = knowledge.facts()
    voice = speech.tts_available()
    # the company essentials (core.py, and facts you pinned) always come first; then what you ask most; then people
    # (up to a quarter of the pack); then your pages, documents and research; meeting facts last (meeting questions
    # are answered online by the server's AI)
    core = [f for f in facts if f.get("core")]
    cid = {f["id"] for f in core}
    often = sorted((f for f in facts if f.get("asked") and f["id"] not in cid), key=lambda f: -f["asked"])
    oid = cid | {f["id"] for f in often}
    people = [f for f in facts if f.get("kind") == "person" and f["id"] not in oid][: max(limit // 4, limit * 2 // 5 if limit > 150 else 0)]
    pid = oid | {f["id"] for f in people}
    rest = sorted((f for f in facts if f["id"] not in pid),
                  key=lambda f: (not f.get("job"), f.get("date") or ""), reverse=True)
    people = core + often + people
    items, seen = [], set()
    for f in people + rest:
        if len(items) >= limit:
            break
        say = [p for p in dict.fromkeys(phrase(x) for x in [f["q"]] + (f.get("alts") or [])) if p and p not in seen]
        if not say:
            continue
        used = sum(len(it["say"]) for it in items)
        if used >= MAX_PHRASES:                     # the watch's speech engine takes 400 phrases in all
            break
        if used + 3 > MAX_PHRASES or (len(items) > limit // 2 and not f.get("core")):
            say = say[:1]                           # a big pack: one way of asking for the later questions
        seen.update(say[:2])
        items.append({"id": f["id"], "q": f["q"], "say": say[:3 if f.get("core") else 2], "a": f["a"][:240],
                      "kind": f.get("kind", "fact"), "date": f.get("date") or ""})
        if f.get("core"):                           # the watch answers these itself, even when online
            items[-1]["core"] = True
            items[-1]["alts"] = [x for x in (f.get("alts") or [])][:6]
            seen.update(say[:3])
        if voice:
            items[-1]["voice"] = voice_tag(f["a"][:240])
    more = []
    if extra:
        used = {it["id"] for it in items}
        room = MAX_PHRASES - sum(len(it["say"]) for it in items)
        for f in people + rest:
            if room <= 0:
                break
            if f["id"] in used:
                continue
            say = [p for p in dict.fromkeys(phrase(x) for x in [f["q"]] + (f.get("alts") or [])) if p and p not in seen][:1]
            if say:
                seen.update(say)
                more.append({"id": f["id"], "q": f["q"], "say": say})
                room -= 1
    out = {"count": len(items), "items": items}
    if extra:
        out["extra"] = more
    out["version"] = hashlib.sha1(json.dumps(out, sort_keys=True).encode()).hexdigest()[:12]
    return out


# ---------------------------------------------------------------- the watch's own words ("Okay.", "Starting the meeting.")
# Spoken with the same voice as the answers (Piper on this server); the watch downloads them once and uses them in
# place of its built-in "Okay." (which stays for when it has none).
PROMPTS = {
    "ok": "Okay.",
    "stopped": "Okay, recording stopped.",
    "marked": "Marked.",
}


def prompt_file(name: str):
    """16 kHz IMA ADPCM WAV of one prompt (silence trimmed), made once and kept until the voice changes."""
    from . import speech
    import io as _io
    import wave as _wave
    import numpy as np
    text = PROMPTS.get(name)
    if not text or not speech.tts_available():
        return None
    tag = voice_tag(text + "|" + str(speech.tts_voices().get("en", "")))
    p = voice_dir() / f"prompt-{name}-{tag}.wav"
    if not p.exists():
        for old in voice_dir().glob(f"prompt-{name}-*.wav"):
            old.unlink(missing_ok=True)
        wav = speech.tts(text)
        with _wave.open(_io.BytesIO(wav), "rb") as r:
            pcm = np.frombuffer(r.readframes(r.getnframes()), dtype="<i2").astype(np.float32)
        loud = np.nonzero(np.abs(pcm) > 0.04 * max(1.0, np.abs(pcm).max()))[0]
        if len(loud):                               # no lead-in silence: it should sound right away
            pcm = pcm[max(0, loud[0] - 160): min(len(pcm), loud[-1] + 1600)]
        pcm = pcm / max(1.0, np.abs(pcm).max()) * 0.85 * 32767
        fade = min(80, len(pcm))
        pcm[:fade] *= np.linspace(0, 1, fade)
        buf = _io.BytesIO()
        with _wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.astype("<i2").tobytes())
        data = speech.wav_to_ima_adpcm(buf.getvalue())
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
    return p


def prompts_info() -> dict:
    """{"voice": tag, "names": [...]} - the watch downloads them again when the tag changes."""
    from . import speech
    if not speech.tts_available():
        return {"voice": "", "names": []}
    return {"voice": voice_tag(json.dumps(PROMPTS, sort_keys=True) + str(speech.tts_voices().get("en", ""))),
            "names": sorted(PROMPTS)}
