"""Wake word models for the watch ("Hey Oracle" + offline commands).

The watch asks GET /api/wakeword every few hours; when the version changes it downloads the files and swaps them in,
so a newly trained "Hey Oracle" reaches the watch without a firmware update. Until you upload your own set (the
training kit in training/wakeword/ does that for you), the watch gets the starter set: "Hey Oracle" + the four commands (also built into the firmware).

An uploaded set is a zip (or separate files) with wake.json and the .tflite files it names:
  {"models": [{"id": "wake", "label": "Hey Oracle", "file": "hey_oracle.tflite", "cutoff": 0.97, "window": 5,
               "arena": 30000},
              {"id": "start_meeting", "label": "Start meeting", "file": "start_meeting.tflite", ...}]}
"wake" is the wake word; the other ids the watch knows are start_meeting, voice_memo, stop_recording, mark_this.
"""
import hashlib
import io
import json
import re
import shutil
import time
import zipfile
from pathlib import Path

from .config import settings

STARTER = Path(__file__).parent / "wakeword_starter"
STARTER_MANIFEST = {"models": [{"id": "wake", "label": "Hey Oracle", "file": "wake.tflite", "cutoff": 0.99, "window": 5, "arena": 30000}, {"id": "start_meeting", "label": "Start meeting", "file": "start_meeting.tflite", "cutoff": 0.97, "window": 5, "arena": 30000}, {"id": "voice_memo", "label": "Voice memo", "file": "voice_memo.tflite", "cutoff": 0.85, "window": 5, "arena": 30000}, {"id": "stop_recording", "label": "Stop recording", "file": "stop_recording.tflite", "cutoff": 0.89, "window": 5, "arena": 30000}, {"id": "mark_this", "label": "Mark this", "file": "mark_this.tflite", "cutoff": 0.84, "window": 5, "arena": 30000}]}
COMMANDS = {"wake", "start_meeting", "voice_memo", "stop_recording", "mark_this"}
FILE_OK = re.compile(r"^[A-Za-z0-9_.-]{1,64}\.tflite$")
MAX_FILE = 1024 * 1024


def _dir() -> Path:
    return settings.data_dir / "wakeword"


def _installed() -> dict | None:
    p = _dir() / "wake.json"
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except ValueError:
        return None


def manifest() -> dict:
    own = _installed()
    m = dict(own or STARTER_MANIFEST)
    folder = _dir() if own else STARTER
    h = hashlib.sha256(json.dumps(m["models"], sort_keys=True).encode())
    for e in m["models"]:
        f = folder / e["file"]
        e["size"] = f.stat().st_size if f.exists() else 0
        if f.exists():
            h.update(f.read_bytes())
    m["version"] = h.hexdigest()[:16]
    m["source"] = "yours" if own else "starter"
    m["uploaded_at"] = (own or {}).get("uploaded_at")
    return m


def file_path(name: str) -> Path | None:
    if not FILE_OK.match(name or ""):
        return None
    own = _installed()
    folder = _dir() if own else STARTER
    names = {e["file"] for e in (own or STARTER_MANIFEST)["models"]}
    p = folder / name
    return p if name in names and p.exists() else None


def _check(models: list, files: dict[str, bytes]) -> list:
    if not isinstance(models, list) or not models:
        raise ValueError("wake.json needs a \"models\" list")
    out, seen, wake = [], set(), False
    for e in models:
        if not isinstance(e, dict):
            raise ValueError("each model is an object")
        mid, f = str(e.get("id", "")), str(e.get("file", ""))
        if mid not in COMMANDS:
            raise ValueError(f"unknown id {mid!r} (use one of {', '.join(sorted(COMMANDS))})")
        if mid in seen:
            raise ValueError(f"{mid} twice")
        seen.add(mid)
        wake |= mid == "wake"
        if not FILE_OK.match(f) or f not in files:
            raise ValueError(f"{f or '(no file)'} is missing")
        b = files[f]
        if len(b) > MAX_FILE or b[4:8] != b"TFL3":
            raise ValueError(f"{f} is not a TensorFlow Lite model")
        cutoff = float(e.get("cutoff", 0.97))
        if not 0.3 <= cutoff <= 0.999:
            raise ValueError(f"{f}: cutoff must be between 0.3 and 0.999")
        out.append({"id": mid, "label": str(e.get("label") or mid.replace("_", " ").capitalize())[:40], "file": f,
                    "cutoff": round(cutoff, 3), "window": max(1, min(16, int(e.get("window", 5)))),
                    "arena": max(8000, min(200000, int(e.get("arena", 30000))))})
    if not wake:
        raise ValueError("no wake word (id \"wake\") in the set")
    return out


def install(uploads: dict[str, bytes]) -> dict:
    """uploads: file name -> bytes (a .zip is unpacked). Replaces the whole set."""
    files: dict[str, bytes] = {}
    for name, data in uploads.items():
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for zi in z.infolist():
                    base = Path(zi.filename).name
                    if base and (base == "wake.json" or base.endswith(".tflite")) and zi.file_size <= MAX_FILE:
                        files[base] = z.read(zi)
        else:
            files[Path(name).name] = data
    if "wake.json" not in files:
        raise ValueError("wake.json is missing")
    try:
        raw = json.loads(files["wake.json"])
    except ValueError:
        raise ValueError("wake.json is not valid JSON")
    models = _check(raw.get("models") if isinstance(raw, dict) else raw, files)
    tmp = _dir().with_name("wakeword.new")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for e in models:
        (tmp / e["file"]).write_bytes(files[e["file"]])
    (tmp / "wake.json").write_text(json.dumps({"models": models, "uploaded_at": int(time.time())}, indent=1))
    old = _dir().with_name("wakeword.old")
    shutil.rmtree(old, ignore_errors=True)
    if _dir().exists():
        _dir().rename(old)
    tmp.rename(_dir())
    shutil.rmtree(old, ignore_errors=True)
    return manifest()


def reset() -> dict:
    shutil.rmtree(_dir(), ignore_errors=True)
    return manifest()


# ---------------------------------------------------------------- your voice ("Teach my voice" on the watch)
# The watch records you saying each phrase (about 20 x "Hey Oracle", 8 x each command) and sends the clips here.
# The training kit fetches them (GET /api/wakeword/samples.zip) and trains with them, so the models know your voice.
SAMPLE_MAX = 400 * 1024


def _samples_dir() -> Path:
    return settings.data_dir / "wakeword_samples"


def add_sample(mid: str, data: bytes) -> dict:
    if mid not in COMMANDS:
        raise ValueError(f"unknown phrase {mid!r}")
    if not (1000 < len(data) <= SAMPLE_MAX) or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("send a short WAV clip")
    d = _samples_dir() / mid
    d.mkdir(parents=True, exist_ok=True)
    name = f"{int(time.time() * 1000)}.wav"
    (d / name).write_bytes(data)
    return samples_summary()


def samples_summary() -> dict:
    out = {}
    for mid in sorted(COMMANDS):
        d = _samples_dir() / mid
        out[mid] = len(list(d.glob("*.wav"))) if d.exists() else 0
    return {"samples": out, "total": sum(out.values())}


def samples_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for mid in sorted(COMMANDS):
            d = _samples_dir() / mid
            for f in sorted(d.glob("*.wav")) if d.exists() else []:
                z.write(f, f"{mid}/{f.name}")
    return buf.getvalue()


def clear_samples(mid: str | None = None) -> dict:
    if mid:
        if mid not in COMMANDS:
            raise ValueError(f"unknown phrase {mid!r}")
        shutil.rmtree(_samples_dir() / mid, ignore_errors=True)
    else:
        shutil.rmtree(_samples_dir(), ignore_errors=True)
    return samples_summary()
