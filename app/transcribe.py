"""Speech-to-text with faster-whisper, plus optional speaker labels (pyannote).

Audio is decoded once to 16 kHz mono float32. Recordings that are still being uploaded ("live") are
transcribed in parts: `partial()` handles the audio received so far, and the final `transcribe()` only
does the rest.
"""
import logging
import struct
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .config import settings

log = logging.getLogger("transcribe")

ProgressFn = Callable[[float, str], None]   # (0..1, stage text)
SR = 16000


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""


@dataclass
class Transcript:
    language: str
    language_prob: float
    duration: float
    model: str
    segments: list[Segment] = field(default_factory=list)
    speakers: dict = field(default_factory=dict)        # "Speaker 1" -> {"seconds": .., "embedding": [..]}

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()


# ---------------------------------------------------------------- audio
def _wav_pcm16(path: str) -> Optional[np.ndarray]:
    """Raw PCM16 WAV -> float32, ignoring the size fields (a WAV still being recorded has them unset)."""
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return None
        fmt = None
        while True:
            ch = f.read(8)
            if len(ch) < 8:
                return None
            cid, size = ch[:4], struct.unpack("<I", ch[4:])[0]
            if cid == b"fmt ":
                fmt = struct.unpack("<HHIIHH", f.read(16))
                f.seek(size - 16, 1)
            elif cid == b"data":
                if not fmt or fmt[0] != 1 or fmt[5] != 16:
                    return None
                raw = f.read()
                break
            else:
                f.seek(size + (size & 1), 1)
    ch, rate = fmt[1], fmt[2]
    raw = raw[: len(raw) // (2 * ch) * 2 * ch]
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if rate != SR:
        n = int(len(a) * SR / rate)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32) if n else a[:0]
    return a


def load_audio(path: str) -> np.ndarray:
    a = None
    try:
        a = _wav_pcm16(path)
    except Exception as e:
        log.debug("not a plain WAV (%s), decoding with ffmpeg", e)
    if a is None:
        from faster_whisper.audio import decode_audio
        a = decode_audio(path, sampling_rate=SR)
    return a


def fix_wav_header(path: str) -> None:
    """Set the RIFF/data sizes from the real file length (needed after a live upload of a growing WAV)."""
    try:
        with open(path, "r+b") as f:
            head = f.read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return
            f.seek(0, 2)
            total = f.tell()
            pos = 12
            while pos + 8 <= total:
                f.seek(pos)
                cid, size = f.read(4), struct.unpack("<I", f.read(4))[0]
                if cid == b"data":
                    f.seek(pos + 4)
                    f.write(struct.pack("<I", total - pos - 8))
                    f.seek(4)
                    f.write(struct.pack("<I", total - 8))
                    return
                pos += 8 + size + (size & 1)
    except OSError as e:
        log.warning("could not fix WAV header of %s: %s", path, e)


# ---------------------------------------------------------------- whisper model (loaded once, kept in RAM)
_model = None
_model_lock = threading.Lock()


def _device_and_compute() -> tuple[str, str]:
    device = settings.whisper_device
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = settings.whisper_compute
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            device, compute = _device_and_compute()
            log.info("loading whisper model %s on %s (%s)", settings.whisper_model, device, compute)
            _model = WhisperModel(
                settings.whisper_model,
                device=device,
                compute_type=compute,
                cpu_threads=settings.whisper_threads,
                download_root=str(settings.model_dir),
            )
            log.info("whisper model ready")
        return _model


def model_loaded() -> bool:
    return _model is not None


def _run(audio: np.ndarray, offset: float, language: Optional[str], prompt: str,
         progress: ProgressFn, total: float) -> tuple[list[Segment], str, float]:
    """Transcribe one stretch of audio; returns (segments with absolute times, language, probability)."""
    if len(audio) < SR // 2:
        return [], language or "", 0.0
    if settings.transcriber == "fake":
        segs, t = [], 0.0
        dur = len(audio) / SR
        while t < dur - 0.01:
            end = min(dur, t + 5)
            i = int(round((offset + t) / 5)) + 1
            segs.append(Segment(round(offset + t, 2), round(offset + end, 2), f"This is test sentence number {i}."))
            progress(min(1.0, (offset + end) / total) if total else 1.0, "transcribing")
            t = end
        return segs, language or settings.default_language or "en", 1.0
    model = get_model()
    kwargs = dict(
        language=language or settings.default_language or None,
        beam_size=settings.whisper_beam,
        vad_filter=settings.vad,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,     # avoids Whisper getting stuck repeating a phrase
    )
    if prompt:
        kwargs["initial_prompt"] = prompt
    segs, info = model.transcribe(audio, **kwargs)
    out = []
    for s in segs:                                   # generator: decoding happens while we iterate
        text = s.text.strip()
        if text:
            out.append(Segment(round(offset + float(s.start), 2), round(offset + float(s.end), 2), text))
        if total > 0:
            progress(min(1.0, (offset + s.end) / total), "transcribing")
    return out, info.language, float(info.language_probability or 0)


def transcribe(path: str, language: Optional[str], progress: ProgressFn, diarize: bool = False,
               prompt: str = "", num_speakers: Optional[int] = None, prior: Optional[dict] = None) -> Transcript:
    """prior = state from partial() for live uploads: only the audio after prior["until"] is transcribed."""
    progress(0.0, "loading audio")
    audio = load_audio(path)
    total = len(audio) / SR
    until = float(prior.get("until", 0)) if prior else 0.0
    segs = [Segment(**s) for s in prior.get("segments", [])] if prior else []
    lang = language or (prior or {}).get("language") or None
    if until > 0:
        progress(min(1.0, until / total) if total else 0, f"transcribing the last {int((total - until) / 60) + 1} min")
    else:
        progress(0.0, "loading model" if settings.transcriber != "fake" and not model_loaded() else "transcribing")
    new, detected, prob = _run(audio[int(until * SR):], until, lang, prompt, progress, total)
    tr = Transcript(lang or detected, prob if not lang else 1.0, total,
                    "fake" if settings.transcriber == "fake" else settings.whisper_model, segs + new)
    if diarize and tr.segments:
        progress(1.0, "labelling speakers")
        assign_speakers(audio, tr, num_speakers)
    return tr


def partial(path: str, state: dict, prompt: str = "", max_part_s: int = 600) -> bool:
    """Live upload: transcribe audio that arrived since the last part, leaving the last ~20 s for later
    (a sentence may still be going on). Updates `state` in place; returns True if anything was done."""
    audio = load_audio(path)
    dur = len(audio) / SR
    until = float(state.get("until", 0))
    cut = min(dur - 20, until + max_part_s)
    if cut - until < min(settings.live_min_new_s, max_part_s):
        return False
    segs, lang, _ = _run(audio[int(until * SR):int(cut * SR)], until, state.get("language") or None, prompt,
                         lambda p, s: None, 0)
    keep = [s for s in segs if s.end <= cut - 3]
    if segs and not keep:                           # one very long segment that started well before the cut
        keep = [s for s in segs if s.start < cut - 30]
    if not keep and segs:                           # speech runs into the cut: wait for more audio
        return False
    new_until = keep[-1].end if keep else cut - 3   # no speech at all: skip the silence
    state["segments"] = state.get("segments", []) + [s.__dict__ for s in keep]
    state["until"] = round(max(until, new_until), 2)
    if lang and not state.get("language"):
        state["language"] = lang
    log.info("live part: %.0f-%.0f s, %d segments", until, state["until"], len(keep))
    return True


# ---------------------------------------------------------------- speaker labels
_diar = None


def _get_diarizer():
    global _diar
    if _diar is None:
        if not settings.hf_token:
            raise RuntimeError("DIARIZE=1 needs HF_TOKEN (and accepting the pyannote model terms on huggingface.co)")
        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise RuntimeError("speaker labels need the image built with INSTALL_DIARIZE=1") from e
        try:
            _diar = Pipeline.from_pretrained(settings.diarize_model, use_auth_token=settings.hf_token)
        except TypeError:                                   # pyannote >= 4 renamed the argument
            _diar = Pipeline.from_pretrained(settings.diarize_model, token=settings.hf_token)
        if _diar is None:
            raise RuntimeError(f"could not load {settings.diarize_model} - accept its terms on huggingface.co")
        try:
            import torch
            if torch.cuda.is_available():
                _diar.to(torch.device("cuda"))
        except Exception:
            pass
    return _diar


def _diarize(audio: np.ndarray, num_speakers: Optional[int]) -> tuple[list[tuple[float, float, str]], dict]:
    """-> ([(start, end, label)], {label: embedding list})"""
    if settings.diarizer == "fake":                 # tests: two people taking turns every 10 s
        dur = len(audio) / SR
        n = max(1, min(num_speakers or 2, 4))
        turns, t, k = [], 0.0, 0
        while t < dur:
            turns.append((t, min(dur, t + 10), f"SPK_{k % n}"))
            t, k = t + 10, k + 1
        vecs = {f"SPK_{i}": [1.0 if j == i else 0.0 for j in range(4)] for i in range(n)}
        return turns, vecs
    import torch
    kw = {"num_speakers": num_speakers} if num_speakers else {}
    inp = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR}
    pipe = _get_diarizer()
    emb = None
    try:
        out = pipe(inp, return_embeddings=True, **kw)
        if isinstance(out, tuple):
            out, emb = out
    except TypeError:
        out = pipe(inp, **kw)
    ann = getattr(out, "speaker_diarization", out)       # pyannote 4 wraps the annotation
    if emb is None:
        emb = getattr(out, "speaker_embeddings", None)
    turns = [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]
    vecs = {}
    if emb is not None:
        for i, label in enumerate(ann.labels()):
            if i < len(emb) and not np.any(np.isnan(emb[i])):
                vecs[label] = [round(float(x), 5) for x in emb[i]]
    return turns, vecs


def assign_speakers(audio: np.ndarray, tr: Transcript, num_speakers: Optional[int] = None) -> None:
    """Give each segment the speaker who talks most during it. Failure only loses the labels."""
    try:
        turns, vecs = _diarize(audio, num_speakers)
    except Exception as e:
        log.warning("speaker labelling failed: %s", e)
        return
    names: dict[str, str] = {}
    secs: dict[str, float] = {}
    for s in tr.segments:
        overlap: dict[str, float] = {}
        for a, b, spk in turns:
            o = min(b, s.end) - max(a, s.start)
            if o > 0:
                overlap[spk] = overlap.get(spk, 0) + o
        if overlap:
            spk = max(overlap, key=overlap.get)
            if spk not in names:
                names[spk] = f"Speaker {len(names) + 1}"
            s.speaker = names[spk]
            secs[names[spk]] = secs.get(names[spk], 0) + (s.end - s.start)
    tr.speakers = {lab: {"seconds": round(secs.get(lab, 0), 1), "embedding": vecs.get(spk)}
                   for spk, lab in names.items()}
