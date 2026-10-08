"""Speech-to-text with faster-whisper, plus optional speaker labels (pyannote)."""
import logging
import threading
import wave
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import settings

log = logging.getLogger("transcribe")

ProgressFn = Callable[[float, str], None]   # (0..1, stage text)


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

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()


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


def transcribe(path: str, language: Optional[str], progress: ProgressFn, diarize: bool = False) -> Transcript:
    if settings.transcriber == "fake":
        tr = _fake_transcribe(path, language, progress)
    else:
        tr = _whisper_transcribe(path, language, progress)
    if diarize and tr.segments:
        progress(1.0, "labelling speakers")
        assign_speakers(path, tr)
    return tr


def _whisper_transcribe(path: str, language: Optional[str], progress: ProgressFn) -> Transcript:
    progress(0.0, "loading model")
    model = get_model()
    progress(0.0, "transcribing")
    kwargs = dict(
        language=language or settings.default_language or None,
        beam_size=settings.whisper_beam,
        vad_filter=settings.vad,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,     # avoids Whisper getting stuck repeating a phrase
    )
    if settings.initial_prompt:
        kwargs["initial_prompt"] = settings.initial_prompt
    segs, info = model.transcribe(path, **kwargs)
    duration = float(info.duration or 0)
    tr = Transcript(info.language, float(info.language_probability or 0), duration, settings.whisper_model)
    for s in segs:                                   # generator: decoding happens while we iterate
        text = s.text.strip()
        if text:
            tr.segments.append(Segment(float(s.start), float(s.end), text))
        if duration > 0:
            progress(min(1.0, s.end / duration), "transcribing")
    return tr


def _fake_transcribe(path: str, language: Optional[str], progress: ProgressFn) -> Transcript:
    """Test stand-in: one segment per 5 s of a WAV file."""
    with wave.open(path, "rb") as w:
        duration = w.getnframes() / float(w.getframerate())
    tr = Transcript(language or settings.default_language or "en", 1.0, duration, "fake")
    t, i = 0.0, 1
    while t < duration:
        end = min(duration, t + 5)
        tr.segments.append(Segment(t, end, f"This is test sentence number {i}."))
        progress(end / duration if duration else 1.0, "transcribing")
        t, i = end, i + 1
    return tr


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


def assign_speakers(path: str, tr: Transcript) -> None:
    """Give each segment the speaker who talks most during it. Failure only loses the labels."""
    try:
        import torch
        from faster_whisper.audio import decode_audio
        audio = decode_audio(path, sampling_rate=16000)
        out = _get_diarizer()({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000})
        ann = getattr(out, "speaker_diarization", out)      # pyannote 4 wraps the annotation
        turns = [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]
    except Exception as e:
        log.warning("speaker labelling failed: %s", e)
        return
    names: dict[str, str] = {}
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
