"""Speech-to-text with faster-whisper, plus optional speaker labels (pyannote).

Audio is decoded once to 16 kHz mono float32. Recordings that are still being uploaded ("live") are
transcribed in parts: `partial()` handles the audio received so far, and the final `transcribe()` only
does the rest.
"""
import os
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
    also: list = field(default_factory=list)            # others talking at the same time (overlapping speech)
    words: list = field(default_factory=list)           # [[start, end, "word"], ...] when known


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
                if not fmt:
                    return None
                raw = f.read()
                if fmt[0] == OPUS_TAG and fmt[4] > 0:          # Opus frames (watch1b's compact recordings)
                    return opus_decode(raw, fmt[4])
                if fmt[0] == 0x11 and fmt[1] == 1:             # IMA ADPCM (the round Idea Saver records this)
                    a = ima_adpcm_decode(raw, fmt[4]).astype(np.float32) / 32768.0
                    return _resample(a, fmt[2])
                if fmt[0] != 1 or fmt[5] != 16:
                    return None
                break
            else:
                f.seek(size + (size & 1), 1)
    ch, rate = fmt[1], fmt[2]
    raw = raw[: len(raw) // (2 * ch) * 2 * ch]
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return _resample(a, rate)


def _resample(a: np.ndarray, rate: int) -> np.ndarray:
    if rate != SR:
        n = int(len(a) * SR / rate)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32) if n else a[:0]
    return a


_SWAP = bytes(((b & 0x0F) << 4) | (b >> 4) for b in range(256))     # WAV IMA: low nibble first


def ima_adpcm_decode(raw: bytes, block_align: int) -> np.ndarray:
    """Mono IMA ADPCM WAV blocks -> int16 samples. A cut-off last block (file still being recorded) is skipped."""
    import audioop                                        # in the stdlib up to 3.12; audioop-lts on 3.13+
    out = []
    n = len(raw) // block_align
    for i in range(n):
        blk = raw[i * block_align:(i + 1) * block_align]
        pred, idx = struct.unpack("<hB", blk[:3])
        idx = min(idx, 88)
        pcm, _ = audioop.adpcm2lin(blk[4:].translate(_SWAP), 2, (pred, idx))
        out.append(struct.pack("<h", pred))
        out.append(pcm)
    return np.frombuffer(b"".join(out), dtype="<i2")


OPUS_TAG = 0x704F       # "Op": the watch's compact format - fixed-size Opus packets (20 ms each) in a WAV data chunk
OPUS_FRAME = 320        # samples per packet at 16 kHz


def opus_decode(raw: bytes, align: int, skip: int = 0) -> np.ndarray:
    """Back-to-back Opus packets of `align` bytes each (CBR, padded) -> 16 kHz float32. A cut-off last packet (file
    still being recorded) is dropped; a damaged packet becomes 20 ms of silence so the timing stays right.
    skip = packets decoded only to warm the decoder up, then dropped."""
    import av
    n = len(raw) // align if align else 0
    if not n:
        return np.zeros(0, np.float32)
    dec = av.CodecContext.create("libopus", "r")
    dec.sample_rate = 48000
    dec.layout = "mono"
    parts, rate = [], 48000
    for i in range(n):
        pk = raw[i * align:(i + 1) * align]
        got = None
        try:
            for fr in dec.decode(av.Packet(pk)):
                a = fr.to_ndarray().reshape(-1)
                if fr.format.name.startswith("s16"):
                    a = a.astype(np.float32) / 32768.0
                elif fr.format.name.startswith("s32"):
                    a = a.astype(np.float32) / 2147483648.0
                else:
                    a = a.astype(np.float32)
                rate = fr.sample_rate or rate
                got = a if got is None else np.concatenate([got, a])
        except Exception as e:                               # noqa: BLE001 - one bad packet must not lose the rest
            log.debug("opus packet %d: %s", i, e)
        if got is None:
            got = np.zeros(rate // 50, np.float32)
        if i >= skip:
            parts.append(got)
    a = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    if rate == 48000:                                         # speech below 8 kHz: every 3rd sample is exact enough
        return np.ascontiguousarray(a[::3])
    return _resample(a, rate)


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


_quick_model = None


def get_quick_model():
    """STT_MODEL (e.g. "small"): a smaller model of its own for short spoken questions - quicker, and it doesn't wait
    for a meeting being transcribed with the main model. "" = the main model."""
    global _quick_model
    if not settings.stt_model or settings.stt_model == settings.whisper_model:
        return get_model()
    with _model_lock:
        if _quick_model is None:
            from faster_whisper import WhisperModel
            device, compute = _device_and_compute()
            log.info("loading quick whisper model %s for spoken questions", settings.stt_model)
            _quick_model = WhisperModel(settings.stt_model, device=device, compute_type=compute,
                                        cpu_threads=settings.whisper_threads, download_root=str(settings.model_dir))
        return _quick_model


def model_loaded() -> bool:
    return _model is not None


quick = threading.local()                           # quick.on: a short spoken question (/api/stt) - speed first


def _run(audio: np.ndarray, offset: float, language: Optional[str], prompt: str,
         progress: ProgressFn, total: float, vad_threshold: Optional[float] = None,
         on_segs: Optional[Callable[[list], None]] = None) -> tuple[list[Segment], str, float]:
    """Transcribe one stretch of audio; returns (segments with absolute times, language, probability).
    vad_threshold = a more sensitive speech detector for one pass (quiet or far-away voices)."""
    if len(audio) < SR // 2:
        return [], language or "", 0.0
    if settings.transcriber == "fake":
        segs, t = [], 0.0
        dur = len(audio) / SR
        while t < dur - 0.01:
            end = min(dur, t + 5)
            i = int(round((offset + t) / 5)) + 1
            text = f"This is test sentence number {i}."
            ws = text.split()
            step = (end - t) / len(ws)
            words = [[round(offset + t + k * step, 2), round(offset + t + (k + 1) * step, 2), w] for k, w in enumerate(ws)]
            segs.append(Segment(round(offset + t, 2), round(offset + end, 2), text, words=words))
            if on_segs:
                on_segs(segs)
            progress(min(1.0, (offset + end) / total) if total else 1.0, "transcribing")
            t = end
        return segs, language or settings.default_language or "en", 1.0
    model = get_quick_model() if getattr(quick, "on", False) else get_model()
    kwargs = dict(
        language=language or settings.default_language or None,
        beam_size=settings.whisper_beam,
        vad_filter=settings.vad,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=400,
                            threshold=settings.vad_threshold if vad_threshold is None else vad_threshold),
        condition_on_previous_text=False,     # avoids Whisper getting stuck repeating a phrase
        word_timestamps=True,                 # lets speaker changes inside a line be split at the right word
    )
    if getattr(quick, "on", False):           # a question to the watch: greedy decoding, no word times (2-3x faster)
        kwargs.update(beam_size=1, word_timestamps=False)
    if prompt:
        kwargs["initial_prompt"] = prompt
    if settings.normalize:
        audio = level_audio(audio)
    segs, info = model.transcribe(audio, **kwargs)
    out = []
    for s in segs:                                   # generator: decoding happens while we iterate
        text = s.text.strip()
        if text:
            words = [[round(offset + float(w.start), 2), round(offset + float(w.end), 2), w.word]
                     for w in (getattr(s, "words", None) or []) if (w.word or "").strip()]
            out.append(Segment(round(offset + float(s.start), 2), round(offset + float(s.end), 2), text, words=words))
            if on_segs:
                on_segs(out)
        if total > 0:
            progress(min(1.0, (offset + s.end) / total), "transcribing")
    return out, info.language, float(info.language_probability or 0)


def transcribe(path: str, language: Optional[str], progress: ProgressFn, diarize: bool = False,
               prompt: str = "", num_speakers: Optional[int] = None, prior: Optional[dict] = None,
               on_text: Optional[Callable[[list, bool], None]] = None) -> Transcript:
    """prior = state from partial() for live uploads: only the audio after prior["until"] is transcribed.
    on_text(segments, done) = the text as it is found, before the (slow) speaker separation: the page shows it
    while the speakers are still being worked out. done=True once all the words are in."""
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
    if on_text and segs:
        on_text(segs, False)
    hook = (lambda out: on_text(segs + out, False)) if on_text else None
    new, detected, prob = _run(audio[int(until * SR):], until, lang, prompt, progress, total, on_segs=hook)
    if on_text:
        on_text(segs + new, True)
    tr = Transcript(lang or detected, prob if not lang else 1.0, total,
                    "fake" if settings.transcriber == "fake" else settings.whisper_model, segs + new)
    if diarize and tr.segments:
        progress(0.0, "separating speakers")
        assign_speakers(audio, tr, num_speakers, lambda f: progress(f, "separating speakers"))
    return tr


def partial(path: str, state: dict, prompt: str = "", max_part_s: int = 600, dur: Optional[float] = None) -> bool:
    """Live upload: transcribe audio that arrived since the last part, leaving the last few seconds (LIVE_HOLD_S) for
    later (a sentence may still be going on). Updates `state` in place; returns True if anything was done.
    `dur` (the length so far, from the file size): only the new stretch is read, not the whole recording again."""
    until = float(state.get("until", 0))
    if dur is None:
        audio = load_audio(path)
        dur = len(audio) / SR
    else:
        audio = None
    cut = min(dur - settings.live_hold_s, until + max_part_s)
    if cut - until < min(settings.live_min_new_s, max_part_s):
        return False
    piece = audio[int(until * SR):int(cut * SR)] if audio is not None else load_range(path, until, cut)
    segs, lang, _ = _run(piece, until, state.get("language") or None, prompt, lambda p, s: None, 0)
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


def level_audio(a: np.ndarray) -> np.ndarray:
    """Bring speech up to a steady level: 2 s windows, gain from each window's loudness (quiet stretches = far voices
    lifted, at most +18 dB), smoothed so it doesn't pump; silence (near the noise floor) is left alone."""
    if len(a) < SR:
        return a
    win = SR * 2
    n = (len(a) + win - 1) // win
    rms = np.array([np.sqrt(np.mean(a[i * win:(i + 1) * win] ** 2) + 1e-12) for i in range(n)])
    floor = np.percentile(rms, 10)
    target = 0.08                                            # about -22 dBFS
    g = np.where(rms > floor * 3, np.clip(target / rms, 1.0, 8.0), 1.0)
    for i in range(1, n):                                    # smooth between windows
        g[i] = 0.6 * g[i] + 0.4 * g[i - 1]
    gain = np.interp(np.arange(len(a)), np.arange(n) * win + win / 2, g)
    out = a * gain
    peak = np.max(np.abs(out)) or 1.0
    return (out / peak * 0.98 if peak > 1.0 else out).astype(np.float32)


# ---------------------------------------------------------------- clips (listen to one stretch of a meeting)
def load_range(path: str, start: float, end: float) -> np.ndarray:
    """Audio from start..end seconds as 16 kHz float32. Plain PCM16 / IMA ADPCM WAVs are read by seeking (fast even
    for a 2-hour meeting); anything else is decoded whole and cut."""
    start, end = max(0.0, start), max(start, end)
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
                fmt = None
                while True:
                    ch = f.read(8)
                    if len(ch) < 8:
                        break
                    cid, size = ch[:4], struct.unpack("<I", ch[4:])[0]
                    if cid == b"fmt ":
                        fmt = struct.unpack("<HHIIHH", f.read(16))
                        f.seek(size - 16 + (size & 1), 1)
                    elif cid == b"data" and fmt:
                        tag, chans, rate, _, align, bits = fmt
                        base = f.tell()
                        if tag == OPUS_TAG and align > 0:                  # Opus packets, 20 ms each
                            f0, f1 = int(start * 50), int(end * 50) + 1
                            warm = min(f0, 5)                              # decoder warm-up before the clip
                            f.seek(base + (f0 - warm) * align)
                            a = opus_decode(f.read((f1 - f0 + warm) * align), align, warm)
                            off = int(start * SR) - f0 * OPUS_FRAME
                            return a[max(0, off):max(0, off) + int(end * SR) - int(start * SR)]
                        if tag == 0x11 and chans == 1:                     # IMA ADPCM blocks
                            spb = (align - 4) * 2 + 1
                            b0, b1 = int(start * rate) // spb, int(end * rate) // spb + 1
                            f.seek(base + b0 * align)
                            a = ima_adpcm_decode(f.read((b1 - b0) * align), align).astype(np.float32) / 32768.0
                            a = _resample(a, rate)
                            off = int(start * SR) - round(b0 * spb * SR / rate)
                            return a[max(0, off):max(0, off) + int(end * SR) - int(start * SR)]
                        if tag == 1 and bits == 16:
                            frame = 2 * chans
                            f.seek(base + int(start * rate) * frame)
                            raw = f.read(int((end - start) * rate) * frame)
                            raw = raw[: len(raw) // frame * frame]
                            a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                            if chans > 1:
                                a = a.reshape(-1, chans).mean(axis=1)
                            return _resample(a, rate)
                        break
                    else:
                        f.seek(size + (size & 1), 1)
    except OSError:
        pass
    a = load_audio(path)
    return a[int(start * SR):int(end * SR)]


def enhance_clip(a: np.ndarray, noise_ref: Optional[np.ndarray] = None) -> np.ndarray:
    """Make a clip easier to listen to (for telling voices apart): rumble filter, spectral noise reduction (noise
    profile from the quietest moments of `noise_ref`, i.e. the clip with some seconds around it), steady loudness and
    a peak limit. Not used for transcription - Whisper gets the original audio."""
    if len(a) < 512:
        return a
    a = a.astype(np.float32)
    # 1. high-pass ~100 Hz (aircon / table / handling rumble)
    k = 0.9615
    y = np.empty_like(a)
    prev_x = prev_y = 0.0
    for i in range(len(a)):                                   # 1st order: y = k * (y + x - x_prev)
        prev_y = k * (prev_y + a[i] - prev_x)
        prev_x = a[i]
        y[i] = prev_y
    a = y
    # 2. spectral gating
    n_fft, hop = 512, 128
    win = np.hanning(n_fft).astype(np.float32)

    def stft(x):
        pad = np.concatenate([np.zeros(n_fft, np.float32), x, np.zeros(n_fft, np.float32)])
        frames = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop] * win
        return np.fft.rfft(frames, axis=1)

    spec = stft(a)
    ref = stft(noise_ref.astype(np.float32)) if noise_ref is not None and len(noise_ref) > n_fft * 4 else spec
    mag_ref = np.abs(ref)
    energy = mag_ref.sum(axis=1)
    quiet = mag_ref[energy <= np.percentile(energy, 20)] if len(energy) > 10 else mag_ref
    noise = np.mean(quiet, axis=0) + 1e-9                    # noise level per frequency
    mag = np.abs(spec)
    gain = np.clip(1.0 - (1.6 * noise / (mag + 1e-9)) ** 2, 0.18, 1.0)   # at most ~15 dB less noise
    for t in range(1, len(gain)):                            # smooth over time (less "musical" noise)
        gain[t] = np.maximum(gain[t], 0.6 * gain[t - 1])
    spec = spec * gain
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * win
    out = np.zeros(len(a) + 2 * n_fft + n_fft, np.float32)
    norm = np.zeros_like(out)
    for t in range(len(frames)):
        out[t * hop:t * hop + n_fft] += frames[t]
        norm[t * hop:t * hop + n_fft] += win ** 2
    out = (out / np.maximum(norm, 1e-6))[n_fft:n_fft + len(a)]
    # 3. loudness: speech to about -20 dBFS, peaks limited
    rms = float(np.sqrt(np.mean(out ** 2)) + 1e-9)
    out = out * min(20.0, 0.1 / rms)
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.95:
        out = np.tanh(out / peak * 1.4) / np.tanh(1.4) * 0.95  # soft limit
    return out.astype(np.float32)


def wav16(a: np.ndarray) -> bytes:
    """float32 16 kHz mono -> a small PCM16 WAV the browser can play."""
    pcm = (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


_embedder = None
diarize_error = ""                                       # why speaker separation failed (shown to you)
embed_error = ""                                         # why voice prints of clips aren't available (shown to you)


def _get_embedder():
    """The speaker-embedding model the diarizer uses (same "voice print" space as its speaker embeddings).
    pyannote 3.x pipelines carry it as ._embedding; otherwise load the model itself."""
    global _embedder
    if _embedder is not None:
        return _embedder
    import torch
    pipe = _get_diarizer()
    e = getattr(pipe, "_embedding", None)
    if e is not None and callable(e):
        _embedder = ("pipeline", e)
        return _embedder
    from pyannote.audio import Inference, Model
    name = getattr(pipe, "embedding", None)
    name = name if isinstance(name, str) and "/" in name else "pyannote/wespeaker-voxceleb-resnet34-LM"
    m = _hf_load(Model, name)
    if m is None:
        raise RuntimeError(f"could not load {name} - accept its terms on huggingface.co")
    inf = Inference(m, window="whole")
    if torch.cuda.is_available():
        inf.to(torch.device("cuda"))
    _embedder = ("inference", inf)
    return _embedder


def embed_clip(a: np.ndarray, t: Optional[float] = None) -> Optional[list[float]]:
    """Voice print of one clip, in the same space as the diarizer's speaker embeddings (None if too short or not
    available - then embed_error says why). t = where the clip starts in the recording (only the test diarizer)."""
    global embed_error
    if len(a) < SR * 1.0:
        return None
    if settings.diarizer == "fake":                      # tests: two people taking turns every 10 s
        k = int((t or 0) // 10) % 2
        return [1.0 if j == k else 0.0 for j in range(4)]
    try:
        import torch
        kind, emb = _get_embedder()
        w = torch.from_numpy(a[: SR * 30].astype(np.float32))
        if kind == "pipeline":
            v = emb(w.reshape(1, 1, -1))
        else:
            v = emb({"waveform": w.reshape(1, -1), "sample_rate": SR})
        v = np.asarray(v.detach().cpu() if hasattr(v, "detach") else v, dtype=np.float32).reshape(-1)
        if not len(v) or np.any(np.isnan(v)):
            return None
        embed_error = ""
        return [round(float(x), 5) for x in v]
    except Exception as e:                                   # noqa: BLE001
        embed_error = str(e)[:200] or type(e).__name__
        log.warning("voice print of a clip failed: %s", e)
        return None


# ---------------------------------------------------------------- speaker labels
_diar = None


def _hf_load(cls, name: str):
    """from_pretrained across pyannote / huggingface_hub versions: pyannote 3 takes use_auth_token, pyannote 4
    takes token, and huggingface_hub >= 1.0 rejects use_auth_token deep inside pyannote 3. The token is also put in
    HF_TOKEN, which huggingface_hub reads by itself, so the last try passes none."""
    os.environ.setdefault("HF_TOKEN", settings.hf_token)
    # torch >= 2.6 loads checkpoints "weights only" by default, which pyannote 3.x model files don't pass
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        import pyannote.audio as pa
        major = int(str(getattr(pa, "__version__", "3")).split(".")[0])
    except Exception:                                       # noqa: BLE001
        major = 3
    tries = (["token", "use_auth_token"] if major >= 4 else ["use_auth_token", "token"]) + [None]
    errs = []
    for kw in tries:
        try:
            return cls.from_pretrained(name, **({kw: settings.hf_token} if kw else {}))
        except TypeError as e:                              # this version doesn't take that argument
            errs.append(f"{kw or 'no token arg'}: {e}")
            log.info("from_pretrained(%s) with %s: %s", name, kw, e)
    raise RuntimeError(f"loading {name} failed ({lib_versions()}) - " + " | ".join(errs))


def lib_versions() -> str:
    """Versions of the speaker-model libraries (they must fit together)."""
    from importlib import metadata
    out = []
    for p in ("pyannote.audio", "torch", "torchaudio", "huggingface_hub", "lightning"):
        try:
            out.append(f"{p} {metadata.version(p)}")
        except Exception:                                   # noqa: BLE001
            out.append(f"{p} -")
    return ", ".join(out)


def _get_diarizer():
    global _diar
    if _diar is None:
        if not settings.hf_token:
            raise RuntimeError("DIARIZE=1 needs HF_TOKEN (and accepting the pyannote model terms on huggingface.co)")
        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise RuntimeError("speaker labels need the image built with INSTALL_DIARIZE=1") from e
        _diar = _hf_load(Pipeline, settings.diarize_model)
        if _diar is None:
            raise RuntimeError(f"could not load {settings.diarize_model} - accept its terms on huggingface.co")
        try:
            import torch
            if torch.cuda.is_available():
                _diar.to(torch.device("cuda"))
        except Exception:
            pass
    return _diar


def _diar_hook(progress):
    """pyannote reports its steps; turn them into 0..1 (finding speech ~30 %, voice prints ~65 %, the rest ~5 %)."""
    if not progress:
        return None
    span = {"segmentation": (0.0, 0.30), "speaker_counting": (0.30, 0.32), "embeddings": (0.32, 0.95),
            "discrete_diarization": (0.95, 1.0)}
    last = [0.0]

    def hook(step_name, step_artifact=None, file=None, total=None, completed=None, **_):
        a, b = span.get(step_name, (last[0], last[0]))
        frac = a + (b - a) * (completed / total if total and completed is not None else 1.0)
        if frac >= last[0] + 0.01 or frac >= 0.999:
            last[0] = frac
            try:
                progress(min(1.0, frac))
            except Exception:                            # noqa: BLE001
                pass
    return hook


def _diarize(audio: np.ndarray, num_speakers: Optional[int], progress=None) -> tuple[list[tuple[float, float, str]], dict]:
    """-> ([(start, end, label)], {label: embedding list}). progress(0..1) while it works."""
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
    hook = _diar_hook(progress)
    if hook:
        kw["hook"] = hook
    try:
        try:
            out = pipe(inp, return_embeddings=True, **kw)
        except TypeError:
            if "hook" not in kw:
                raise
            kw.pop("hook")                               # an older pipeline without progress reports
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


def _who(turns, a: float, b: float) -> dict:
    """Seconds each speaker talks between a and b."""
    out: dict[str, float] = {}
    for x, y, spk in turns:
        o = min(y, b) - max(x, a)
        if o > 0:
            out[spk] = out.get(spk, 0) + o
    return out


def split_by_speaker(s: Segment, turns) -> list[Segment]:
    """One Whisper line spoken by two people in turn ("Ship Friday?" "Yes, if QA passes") -> one line each,
    cut at the word where the speaker changes. Needs word timestamps; tiny pieces stay with their neighbour."""
    if len(s.words) < 4:
        return [s]
    tags = []
    for ws, we, w in s.words:
        who = _who(turns, ws, max(we, ws + 0.05))
        tags.append(max(who, key=who.get) if who else None)
    for i in range(len(tags)):                                 # words in a gap: same as the word before
        if tags[i] is None:
            tags[i] = tags[i - 1] if i else next((t for t in tags if t), None)
    runs = []                                                  # [speaker, first word, last word]
    for i, t in enumerate(tags):
        if runs and runs[-1][0] == t:
            runs[-1][2] = i
        else:
            runs.append([t, i, i])
    changed = True                                             # pieces under 2 words / 0.7 s join a neighbour
    while changed and len(runs) > 1:
        changed = False
        for k, (t, a, b) in enumerate(runs):
            if b - a + 1 < 2 or s.words[b][1] - s.words[a][0] < 0.7:
                j = k - 1 if k > 0 else k + 1
                lo, hi = min(runs[j][1], a), max(runs[j][2], b)
                runs[j] = [runs[j][0], lo, hi]
                del runs[k]
                changed = True
                break
    if len(runs) < 2:
        return [s]
    out = []
    for t, a, b in runs:
        ws = s.words[a:b + 1]
        text = "".join(w[2] for w in ws).strip() if any(w[2].startswith(" ") for w in ws) else " ".join(w[2].strip() for w in ws)
        out.append(Segment(ws[0][0], ws[-1][1], text, words=ws))
    return out


def label_lines(audio: np.ndarray, segs: list[dict], num_speakers: Optional[int] = None, first: int = 1, progress=None):
    """Speaker separation for a meeting that already has its lines (they're not split, so line numbers stay):
    -> ([(speaker, also)] per line, {label: {"seconds", "embedding"}}). Labels start at "Speaker <first>"."""
    global diarize_error
    try:
        turns, vecs = _diarize(audio, num_speakers, progress)
        diarize_error = ""
    except Exception as e:
        diarize_error = str(e)[:200] or type(e).__name__
        raise
    names: dict[str, str] = {}
    secs: dict[str, float] = {}

    def label(spk):
        if spk not in names:
            names[spk] = f"Speaker {first + len(names)}"
        return names[spk]

    out = []
    for s in segs:
        a, b = float(s["start"]), float(s["end"])
        overlap = _who(turns, a, b)
        if not overlap:
            out.append(("", []))
            continue
        ranked = sorted(overlap, key=overlap.get, reverse=True)
        who = label(ranked[0])
        secs[who] = secs.get(who, 0) + (b - a)
        dur = max(0.1, b - a)
        also = [label(ranked[1])] if (len(ranked) > 1 and overlap[ranked[1]] >= 0.3 * dur
                                      and overlap[ranked[0]] + overlap[ranked[1]] > 1.15 * dur) else []
        out.append((who, also))
    spk = {lab: {"seconds": round(secs.get(lab, 0), 1), "embedding": vecs.get(k)} for k, lab in names.items()}
    return out, spk


def assign_speakers(audio: np.ndarray, tr: Transcript, num_speakers: Optional[int] = None, progress=None) -> None:
    """Give each line the speaker who talks most during it; split lines where the speaker changes mid-line;
    mark lines where a second person talks at the same time ("also"). Failure only loses the labels."""
    global diarize_error
    try:
        turns, vecs = _diarize(audio, num_speakers, progress)
        diarize_error = ""
    except Exception as e:
        diarize_error = str(e)[:200] or type(e).__name__
        log.warning("speaker labelling failed: %s", e)
        return
    segs = []
    for s in tr.segments:
        segs += split_by_speaker(s, turns)
    tr.segments = segs
    names: dict[str, str] = {}
    secs: dict[str, float] = {}

    def label(spk):
        if spk not in names:
            names[spk] = f"Speaker {len(names) + 1}"
        return names[spk]

    for s in tr.segments:
        overlap = _who(turns, s.start, s.end)
        if overlap:
            ranked = sorted(overlap, key=overlap.get, reverse=True)
            s.speaker = label(ranked[0])
            secs[s.speaker] = secs.get(s.speaker, 0) + (s.end - s.start)
            dur = max(0.1, s.end - s.start)
            # someone else talking over them for a good part of the line: overlapping speech
            if len(ranked) > 1 and overlap[ranked[1]] >= 0.3 * dur and overlap[ranked[0]] + overlap[ranked[1]] > 1.15 * dur:
                s.also = [label(ranked[1])]
    tr.speakers = {lab: {"seconds": round(secs.get(lab, 0), 1), "embedding": vecs.get(spk)}
                   for spk, lab in names.items()}
