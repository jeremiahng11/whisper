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
            text = f"This is test sentence number {i}."
            ws = text.split()
            step = (end - t) / len(ws)
            words = [[round(offset + t + k * step, 2), round(offset + t + (k + 1) * step, 2), w] for k, w in enumerate(ws)]
            segs.append(Segment(round(offset + t, 2), round(offset + end, 2), text, words=words))
            progress(min(1.0, (offset + end) / total) if total else 1.0, "transcribing")
            t = end
        return segs, language or settings.default_language or "en", 1.0
    model = get_model()
    kwargs = dict(
        language=language or settings.default_language or None,
        beam_size=settings.whisper_beam,
        vad_filter=settings.vad,
        vad_parameters=dict(min_silence_duration_ms=500, threshold=settings.vad_threshold, speech_pad_ms=400),
        condition_on_previous_text=False,     # avoids Whisper getting stuck repeating a phrase
        word_timestamps=True,                 # lets speaker changes inside a line be split at the right word
    )
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


def embed_clip(a: np.ndarray, t: Optional[float] = None) -> Optional[list[float]]:
    """Voice print of one clip, in the same space as the diarizer's speaker embeddings (None if not available).
    t = where the clip starts in the recording (only the test diarizer uses it)."""
    if len(a) < SR * 1.2:
        return None
    if settings.diarizer == "fake":                      # tests: two people taking turns every 10 s
        k = int((t or 0) // 10) % 2
        return [1.0 if j == k else 0.0 for j in range(4)]
    try:
        import torch
        pipe = _get_diarizer()
        emb_model = getattr(pipe, "_embedding", None)
        if emb_model is None:
            return None
        w = torch.from_numpy(a[: SR * 30].astype(np.float32)).reshape(1, 1, -1)
        v = emb_model(w)
        v = np.asarray(v.detach().cpu() if hasattr(v, "detach") else v).reshape(-1)
        if not len(v) or np.any(np.isnan(v)):
            return None
        return [round(float(x), 5) for x in v]
    except Exception as e:                                   # noqa: BLE001
        log.warning("voice print of a clip failed: %s", e)
        return None


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


def assign_speakers(audio: np.ndarray, tr: Transcript, num_speakers: Optional[int] = None) -> None:
    """Give each line the speaker who talks most during it; split lines where the speaker changes mid-line;
    mark lines where a second person talks at the same time ("also"). Failure only loses the labels."""
    try:
        turns, vecs = _diarize(audio, num_speakers)
    except Exception as e:
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
