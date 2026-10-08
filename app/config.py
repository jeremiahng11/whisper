"""All settings come from environment variables (set them in Coolify)."""
import os
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    return int(v) if v.strip() else default


class Settings:
    # --- security ---
    api_key: str = os.getenv("API_KEY", "")

    # --- storage ---
    data_dir: Path = Path(os.getenv("DATA_DIR", "/data"))
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 2048)
    keep_audio_days: int = _int("KEEP_AUDIO_DAYS", 30)      # 0 = keep forever
    keep_jobs_days: int = _int("KEEP_JOBS_DAYS", 0)         # 0 = keep forever

    # --- transcription ---
    transcriber: str = os.getenv("TRANSCRIBER", "faster-whisper")   # "fake" is only for tests
    whisper_model: str = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto")       # auto | cpu | cuda
    whisper_compute: str = os.getenv("WHISPER_COMPUTE", "auto")     # auto -> int8 on CPU, float16 on GPU
    whisper_threads: int = _int("WHISPER_THREADS", 0)               # 0 = all cores
    whisper_beam: int = _int("WHISPER_BEAM", 5)
    default_language: str = os.getenv("DEFAULT_LANGUAGE", "")       # "" = auto-detect, e.g. "en"
    initial_prompt: str = os.getenv("INITIAL_PROMPT", "")           # vocabulary hints: names, products, jargon
    vad: bool = _bool("VAD", True)

    # --- speaker labels (needs image built with INSTALL_DIARIZE=1 and a Hugging Face token) ---
    diarize: bool = _bool("DIARIZE", False)
    hf_token: str = os.getenv("HF_TOKEN", "")
    diarize_model: str = os.getenv("DIARIZE_MODEL", "pyannote/speaker-diarization-3.1")

    # --- summary ---
    summary_backend: str = os.getenv("SUMMARY_BACKEND", "ollama")   # ollama | openai | none
    summary_language: str = os.getenv("SUMMARY_LANGUAGE", "")       # "" = same language as the recording
    ollama_url: str = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    ollama_num_ctx: int = _int("OLLAMA_NUM_CTX", 16384)
    ollama_auto_pull: bool = _bool("OLLAMA_AUTO_PULL", True)
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    summary_chunk_chars: int = _int("SUMMARY_CHUNK_CHARS", 24000)   # longer transcripts are summarised in parts
    llm_timeout_s: int = _int("LLM_TIMEOUT_S", 1800)

    # --- calendar (device Agenda app) ---
    calendar_ics_urls: str = os.getenv("CALENDAR_ICS_URLS", "")   # private ICS links, comma separated
    calendar_tz: str = os.getenv("CALENDAR_TZ", os.getenv("TZ", "Asia/Singapore"))
    calendar_cache_s: int = _int("CALENDAR_CACHE_S", 300)

    # --- file sync (device notes backup) ---
    sync_max_kb: int = _int("SYNC_MAX_KB", 1024)                    # largest text file accepted

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def result_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.db"

    def ensure_dirs(self) -> None:
        for d in (self.audio_dir, self.upload_dir, self.result_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
