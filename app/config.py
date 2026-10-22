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
    stt_model: str = os.getenv("STT_MODEL", "distil-small.en")      # short spoken questions (/api/stt): e.g. "small" - quicker
    default_language: str = os.getenv("DEFAULT_LANGUAGE", "")       # "" = auto-detect, e.g. "en"
    initial_prompt: str = os.getenv("INITIAL_PROMPT", "")           # vocabulary hints: names, products, jargon
    region: str = os.getenv("REGION", "").strip().lower()           # "sg" = Singapore: mixed English/Singlish/Mandarin/Malay
    vad: bool = _bool("VAD", True)
    # speech detection before Whisper: lower = keeps quieter / far-away voices (Silero default 0.5)
    vad_threshold: float = float(os.getenv("VAD_THRESHOLD", "0.35"))
    normalize: bool = _bool("NORMALIZE", True)        # level quiet recordings up before transcribing

    # --- speaker labels (needs image built with INSTALL_DIARIZE=1 and a Hugging Face token) ---
    diarize: bool = _bool("DIARIZE", False)
    hf_token: str = os.getenv("HF_TOKEN", "")
    diarize_model: str = os.getenv("DIARIZE_MODEL", "pyannote/speaker-diarization-3.1")
    diarizer: str = os.getenv("DIARIZER", "pyannote")             # "fake" is only for tests
    voice_match: float = float(os.getenv("VOICE_MATCH", "0.70") or 0.70)
    autotag_match: float = float(os.getenv("AUTOTAG_MATCH", "0.50") or 0.50)    # same meeting: a line sounds like X
    autotag_margin: float = float(os.getenv("AUTOTAG_MARGIN", "0.03") or 0.03)  # ... and more like X than anyone else
    autotag_group: float = float(os.getenv("AUTOTAG_GROUP", "0.50") or 0.50)    # a whole speaker group sounds like X
    autotag_new: bool = _bool("AUTOTAG_NEW", True)       # new meetings: name lines of voices already remembered
    autotag_sweep: bool = _bool("AUTOTAG_SWEEP", True)
    people_sweep: bool = _bool("PEOPLE_SWEEP", True)
    watchai_threads: int = int(os.getenv("WATCHAI_THREADS", "0"))    # CPU threads for training the watch AI (0 = half)
    watchai_nightly: bool = _bool("WATCHAI_NIGHTLY", True)   # retrain the watch model at night when facts changed
    watchai_minutes: float = float(os.getenv("WATCHAI_MINUTES", "120"))  # longest a training run may take
    research_auto: bool = _bool("RESEARCH_AUTO", True)        # research your company's topics on the web by itself
    research_live: bool = _bool("RESEARCH_LIVE", True)        # Ask: a quick web search when nothing is known yet
    bundled_firmware: bool = _bool("BUNDLED_FIRMWARE", True)  # offer the watch firmware that comes with this version
    # new watch firmware straight from GitHub (no redeploy): the .bin files in this folder of this repo. "" = off
    firmware_repo: str = os.getenv("FIRMWARE_REPO", "jeremiahng11/whisper").strip()
    firmware_branch: str = os.getenv("FIRMWARE_BRANCH", "main").strip() or "main"
    firmware_path: str = os.getenv("FIRMWARE_PATH", "app/watch_files/firmware").strip().strip("/")
    firmware_check_min: float = float(os.getenv("FIRMWARE_CHECK_MIN", "5") or 5)   # how often to look
    github_token: str = (os.getenv("FIRMWARE_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()   # private repo
    home_city: str = os.getenv("HOME_CITY", "Singapore")          # weather / "the time" when no city is named
    default_tz_min: int = _int("TZ_OFFSET_MIN", 480)              # the time zone when the watch doesn't say (+8)
    research_questions: int = int(os.getenv("RESEARCH_QUESTIONS", "4"))   # unanswered questions researched per day
    knowledge_sweep: bool = _bool("KNOWLEDGE_SWEEP", True)   # learn facts from new meetings in the background
    vocab_packs: str = os.getenv("VOCAB_PACKS", "all")       # built-in vocabularies: payments,tech,finance | none
    admin_id: str = os.getenv("ADMIN_ID", "").strip()        # web page sign-in (both set = sign-in required)
    admin_pass: str = os.getenv("ADMIN_PASS", "")
    session_secret: str = os.getenv("SESSION_SECRET", "")    # optional; otherwise a random key kept in /data
    public_url: str = os.getenv("PUBLIC_URL", "").strip()    # e.g. https://whisper.example.com (passkeys), optional     # idle server: update each person's profile   # idle server: go through older meetings with new voices   # cosine similarity to recognise a known voice

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
    # "Best quality" minutes use a bigger model (slower). Empty = same model as normal.
    ollama_model_best: str = os.getenv("OLLAMA_MODEL_BEST", "")       # e.g. qwen2.5:14b-instruct
    # live questions (the watch, Ask): a small quick model of their own, so they never wait behind the background
    # reading (meetings, sources, research) the main model does; "" = use the main model
    ollama_model_fast: str = os.getenv("OLLAMA_MODEL_FAST", "qwen2.5:3b-instruct")
    ollama_fast_ctx: int = _int("OLLAMA_FAST_CTX", 6144)
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")   # find by meaning (rag.py)
    rag: bool = _bool("RAG", True)
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.45"))
    openai_model_best: str = os.getenv("OPENAI_MODEL_BEST", "")
    openai_model_fast: str = os.getenv("OPENAI_MODEL_FAST", "")
    # Questions you ask the watch or Ask (not meetings) on a fast cloud AI - any OpenAI-compatible API (OpenAI, Groq,
    # OpenRouter, Anthropic's OpenAI endpoint...). Off unless LIVE_LLM_API_KEY is set: your question and the company
    # context sent with it then leave this server. Meetings and minutes always stay on SUMMARY_BACKEND.
    live_llm_base_url: str = os.getenv("LIVE_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    live_llm_api_key: str = os.getenv("LIVE_LLM_API_KEY", "")
    live_llm_model: str = os.getenv("LIVE_LLM_MODEL", "gpt-4o-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    summary_chunk_chars: int = _int("SUMMARY_CHUNK_CHARS", 24000)   # longer transcripts are summarised in parts
    llm_timeout_s: int = _int("LLM_TIMEOUT_S", 1800)

    # --- privacy ---
    redact: bool = _bool("REDACT", True)          # hide NRIC/FIN, card and bank account numbers in transcripts

    # --- live upload (device uploads while it records; transcribed in parts) ---
    live_min_new_s: int = _int("LIVE_MIN_NEW_S", 20)      # transcribe a part once this much new audio arrived
    live_hold_s: int = _int("LIVE_HOLD_S", 8)             # the last seconds wait for the next part (a sentence may go on)
    live_tag: bool = os.getenv("LIVE_TAG", "1") not in ("0", "false", "no")   # who's talking, while it streams in
    live_same_voice: float = float(os.getenv("LIVE_SAME_VOICE", "0.55") or 0.55)   # this close = the same person

    # --- voice (round Idea Saver): text-to-speech ---
    tts_backend: str = os.getenv("TTS", "piper").lower()          # piper | none
    tts_voices: str = os.getenv("TTS_VOICES", "en=en_US-lessac-medium,zh=zh_CN-huayan-medium")
    # ElevenLabs voice for everything spoken (answers, the morning brief, the watch's offline answers and prompts):
    # set the API key and the voice id (ElevenLabs > Voices > your voice > ID). Piper stays as the fallback.
    elevenlabs_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    elevenlabs_model: str = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
    # answers read out as you wait (Ask on the watch): ElevenLabs' fast model - the same voice, starts in a fraction of
    # a second (the big models take several seconds per answer). The watch's saved answers use ELEVENLABS_MODEL.
    # "" = ELEVENLABS_MODEL for everything.
    elevenlabs_model_live: str = os.getenv("ELEVENLABS_MODEL_LIVE", "eleven_flash_v2_5")
    elevenlabs_cache_mb: int = int(os.getenv("ELEVENLABS_CACHE_MB", "300") or 300)

    # --- calendar (device Agenda app) ---
    calendar_ics_urls: str = os.getenv("CALENDAR_ICS_URLS", "")   # private ICS links, comma separated
    calendar_tz: str = os.getenv("CALENDAR_TZ", os.getenv("TZ", "Asia/Singapore"))
    calendar_cache_s: int = _int("CALENDAR_CACHE_S", 300)
    # Microsoft 365 / Teams calendar read directly (Microsoft Graph, app-only, read-only): no public link needed
    ms_tenant_id: str = os.getenv("MS_TENANT_ID", "")
    ms_client_id: str = os.getenv("MS_CLIENT_ID", "")
    ms_client_secret: str = os.getenv("MS_CLIENT_SECRET", "")
    ms_calendar_user: str = os.getenv("MS_CALENDAR_USER", "")    # mailbox(es) to read, e.g. you@company.com

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

    @property
    def glossary_path(self) -> Path:
        return self.data_dir / "glossary.txt"

    def ensure_dirs(self) -> None:
        for d in (self.audio_dir, self.upload_dir, self.result_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
