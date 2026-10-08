"""Background worker: takes queued jobs one at a time -> transcribe -> summarise -> save results."""
import json
import logging
import os
import shutil
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from . import db, render, summarize, transcribe
from .config import settings

log = logging.getLogger("worker")
_wake = threading.Event()
_stop = threading.Event()
_thread: threading.Thread | None = None
current_job: str | None = None


def wake() -> None:
    _wake.set()


def result_dir(jid: str) -> Path:
    return settings.result_dir / jid


def load_transcript(jid: str) -> dict | None:
    p = result_dir(jid) / "transcript.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_summary(jid: str) -> str | None:
    p = result_dir(jid) / "summary.md"
    return p.read_text() if p.exists() else None


def _write(jid: str, name: str, data: str) -> None:
    d = result_dir(jid)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, d / name)


def build_outputs(jid: str) -> None:
    """(Re)write result.md / .txt / .srt from the stored transcript + summary."""
    job = db.get(jid)
    tr = load_transcript(jid)
    if not job or tr is None:
        return
    _write(jid, "result.md", render.markdown(job, tr, load_summary(jid), job.get("title", "")))
    _write(jid, "transcript.txt", render.text(tr))
    _write(jid, "transcript.srt", render.srt(tr))


def run_summary(jid: str) -> None:
    job = db.get(jid)
    tr = load_transcript(jid)
    if not summarize.enabled() or not job.get("options", {}).get("summarize", True) or not tr or not tr["segments"]:
        return
    db.update(jid, status="summarizing", stage="writing summary", progress=0)
    try:
        title, body = summarize.summarize(
            render.transcript_for_llm(tr), tr.get("language", ""),
            lambda p, s: db.update(jid, progress=round(p, 3), stage=s),
        )
        _write(jid, "summary.md", body)
        user_title = job.get("options", {}).get("title")
        db.update(jid, title=user_title or title, summary_error="")
    except Exception as e:
        log.warning("summary failed for %s: %s", jid, e)
        db.update(jid, summary_error=str(e)[:500])


def process(job: dict) -> None:
    jid = job["id"]
    opts = job.get("options", {})
    if opts.pop("resummarize", False) and load_transcript(jid) is not None:   # only redo the summary
        db.update(jid, options=opts)
        run_summary(jid)
        build_outputs(jid)
        db.update(jid, status="done", stage="done", progress=1, finished_at=time.time())
        return
    db.update(jid, status="transcribing", stage="starting", progress=0, error="", started_at=time.time())
    last = [0.0]

    def prog(p: float, stage: str) -> None:
        if time.time() - last[0] > 2 or p >= 1:      # don't hammer SQLite
            last[0] = time.time()
            db.update(jid, progress=round(p, 3), stage=stage)

    tr = transcribe.transcribe(job["audio_path"], opts.get("language"), prog,
                               diarize=opts.get("diarize", settings.diarize))
    data = asdict(tr)
    _write(jid, "transcript.json", json.dumps(data, ensure_ascii=False, indent=1))
    db.update(jid, language=tr.language, duration=tr.duration, title=opts.get("title", ""))
    run_summary(jid)
    build_outputs(jid)                              # files first, so "done" always means results are ready
    db.update(jid, status="done", stage="done", progress=1, finished_at=time.time())


def cleanup() -> None:
    now = time.time()
    if settings.keep_audio_days > 0:
        for j in db.older_than(now - settings.keep_audio_days * 86400):
            p = j.get("audio_path")
            if p and os.path.exists(p):
                os.remove(p)
                log.info("removed old audio %s", p)
    if settings.keep_jobs_days > 0:
        for j in db.older_than(now - settings.keep_jobs_days * 86400):
            delete_job_files(j)
            db.delete(j["id"])
    # abandoned partial uploads (no activity for 7 days)
    for f in settings.upload_dir.glob("*.part"):
        if now - f.stat().st_mtime > 7 * 86400:
            f.unlink(missing_ok=True)


def delete_job_files(job: dict) -> None:
    p = job.get("audio_path")
    if p and os.path.exists(p):
        os.remove(p)
    shutil.rmtree(result_dir(job["id"]), ignore_errors=True)


def _loop() -> None:
    global current_job
    if summarize.enabled() and settings.summary_backend.lower() == "ollama" and settings.ollama_auto_pull:
        threading.Thread(target=summarize.ensure_ollama_model, daemon=True).start()
    if settings.transcriber != "fake" and os.getenv("PRELOAD_MODEL", "1") == "1":
        try:
            transcribe.get_model()                  # first start downloads the model (~1.6 GB for turbo)
        except Exception as e:
            log.error("could not load whisper model: %s", e)
    last_cleanup = 0.0
    while not _stop.is_set():
        if time.time() - last_cleanup > 3600:
            last_cleanup = time.time()
            try:
                cleanup()
            except Exception as e:
                log.warning("cleanup failed: %s", e)
        job = db.next_queued()
        if not job:
            _wake.wait(5)
            _wake.clear()
            continue
        current_job = job["id"]
        log.info("job %s: %s", job["id"], job["filename"])
        try:
            process(job)
            log.info("job %s done", job["id"])
        except Exception as e:
            log.error("job %s failed: %s\n%s", job["id"], e, traceback.format_exc())
            db.update(job["id"], status="error", error=str(e)[:1000], stage="failed", finished_at=time.time())
        finally:
            current_job = None


def start() -> None:
    global _thread
    if _thread is None or not _thread.is_alive():
        _stop.clear()
        _thread = threading.Thread(target=_loop, name="worker", daemon=True)
        _thread.start()


def stop() -> None:
    _stop.set()
    _wake.set()
