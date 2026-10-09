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

from . import db, redact, render, search, summarize, transcribe, voices
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


def load_json(jid: str, name: str, default=None):
    p = result_dir(jid) / name
    try:
        return json.loads(p.read_text()) if p.exists() else default
    except ValueError:
        return default


def meeting_date(job: dict) -> str:
    return render.recorded_at(job["filename"], job["created_at"]).date().isoformat()


def build_outputs(jid: str) -> None:
    """(Re)write result.md / .txt / .srt / actions.json from the stored transcript + summary, with current names."""
    job = db.get(jid)
    tr = load_transcript(jid)
    if not job or tr is None:
        return
    opts = job.get("options") or {}
    names = render.names_of(job)
    named = render.apply_names(tr, names)
    _write(jid, "result.md", render.markdown(job, tr, load_summary(jid), job.get("title", "")))
    _write(jid, "transcript.txt", render.text(named))
    _write(jid, "transcript.srt", render.srt(named))
    raw = load_json(jid, "actions_raw.json")
    if raw is not None:
        at = opts.get("names_at_summary")
        acts = [{**a, "owner": render.rename_text(a.get("owner", ""), names, at),
                 "task": render.rename_text(a.get("task", ""), names, at)} for a in raw]
        _write(jid, "actions.json", json.dumps(acts, ensure_ascii=False, indent=1))
    try:
        summary = render.rename_text(load_summary(jid) or "", names, opts.get("names_at_summary"))
        search.index(jid, f"{job.get('title', '')}\n{summary}", render.paragraphs(named.get("segments", [])))
    except Exception as e:
        log.warning("search index failed for %s: %s", jid, e)


def run_summary(jid: str) -> None:
    job = db.get(jid)
    tr = load_transcript(jid)
    opts = job.get("options", {})
    if not summarize.enabled() or not opts.get("summarize", True) or not tr or not tr["segments"]:
        return
    db.update(jid, status="summarizing", stage="writing minutes", progress=0)
    names = render.names_of(job)
    named = render.apply_names(tr, names)
    best = bool(opts.get("best"))
    try:
        title, body = summarize.summarize(
            render.transcript_for_llm(named), tr.get("language", ""),
            lambda p, s: db.update(jid, progress=round(p, 3), stage=s),
            user_notes=opts.get("notes", ""), template=opts.get("template", "general"), best=best,
            attendees=opts.get("attendees"), glossary=voices.glossary(),
            flagged=render.flagged_text(named, render.parse_marks(opts.get("notes", ""))),
            meeting_date=meeting_date(job),
        )
        _write(jid, "summary.md", body)
        opts = db.get(jid).get("options", {})
        opts["names_at_summary"] = names
        db.update(jid, title=opts.get("title") or title, summary_error="", options=opts)
        db.update(jid, stage="listing action items")
        acts = summarize.extract_actions(body, meeting_date(job), best)
        _write(jid, "actions_raw.json", json.dumps(acts, ensure_ascii=False, indent=1))
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

    speakers = opts.get("speakers")
    tr = transcribe.transcribe(job["audio_path"], opts.get("language"), prog,
                               diarize=opts.get("diarize", settings.diarize),
                               prompt=voices.whisper_prompt(opts.get("attendees")),
                               num_speakers=int(speakers) if speakers else None,
                               prior=load_json(jid, "live.json"))
    if opts.get("redact", settings.redact):
        for s in tr.segments:
            s.text = redact.redact(s.text)
    data = asdict(tr)
    spk = data.pop("speakers", {}) or {}
    _write(jid, "transcript.json", json.dumps(data, ensure_ascii=False, indent=1))
    _write(jid, "speakers.json", json.dumps(spk))
    opts = db.get(jid).get("options", {})
    auto = {k: v for k, v in voices.match(spk).items() if k not in (opts.get("speaker_names") or {})}
    if auto:
        opts["speaker_names"] = {**auto, **(opts.get("speaker_names") or {})}
        opts["speaker_auto"] = sorted(auto)
    db.update(jid, language=tr.language, duration=tr.duration, title=opts.get("title", ""), options=opts)
    run_summary(jid)
    build_outputs(jid)                              # files first, so "done" always means results are ready
    db.update(jid, status="done", stage="done", progress=1, finished_at=time.time())


# ---------------------------------------------------------------- live uploads (transcribed while recording)
def live_state_path(rid: str) -> Path:
    return settings.upload_dir / f"{rid}.live.json"


def live_pass() -> bool:
    """Transcribe the next part of one live upload. True if it did some work."""
    for st_path in sorted(settings.upload_dir.glob("*.live.json"), key=lambda p: p.stat().st_mtime):
        rid = st_path.name[:-len(".live.json")]
        part = settings.upload_dir / f"{rid}.part"
        if not part.exists() or time.time() - part.stat().st_mtime > 3 * 3600:
            continue
        try:
            state = json.loads(st_path.read_text())
        except ValueError:
            state = {}
        global current_job
        current_job = f"live:{rid}"
        try:
            did = transcribe.partial(str(part), state, voices.whisper_prompt())
        except Exception as e:
            log.warning("live part for %s failed: %s", rid, e)
            did = False
        finally:
            current_job = None
        if did and st_path.exists():                # it may have been completed meanwhile
            tmp = st_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            os.replace(tmp, st_path)
            return True
    return False


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
    for f in list(settings.upload_dir.glob("*.part")) + list(settings.upload_dir.glob("*.live.json")):
        if now - f.stat().st_mtime > 7 * 86400:
            f.unlink(missing_ok=True)


def delete_job_files(job: dict) -> None:
    try:
        search.remove(job["id"])
    except Exception:
        pass
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
            if live_pass():
                continue
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
