"""Background worker: takes queued jobs one at a time -> transcribe -> summarise -> save results."""
import json
import re
import logging
import os
import shutil
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from . import db, people, redact, render, search, summarize, terms, transcribe, voices
from .config import settings

log = logging.getLogger("worker")
_wake = threading.Event()
_stop = threading.Event()
_thread: threading.Thread | None = None
current_job: str | None = None
live_info: dict = {}                # the live recording being transcribed now: {"rid", "until", "received"}


def wav_seconds(path) -> float:
    """Length of a (still growing) WAV from its size: IMA ADPCM (the watch) or PCM."""
    try:
        with open(path, "rb") as f:
            h = f.read(48)
        size = os.path.getsize(path)
        fmt = int.from_bytes(h[20:22], "little") if len(h) >= 22 else 1
        rate = int.from_bytes(h[24:28], "little") or 16000
        if fmt == 0x11:                             # IMA ADPCM: 256-byte blocks of 505 samples (mono)
            return max(0.0, (size - 48) / 256 * 505 / rate)
        ch = int.from_bytes(h[22:24], "little") or 1
        bits = int.from_bytes(h[34:36], "little") or 16
        return max(0.0, (size - 44) / (rate * ch * bits / 8))
    except OSError:
        return 0.0


def progress_info() -> dict | None:
    """What the transcriber is doing right now, with a percentage (for the web page header)."""
    cj = current_job
    if not cj:
        return None
    if cj.startswith("live:"):
        li = dict(live_info)
        rec = li.get("received") or 0
        done = li.get("until") or 0
        return {"kind": "live", "title": cj[5:], "stage": "transcribing while recording",
                "pct": round(100 * min(1.0, done / rec)) if rec else None,
                "done_min": round(done / 60, 1), "received_min": round(rec / 60, 1)}
    j = db.get(cj) or {}
    return {"kind": "job", "id": cj, "title": j.get("title") or j.get("filename") or cj, "stage": j.get("stage") or j.get("status"),
            "pct": round(100 * float(j.get("progress") or 0)), "status": j.get("status")}


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
    tmp = d / f"{name}.{os.getpid()}.{threading.get_ident()}.tmp"   # unique: two writers never share a temp file
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
    known = terms.known_terms()
    summ = load_summary(jid)
    fixes = {**terms.corrections(), **terms.meeting_fixes(tr)}   # words you corrected, here or in any meeting
    if summ:                                        # minutes written before a name/term/fix was added: follow it
        summ = "\n".join(terms.fix_text(terms.apply_corrections(line, fixes), known) for line in summ.split("\n"))
    _write(jid, "result.md", render.markdown(job, tr, summ, job.get("title", "")))
    _write(jid, "transcript.txt", render.text(named))
    _write(jid, "transcript.srt", render.srt(named))
    raw = load_json(jid, "actions_raw.json")
    if raw is not None:
        at = opts.get("names_at_summary")
        acts = [{**a, "owner": render.rename_text(a.get("owner", ""), names, at),
                 "task": terms.apply_corrections(render.rename_text(a.get("task", ""), names, at), fixes)} for a in raw]
        _write(jid, "actions.json", json.dumps(acts, ensure_ascii=False, indent=1))
    try:
        summary = render.rename_text(load_summary(jid) or "", names, opts.get("names_at_summary"))
        search.index(jid, f"{job.get('title', '')}\n{summary}", render.paragraphs(named.get("segments", [])))
    except Exception as e:
        log.warning("search index failed for %s: %s", jid, e)


def fix_terms(jid: str) -> int:
    from . import autotag
    with autotag.job_lock(jid):                     # a tag being saved writes the same file
        tr = load_transcript(jid)
        if not tr:
            return 0
        n = terms.fix_transcript(tr)
        if n:
            _write(jid, "transcript.json", json.dumps(tr, ensure_ascii=False, indent=1))
        return n


def refresh_terms_everywhere() -> int:
    """Names & terms changed: fix the spelling in every finished meeting (transcripts and minutes)."""
    jobs = db.list_jobs(100000, "done")

    def run():
        for j in jobs:
            try:
                fix_terms(j["id"])
                build_outputs(j["id"])
            except Exception as e:                  # noqa: BLE001
                log.warning("names & terms for %s: %s", j["id"], e)
    threading.Thread(target=run, daemon=True).start()
    return len(jobs)


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
            attendees=opts.get("attendees"), glossary=terms.known_terms(),
            flagged=render.flagged_text(named, render.parse_marks(opts.get("notes", ""))),
            meeting_date=meeting_date(job),
            people=people.context_for(sorted(set(names.values()))),
            minutes_lang=opts.get("minutes_lang", ""),
        )
        _write(jid, "summary.md", body)
        opts = db.get(jid).get("options", {})
        opts["names_at_summary"] = names
        opts["summary_title"] = title                       # what the minutes called it (your own title wins)
        db.update(jid, title=opts.get("title") or title, summary_error="", options=opts)
        db.update(jid, stage="listing action items")
        acts = summarize.extract_actions(summarize.primary_part(body), meeting_date(job), best)
        _write(jid, "actions_raw.json", json.dumps(acts, ensure_ascii=False, indent=1))
    except Exception as e:
        log.warning("summary failed for %s: %s", jid, e)
        db.update(jid, summary_error=summarize.explain_llm_error(e)[:500])


def process(job: dict) -> None:
    jid = job["id"]
    opts = job.get("options", {})
    if opts.get("resummarize") and load_transcript(jid) is not None:   # only redo the summary
        # re-read the options: names / tags you set after pressing "Redo minutes" must not be lost
        opts = dict((db.get(jid) or {}).get("options") or opts)
        opts.pop("resummarize", None)
        db.update(jid, options=opts)
        fix_terms(jid)                               # Names & terms you added since: into the transcript too
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
    terms.fix_transcript(data)                      # mis-heard names & terms get your spelling
    _write(jid, "transcript.json", json.dumps(data, ensure_ascii=False, indent=1))
    _write(jid, "speakers.json", json.dumps(spk))
    opts = db.get(jid).get("options", {})
    if opts.get("diarize", settings.diarize) and not spk and transcribe.diarize_error:
        opts["diarize_error"] = transcribe.diarize_error          # shown on the web, retried later
        opts["diarize_failed_at"] = int(time.time())
    auto = {k: v for k, v in voices.match(spk).items() if k not in (opts.get("speaker_names") or {})}
    if auto:
        opts["speaker_names"] = {**auto, **(opts.get("speaker_names") or {})}
        opts["speaker_auto"] = sorted(auto)
    db.update(jid, language=tr.language, duration=tr.duration, title=opts.get("title", ""), options=opts)
    run_summary(jid)
    build_outputs(jid)                              # files first, so "done" always means results are ready
    db.update(jid, status="done", stage="done", progress=1, finished_at=time.time())
    if opts.get("diarize", settings.diarize):
        from . import autotag
        autotag.after_transcription(jid)            # people you've named before: their lines get their names


# ---------------------------------------------------------------- live uploads (transcribed while recording)
live_lock = threading.Lock()       # a live part being saved vs the recording being completed (moves the state)


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
        live_info.clear()
        live_info.update(rid=rid, until=float(state.get("until", 0)), received=wav_seconds(part))
        try:
            did = transcribe.partial(str(part), state, voices.whisper_prompt(), dur=live_info["received"])
        except Exception as e:
            log.warning("live part for %s failed: %s", rid, e)
            did = False
        finally:
            current_job = None
        with live_lock:
            if did and st_path.exists():            # it may have been completed meanwhile
                tmp = st_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(state))
                os.replace(tmp, st_path)
                return True
    return False


def live_now() -> list[dict]:
    """Recordings streaming in from a watch right now (for the page): how much arrived, when the last audio came,
    how far the live transcript is, and the transcript so far."""
    out = []
    now = time.time()
    for st_path in settings.upload_dir.glob("*.live.json"):
        rid = st_path.name[:-len(".live.json")]
        part = settings.upload_dir / f"{rid}.part"
        if not part.exists():
            continue
        age = now - part.stat().st_mtime
        if age > 3 * 3600:
            continue
        try:
            state = json.loads(st_path.read_text())
        except ValueError:
            state = {}
        m = re.match(r"^(.*?)-((?:memo_|dict_)?\d{4}-\d{2}-\d{2}_\d{4})$", rid)
        segs = state.get("segments") or []
        out.append({
            "rid": rid, "device": m.group(1) if m else rid, "name": m.group(2) if m else "",
            "started": st_path.stat().st_ctime, "received_s": round(wav_seconds(part), 1),
            "last_audio_s_ago": round(age, 1), "receiving": age < 30,
            "transcribed_s": float(state.get("until", 0)), "language": state.get("language", ""),
            "transcribing": current_job == f"live:{rid}",
            "segments": [{"start": round(s.get("start", 0), 1), "text": (s.get("text") or "").strip()} for s in segs[-300:]],
        })
    out.sort(key=lambda x: x["started"])
    return out


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
        def pull_all():
            summarize.ensure_ollama_model()
            for m in (settings.ollama_model_fast, settings.ollama_embed_model if settings.rag else ""):
                if m:
                    summarize.ensure_ollama_model(m)   # the quick model for live questions, the embedding model
        threading.Thread(target=pull_all, daemon=True).start()
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


_REACH = ("ollama", "name or service", "refused", "timed out", "timeout", "connect", "unreachable", "502", "503")


def retry_failed_summaries(limit: int = 20) -> int:
    """Meetings whose minutes failed because the AI couldn't be reached are written again once it answers (at most
    once every 6 hours each). Returns how many were queued."""
    if not summarize.enabled() or summarize.llm_status() != "ok":
        return 0
    n = 0
    now = time.time()
    for j in db.list_jobs(500, "done"):
        err = (j.get("summary_error") or "").lower()
        if not err or not any(k in err for k in _REACH):
            continue
        opts = dict(j.get("options") or {})
        if now - float(opts.get("summary_retry_at") or 0) < 6 * 3600:
            continue
        opts.update(summarize=True, resummarize=True, summary_retry_at=now)
        db.update(j["id"], status="queued", stage="summary queued", progress=0, options=opts)
        n += 1
        if n >= limit:
            break
    if n:
        log.info("writing the minutes again for %d meetings (the AI is reachable again)", n)
        wake()
    return n
