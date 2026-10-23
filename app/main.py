"""HTTP API + small web page. Run: uvicorn app.main:app --host 0.0.0.0 --port 8000"""
import importlib.util
import logging
import os
import json
import re
import secrets
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse

from . import agenda, autotag, chat, db, devices, dynamics, knowledge, login, sources, watchlm, watchpack, people, rag, research, vocab, redact, render, search, speech, summarize, terms, transcribe, voices, wakeword, worker
from .config import settings

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
VERSION = "1.38.2"
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.api_key:
        raise RuntimeError("Set the API_KEY environment variable (any long random string).")
    db.init()
    worker.start()
    autotag.start_sweeper()
    people.start_sweeper()
    knowledge.start_sweeper()
    try:
        if settings.bundled_firmware:
            devices.import_bundled()                 # the watch firmware that comes with this version
        devices.prune_firmware()                     # only the newest version and the one before
    except Exception as e:                           # noqa: BLE001
        log.warning("bundled firmware: %s", e)
    if settings.bundled_firmware:
        devices.start_github_watcher()               # and new firmware pushed to GitHub, without a redeploy
    if speech.tts_available():                       # load the voices now, not on the first spoken answer
        import threading
        threading.Thread(target=speech.warm_voices, daemon=True).start()
    log.info("whisper service %s ready - model %s, summary %s", VERSION, settings.whisper_model, settings.summary_backend)
    yield
    worker.stop()


app = FastAPI(title="Oracle Minutes", version=VERSION, lifespan=lifespan)


# ---------------------------------------------------------------- auth
def auth(request: Request, authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)) -> None:
    """The API key (watch, scripts) or a signed-in web page."""
    key = x_api_key or ""
    if authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    if key and secrets.compare_digest(key, settings.api_key):
        return
    if login.session_user(request):
        if request.method in ("GET", "HEAD") or login.same_origin(request):
            return
        raise HTTPException(403, "request from another site")
    raise HTTPException(401, "sign in, or send the API key")


# ---------------------------------------------------------------- helpers
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def check_id(rid: str) -> str:
    if not SAFE_ID.match(rid or ""):
        raise HTTPException(400, "recording id may only use letters, digits, . _ - (max 128)")
    return rid


def safe_name(name: str) -> str:
    name = os.path.basename(name or "").strip() or "recording.wav"
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:120]


def limit_bytes() -> int:
    return settings.max_upload_mb * 1024 * 1024


def job_out(j: dict) -> dict:
    base = f"/api/jobs/{j['id']}"
    done = j["status"] == "done"
    return {
        "id": j["id"],
        "recording_id": j["recording_id"],
        "filename": j["filename"],
        "status": j["status"],                 # queued | transcribing | summarizing | done | error
        "stage": j["stage"],
        "progress": j["progress"],
        "queue_position": db.queue_position(j["id"]),
        "error": j["error"],
        "summary_error": j["summary_error"],
        "title": j["title"],
        "language": j["language"],
        "duration": j["duration"],
        "created_at": j["created_at"],
        "finished_at": j["finished_at"],
        "template": (j.get("options") or {}).get("template", "general"),
        "best": bool((j.get("options") or {}).get("best")),
        "minutes_lang": (j.get("options") or {}).get("minutes_lang", ""),
        "speakers": (j.get("options") or {}).get("speakers"),
        "speaker_names": (j.get("options") or {}).get("speaker_names") or {},
        "speaker_auto": (j.get("options") or {}).get("speaker_auto") or [],
        "voice_work": autotag.active(j["id"]),       # speakers / voices being worked on in the background
        "results": {
            "markdown": f"{base}/result.md",
            "text": f"{base}/transcript.txt",
            "srt": f"{base}/transcript.srt",
            "json": f"{base}/transcript.json",
            "actions": f"{base}/actions.json",
            "docx": f"{base}/result.docx",
        } if done else None,
    }


def options(language: Optional[str], summarize_: bool, diarize: Optional[bool], title: Optional[str],
            template: Optional[str] = None, speakers: Optional[int] = None, best: Optional[bool] = None,
            redact_: Optional[bool] = None, attendees: Optional[str] = None,
            minutes_lang: Optional[str] = None) -> dict:
    o = {"summarize": summarize_, "diarize": settings.diarize if diarize is None else diarize,
         "redact": settings.redact if redact_ is None else redact_}
    if language and language.lower() != "auto":
        o["language"] = language.lower()
    if title:
        o["title"] = title[:200]
    if template:
        if template not in summarize.TEMPLATES:
            raise HTTPException(400, f"template must be one of {', '.join(summarize.TEMPLATES)}")
        o["template"] = template
    if speakers:
        if not 1 <= speakers <= 20:
            raise HTTPException(400, "speakers must be 1-20")
        o["speakers"] = speakers
    if best:
        o["best"] = True
    if attendees:
        o["attendees"] = [a.strip()[:60] for a in attendees.split(",") if a.strip()][:30]
    if minutes_lang:
        o["minutes_lang"] = check_minutes_lang(minutes_lang)
    return o


def check_minutes_lang(v: str) -> str:
    """"" / "auto" = same as speech, "en", "zh", "both" (meeting language + English), "zh+en"."""
    v = str(v or "").strip().lower()
    if v in ("", "auto", "same"):
        return ""
    if v in ("both", "bilingual") or re.fullmatch(r"[a-z]{2,3}(\+[a-z]{2,3})?", v):
        return "both" if v == "bilingual" else v
    raise HTTPException(400, "minutes_lang must be empty, a language code like en / zh, both, or zh+en")


def apply_extra(o: dict, extra: dict) -> dict:
    """Merge the JSON body the device sends on complete: notes, attendees."""
    notes = extra.get("notes") if isinstance(extra.get("notes"), str) else ""
    if notes.strip():
        o["notes"] = redact.redact(notes[:20000]) if o.get("redact") else notes[:20000]
    att = extra.get("attendees")
    if isinstance(att, list) and att:
        o["attendees"] = [str(a).strip()[:60] for a in att if str(a).strip()][:30]
    return o


def existing_for(rid: Optional[str]) -> Optional[dict]:
    """Same recording uploaded again: reuse the job, unless it failed (then start over)."""
    if not rid:
        return None
    j = db.get_by_recording(rid)
    if j and j["status"] == "error":
        worker.delete_job_files(j)
        db.delete(j["id"])
        return None
    return j


def new_job(src: Path, filename: str, opts: dict, rid: Optional[str], priority: str = "") -> dict:
    if src.stat().st_size == 0:
        src.unlink(missing_ok=True)
        raise HTTPException(400, "empty upload")
    ext = Path(filename).suffix.lower() or ".wav"
    dest = settings.audio_dir / f"{int(time.time())}_{secrets.token_hex(4)}{ext}"
    shutil.move(str(src), dest)
    if ext == ".wav":
        transcribe.fix_wav_header(str(dest))          # a live upload's header still says "0 bytes"
    j = db.create(filename, str(dest), opts, rid, 1 if priority == "high" else 0)
    with worker.live_lock:
        if rid and (live := worker.live_state_path(rid)).exists():
            d = worker.result_dir(j["id"])
            d.mkdir(parents=True, exist_ok=True)
            shutil.move(str(live), d / "live.json")  # parts already transcribed while it was recording
    worker.wake()
    return j


async def save_body(request: Request, dest: Path, mode: str = "wb", already: int = 0) -> int:
    n = already
    with open(dest, mode) as f:
        async for chunk in request.stream():
            n += len(chunk)
            if n > limit_bytes():
                raise HTTPException(413, f"upload bigger than MAX_UPLOAD_MB={settings.max_upload_mb}")
            f.write(chunk)
    return n


# ---------------------------------------------------------------- simple upload (one request)
@app.post("/api/jobs", dependencies=[Depends(auth)], status_code=201)
async def create_job(
    request: Request,
    language: Optional[str] = Query(None, description="e.g. en, zh, ms - empty/auto = detect"),
    summarize_: bool = Query(True, alias="summarize"),
    diarize: Optional[bool] = Query(None, description="speaker labels (needs DIARIZE build)"),
    title: Optional[str] = Query(None),
    priority: str = Query("", description="'high' = jump the queue (short dictation clips)"),
    template: Optional[str] = Query(None, description="general, standup, client, one_on_one, interview, board, memo"),
    speakers: Optional[int] = Query(None, description="how many people talk (helps speaker labels)"),
    best: Optional[bool] = Query(None, description="best-quality minutes (bigger model, slower)"),
    redact_: Optional[bool] = Query(None, alias="redact"),
    attendees: Optional[str] = Query(None, description="comma separated names"),
    minutes_lang: Optional[str] = Query(None, description="minutes in: en, zh, both (meeting language + English), zh+en"),
    x_recording_id: Optional[str] = Header(None, description="your own id; re-uploads return the same job"),
    x_filename: Optional[str] = Header(None),
):
    """Send the audio as the raw request body (Content-Type audio/wav etc.) or as multipart field `file`."""
    rid = check_id(x_recording_id) if x_recording_id else None
    if (j := existing_for(rid)):
        return JSONResponse(job_out(j), status_code=200)
    tmp = settings.upload_dir / f"tmp_{secrets.token_hex(8)}"
    filename = safe_name(x_filename or "")
    try:
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            up = form.get("file")
            if up is None or not hasattr(up, "read"):
                raise HTTPException(400, "multipart upload needs a field named 'file'")
            filename = safe_name(x_filename or up.filename or "")
            with open(tmp, "wb") as f:
                shutil.copyfileobj(up.file, f, 1024 * 1024)
            if tmp.stat().st_size > limit_bytes():
                raise HTTPException(413, f"upload bigger than MAX_UPLOAD_MB={settings.max_upload_mb}")
        else:
            await save_body(request, tmp)
        opts = options(language, summarize_, diarize, title, template, speakers, best, redact_, attendees, minutes_lang)
        return job_out(new_job(tmp, filename, opts, rid, priority))
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------- resumable upload (for the device)
def part_path(rid: str) -> Path:
    return settings.upload_dir / f"{rid}.part"


@app.get("/api/uploads/{rid}", dependencies=[Depends(auth)])
def upload_status(rid: str):
    """How many bytes the server already has -> continue the upload from there."""
    check_id(rid)
    j = db.get_by_recording(rid)
    p = part_path(rid)
    return {"recording_id": rid, "received": p.stat().st_size if p.exists() else 0,
            "job": job_out(j) if j else None}


@app.put("/api/uploads/{rid}", dependencies=[Depends(auth)])
async def upload_chunk(rid: str, request: Request, x_offset: int = Header(0), x_live: Optional[str] = Header(None),
                       x_live_state: Optional[str] = Header(None), x_live_why: Optional[str] = Header(None),
                       language: Optional[str] = Query(None)):
    """Append the request body at byte X-Offset. Wrong offset -> 409 with the size the server has."""
    check_id(rid)
    if (j := existing_for(rid)):
        return {"recording_id": rid, "received": None, "job": job_out(j)}
    p = part_path(rid)
    have = p.stat().st_size if p.exists() else 0
    if x_offset == 0 and have:                      # client restarts from scratch
        p.unlink()
        worker.live_state_path(rid).unlink(missing_ok=True)
        have = 0
    if x_offset != have:
        return JSONResponse({"recording_id": rid, "received": have, "error": "offset mismatch"}, status_code=409)
    n = await save_body(request, p, "ab", have)
    if x_live == "1" and not worker.live_state_path(rid).exists():   # still recording: transcribe as it arrives
        import json as _json
        st = {"until": 0, "segments": []}
        if language and language.lower() != "auto":
            st["language"] = language.lower()
        worker.live_state_path(rid).write_text(_json.dumps(st))
    if x_live == "1" and x_live_state:                # a heartbeat: still recording, nothing new to send (quiet room,
        import json as _json                          # silence skipped) - the page shows "quiet", not "paused"
        (settings.upload_dir / f"{rid}.beat").write_text(_json.dumps(
            {"at": time.time(), "state": x_live_state[:20], "why": (x_live_why or "")[:160]}))
    out = {"recording_id": rid, "received": n, "job": None}
    if worker.live_state_path(rid).exists():
        worker.wake()
        if x_live == "1" and (b := _brief(rid)) is not None:
            out["brief"] = {k: v for k, v in b.items() if k != "rid"}   # the watch's caption, no extra request
    return out


async def body_extra(request: Request) -> dict:
    """Optional JSON body {"notes": "...", "attendees": [...]}: what the device knows about the meeting."""
    try:
        raw = await request.body()
        if raw.strip():
            import json as _json
            v = _json.loads(raw)
            if isinstance(v, dict):
                return v
    except Exception:
        pass
    return {}


@app.post("/api/uploads/{rid}/complete", dependencies=[Depends(auth)], status_code=201)
async def upload_complete(
    rid: str,
    request: Request,
    language: Optional[str] = Query(None),
    summarize_: bool = Query(True, alias="summarize"),
    diarize: Optional[bool] = Query(None),
    title: Optional[str] = Query(None),
    filename: Optional[str] = Query(None),
    priority: str = Query(""),
    template: Optional[str] = Query(None),
    speakers: Optional[int] = Query(None),
    best: Optional[bool] = Query(None),
    redact_: Optional[bool] = Query(None, alias="redact"),
    attendees: Optional[str] = Query(None),
    minutes_lang: Optional[str] = Query(None, description="minutes in: en, zh, both (meeting language + English), zh+en"),
    x_total_size: Optional[int] = Header(None, description="optional: full file size, checked before queueing"),
):
    check_id(rid)
    extra = await body_extra(request)
    if (j := existing_for(rid)):
        return JSONResponse(job_out(j), status_code=200)
    p = part_path(rid)
    if not p.exists():
        raise HTTPException(404, "nothing uploaded for this recording id")
    if x_total_size is not None and p.stat().st_size != x_total_size:
        return JSONResponse({"recording_id": rid, "received": p.stat().st_size, "error": "size mismatch"}, status_code=409)
    opts = apply_extra(options(language, summarize_, diarize, title, template, speakers, best, redact_, attendees, minutes_lang), extra)
    return job_out(new_job(p, safe_name(filename or rid), opts, rid, priority))


# ---------------------------------------------------------------- jobs
def _job(jid: str) -> dict:
    j = db.get(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return j


@app.get("/api/jobs", dependencies=[Depends(auth)])
def list_jobs(limit: int = Query(50, le=500), status: Optional[str] = None):
    return [job_out(j) for j in db.list_jobs(limit, status)]


@app.get("/api/jobs/{jid}", dependencies=[Depends(auth)])
def get_job(jid: str):
    return job_out(_job(jid))


@app.get("/api/recordings/{rid}", dependencies=[Depends(auth)])
def get_by_recording(rid: str):
    """Look a job up by the device's own recording id."""
    j = db.get_by_recording(check_id(rid))
    if not j:
        raise HTTPException(404, "no job for this recording id")
    return job_out(j)


RESULT_FILES = {
    "result.md": ("result.md", "text/markdown; charset=utf-8"),
    "transcript.txt": ("transcript.txt", "text/plain; charset=utf-8"),
    "transcript.srt": ("transcript.srt", "application/x-subrip; charset=utf-8"),
    "transcript.json": ("transcript.json", "application/json"),
    "summary.md": ("summary.md", "text/markdown; charset=utf-8"),
    "actions.json": ("actions.json", "application/json"),
}


def export_name(j: dict, kind: str = "") -> str:
    """File name for downloads: Title + date, e.g. "MANCO_14-Oct-2026" (+ "_transcript" etc.)."""
    when = render.recorded_at(j["filename"], j["created_at"])
    title = (j.get("title") or Path(j["filename"]).stem).strip()
    date = when.strftime("%d-%b-%Y").lstrip("0")
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_.")[:70] or "Minutes"
    if not re.search(when.strftime("%Y-%m-%d") + r"|" + when.strftime("%b") + r"[_-]?" + when.strftime("%Y"), t, re.I) \
            and not re.search(r"\d{1,2}[_-]?" + when.strftime("%b"), t, re.I):
        t = f"{t}_{date}"
    return t + (f"_{kind}" if kind else "")


def _result(j: dict, name: str, analysis: bool = False):
    if name == "result.docx":
        if j["status"] != "done":
            raise HTTPException(409, f"job is {j['status']}")
        p = worker.result_dir(j["id"]) / "result.md"
        if not p.exists():
            raise HTTPException(404, "result not available")
        md = p.read_text(encoding="utf-8")
        if analysis:                                 # Oracle AI analysis below the minutes
            md = dynamics.with_analysis(md, j["id"])
        data = render.docx(md, speaker_colours(j))
        stem = export_name(j) + ("_with_Analysis" if analysis else "")
        return Response(data, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.docx"'})
    if name not in RESULT_FILES:
        raise HTTPException(404, "unknown result file")
    if j["status"] != "done":
        raise HTTPException(409, f"job is {j['status']}")
    fname, ctype = RESULT_FILES[name]
    p = worker.result_dir(j["id"]) / fname
    if not p.exists():
        raise HTTPException(404, f"{name} not available")
    kinds = {"result.md": ("", ".md"), "transcript.txt": ("transcript", ".txt"), "transcript.srt": ("subtitles", ".srt"),
             "transcript.json": ("transcript", ".json"), "summary.md": ("summary", ".md"), "actions.json": ("actions", ".json")}
    kind, ext = kinds.get(name, (Path(name).stem, Path(name).suffix))
    if analysis and name == "result.md":             # minutes + Oracle AI analysis
        md = dynamics.with_analysis(p.read_text(encoding="utf-8"), j["id"])
        return Response(md, media_type=ctype, headers={
            "Content-Disposition": f'inline; filename="{export_name(j)}_with_Analysis.md"'})
    dl = export_name(j, kind) + ext
    return FileResponse(p, media_type=ctype, filename=dl, content_disposition_type="inline")


@app.get("/api/jobs/{jid}/{name}", dependencies=[Depends(auth)])
def job_result(jid: str, name: str, start: float = Query(0.0, ge=0), end: float = Query(0.0, ge=0),
               enhance: bool = Query(True), analysis: bool = Query(False)):
    if name == "speakers":
        return get_speakers(jid)
    if name == "clip":
        return job_clip(jid, start, end, enhance)
    if name == "autotag":
        _job(jid)
        return autotag.status(jid)
    if name == "draft":
        return job_draft(jid)
    if name == "dynamics":
        if _job(jid)["status"] != "done":
            raise HTTPException(409, "not finished yet")
        return dynamics.get(jid)
    return _result(_job(jid), name, analysis)


def job_draft(jid: str):
    """The transcript so far of a recording that isn't finished: the words as they're found, then (while the minutes
    are written) the transcript with its speakers. Live recordings start with what was transcribed while streaming."""
    j = _job(jid)
    names = (j.get("options") or {}).get("speaker_names") or {}
    tr = worker.load_transcript(jid)
    if tr:
        segs = [{"start": s["start"], "end": s["end"], "text": s["text"],
                 "speaker": names.get(s.get("speaker") or "", s.get("speaker") or "")} for s in tr.get("segments", [])]
        return {"segments": segs, "words_done": True, "speakers_done": True, "status": j["status"], "stage": j["stage"]}
    d = worker.load_json(jid, "draft.json") or {}
    segs = d.get("segments")
    if segs is None:                                  # not started yet: what came in live, if anything
        live = worker.load_json(jid, "live.json") or {}
        ln = live.get("names") or {}
        segs = [{"start": s["start"], "end": s["end"], "text": s["text"],
                 "speaker": ln.get(s.get("speaker") or "", s.get("speaker") or "")} for s in live.get("segments", [])]
        if (j.get("options") or {}).get("redact", settings.redact):
            segs = [{**s, "text": redact.redact(s["text"])} for s in segs]
    return {"segments": segs, "words_done": bool(d.get("words_done")), "speakers_done": False,
            "status": j["status"], "stage": j["stage"]}


def job_clip(jid: str, start: float, end: float, enhance: bool = True):
    """GET /api/jobs/{id}/clip?start=12.5&end=20 -> that stretch of the meeting as a WAV (at most 60 s).
    enhance (default): noise reduction + steady volume, and 0.3 s more on each side so words aren't cut."""
    j = _job(jid)
    if not j["audio_path"] or not os.path.exists(j["audio_path"]):
        raise HTTPException(410, "the audio of this recording is no longer kept")
    if end <= start:
        end = start + 8
    end = min(end, start + 60)
    if not enhance:
        a = transcribe.load_range(j["audio_path"], start, end)
        return Response(transcribe.wav16(a), media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})
    s, e = max(0.0, start - 0.3), end + 0.3
    ctx = transcribe.load_range(j["audio_path"], max(0.0, s - 4), e + 4)      # some seconds around: noise profile
    off = int((s - max(0.0, s - 4)) * transcribe.SR)
    a = ctx[off:off + int((e - s) * transcribe.SR)]
    a = transcribe.enhance_clip(a, ctx)
    return Response(transcribe.wav16(a), media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/recordings/{rid}/{name}", dependencies=[Depends(auth)])
def recording_result(rid: str, name: str):
    j = db.get_by_recording(check_id(rid))
    if not j:
        raise HTTPException(404, "no job for this recording id")
    return _result(j, name)


@app.post("/api/jobs/{jid}/summarize", dependencies=[Depends(auth)])
async def resummarize(jid: str, request: Request):
    """Write the minutes again, optionally with {"template": "...", "best": true}. Runs in the background."""
    j = _job(jid)
    extra = await body_extra(request)
    if j["status"] not in ("done",):
        raise HTTPException(409, f"job is {j['status']}")
    if not summarize.enabled():
        raise HTTPException(400, "SUMMARY_BACKEND is none")
    opts = dict(j["options"], summarize=True, resummarize=True)
    if extra.get("template"):
        if extra["template"] not in summarize.TEMPLATES:
            raise HTTPException(400, "unknown template")
        opts["template"] = extra["template"]
    if "best" in extra:
        opts["best"] = bool(extra["best"])
    if "minutes_lang" in extra:
        ml = check_minutes_lang(extra["minutes_lang"])
        if ml:
            opts["minutes_lang"] = ml
        else:
            opts.pop("minutes_lang", None)
    db.update(jid, status="queued", stage="summary queued", progress=0, options=opts)
    worker.wake()
    return job_out(db.get(jid))


@app.post("/api/jobs/{jid}/title", dependencies=[Depends(auth)])
async def set_title(jid: str, request: Request):
    """Rename the meeting: {"title": "CTRIP dashboard review"}. Kept when the minutes are redone; "" = back to the
    title the minutes were written with."""
    j = _job(jid)
    data = await body_extra(request)
    title = re.sub(r"\s+", " ", str(data.get("title") or "")).strip()[:120]
    with autotag.job_lock(jid):
        opts = dict((db.get(jid) or {}).get("options") or {})
        if title:
            opts["title"] = title
        else:
            opts.pop("title", None)
        db.update(jid, title=title or opts.get("summary_title", j.get("title", "")), options=opts)
    if j["status"] == "done":
        worker.build_outputs(jid)
    return job_out(db.get(jid))


@app.post("/api/jobs/{jid}/retry", dependencies=[Depends(auth)])
def retry(jid: str):
    j = _job(jid)
    if j["status"] != "error":
        raise HTTPException(409, f"job is {j['status']}")
    if not j["audio_path"] or not os.path.exists(j["audio_path"]):
        raise HTTPException(410, "audio file is gone - upload it again")
    db.update(jid, status="queued", stage="retry", progress=0, error="")
    worker.wake()
    return job_out(db.get(jid))


@app.delete("/api/jobs/{jid}", dependencies=[Depends(auth)])
def delete_job(jid: str):
    j = _job(jid)
    if worker.current_job == jid:
        raise HTTPException(409, "job is running")
    worker.delete_job_files(j)
    db.delete(jid)
    knowledge.forget_meeting(jid)
    return {"deleted": jid}


# ---------------------------------------------------------------- chat (Ask AI on the device)
MEETING_SYSTEM = ("You answer questions about the user's recorded meetings, using only the material below. "
                  "Be brief (a few sentences or a short list). Mention the time [mm:ss] and, when there are several "
                  "meetings, the meeting title and date for each fact. If the answer isn't in the material, say so. "
                  "Plain text only.")
EMAIL_TASK = ("Draft a short follow-up email to the people in this meeting: a subject line first (\"Subject: ...\"), "
              "then a thank-you line, a 2-3 sentence recap, the decisions, and the action items with owners and due "
              "dates. Professional and friendly. Plain text, no Markdown.")


def _job_for(ref: str) -> dict:
    j = db.get(ref) or db.get_by_recording(ref)
    if not j:
        raise HTTPException(404, "unknown recording")
    if j["status"] != "done":
        raise HTTPException(409, f"recording is {j['status']}")
    return j


def meeting_context(j: dict, limit: int = 0) -> str:
    limit = limit or max(8000, settings.ollama_num_ctx * 3 - 6000)
    p = worker.result_dir(j["id"]) / "result.md"
    md = p.read_text(encoding="utf-8") if p.exists() else ""
    if len(md) > limit:
        head, _, tr = md.partition("## Transcript")
        md = head + "## Transcript (shortened)\n" + tr[: max(2000, limit - len(head))]
    return "Meeting minutes and transcript:\n\"\"\"\n" + md + "\n\"\"\""


MEETING_LIVE_SYSTEM = MEETING_SYSTEM + (" The meeting is still going on: the transcript below is what has been said so "
                                        "far, the newest at the end. Answer from that; if something hasn't come up yet, "
                                        "say so.")


def recording_context(ref: str) -> tuple[str, str]:
    """(context, system) for a question about one recording: the minutes when it's done; while it's still streaming
    in from a watch or being processed, the transcript so far."""
    j = db.get(ref) or db.get_by_recording(ref)
    if j and j["status"] == "done":
        return meeting_context(j), MEETING_SYSTEM
    segs, names = None, {}
    if SAFE_ID.match(ref) and (lp := worker.live_state_path(ref)).exists():
        try:
            st = json.loads(lp.read_text())
        except (OSError, ValueError):
            st = {}
        segs, names = st.get("segments") or [], st.get("names") or {}
    elif j:
        tr = worker.load_transcript(j["id"])
        if tr:
            segs, names = tr.get("segments") or [], (j.get("options") or {}).get("speaker_names") or {}
        elif (d := worker.load_json(j["id"], "draft.json")) and d.get("segments") is not None:
            segs = d["segments"]
        else:
            lv = worker.load_json(j["id"], "live.json") or {}
            segs, names = lv.get("segments") or [], lv.get("names") or {}
    if segs is None:
        raise HTTPException(404, "unknown recording")
    lines = []
    for x in segs:
        who = names.get(x.get("speaker") or "", x.get("speaker") or "")
        lines.append(f"[{render.ts(float(x.get('start') or 0))}] " + (f"{who}: " if who else "") + str(x.get("text") or "").strip())
    text = "\n".join(lines) or "(nothing has been transcribed yet)"
    limit = max(8000, settings.ollama_num_ctx * 3 - 6000)
    if len(text) > limit:                            # keep the most recent part
        text = "(earlier part left out)\n" + text[-limit:]
    return "Meeting in progress - transcript so far:\n\"\"\"\n" + text + "\n\"\"\"", MEETING_LIVE_SYSTEM


def search_context(q: str) -> str:
    hits = search.search(q, 14)
    if not hits:
        return "No meeting matched the question's words."
    lines = []
    for h in hits:
        where = "minutes" if h["start"] < 0 else render.ts(h["start"])
        who = f"{h['speaker']}: " if h["speaker"] else ""
        lines.append(f"[{h['title']} - {h['date']} - {where}] {who}{h['text'][:1500]}")
    return "Excerpts from the user's meetings (best matches first):\n\"\"\"\n" + "\n\n".join(lines) + "\n\"\"\""


COMPANY_NOTE = ("Below is what the user's company knowledge base knows (learned from their meetings, website, product "
                "documents and the people they work with). When the question is about the company, its people, products, "
                "services or meetings, answer from it - it is more current than your own knowledge - in a sentence or "
                "two, naming people and products as written there. Read all of it before answering: the answer is often "
                "in a passage further down. Only if nothing below answers the question, answer from what you know "
                "(say briefly that it's not from the company's notes).")


def company_context(q: str, limit: int = 6000) -> str:
    """Facts, passages close in meaning (rag.py), people and (when facts are few) meeting excerpts matching the
    question - or "" if nothing matches."""
    parts = []
    kb = knowledge.context_for(q, 1800)
    if kb:
        parts.append(kb)
    near = rag.context_for(q, 3000)
    if near:
        parts.append(near)
    who = [n for n in people.all_names() if re.search(r"\b" + re.escape(n) + r"\b", q, re.I)]
    if who:
        pc = people.context_for(who, 1200)
        if pc:
            parts.append("About these people (from all meetings):\n" + pc)
    if len(knowledge.find(q, 4)) < 2 and search.search(q, 1):
        parts.append(search_context(q)[:1500])
    if not parts:
        return ""
    return (COMPANY_NOTE + "\n\n" + "\n\n".join(parts))[:limit]


def about_company(q: str) -> bool:
    """Does the question name the company, its topics or its people, or talk about "us / our meeting"? Otherwise it's a
    question about the world, and the web is searched for it as asked."""
    from . import worldinfo
    if worldinfo.OURS.search(q):
        return True
    words = knowledge._words(q)
    try:
        d = research.load()
        for t in (d.get("topics") or []) + (d.get("found") or [])[:3]:
            tw = knowledge._words(t)
            if tw and (tw <= words or (len(tw) > 1 and len(tw & words) >= 2)):
                return True
    except Exception:                                # noqa: BLE001
        pass
    if any(re.search(r"\b" + re.escape(n.split(" ")[0]) + r"\b", q, re.I) for n in people.all_names() if len(n) > 2):
        return True
    f = knowledge.best_answer(q)                     # a fact that clearly answers it
    return bool(f)


WEB_NOTE = ("If the company information above doesn't answer the question, answer from these web results - give the "
            "answer itself in a sentence or two, not a list of links; never say you don't know when they answer it.")


@app.post("/api/ask", dependencies=[Depends(auth)], status_code=202)
async def ask(request: Request, wait: float = 0):
    """{"messages": [...]} -> {"id", "status": "pending"}; poll GET /api/ask/{id}.
    Optional: "recording": <recording or job id> (ask about one meeting), "scope": "meetings" (search all
    meetings), "task": "email" (draft a follow-up email for "recording"), "best": true, "speak": true (the answer
    will be read out: its voice is made as soon as it's ready). ?wait=<s> (up to 15): answer in this reply if it's
    ready by then (one request instead of several polls)."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "body must be JSON")
    if not isinstance(data, dict):
        raise HTTPException(400, "body must be a JSON object")
    from starlette.concurrency import run_in_threadpool
    r = await run_in_threadpool(_ask_start, data)       # (a forecast lookup may take a second: not on the event loop)
    if data.get("speak"):
        chat.speak_when_done(r["id"])
    r = await _wait_answer(r, wait)
    if r.get("status") != "pending":                 # answered already: 200 (the watch takes an answer only with 200)
        return JSONResponse(r, status_code=200)
    return r


async def _wait_answer(r: dict, wait: float) -> dict:
    import asyncio
    t_end = time.time() + max(0.0, min(float(wait or 0), 15.0))
    while r.get("status") == "pending" and time.time() < t_end:
        await asyncio.sleep(0.15)
        r = chat.get(r["id"]) or r
    return r


class _Plan(Exception):
    """_ask_start(plan=True): what it would do, instead of starting it (the one-trip /api/voice streams it itself)."""
    def __init__(self, **kw):
        self.kw = kw


def _ask_plan(data: dict) -> dict:
    """{"instant": answer} or {"messages", "context", "system", "late", "best"} for a question."""
    try:
        _ask_start(data, plan=True)
    except _Plan as p:
        return p.kw
    raise RuntimeError("no plan")


def _ask_start(data: dict, plan: bool = False) -> dict:
    def instant(a: str) -> dict:
        if plan:
            raise _Plan(instant=a)
        return chat.instant(a)

    def go(clean, context, best, system, late=None) -> dict:
        if plan:
            raise _Plan(messages=clean, context=context, best=best, system=system, late=late)
        return chat.start(clean, context, best, system, late=late)
    msgs = data.get("messages")
    if not msgs and data.get("prompt"):
        msgs = [{"role": "user", "content": str(data["prompt"])}]
    if data.get("task") == "email":
        if not data.get("recording"):
            raise HTTPException(400, "task=email needs 'recording'")
        msgs = [{"role": "user", "content": EMAIL_TASK}]
    if not msgs or not isinstance(msgs, list):
        raise HTTPException(400, "send {'messages': [...]} or {'prompt': '...'}")
    sys_given = " ".join(str(m.get("content", ""))[:4000] for m in msgs if isinstance(m, dict) and m.get("role") == "system")
    clean = [{"role": m.get("role") if m.get("role") in ("user", "assistant") else "user", "content": str(m.get("content", ""))[:8000]}
             for m in msgs[-20:] if isinstance(m, dict) and m.get("role") != "system"]
    if not clean:
        raise HTTPException(400, "no user message")
    if not summarize.enabled():
        raise HTTPException(400, "SUMMARY_BACKEND is none - no LLM configured")
    context, system = "", sys_given
    q = " ".join(m["content"] for m in clean if m["role"] == "user")[-1000:]
    from . import worldinfo
    try:
        tz = int(data["tz"]) if data.get("tz") is not None else None
    except (TypeError, ValueError):
        tz = None
    city = str(data.get("city") or "")[:60]
    # the date and time go with every question (the AI doesn't know them); the last question decides the rest
    last_q = next((m["content"] for m in reversed(clean) if m["role"] == "user"), "")
    now = worldinfo.now_line(tz, city)
    if data.get("recording"):
        context, system = recording_context(str(data["recording"]))
    elif data.get("scope") == "meetings":
        context, system = search_context(q), MEETING_SYSTEM
    if data.get("recording") or data.get("scope") == "meetings":
        who = [n for n in people.all_names() if re.search(r"\b" + re.escape(n) + r"\b", q, re.I)]
        if who:                                      # "what does Priscilla usually raise?"
            context += "\n\nWhat Oracle Minutes knows about these people from all meetings:\n" + people.context_for(who, 3000)
    if not data.get("recording") and not data.get("task"):
        last = next((m["content"] for m in reversed(clean) if m["role"] == "user"), "")
        threading.Thread(target=knowledge.note_question, args=(last, "ask"), daemon=True).start()   # learn your wording
    if data.get("scope") == "meetings":
        kb = knowledge.context_for(q)                # short facts learned from all meetings (the knowledge base)
        if kb:
            context += "\n\n" + kb
    elif not data.get("recording") and data.get("task") != "email":
        if len(clean) == 1 and not data.get("best") and not data.get("offline"):
            # "how's the weather?", "will it rain tomorrow?": straight from the forecast, like a phone assistant
            t0 = time.time()
            wa = worldinfo.weather_answer(last_q, city)
            if wa:
                log.info("ask: weather answered directly in %.1fs", time.time() - t0)
                from . import timing
                timing.note("forecast", time.time() - t0)
                return instant(wa)
        if len(clean) == 1 and not data.get("best"):
            # "who is Ryan", "who is our CTO", "what products do we have": a company essential or a who-is answer
            # that clearly matches is the answer - at once, without waiting for the AI (on a home server that's
            # the difference between a second and half a minute)
            f = knowledge.best_answer(q)
            if f and (f.get("core") or f.get("edited") or f.get("kind") == "person") and len(knowledge._words(q)) <= 6 \
                    and not worldinfo.is_world(last_q):
                return instant(f["a"])
        # any other question (the watch's "Hey Oracle ...", Ask on the web): your company's knowledge goes along when
        # something matches - people, products, services, decisions - so "what is Aleta Planet" or "who is Ryan"
        # is answered from your meetings, website and documents instead of the AI's general knowledge
        web = len(knowledge.find(q, 4)) < 2 and not data.get("offline")
        world = worldinfo.is_world(last_q) and not data.get("offline")

        def late() -> str:
            # found in the answer's thread (finding by meaning asks the embedding model): the company's knowledge
            # closest to the question, and - when little is known - a quick web search, so "what's new with Sanrio
            # Aleta Adventure" is answered from today's web, not only the AI's old general knowledge.
            # A question about the world ("weather tomorrow", "who won the match", "price of gold"): the weather
            # and a web search for the question as asked, like a phone assistant
            if world:
                out = worldinfo.context(last_q, tz, city)
                if out:
                    return out + "\n\nAnswer from this (it is current); say where it's from in a few words."
            ours = about_company(last_q)
            if data.get("offline"):
                return company_context(q)
            # every question is also searched on the web as asked, at the same time as the company's knowledge is
            # looked through: the AI answers from your knowledge when that has it, else from the web ("how many payment
            # companies are there in Singapore" - before, a question that looked a little like the company's got only
            # the company's notes, and the answer was "I don't have that")
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(2) as ex:
                fw = ex.submit(research.live_context, last_q, 2500, True)
                out = company_context(q)
                w = fw.result()
            if ours and web and "closest in meaning" not in out and not w:   # about us, nothing found either way
                w = research.live_context(q)
            if w:
                out = (out + "\n\n" if out else "") + w + "\n\n" + WEB_NOTE
            return out
        return go(clean, context, bool(data.get("best")), (system or chat.SYSTEM) + "\n\n" + now, late=late)
    return go(clean, context, bool(data.get("best")), (system or chat.SYSTEM) + "\n\n" + now)


@app.get("/api/ask/{aid}", dependencies=[Depends(auth)])
async def ask_result(aid: str, wait: float = 0):
    """?wait=<s> (up to 15): reply as soon as the answer is ready (or after that long)."""
    r = chat.get(aid)
    if not r:
        raise HTTPException(404, "unknown or expired")
    return await _wait_answer(r, wait)

# ---------------------------------------------------------------- speakers, voices, glossary, search, templates
@app.post("/api/jobs/{jid}/speakers", dependencies=[Depends(auth)])
async def name_speakers(jid: str, request: Request):
    """{"names": {"Speaker 1": "San", "Speaker 2": ""}, "remember": true} - "" goes back to the label.
    remember = keep their voices so later recordings are named automatically."""
    j = _job(jid)
    data = await body_extra(request)
    names = data.get("names")
    if not isinstance(names, dict):
        raise HTTPException(400, "send {'names': {'Speaker 1': 'Name'}}")
    opts = dict(j["options"])
    cur = dict(opts.get("speaker_names") or {})
    auto = set(opts.get("speaker_auto") or [])
    spk = worker.load_json(jid, "speakers.json", {}) or {}
    tr0 = worker.load_transcript(jid) or {}
    was = data.get("was") if isinstance(data.get("was"), dict) else {}
    skipped = []
    for label, name in names.items():
        label, name = str(label)[:40], terms.spelled(re.sub(r"\s+", " ", str(name or "")).strip()[:60])
        if not re.match(r"^Speaker \d+$", label):
            raise HTTPException(400, f"unknown speaker label {label!r}")
        old = cur.get(label, "")
        # the page sends what it showed for this speaker ("was"): if the speakers were renumbered or joined since
        # (another save, a voice check), this label is someone else now - don't rename or clear the wrong person
        if label in was and (old or "").lower() != (str(was[label] or "")).strip().lower() and name.lower() != old.lower():
            skipped.append(label)
            continue
        if old and old.lower() != name.lower():          # "that's not James": James's samples from these lines go
            lines = [i for i, s in enumerate(tr0.get("segments", [])) if s.get("speaker") == label]
            people.unlearn(old, jid, lines + [-1])       # (-1 = the whole speaker's voice print)
        if name:
            cur[label] = name
            if data.get("remember", True) and (spk.get(label) or {}).get("embedding"):
                voices.remember(name, spk[label]["embedding"], jid, -1)
        else:
            cur.pop(label, None)
        auto.discard(label)
    with autotag.job_lock(jid):                      # an auto-tag run may be saving at the same moment
        o2 = dict((db.get(jid) or {}).get("options") or opts)
        o2["speaker_names"], o2["speaker_auto"] = cur, sorted(auto)
        db.update(jid, options=o2)
    joined = autotag.merge_same_names(jid)           # the same name twice = one person: join them
    if jid not in autotag._running:                  # (a running voice check renumbers when it's done)
        autotag.renumber(jid)                        # Speaker 1, 2, 3 ... without gaps
    if j["status"] == "done":
        worker.build_outputs(jid)
        if joined:
            autotag.start(jid)                       # their joined voice finds more of their lines
    out = job_out(db.get(jid))
    out["joined"] = joined
    out["skipped"] = skipped
    return out


@app.post("/api/jobs/{jid}/segments/{idx}", dependencies=[Depends(auth)])
async def tag_segment(jid: str, idx: int, request: Request):
    """Say who really spoke one transcript line: {"name": "Alan", "learn": true} or {"speaker": "Speaker 2"}.
    A name used in this meeting goes to that speaker; a new name becomes a new speaker. learn = also add this
    clip to that person's voice print, so later meetings recognise them better.
    {"also": "Wei Ling"} = a second person talking at the same time ("" = nobody); can be sent with or without a name."""
    j = _job(jid)
    if j["status"] != "done":
        raise HTTPException(409, f"job is {j['status']}")
    data = await body_extra(request)
    from starlette.concurrency import run_in_threadpool
    out = await run_in_threadpool(_tag_locked, jid, idx, data)
    # learn the voice from this line (outside the lock: it can take a moment)
    who, label, segs_idx = out.pop("_who"), out["speaker"], out.pop("_seg")
    if data.get("learn", True) and who and not segs_idx.get("also") and j["audio_path"] and os.path.exists(j["audio_path"]):
        clip = await run_in_threadpool(transcribe.load_range, j["audio_path"], float(segs_idx["start"]), float(segs_idx["end"]))
        emb = await run_in_threadpool(transcribe.embed_clip, clip, float(segs_idx["start"]))
        if emb:
            voices.remember(who, emb, jid, idx)      # kept with the line it came from (so a correction can undo it)
            out["learned"] = True
    if out.pop("_spread"):
        autotag.start(jid)                           # lines elsewhere that sound like them follow (background)
    return out


def _tag_locked(jid: str, idx: int, data: dict) -> dict:
    with autotag.job_lock(jid):
        return _tag(jid, idx, data)


def _tag(jid: str, idx: int, data: dict) -> dict:
    j = _job(jid)
    tr = worker.load_transcript(jid)
    segs = (tr or {}).get("segments", [])
    if not 0 <= idx < len(segs):
        raise HTTPException(404, "no such line")
    opts = dict(j["options"])
    names = dict(opts.get("speaker_names") or {})
    labels = {s.get("speaker") for s in segs if s.get("speaker")} | set(names)

    def resolve(name: str, label: str = "") -> str:
        name = terms.spelled(re.sub(r"\s+", " ", str(name or "")).strip()[:60])   # spelled as in Names & terms
        if name:
            label = next((lab for lab, n in names.items() if n.lower() == name.lower()), "")
            if not label and re.match(r"^Speaker \d+$", name):
                label = name
            if not label:                                # someone new in this meeting
                nums = [int(x.split()[-1]) for x in labels | set(names) if re.match(r"^Speaker \d+$", x or "")]
                label = f"Speaker {max(nums, default=0) + 1}"
                names[label] = name
                labels.add(label)
        return label

    group = segs[idx].get("orig") or segs[idx].get("speaker", "")   # the speaker-separation group of this line
    label = segs[idx].get("speaker", "")
    if data.get("name") or data.get("speaker"):
        label = resolve(data.get("name"), str(data.get("speaker") or "").strip())
        if not re.match(r"^Speaker \d+$", label):
            raise HTTPException(400, "send {'name': 'Alan'} or {'speaker': 'Speaker 2'}")
    elif "also" not in data:
        raise HTTPException(400, "send {'name': 'Alan'}, {'speaker': 'Speaker 2'} or {'also': 'Wei Ling'}")
    was = names.get(segs[idx].get("speaker", ""), "")
    if segs[idx].get("speaker") and segs[idx].get("speaker") != label:
        segs[idx].setdefault("orig", segs[idx]["speaker"])
        if was and was.lower() != (names.get(label, "") or "").lower():
            people.unlearn(was, jid, [idx])          # this line isn't them: don't keep it as their voice
    segs[idx]["speaker"] = label
    manual = dict(opts.get("manual_lines") or {})
    auto_lines = set(opts.get("auto_lines") or [])
    moved = 0
    # the rest of this line's group goes with it: the speaker separation already put the same voice together.
    # Not when you already named that group as someone else, or tagged a line of it as someone else
    # (a name the server guessed from a remembered voice doesn't count: you're correcting it).
    if (data.get("name") or data.get("speaker")) and data.get("spread", True) and group and group != label \
            and re.match(r"^Speaker \d+$", group) and (not names.get(group) or group in (opts.get("speaker_auto") or [])):
        conflict = any(i != str(idx) and lab != label and (segs[int(i)].get("orig") or "") == group
                       for i, lab in manual.items() if int(i) < len(segs))
        if not conflict:
            for k, s in enumerate(segs):
                if k == idx or str(k) in manual or s.get("also"):
                    continue
                if (s.get("orig") or s.get("speaker")) == group and s.get("speaker") == group:
                    s["orig"] = group
                    s["speaker"] = label
                    auto_lines.add(k)
                    moved += 1
            merged = dict(opts.get("merged") or {})
            merged[group] = label
            opts["merged"] = merged
    if "also" in data:
        other = resolve(data.get("also")) if data.get("also") else ""
        segs[idx]["also"] = [other] if other and other != label else []
    worker._write(jid, "transcript.json", json.dumps(tr, ensure_ascii=False, indent=1))
    learned = False
    who = names.get(label, "")
    opts["speaker_names"] = names
    opts["speaker_auto"] = sorted(set(opts.get("speaker_auto") or []) - {label})
    named_now = bool(data.get("name") or data.get("speaker"))
    if named_now:                                    # you chose who said it: auto-tagging never changes this line
        manual[str(idx)] = label
        auto_lines.discard(idx)
    opts["manual_lines"] = manual                    # (only the "+ second person" changed: the name stays as it was)
    opts["auto_lines"] = sorted(auto_lines)
    db.update(jid, options=opts)
    worker.build_outputs(jid)
    spread = named_now and bool(data.get("spread", True)) and bool(who) and bool(j["audio_path"]) and os.path.exists(j["audio_path"])
    return {"index": idx, "speaker": label, "name": who, "learned": learned, "moved": moved, "also": segs[idx].get("also") or [],
            "also_names": [names.get(a, a) for a in segs[idx].get("also") or []], "autotag": "started" if spread else "off",
            "_who": who if named_now else "", "_seg": dict(segs[idx]), "_spread": spread}


@app.post("/api/jobs/{jid}/segments/{idx}/text", dependencies=[Depends(auth)])
async def edit_line(jid: str, idx: int, request: Request):
    """Fix the words of one line: {"text": "...", "learn": true}. learn = remember the words you changed (heard ->
    right) and fix the same mistake in every meeting, and in new ones as they're transcribed."""
    j = _job(jid)
    if j["status"] != "done":
        raise HTTPException(409, f"job is {j['status']}")
    data = await body_extra(request)
    text = re.sub(r"\s+", " ", str(data.get("text") or "")).strip()[:2000]
    if not text:
        raise HTTPException(400, "send {'text': '...'}")
    with autotag.job_lock(jid):
        tr = worker.load_transcript(jid) or {}
        segs = tr.get("segments", [])
        if not 0 <= idx < len(segs):
            raise HTTPException(404, "no such line")
        before = segs[idx].get("text", "")
        segs[idx].setdefault("heard", before)
        segs[idx]["text"], segs[idx]["edited"] = text, True
        worker._write(jid, "transcript.json", json.dumps(tr, ensure_ascii=False, indent=1))
    learned = terms.learn(before, text) if data.get("learn", True) else []
    worker.build_outputs(jid)
    if learned:
        worker.refresh_terms_everywhere()            # the same mistake in your other meetings
    return {"index": idx, "text": text, "learned": [{"heard": h, "right": r} for h, r in learned]}


# ---------------------------------------------------------------- knowledge base (facts from all meetings)
@app.get("/api/knowledge", dependencies=[Depends(auth)])
def get_knowledge(q: str = "", limit: int = 500):
    """What the knowledge base knows: {"facts": n, "kinds", "meetings_read", "state", ..., "items": [...]}.
    ?q= returns the facts best matching a question."""
    items = knowledge.find(q, min(limit, 50)) if q else knowledge.facts()[: max(1, min(limit, 5000))]
    out = {**knowledge.summary(), "items": items}
    if q:
        out["answer"] = knowledge.best_answer(q)     # the fact that clearly answers it (None: ask the AI)
    return out


@app.post("/api/knowledge/learn", dependencies=[Depends(auth)], status_code=202)
async def learn_knowledge(request: Request):
    """Read the meetings that are new or changed (or all of them with {"all": true}) in the background."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    if not summarize.enabled():
        raise HTTPException(400, "SUMMARY_BACKEND is none - no LLM configured")
    started = knowledge.start_learning(bool((data or {}).get("all")))
    return {"started": started, **knowledge.summary()}


def _learn_source_bg(sid: str, force: bool = False) -> None:
    def run():
        try:
            sources.learn(sid, force)
        except Exception as e:                       # noqa: BLE001 - shown on the source (state "error")
            log.warning("source %s: %s", sid, e)
    if summarize.enabled():
        threading.Thread(target=run, daemon=True).start()


@app.get("/api/knowledge/sources", dependencies=[Depends(auth)])
def list_sources():
    """Your web pages, files and texts the knowledge base learns from."""
    return {"items": sources.listing()}


@app.post("/api/knowledge/sources", dependencies=[Depends(auth)], status_code=201)
async def add_source(request: Request):
    """{"url": "https://...", "crawl": 20} (the page + up to 20 more under it), {"news": "Aleta Planet"} (follow the
    news about it: searched and read daily), or {"text": "...", "title": ""};
    or a file (PDF, Word, text, Markdown, CSV, HTML) as multipart field "file". Read by the AI in the background."""
    ctype = request.headers.get("content-type", "")
    try:
        if ctype.startswith("multipart/"):
            form = await request.form()
            f = form.get("file")
            if not f or not hasattr(f, "read"):
                raise ValueError("send the file as the field 'file'")
            data = await f.read()
            if len(data) > 30 * 1024 * 1024:
                raise ValueError("files up to 30 MB")
            m = sources.add("file", str(form.get("title") or ""), filename=f.filename or "upload.txt", data=data)
        else:
            d = await request.json()
            if d.get("news"):
                m = sources.add("news", str(d.get("title") or ""), url=str(d["news"]))
            elif d.get("url"):
                m = sources.add("url", str(d.get("title") or ""), url=str(d["url"]), crawl=int(d.get("crawl") or 0))
            else:
                m = sources.add("text", str(d.get("title") or ""), text=str(d.get("text") or ""))
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    _learn_source_bg(m["id"])
    return {k: v for k, v in m.items() if k != "facts"}


@app.delete("/api/knowledge/sources/{sid}", dependencies=[Depends(auth)])
def delete_source(sid: str):
    if not sources.remove(sid):
        raise HTTPException(404, "no such source")
    return {"deleted": sid}


@app.post("/api/knowledge/sources/{sid}/refresh", dependencies=[Depends(auth)], status_code=202)
def refresh_source(sid: str):
    """Fetch / read it again now (the AI re-reads it even if the text is the same)."""
    if not sources.get(sid):
        raise HTTPException(404, "no such source")
    _learn_source_bg(sid, True)
    return {"started": summarize.enabled()}


# ---------------------------------------------------------------- research on its own (research.py)
@app.get("/api/research", dependencies=[Depends(auth)])
def research_state():
    """Topics researched on the web (yours and the ones it found), questions it researched, what's running."""
    return research.summary()


@app.put("/api/research", dependencies=[Depends(auth)])
async def research_set(request: Request):
    """{"auto": false} - stop finding topics and researching unanswered questions by itself."""
    d = await body_extra(request)
    if "auto" in d:
        research.set_auto(bool(d["auto"]))
    return research.summary()


@app.post("/api/research/topics", dependencies=[Depends(auth)], status_code=201)
async def research_add(request: Request):
    """{"topic": "Sanrio Aleta Adventure"} - research it on the web (now, then weekly)."""
    d = await body_extra(request)
    try:
        m = research.add_topic(str(d.get("topic") or ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    if m.get("state") in ("new", "changed"):
        _learn_source_bg(m["id"])
    return research.summary()


@app.delete("/api/research/topics", dependencies=[Depends(auth)])
def research_remove(topic: str):
    """Stop researching it (a topic it found isn't found again); what it learned goes too."""
    if not research.remove_topic(topic):
        raise HTTPException(404, "no such topic")
    return research.summary()


@app.post("/api/research/now", dependencies=[Depends(auth)], status_code=202)
def research_go():
    """Find topics, read every topic again and research the questions nobody could answer - now."""
    if not summarize.enabled():
        raise HTTPException(400, "SUMMARY_BACKEND is none - no LLM configured")
    return {"started": research.research_now()}


@app.put("/api/knowledge/auto", dependencies=[Depends(auth)])
async def knowledge_auto(request: Request):
    """{"on": false} - stop learning by itself (facts from new meetings, who-is, company profile, re-reading pages,
    research, watch-model training); {"on": true} - start again. The buttons still work either way."""
    d = await body_extra(request)
    if "on" not in d:
        raise HTTPException(400, "send {'on': true|false}")
    knowledge.set_auto_learning(bool(d["on"]))
    return knowledge.summary()


@app.post("/api/knowledge/core", dependencies=[Depends(auth)], status_code=202)
def core_now():
    """Write the company profile (the essentials always on the watch) again now, from everything known."""
    if not summarize.enabled():
        raise HTTPException(400, "SUMMARY_BACKEND is none - no LLM configured")
    from . import core

    def run():
        try:
            core.build(force=True)
        except Exception as e:                       # noqa: BLE001
            log.warning("company profile: %s", e)
    threading.Thread(target=run, daemon=True).start()
    return {"started": True}


@app.put("/api/knowledge/{fid}", dependencies=[Depends(auth)])
async def fix_fact(fid: str, request: Request):
    """{"a": "the right answer"} or {"hidden": true} (wrong or private fact) or {} (back to what the AI read)."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "body must be JSON")
    if not re.fullmatch(r"[0-9a-f]{8}", fid):
        raise HTTPException(400, "bad fact id")
    if "pinned" in data:                             # {"pinned": true} - always on the watch (like the company profile)
        knowledge.set_override(fid, pinned=bool(data["pinned"]))
    else:
        alts = data.get("alts")
        if alts is not None and not isinstance(alts, list):
            raise HTTPException(400, "alts must be a list of questions")
        knowledge.set_override(fid, str(data.get("a") or "") or None, bool(data.get("hidden")),
                               alts=[str(x) for x in alts] if alts is not None else None)
    return {"ok": True, **knowledge.summary()}


@app.post("/api/knowledge/correct", dependencies=[Depends(auth)])
async def correct_answer(request: Request):
    """"Edit this answer": {"q": "who is ryan", "a": "the right answer"} -> the matching fact gets it (or it becomes
    your own question). Oracle on the web and the watch answer with it from then on."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "body must be JSON")
    try:
        return {"ok": True, **knowledge.correct(str(data.get("q") or ""), str(data.get("a") or ""))}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/knowledge/custom", dependencies=[Depends(auth)])
async def add_question(request: Request):
    """Your own question for the watch: {"q": "Who is Oracle AI", "a": "...", "alts": ["other ways to ask"]} - always
    on the watch and answered there."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "body must be JSON")
    try:
        fid = knowledge.add_custom(str(data.get("q") or ""), str(data.get("a") or ""),
                                   [str(x) for x in (data.get("alts") or [])] if isinstance(data.get("alts"), list) else [])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "id": fid}


@app.delete("/api/knowledge/custom/{fid}", dependencies=[Depends(auth)])
def delete_question(fid: str):
    if not re.fullmatch(r"[0-9a-f]{8}", fid) or not knowledge.delete_custom(fid):
        raise HTTPException(404, "no such question of yours")
    return {"ok": True}


@app.get("/api/watch/pack", dependencies=[Depends(auth)])
def watch_pack(limit: int = watchpack.MAX_ITEMS, extra: bool = False):
    """The watch's offline answers: {"version", "count", "items": [{"id", "q", "say": [phrases], "a", ...}]};
    extra=1 also lists more questions (no answers) for the watch's own model: "extra": [{"id", "q", "say"}]."""
    return watchpack.build(max(1, min(limit, 400)), extra)


@app.get("/api/watch/voice/{fid}", dependencies=[Depends(auth)])
def watch_voice(fid: str):
    """A pack answer read out (IMA ADPCM WAV), for the watch to play offline. Made on first request, then kept."""
    p = watchpack.voice_file(fid)
    if not p:
        raise HTTPException(404, "no such answer, or no voice (TTS) on this server")
    return FileResponse(p, media_type="audio/wav")


@app.post("/api/watch/crash", dependencies=[Depends(auth)])
async def watch_crash(request: Request):
    """The watch's last crash (task, address, backtrace) after it restarted: {"fw", "text"} - kept to look into."""
    d = await body_extra(request)
    text = str(d.get("text") or "")[:2000]
    if not text:
        raise HTTPException(400, "send {'text': ...}")
    from . import crashdecode
    from starlette.concurrency import run_in_threadpool
    fw = str(d.get("fw") or "")[:20]
    dec = await run_in_threadpool(crashdecode.decode, text, fw)
    p = settings.data_dir / "watch_crashes.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps({"at": time.time(), "fw": fw, "text": text, "device": str(d.get("device") or "")[:40],
                            "decoded": dec}) + "\n")
    log.warning("watch crash report: %s | %s", text, " < ".join(dec.get("frames") or [dec.get("error", "")]))
    return {"ok": True, "decoded": dec}


@app.get("/api/watch/crashes", dependencies=[Depends(auth)])
def watch_crashes():
    """The watch's crash reports, newest first."""
    p = settings.data_dir / "watch_crashes.jsonl"
    try:
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        rows = []
    rows = rows[::-1][:50]
    from . import crashdecode
    for r in rows[:5]:                                   # reported before the symbols were there: decode now
        if not (r.get("decoded") or {}).get("frames"):
            r["decoded"] = crashdecode.decode(r.get("text", ""), r.get("fw", ""))
    return {"items": rows}


@app.get("/api/watch/weather", dependencies=[Depends(auth)])
def watch_weather(city: str = Query("")):
    """Today's and tomorrow's weather as spoken sentences, kept on the watch (every 2 hours) so "what's the weather"
    is answered even offline."""
    from . import worldinfo
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:
        t = ex.submit(worldinfo.weather_answer, "how's the weather today", city)
        m = ex.submit(worldinfo.weather_answer, "what's the weather tomorrow", city)
        today, tomorrow = t.result(), m.result()
    if not today:
        raise HTTPException(503, "no weather right now")
    return {"today": today, "tomorrow": tomorrow, "city": city or worldinfo.home_city()}


@app.get("/api/watch/prompts", dependencies=[Depends(auth)])
def watch_prompts():
    """The watch's own spoken words ("Okay.", "Starting the meeting."), in the answers' voice: {"voice", "names"}."""
    return watchpack.prompts_info()


@app.get("/api/watch/prompts/{name}", dependencies=[Depends(auth)])
def watch_prompt(name: str):
    p = watchpack.prompt_file(name)
    if not p:
        raise HTTPException(404, "no such prompt, or no voice (TTS) on this server")
    return FileResponse(p, media_type="audio/wav")


@app.post("/api/watch/asked", dependencies=[Depends(auth)])
async def watch_asked(request: Request):
    """The questions asked on the watch while offline: {"items": [{"id": "<fact id>", "q": "who is ryan"}]}."""
    try:
        d = await request.json()
    except Exception:
        raise HTTPException(400, "body must be JSON")
    n = 0
    for it in (d.get("items") or [])[:200]:
        if isinstance(it, dict) and it.get("q"):
            knowledge.note_question(str(it["q"]), "watch", str(it.get("id") or ""))
            n += 1
    return {"noted": n}


@app.get("/api/watch/speech", dependencies=[Depends(auth)])
def watch_speech_model():
    """The watch's offline speech models: ESP-SR's srmodels.bin exactly as Espressif ships it (MultiNet7 English and
    what it uses with it, 3.3 MB) - the trimmed copy of 1.12 (MultiNet only) crashed the watch."""
    return FileResponse(Path(__file__).parent / "watch_files" / "srmodels.bin", media_type="application/octet-stream")


@app.get("/api/watch/model", dependencies=[Depends(auth)])
def watch_model():
    """The watch's own AI model: {"training": bool, "status": {...progress}, "model": {version, size, scores...} | null}."""
    return watchlm.state()


@app.post("/api/watch/model/train", dependencies=[Depends(auth)], status_code=202)
def train_watch_model():
    """Train it again from the knowledge base (background, low priority; minutes to an hour or two)."""
    try:
        return watchlm.start(knowledge.facts())
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/watch/model/train", dependencies=[Depends(auth)])
def stop_watch_model():
    watchlm.stop()
    return watchlm.state()


@app.get("/api/watch/model/file", dependencies=[Depends(auth)])
def watch_model_file():
    p = watchlm.model_file()
    if not p:
        raise HTTPException(404, "no model trained yet")
    return FileResponse(p, media_type="application/octet-stream", filename="oracle_watch.owl")


@app.get("/api/wakeword", dependencies=[Depends(auth)])
def get_wakeword():
    """The watch's wake word set ("Hey Oracle" + offline commands): manifest with a version the watch compares."""
    return wakeword.manifest()


@app.get("/api/wakeword/files/{name}", dependencies=[Depends(auth)])
def get_wakeword_file(name: str):
    p = wakeword.file_path(name)
    if not p:
        raise HTTPException(404, "no such model file")
    return FileResponse(p, media_type="application/octet-stream")


@app.post("/api/wakeword", dependencies=[Depends(auth)])
async def put_wakeword(request: Request):
    """Install a trained set: a zip (or several files) with wake.json and the .tflite files it names, as multipart
    form fields, or the zip as the raw body. The watch picks it up at its next check (within a few hours, or
    Settings > Hey Oracle)."""
    uploads: dict[str, bytes] = {}
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/"):
        form = await request.form()
        for _, v in form.multi_items():
            if hasattr(v, "read"):
                uploads[v.filename or "upload.zip"] = await v.read()
    else:
        uploads["upload.zip"] = await request.body()
    if not uploads or sum(len(b) for b in uploads.values()) > 20 * 1024 * 1024:
        raise HTTPException(400, "send the wake word zip (max 20 MB)")
    try:
        return wakeword.install(uploads)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:                       # noqa: BLE001  (a broken zip)
        raise HTTPException(400, f"could not read the upload: {e}")


@app.post("/api/wakeword/samples/{mid}", dependencies=[Depends(auth)])
async def add_wake_sample(mid: str, request: Request):
    """One clip of you saying a phrase (WAV body), from the watch's "Teach my voice"."""
    data = await request.body()
    try:
        return wakeword.add_sample(mid, data)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/wakeword/samples", dependencies=[Depends(auth)])
def wake_samples():
    return wakeword.samples_summary()


@app.get("/api/wakeword/samples.zip", dependencies=[Depends(auth)])
def wake_samples_zip():
    """All your clips, for the training kit."""
    return Response(wakeword.samples_zip(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="my_voice_samples.zip"'})


@app.delete("/api/wakeword/samples", dependencies=[Depends(auth)])
def clear_wake_samples(phrase: Optional[str] = Query(None)):
    try:
        return wakeword.clear_samples(phrase)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/wakeword", dependencies=[Depends(auth)])
def reset_wakeword():
    """Back to the starter wake word."""
    return wakeword.reset()


@app.get("/api/vocab", dependencies=[Depends(auth)])
def get_vocab():
    """Built-in vocabularies (payments / tech / finance) and which are on."""
    on = set(vocab.enabled_packs())
    return {"packs": [{"id": k, "label": v[0], "on": k in on, "terms": vocab._split(v[1])} for k, v in vocab.PACKS.items()]}


@app.put("/api/vocab", dependencies=[Depends(auth)])
async def set_vocab(request: Request):
    """{"on": ["payments", "tech"]} - the rest are turned off; meetings are re-spelled in the background."""
    data = await body_extra(request)
    on = set(data.get("on") or [])
    vocab.set_off([k for k in vocab.PACKS if k not in on])
    vocab._recent["at"] = 0
    worker.refresh_terms_everywhere()
    return get_vocab()


@app.get("/api/corrections", dependencies=[Depends(auth)])
def get_corrections():
    """Words you corrected: {"tapin": "DABIN"} - fixed in every meeting."""
    return {"corrections": terms.corrections()}


@app.delete("/api/corrections/{heard}", dependencies=[Depends(auth)])
def delete_correction(heard: str):
    c = terms.corrections()
    if c.pop(heard.lower(), None) is None:
        raise HTTPException(404, "no such correction")
    terms.save_corrections(c)
    return {"corrections": c}


@app.post("/api/jobs/{jid}/fill", dependencies=[Depends(auth)])
async def fill_gap(jid: str, request: Request):
    """Nothing was written down between start and end: {"start": 242, "end": 511} transcribes that stretch again
    with a more sensitive speech detector, adds what it finds, then runs the voice check on the new lines."""
    j = _job(jid)
    if j["status"] != "done":
        raise HTTPException(409, f"job is {j['status']}")
    if not j["audio_path"] or not os.path.exists(j["audio_path"]):
        raise HTTPException(410, "the audio of this recording is no longer kept")
    data = await body_extra(request)
    try:
        a, b = float(data.get("start")), float(data.get("end"))
    except (TypeError, ValueError):
        raise HTTPException(400, "send {'start': seconds, 'end': seconds}")
    if b - a < 1 or b - a > 3600:
        raise HTTPException(400, "pick a stretch between 1 second and 1 hour")
    autotag.start(jid, fill=(a, b))
    return autotag.status(jid)


@app.post("/api/jobs/{jid}/autotag/accept", dependencies=[Depends(auth)])
async def accept_autotag(jid: str, request: Request):
    """The "auto" lines are right: mark them ✓ (yours). {"name": "Priscilla"} = only hers."""
    _job(jid)
    data = await body_extra(request)
    from starlette.concurrency import run_in_threadpool
    r = await run_in_threadpool(autotag.accept, jid, str(data.get("name") or ""))
    await run_in_threadpool(worker.build_outputs, jid)
    return r


@app.post("/api/jobs/{jid}/autotag", dependencies=[Depends(auth)])
def run_autotag(jid: str, loose: Optional[bool] = None):
    """Look again for lines that sound like the people you tagged here or anyone remembered from other meetings."""
    j = _job(jid)
    if j["status"] != "done":
        raise HTTPException(409, f"job is {j['status']}")
    if not j["audio_path"] or not os.path.exists(j["audio_path"]):
        raise HTTPException(410, "the audio of this recording is no longer kept")
    if (not (j.get("options") or {}).get("manual_lines") and not (j.get("options") or {}).get("speaker_names")
            and autotag.voices_signature().startswith("0:")):
        raise HTTPException(400, "tag a line or name a speaker first")
    if loose is not None:                                  # "Match more": accept closer calls for this meeting
        with autotag.job_lock(jid):
            o = dict((db.get(jid) or {}).get("options") or {})
            o["autotag_loose"] = bool(loose)
            db.update(jid, options=o)
    autotag.start(jid)
    return autotag.status(jid)


def speaker_labels(j: dict) -> list[str]:
    """Speakers of a meeting in the order the web page colours them."""
    jid = j["id"]
    spk = worker.load_json(jid, "speakers.json", {}) or {}
    names = (j.get("options") or {}).get("speaker_names") or {}
    tr = worker.load_transcript(jid) or {}
    labels = sorted(set(spk) | {s.get("speaker") for s in tr.get("segments", []) if s.get("speaker")}
                    | {a for s in tr.get("segments", []) for a in (s.get("also") or []) if a},
                    key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 99)
    used = {s.get("speaker") for s in tr.get("segments", [])} | {a for s in tr.get("segments", []) for a in (s.get("also") or [])}
    merged = (j.get("options") or {}).get("merged") or {}
    return [lab for lab in labels if lab in used or (lab not in merged and lab in names)]


def speaker_colours(j: dict) -> list:
    """[(name, colour, share of talking time)] for the Word file, most talking first."""
    names = (j.get("options") or {}).get("speaker_names") or {}
    tr = worker.load_transcript(j["id"]) or {}
    secs: dict[str, float] = {}
    for s in tr.get("segments", []):
        secs[s.get("speaker", "")] = secs.get(s.get("speaker", ""), 0) + float(s["end"]) - float(s["start"])
    total = sum(secs.values()) or 1
    out = []
    for i, lab in enumerate(speaker_labels(j)):
        out.append((names.get(lab) or lab, render.SPEAKER_COLOURS[i % 8], secs.get(lab, 0) / total))
    return sorted([o for o in out if o[2] > 0], key=lambda o: -o[2])


@app.get("/api/jobs/{jid}/speakers", dependencies=[Depends(auth)])
def get_speakers(jid: str):
    j = _job(jid)
    spk = worker.load_json(jid, "speakers.json", {}) or {}
    names = (j.get("options") or {}).get("speaker_names") or {}
    auto = set((j.get("options") or {}).get("speaker_auto") or [])
    tr = worker.load_transcript(jid) or {}
    sample = {}
    for s in tr.get("segments", []):               # a line each person said, to help tell them apart
        if s.get("speaker") and len(s["text"]) > len(sample.get(s["speaker"], "")) and len(s["text"]) < 200:
            sample[s["speaker"]] = s["text"]
    labels = speaker_labels(j)
    own, over = {}, {}
    for s in tr.get("segments", []):
        own[s.get("speaker")] = own.get(s.get("speaker"), 0) + 1
        for a in s.get("also") or []:
            over[a] = over.get(a, 0) + 1
            if a not in sample and len(s["text"]) < 200:        # only heard over someone else: that line
                sample[a] = s["text"]
    # nobody's: no line of their own, not over anyone, no name -> not worth a row
    labels = [lab for lab in labels if own.get(lab) or over.get(lab) or names.get(lab)]
    return {"speakers": [{"label": lab, "name": names.get(lab, ""), "auto": lab in auto,
                          "seconds": (spk.get(lab) or {}).get("seconds", 0), "has_voice": bool((spk.get(lab) or {}).get("embedding")),
                          "sample": sample.get(lab, ""), "lines": own.get(lab, 0), "over_lines": over.get(lab, 0)}
                         for lab in labels]}


@app.get("/api/voices", dependencies=[Depends(auth)])
def get_voices():
    return {"voices": voices.list_voices(), "threshold": settings.voice_match}


@app.delete("/api/voices/{name}", dependencies=[Depends(auth)])
def delete_voice(name: str):
    if not voices.forget(name):
        raise HTTPException(404, "unknown voice")
    return {"deleted": name}


@app.get("/api/me", dependencies=[Depends(auth)])
def get_me():
    """Your own name, for "Me" when tagging speakers."""
    p = settings.data_dir / "me.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {"name": os.getenv("ME_NAME", "")}
    except ValueError:
        return {"name": ""}


@app.put("/api/me", dependencies=[Depends(auth)])
async def set_me(request: Request):
    data = await body_extra(request)
    name = re.sub(r"\s+", " ", str(data.get("name") or "")).strip()[:60]
    (settings.data_dir / "me.json").write_text(json.dumps({"name": name}))
    return {"name": name}


@app.get("/api/backup", dependencies=[Depends(auth)])
def backup():
    """Everything you taught the server, in one file: Names & terms and the remembered voices."""
    with db._lock:
        rows = db._conn.execute("SELECT name, embedding, samples, updated_at FROM voices ORDER BY name").fetchall()
    p = settings.glossary_path
    data = {"kind": "whisper-minutes-backup", "version": VERSION, "made_at": int(time.time()),
            "glossary": p.read_text(encoding="utf-8") if p.exists() else "",
            "corrections": terms.corrections(),
            "voices": [{"name": r["name"], "embedding": json.loads(r["embedding"]), "samples": r["samples"],
                        "updated_at": r["updated_at"]} for r in rows]}
    return JSONResponse(data, headers={"Content-Disposition": 'attachment; filename="whisper-minutes-backup.json"'})


@app.post("/api/backup", dependencies=[Depends(auth)])
async def restore(request: Request):
    """Load a backup: Names & terms are merged (new lines added), voices added or replaced by name."""
    data = await body_extra(request)
    if data.get("kind") != "whisper-minutes-backup":
        raise HTTPException(400, "not an Oracle Minutes backup file")
    old = voices.glossary()
    new = [t for t in str(data.get("glossary", "")).splitlines() if t.strip() and not t.strip().startswith("#")]
    p = settings.glossary_path
    have = p.read_text(encoding="utf-8") if p.exists() else ""
    add = [t for t in new if t.strip() not in old]
    if add:
        voices.set_glossary((have.rstrip() + "\n" if have.strip() else "") + "\n".join(add))
    n = 0
    for v in data.get("voices") or []:
        name, emb = str(v.get("name", "")).strip()[:60], v.get("embedding")
        if not name or not isinstance(emb, list) or not emb:
            continue
        with db._lock:
            db._conn.execute("INSERT OR REPLACE INTO voices (name, embedding, samples, updated_at) VALUES (?,?,?,?)",
                             (name, json.dumps(emb), int(v.get("samples") or 1), float(v.get("updated_at") or time.time())))
            db._conn.commit()
        n += 1
    corr = data.get("corrections") or {}
    if isinstance(corr, dict) and corr:
        c = terms.corrections()
        c.update({str(k).lower(): str(v) for k, v in corr.items() if k and v})
        terms.save_corrections(c)
    return {"terms_added": len(add), "voices": n, "corrections": len(corr) if isinstance(corr, dict) else 0}


@app.post("/api/jobs/{jid}/speakers/redo", dependencies=[Depends(auth)])
def redo_speakers(jid: str):
    """Separate the speakers again (e.g. it failed when the meeting was transcribed). Your tagged lines stay."""
    j = _job(jid)
    if j["status"] != "done":
        raise HTTPException(409, f"job is {j['status']}")
    if not j["audio_path"] or not os.path.exists(j["audio_path"]):
        raise HTTPException(410, "the audio of this recording is no longer kept")
    autotag.start(jid, speakers=True)
    return autotag.status(jid)


@app.post("/api/voices/relearn", dependencies=[Depends(auth)])
def relearn_voices():
    """Learn voices again from the lines you tagged (for names that have no remembered voice)."""
    return autotag.relearn()


@app.post("/api/jobs/{jid}/dynamics", dependencies=[Depends(auth)])
async def meeting_dynamics(jid: str, request: Request):
    """{"external": true|false} = was this with a vendor / partner / client; {"write": true} = (re)write the AI
    read of how the meeting ran (background; poll GET .../dynamics)."""
    j = _job(jid)
    data = await body_extra(request)
    if "external" in data:
        with autotag.job_lock(jid):
            o = dict((db.get(jid) or {}).get("options") or {})
            o["external"] = bool(data["external"])
            db.update(jid, options=o)
    if data.get("write") and j["status"] == "done" and summarize.enabled():
        dynamics.write_note(jid)
    return dynamics.get(jid)


@app.get("/api/people", dependencies=[Depends(auth)])
def list_people():
    """Everyone you've named: meetings, talk time, pace, pitch (profiles are updated in the background)."""
    return {"people": people.listing()}


@app.get("/api/people/{name}", dependencies=[Depends(auth)])
def get_person(name: str):
    p = people.get(name)
    if not p:
        if name.lower() not in {n.lower() for n in people.all_names()}:
            raise HTTPException(404, "no such person")
        p = people.build(name, with_note=False)      # first look: the numbers now, the AI note in the background
        threading.Thread(target=people.build, args=(name,), daemon=True).start()
    p = dict(p, samples=people.sample_count(name))     # live: samples are added in the background
    return p


@app.get("/api/people/{name}/samples", dependencies=[Depends(auth)])
def person_samples(name: str):
    """The voice samples kept for this person and the line each came from (to play, and remove wrong ones)."""
    if name.lower() not in {n.lower() for n in people.all_names()}:
        raise HTTPException(404, "no such person")
    return {"items": people.sample_list(name)}


@app.post("/api/people/{name}/not-them", dependencies=[Depends(auth)])
async def person_not_them(name: str, request: Request):
    """{"job": "<meeting id>", "i": <line>} - that clip isn't them: not used for them any more (Hear them, their
    profile and voice), and the voice samples taken from it go. Their profile is rebuilt in the background."""
    d = await body_extra(request)
    jid, i = str(d.get("job") or ""), d.get("i")
    if not jid or not isinstance(i, int):
        raise HTTPException(400, "send {'job': ..., 'i': <line number>}")
    if name.lower() not in {n.lower() for n in people.all_names()}:
        raise HTTPException(404, "no such person")
    removed = people.not_them(name, jid, i)
    prof = people.build(name, with_note=False)
    return {"removed": removed, "samples": people.sample_count(name), "sample": prof.get("sample")}


@app.post("/api/people/{name}/relearn", dependencies=[Depends(auth)])
def relearn_person(name: str):
    """Rebuild this person's voice only from lines named as them now (e.g. after fixing lines that weren't them)."""
    if name.lower() not in {n.lower() for n in people.all_names()}:
        raise HTTPException(404, "no such person")
    return people.relearn_person(name)


@app.post("/api/people/{name}/refresh", dependencies=[Depends(auth)])
def refresh_person(name: str):
    """Rebuild this person's profile now (numbers, words, voice, and a new AI note)."""
    if name.lower() not in {n.lower() for n in people.all_names()}:
        raise HTTPException(404, "no such person")
    old = people.get(name) or {}
    if old:
        old["note_sig"] = ""                         # force a new note
        with db._lock:
            db._conn.execute("UPDATE people SET profile=? WHERE lower(name)=lower(?)", (json.dumps(old), name))
            db._conn.commit()
    return people.build(name)


@app.get("/api/glossary", dependencies=[Depends(auth)])
def get_glossary():
    p = settings.glossary_path
    g = voices.glossary()
    low = {t.lower() for t in g}
    vs = [v["name"] for v in voices.list_voices()]
    return {"text": p.read_text(encoding="utf-8") if p.exists() else "", "terms": g,
            "voices_linked": [n for n in vs if n.lower() in low],      # a name here + a remembered voice
            "voices_only": [n for n in vs if n.lower() not in low]}    # remembered voices: spelled like this too


@app.put("/api/glossary", dependencies=[Depends(auth)])
async def put_glossary(request: Request):
    raw = (await request.body()).decode("utf-8", "replace")
    try:
        import json as _json
        v = _json.loads(raw)
        if isinstance(v, dict):
            raw = str(v.get("text", ""))
    except ValueError:
        pass
    t = voices.set_glossary(raw)
    return {"terms": t, "meetings": worker.refresh_terms_everywhere()}


@app.get("/api/search", dependencies=[Depends(auth)])
def search_meetings(q: str = Query(..., min_length=2), limit: int = Query(20, le=50)):
    return {"hits": search.search(q, limit)}


@app.get("/api/templates")
def templates():
    return {"templates": [{"id": k, "label": v[0]} for k, v in summarize.TEMPLATES.items()],
            "best_available": bool(summarize.best_model())}


# ---------------------------------------------------------------- voice (round Idea Saver)
@app.post("/api/stt", dependencies=[Depends(auth)])
async def speech_to_text(request: Request, language: Optional[str] = Query(None), languages: Optional[str] = Query(None)):
    """Short clip (WAV, up to ~2 min) -> {"text", "language", "duration"} right away, outside the job queue.
    language=en forces one language; languages=en,zh detects but only picks from that list."""
    tmp = settings.upload_dir / f"stt_{secrets.token_hex(6)}.wav"
    try:
        n = await save_body(request, tmp)
        if n > 8 * 1024 * 1024:
            raise HTTPException(413, "clip too long for /api/stt - use the upload API")
        if n < 100:
            raise HTTPException(400, "empty audio")
        transcribe.fix_wav_header(str(tmp))
        from starlette.concurrency import run_in_threadpool
        lang = language if language and language.lower() != "auto" else None
        cands = [c for c in (languages or "").split(",") if c.strip()] or None
        return await run_in_threadpool(speech.stt, str(tmp), lang, cands)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------- a spoken question in one trip (the watch)
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def _voice_bytes(text: str) -> bytes:
    """The text read out (ElevenLabs' fast model / Piper) as IMA ADPCM blocks (256 bytes, 505 samples each, 16 kHz) -
    blocks stand on their own, so the watch plays them as they come. b"" if it couldn't be made."""
    t0 = time.time()
    try:
        data = speech.wav_to_ima_adpcm(speech.tts(text, live=True))[48:]
    except Exception as e:                           # noqa: BLE001 - the text still shows
        log.warning("voice for %r: %s", text[:40], e)
        return b""
    from . import timing
    timing.note("voice part", time.time() - t0, f"{len(text)} characters")
    return data


def _b_lines(data: bytes):
    import base64 as _b64
    for i in range(0, len(data), 4096):
        yield "B " + _b64.b64encode(data[i:i + 4096]).decode() + "\n"


def _voice_lines(text: str):
    yield from _b_lines(_voice_bytes(text))


class _FirstVoice:
    """The first sentence's voice, sent while ElevenLabs is still making it (its first sound ~0.2-0.3 s sooner than
    waiting for the whole sentence). take() = the B lines ready so far; done when all of it went."""

    def __init__(self, text: str):
        self.parts: list[bytes] = []
        self.done = False
        self._lock = threading.Lock()
        self.t0 = time.time()
        self.text = text
        threading.Thread(target=self._run, daemon=True, name="voice1").start()

    def _run(self):
        import numpy as _np
        held = _np.zeros(0, dtype="<i2")
        try:
            for pcm in speech.tts_stream(self.text):
                held = _np.concatenate([held, pcm])
                n = len(held) // 505 * 505
                if n:
                    b = speech.adpcm_blocks(held[:n])
                    held = held[n:]
                    with self._lock:
                        self.parts.append(b)
            if len(held):
                with self._lock:
                    self.parts.append(speech.adpcm_blocks(held))
        except Exception as e:                       # noqa: BLE001 - the text still shows
            log.warning("first voice for %r: %s", self.text[:40], e)
        from . import timing
        timing.note("voice part", time.time() - self.t0, f"{len(self.text)} characters, streamed")
        self.done = True

    def take(self):
        with self._lock:
            parts, self.parts = self.parts, []
        for b in parts:
            yield from _b_lines(b)

    def finish(self):
        while not self.done:
            yield from self.take()
            time.sleep(0.03)
        yield from self.take()


def _first_split(sent: str) -> list[str]:
    """A long first sentence is voiced in two (at a comma near the middle): its first half is heard sooner."""
    if len(sent) < 90:
        return [sent]
    best = None
    for m in re.finditer(r"[,;:]\s", sent):
        if 25 <= m.start() <= len(sent) - 25 and (best is None or abs(m.start() - len(sent) / 2) < abs(best - len(sent) / 2)):
            best = m.start()
    return [sent] if best is None else [sent[:best + 1].strip(), sent[best + 1:].strip()]


_FILLERS = ("Let me check.", "One moment.", "Let me look that up.")
_STOP_WORDS = re.compile(r"(?:(?:hey )?oracle )?(?:stop|stop it|stop talking|cancel|never ?mind|be quiet|quiet|shut up|"
                         r"that'?s all|that is all|nothing|no thanks|no thank you|thank you|thanks|ok(?:ay)? thanks|"
                         r"goodbye|bye)")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")
from concurrent.futures import TimeoutError as FutureTimeout  # noqa: E402


# the sentences' voices are made side by side (ElevenLabs takes a few hundred ms each) and sent in order
_TTS_POOL = None


def _tts_pool():
    global _TTS_POOL
    if _TTS_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _TTS_POOL = ThreadPoolExecutor(3, thread_name_prefix="voice")
    return _TTS_POOL


_LATE_POOL = None


def _late_pool():
    global _LATE_POOL
    if _LATE_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _LATE_POOL = ThreadPoolExecutor(4, thread_name_prefix="late")
    return _LATE_POOL


def _sentences(text: str) -> list[str]:
    """A spoken answer in sentences, short ones joined to the next (the first comes out on its own: it starts the voice)."""
    out, cur = [], ""
    for part in _SENT_END.split(text.strip()):
        cur = (cur + " " + part).strip()
        if len(cur) >= 12:
            out.append(cur)
            cur = ""
    if cur:
        if out:
            out[-1] += " " + cur
        else:
            out.append(cur)
    return out


@app.post("/api/voice", dependencies=[Depends(auth)])
async def voice_question(request: Request, language: Optional[str] = Query(None), speak: int = Query(1),
                         tz: Optional[int] = Query(None), city: str = Query(""), scope: str = Query(""),
                         intents: int = Query(1)):
    """A spoken question in one trip (the watch): body = the recording (WAV); header X-Ask = base64 of JSON
    {"messages": [system + the conversation so far]}. The reply is a stream of lines, as each part is ready:
      T {"text", "language"}   what you said (at once - the watch may answer it itself and hang up)
      X {"type", ...}          a reminder / timer / task / note / meeting / translation to do on the watch (then E)
      P {"text"}               the answer so far, a sentence at a time
      B <base64>               its voice: IMA ADPCM blocks (16 kHz, 256-byte blocks), playable as they come
      A {"answer"}             the whole answer
      E {} / E {"error"}       the end
    The voice of the first sentence comes while the AI is still writing the rest."""
    tmp = settings.upload_dir / f"voice_{secrets.token_hex(6)}.wav"
    t_in = time.time()
    summarize.warm_cloud()                           # while you're still talking (the watch sends as you speak):
    speech.warm_voice()                              # the connections to Groq and ElevenLabs open now
    n = await save_body(request, tmp)
    from . import timing as _timing
    _timing.note("voice question in", time.time() - t_in, f"{n // 1024} KB, one trip")
    if n < 100:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "empty audio")
    if n > 8 * 1024 * 1024:
        tmp.unlink(missing_ok=True)
        raise HTTPException(413, "clip too long")
    ask: dict = {}
    if request.headers.get("x-ask"):
        import base64 as _b64
        try:
            ask = json.loads(_b64.b64decode(request.headers["x-ask"]).decode("utf-8", "replace"))
        except (ValueError, TypeError):
            ask = {}
    if not isinstance(ask, dict):
        ask = {}

    def line(kind: str, obj: dict) -> str:
        return f"{kind} {json.dumps(obj, ensure_ascii=False)}\n"

    def gen():
        from . import timing
        t0 = time.time()
        try:
            transcribe.fix_wav_header(str(tmp))
            lang = language if language and language.lower() != "auto" else None
            r = speech.stt(str(tmp), lang, None)
        except Exception as e:                       # noqa: BLE001
            yield line("E", {"error": f"speech to text failed: {str(e)[:160]}"})
            return
        finally:
            tmp.unlink(missing_ok=True)
        text = speech.strip_wake((r.get("text") or "").strip())
        timing.note("heard", 0, text[:60])
        yield line("T", {"text": text, "language": r.get("language") or ""})
        if not text or _STOP_WORDS.fullmatch(re.sub(r"[^a-z' ]", "", text.lower()).strip()):
            yield line("E", {})                      # "stop", "never mind", "thanks": nothing to answer (the watch goes quiet)
            return
        tzm = tz if tz is not None else settings.default_tz_min
        if intents and scope != "meetings" and not speech.plain_question(text):
            # reminders, timers, tasks, notes, translations: by the rules only - asking the AI what you meant first
            # cost 2-4 s on every question that isn't plainly a question
            from datetime import datetime as _dt, timedelta as _td, timezone as _tz
            it = speech.rule_intent(text, _dt.now(_tz(_td(minutes=tzm)))) or {}
            if it.get("type") not in (None, "", "question", "none"):
                yield line("X", it)
                yield line("E", {})
                return
        msgs = [m for m in (ask.get("messages") or []) if isinstance(m, dict)][-12:]
        data = {"messages": msgs + [{"role": "user", "content": text}], "tz": tzm, "city": city or ask.get("city") or ""}
        if scope:
            data["scope"] = scope
        try:
            plan = _ask_plan(data)
        except HTTPException as e:
            yield line("E", {"error": str(e.detail)})
            return
        if "instant" in plan:
            a = plan["instant"]
            yield line("A", {"answer": a})
            if speak:                                    # sentence by sentence: the first one starts sooner
                ss = _sentences(a)
                if ss:
                    ss = _first_split(ss[0]) + ss[1:]
                    fv = _FirstVoice(ss[0])              # (streamed as it's made)
                    futs = [_tts_pool().submit(_voice_bytes, s_) for s_ in ss[1:]]
                    yield from fv.finish()
                    for f in futs:
                        yield from _b_lines(f.result())
            timing.note("voice trip", time.time() - t0, "without the AI")
            yield line("E", {})
            return
        system = plan["system"] + (("\n\n" + plan["context"]) if plan["context"] else "")
        if plan.get("late"):
            # the web search / company knowledge for the answer: when it takes a moment, say so ("Let me check")
            # instead of a silence - like a phone assistant (the phrases are voiced once and kept)
            fl = _late_pool().submit(plan["late"])
            try:
                extra = fl.result(timeout=settings.filler_after_s) if speak and settings.filler_after_s > 0 else fl.result()
            except FutureTimeout:
                yield from _b_lines(_voice_bytes(_FILLERS[int(time.time()) % len(_FILLERS)]))
                timing.note("said", 0, "let me check")
                try:
                    extra = fl.result()
                except Exception:                    # noqa: BLE001
                    extra = ""
            except Exception:                        # noqa: BLE001
                extra = ""
            if extra:
                system += "\n\n" + extra
        full, pending, shown, first = "", "", "", None
        from collections import deque
        futs: deque = deque()                        # voices being made, in the answer's order
        fv: Optional[_FirstVoice] = None             # the first sentence's, streamed (before all the others)
        try:
            with chat._sem:
                for piece in summarize.chat_stream([{"role": "system", "content": system}] + data["messages"]):
                    full += piece
                    pending += piece
                    while True:
                        m = _SENT_END.search(pending)
                        if (not m or len(pending[:m.start()].strip()) < 12) and fv is None and speak:
                            # no sentence end yet: the first words go out at the first comma (8+ words in) - the voice
                            # starts ~0.3-0.5 s sooner on a long first sentence
                            c = next((x for x in _CLAUSE.finditer(pending) if x.start() >= 40), None)
                            if c and "<think>" not in pending:
                                m = c
                        if not m or len(pending[:m.start()].strip()) < 12:
                            break
                        sent, pending = pending[:m.start()].strip(), pending[m.end():]
                        shown = (shown + " " + sent).strip()
                        if first is None:
                            first = time.time() - t0
                        yield line("P", {"text": shown})
                        if speak:                    # made while the AI writes on
                            if fv is None:
                                parts = _first_split(sent)
                                fv = _FirstVoice(parts[0])
                                for p_ in parts[1:]:
                                    futs.append(_tts_pool().submit(_voice_bytes, p_))
                            else:
                                futs.append(_tts_pool().submit(_voice_bytes, sent))
                    if fv is not None:
                        yield from fv.take()
                    while futs and futs[0].done() and (fv is None or fv.done):   # the voices ready so far, in order
                        if fv is not None:
                            yield from fv.take()
                        yield from _b_lines(futs.popleft().result())
        except Exception as e:                       # noqa: BLE001
            if full.strip() or not speak:            # (part of it was said: say why it stopped)
                yield line("E", {"error": summarize.explain_llm_error(e)[:200]})
                return
            # nothing said yet: a spoken "busy" answer instead of an error and a silence (watch waits no longer)
            log.warning("voice: the AI failed: %s", e)
            msg = ("Sorry, the AI is busy right now. Please try again in a minute." if "busy" in str(e)
                   else "Sorry, I couldn't reach the AI just now. Please try again.")
            yield line("A", {"answer": msg})
            yield from _b_lines(_voice_bytes(msg))
            timing.note("voice trip", time.time() - t0, "AI failed: " + str(e)[:60])
            yield line("E", {})
            return
        rest = re.sub(r"<think>.*?</think>", "", pending, flags=re.S).strip()
        answer = re.sub(r"<think>.*?</think>", "", full, flags=re.S).strip()
        if speak and rest:
            if fv is None:
                fv = _FirstVoice(rest)
            else:
                futs.append(_tts_pool().submit(_voice_bytes, rest))
        if fv is not None:
            yield from fv.take()
        while futs and futs[0].done() and (fv is None or fv.done):
            yield from _b_lines(futs.popleft().result())
        yield line("A", {"answer": answer})
        if fv is not None:
            yield from fv.finish()
        while futs:
            yield from _b_lines(futs.popleft().result())
        timing.note("voice trip", time.time() - t0, f"first sentence at {first or time.time() - t0:.1f}s")
        yield line("E", {})

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/intent", dependencies=[Depends(auth)])
async def voice_intent(request: Request):
    """{"text": "remind me at 3 to call San", "now": <epoch>, "tz": <minutes from UTC>} -> an action:
    reminder {text, at, repeat} | timer {seconds, label} | task {text} | note {text} | meeting {title} |
    translate {text, lang} | question."""
    data = await body_extra(request)
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(speech.intent, str(data.get("text", ""))[:1000],
                                   int(data["now"]) if str(data.get("now", "")).isdigit() else None,
                                   int(data.get("tz", 480)) if str(data.get("tz", "480")).lstrip("-").isdigit() else 480)


@app.post("/api/tts", dependencies=[Depends(auth)])
async def text_to_speech(request: Request):
    """{"text": "...", "lang": "en", "format": "adpcm"} -> audio/wav (16 kHz mono; "adpcm" = IMA ADPCM, 4x smaller,
    what the round Idea Saver asks for). 503 when no TTS is installed."""
    data = await body_extra(request)
    if not speech.tts_available():
        raise HTTPException(503, "text-to-speech not installed (build with INSTALL_TTS=1, TTS=piper)")
    from starlette.concurrency import run_in_threadpool
    try:
        adpcm = str(data.get("format", "")).lower() == "adpcm"
        from . import timing
        t0 = time.time()
        wav = await run_in_threadpool(chat.spoken, str(data.get("text", "")), str(data.get("lang", "en")), adpcm)   # made ahead
        if wav is not None:
            timing.note("voice sent", time.time() - t0, "made ahead")
        else:
            wav = await run_in_threadpool(lambda: speech.tts(str(data.get("text", "")), str(data.get("lang", "en")), live=True))
            if adpcm:
                wav = await run_in_threadpool(speech.wav_to_ima_adpcm, wav)
            timing.note("voice sent", time.time() - t0, "made now")
    except Exception as e:
        raise HTTPException(500, f"tts failed: {str(e)[:200]}")
    return Response(wav, media_type="audio/wav")


# ---------------------------------------------------------------- calendar (device Agenda app)
@app.get("/api/agenda", dependencies=[Depends(auth)])
def get_agenda(days: int = Query(7, ge=1, le=60)):
    if not agenda.sources():
        raise HTTPException(503, "no calendars configured - set CALENDAR_ICS_URLS or MS_TENANT_ID/MS_CLIENT_ID/MS_CLIENT_SECRET/MS_CALENDAR_USER")
    return agenda.agenda(days)


# ---------------------------------------------------------------- file sync (notes backup, two-way)
SAFE_PATH = re.compile(r"^[A-Za-z0-9 ._()&,'+@#!-]+(/[A-Za-z0-9 ._()&,'+@#!-]+)*$")
SYNC_EXT = (".md", ".txt", ".csv", ".json")


def fnv1a(data: bytes) -> str:
    """Same 32-bit FNV-1a hash the device uses to spot changed files."""
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def check_path(path: str) -> str:
    path = (path or "").strip().lstrip("/")
    if (not SAFE_PATH.match(path) or len(path) > 200 or any(p in ("", ".", "..") for p in path.split("/"))
            or not path.lower().endswith(SYNC_EXT)):
        raise HTTPException(400, "bad file path (text files .md .txt .csv .json, no ..)")
    return path


def check_device(device: str) -> str:
    if not SAFE_ID.match(device or ""):
        raise HTTPException(400, "device id may only use letters, digits, . _ -")
    return device


@app.get("/api/files", dependencies=[Depends(auth)])
def files_list(device: str = Query("")):
    if not device:
        return {"devices": db.file_devices()}
    return {"device": check_device(device), "files": db.file_list(device)}


@app.get("/api/files/{path:path}", dependencies=[Depends(auth)])
def file_read(path: str, device: str = Query(...)):
    meta, data = db.file_get(check_device(device), check_path(path))
    if not meta or meta["deleted"]:
        raise HTTPException(404, "no such file")
    return Response(data, media_type="text/plain; charset=utf-8",
                    headers={"X-Rev": str(meta["rev"]), "X-Hash": meta["hash"]})


def _conflict(meta: Optional[dict]) -> JSONResponse:
    return JSONResponse({"error": "conflict - file changed elsewhere",
                         "rev": meta["rev"] if meta else 0, "hash": meta["hash"] if meta else "",
                         "deleted": meta["deleted"] if meta else True}, status_code=409)


def _base_ok(meta: Optional[dict], base_rev: int) -> bool:
    cur = meta["rev"] if meta else 0
    return base_rev == cur or (base_rev == 0 and (meta is None or meta["deleted"]))


@app.put("/api/files/{path:path}", dependencies=[Depends(auth)])
async def file_write(path: str, request: Request, device: str = Query(...), base_rev: int = Query(0),
                     by: str = Query("device")):
    device, path = check_device(device), check_path(path)
    data = await request.body()
    if len(data) > settings.sync_max_kb * 1024:
        raise HTTPException(413, f"file bigger than SYNC_MAX_KB={settings.sync_max_kb}")
    h = fnv1a(data)
    meta, _ = db.file_get(device, path)
    if meta and not meta["deleted"] and meta["hash"] == h:          # same content: nothing to do
        return {"path": path, "rev": meta["rev"], "hash": h, "unchanged": True}
    if not _base_ok(meta, base_rev):
        return _conflict(meta)
    m = db.file_put(device, path, data, h, by[:20])
    return {"path": path, "rev": m["rev"], "hash": h}


@app.delete("/api/files/{path:path}", dependencies=[Depends(auth)])
def file_delete(path: str, device: str = Query(...), base_rev: int = Query(0), by: str = Query("device")):
    device, path = check_device(device), check_path(path)
    meta, _ = db.file_get(device, path)
    if not meta or meta["deleted"]:
        return {"path": path, "rev": meta["rev"] if meta else 0, "deleted": True}
    if base_rev != meta["rev"]:
        return _conflict(meta)
    m = db.file_put(device, path, None, "", by[:20])
    return {"path": path, "rev": m["rev"], "deleted": True}


# ---------------------------------------------------------------- misc
def storage_kind(mountinfo: Optional[str] = None) -> str:
    """Is DATA_DIR kept across redeploys? "temporary" = an anonymous Docker volume (a new, empty one each deploy,
    e.g. Coolify's Dockerfile build pack without Persistent Storage), "persistent" = a named volume or bind mount."""
    try:
        if mountinfo is None:
            with open("/proc/self/mountinfo") as f:
                mountinfo = f.read()
    except OSError:
        return "unknown"
    target = os.path.realpath(str(settings.data_dir))
    best = None
    for line in mountinfo.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        root, point = parts[3], parts[4]
        if (target == point or target.startswith(point.rstrip("/") + "/")) and (best is None or len(point) > len(best[1])):
            best = (root, point)
    if not best or best[1] == "/":
        return "temporary"                           # not a mount at all: inside the container's own layer
    if re.search(r"/volumes/[0-9a-f]{64}/_data", best[0]):
        return "temporary"                           # anonymous volume (64-hex name)
    return "persistent"


@app.get("/api/live", dependencies=[Depends(auth)])
def live_now():
    """Recordings streaming in from a watch right now, with the live transcript so far."""
    return {"live": worker.live_now()}


@app.get("/api/live/{rid}/brief", dependencies=[Depends(auth)])
def live_brief(rid: str):
    """For the watch while it records: the last line heard, who said it, who has talked so far. Small and quick."""
    b = _brief(check_id(rid))
    if b is None:
        raise HTTPException(404, "not streaming")
    return b


def _brief(rid: str) -> Optional[dict]:
    lp = worker.live_state_path(rid)
    if not lp.exists():
        return None
    try:
        st = json.loads(lp.read_text())
    except (OSError, ValueError):
        st = {}
    names = st.get("names") or {}
    segs = st.get("segments") or []
    last = segs[-1] if segs else {}
    part = settings.upload_dir / f"{rid}.part"
    rec = round(worker.wav_seconds(part), 1) if part.exists() else 0.0
    done = float(st.get("until", 0))
    lab = last.get("speaker") or ""
    sp = sorted(st.get("voices") or [], key=lambda v: v.get("secs", 0), reverse=True)[:4]
    return {"rid": rid, "line": (last.get("text") or "").strip()[:140], "speaker": names.get(lab, lab),
            "speakers": [{"name": names.get(v["label"], v["label"]), "seconds": round(v.get("secs", 0))} for v in sp],
            "transcribed_s": done, "received_s": rec, "behind_s": round(max(0.0, rec - done), 1)}


# ---------------------------------------------------------------- watches: logs, status, firmware updates
def _did(did: str) -> str:
    if not devices.DEV_ID.match(did or ""):
        raise HTTPException(400, "device id may only use letters, digits, _ - (max 40)")
    return did


@app.post("/api/devices/{did}/log", dependencies=[Depends(auth)])
async def device_log(did: str, request: Request, x_status: Optional[str] = Header(None)):
    """The watch's log lines (text/plain) and, in X-Status, its state (battery, WiFi, memory...) - every minute."""
    _did(did)
    body = await request.body()
    if len(body) > 256 * 1024:
        raise HTTPException(413, "send at most 256 KB of log at a time")
    status = None
    if x_status:
        try:
            status = json.loads(x_status)
            if not isinstance(status, dict):
                status = None
        except ValueError:
            status = None
    devices.add_log(did, body.decode("utf-8", "replace"), status)
    return {"ok": True, "next_s": devices.POST_EVERY_S}


@app.get("/api/devices", dependencies=[Depends(auth)])
def devices_list():
    return {"devices": devices.list_devices()}


@app.get("/api/devices/{did}/stability", dependencies=[Depends(auth)])
def device_stability(did: str, days: int = Query(7, ge=1, le=60)):
    """How the watch has been, day by day: restarts, crashes, failed sends, memory running out, questions, slowest."""
    return {"days": devices.stability(_did(did), days, settings.default_tz_min)}


@app.get("/api/devices/{did}/log", dependencies=[Depends(auth)])
def device_log_read(did: str, lines: int = Query(500, ge=1, le=5000)):
    return PlainTextResponse(devices.read_log(_did(did), lines))


@app.delete("/api/devices/{did}/log", dependencies=[Depends(auth)])
def device_log_clear(did: str):
    devices.clear_log(_did(did))
    return {"ok": True}


@app.post("/api/firmware", dependencies=[Depends(auth)])
async def firmware_upload(request: Request, notes: str = Query("")):
    """A firmware .bin for watches to install over WiFi; its board and version are read from the file."""
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > devices.FW_MAX:
            raise HTTPException(413, "firmware bigger than 8 MB")
    try:
        m = devices.save_firmware(bytes(data), notes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"board": m["board"], "version": m["version"], "size": m["size"], "sha256": m["sha256"]}


@app.get("/api/firmware", dependencies=[Depends(auth)])
def firmware_list():
    threading.Thread(target=devices.github_check, args=(120,), daemon=True).start()   # opening the page looks too
    return {"items": devices.list_firmware(), "github": devices.gh_status}


@app.post("/api/firmware/github", dependencies=[Depends(auth)])
async def firmware_github():
    """Look at GitHub for new watch firmware now."""
    import asyncio
    added = await asyncio.to_thread(devices.github_check)
    return {"added": added, "github": devices.gh_status}


@app.get("/api/firmware/latest", dependencies=[Depends(auth)])
async def firmware_latest(board: str = Query(...), current: str = Query("")):
    if not devices.BOARD.match(board):
        raise HTTPException(400, "bad board")
    # Check for updates on the watch: GitHub is looked at too, in the background - the watch gets its answer at once
    # (a download from GitHub can take longer than the watch waits); a version just found is there on the next check
    th = threading.Thread(target=devices.github_check, args=(60,), daemon=True)
    th.start()
    import asyncio
    await asyncio.to_thread(th.join, 12)           # usually done by then: what was just pushed is offered now
    r = devices.latest(board, current)
    if not r:
        raise HTTPException(404, "no firmware for this board")
    if current and not r["newer"]:
        # up to date (W1B-1.5.8 and older can't compare versions themselves: they'd be offered their own again)
        raise HTTPException(404, "up to date")
    return r


def _fw(board: str, version: str) -> tuple[str, str]:
    if not devices.BOARD.match(board or "") or not devices.VERSION.match(version or ""):
        raise HTTPException(400, "bad board or version")
    return board, version


@app.get("/api/firmware/{board}/{fname}", dependencies=[Depends(auth)])
def firmware_file(board: str, fname: str):
    if not fname.endswith(".bin"):
        raise HTTPException(404, "not found")
    p = devices.firmware_path(*_fw(board, fname[:-4]))
    if not p:
        raise HTTPException(404, "no such firmware")
    return FileResponse(p, media_type="application/octet-stream", filename=f"{board}-{fname}")


@app.post("/api/firmware/{board}/{version}", dependencies=[Depends(auth)])
async def firmware_set(board: str, version: str, request: Request):
    _fw(board, version)
    d = await body_extra(request)
    m = devices.set_active(board, version, bool(d.get("active", True)))
    if not m:
        raise HTTPException(404, "no such firmware")
    return m


@app.delete("/api/firmware/{board}/{version}", dependencies=[Depends(auth)])
def firmware_delete(board: str, version: str):
    if not devices.delete_firmware(*_fw(board, version)):
        raise HTTPException(404, "no such firmware")
    return {"deleted": f"{board}/{version}"}


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "version": VERSION,
        "model": settings.whisper_model,
        "model_loaded": transcribe.model_loaded() or settings.transcriber == "fake",
        "summary_backend": settings.summary_backend,
        "summary_model": settings.ollama_model if settings.summary_backend == "ollama" else settings.openai_model,
        "llm": summarize.llm_status(),
        "default_language": settings.default_language or "auto",
        "region": settings.region or "none",
        "storage": storage_kind(),
        "summary_language": settings.summary_language or "same as speech",
        "diarize_default": settings.diarize,
        "diarize_installed": importlib.util.find_spec("pyannote") is not None,
        "hf_token_set": bool(settings.hf_token),
        "calendars": agenda.sources(),
        "best_model": summarize.best_model(),
        "redact": settings.redact,
        "voices": len(voices.list_voices()),
        "voice_prints": ("error: " + transcribe.embed_error) if transcribe.embed_error else "ok",
        "speaker_separation": ("error: " + transcribe.diarize_error) if transcribe.diarize_error else "ok",
        "speaker_libs": transcribe.lib_versions(),
        "tts": speech.tts_available(),
        "tts_error": speech.tts_error,
        "timing": __import__("app.timing", fromlist=["recent"]).recent(),
        "question_stt": ((lambda t: f"cloud {t[2]} at {t[0].split('//')[-1].split('/')[0]}, else "
                          if t else "")(speech.cloud_stt_target())) + str(transcribe.quick_model_status()),
        "web_search": __import__("app.sources", fromlist=["search_status"]).search_status,
        "live_ai_fallback": {"model": summarize.fallback_model(), **summarize.cloud_status},
        "live_ai": (settings.live_llm_model + " at " + settings.live_llm_base_url.split("//")[-1].split("/")[0])
                   if settings.live_llm_api_key else "",
        "tts_voice": ("ElevenLabs " + settings.elevenlabs_model) if speech.eleven_on() else
                     ("Piper " + str(speech.tts_voices().get("en", ""))) if speech.tts_available() else "",
        "jobs": db.counts(),
        "running": worker.current_job,
        "progress": worker.progress_info(),
        "voice_work": [{"id": k, "title": ((db.get(k) or {}).get("title") or (db.get(k) or {}).get("filename") or k), **v}
                       for k, v in autotag.active().items()],
    }


app.include_router(login.router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    if login.enabled() and not login.session_user(request):
        return RedirectResponse("/login", 303)
    # always the page of the version running (no hard refresh needed after an update)
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return PlainTextResponse("", status_code=204)
