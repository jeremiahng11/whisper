"""HTTP API + small web page. Run: uvicorn app.main:app --host 0.0.0.0 --port 8000"""
import logging
import os
import re
import secrets
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from . import chat, db, summarize, transcribe, worker
from .config import settings

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
VERSION = "1.0.0"
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.api_key:
        raise RuntimeError("Set the API_KEY environment variable (any long random string).")
    db.init()
    worker.start()
    log.info("whisper service %s ready - model %s, summary %s", VERSION, settings.whisper_model, settings.summary_backend)
    yield
    worker.stop()


app = FastAPI(title="Whisper meeting service", version=VERSION, lifespan=lifespan)


# ---------------------------------------------------------------- auth
def auth(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)) -> None:
    key = x_api_key or ""
    if authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    if not key or not secrets.compare_digest(key, settings.api_key):
        raise HTTPException(401, "missing or wrong API key")


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
        "results": {
            "markdown": f"{base}/result.md",
            "text": f"{base}/transcript.txt",
            "srt": f"{base}/transcript.srt",
            "json": f"{base}/transcript.json",
        } if done else None,
    }


def options(language: Optional[str], summarize_: bool, diarize: Optional[bool], title: Optional[str]) -> dict:
    o = {"summarize": summarize_, "diarize": settings.diarize if diarize is None else diarize}
    if language and language.lower() != "auto":
        o["language"] = language.lower()
    if title:
        o["title"] = title[:200]
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
    j = db.create(filename, str(dest), opts, rid, 1 if priority == "high" else 0)
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
        return job_out(new_job(tmp, filename, options(language, summarize_, diarize, title), rid, priority))
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
async def upload_chunk(rid: str, request: Request, x_offset: int = Header(0)):
    """Append the request body at byte X-Offset. Wrong offset -> 409 with the size the server has."""
    check_id(rid)
    if (j := existing_for(rid)):
        return {"recording_id": rid, "received": None, "job": job_out(j)}
    p = part_path(rid)
    have = p.stat().st_size if p.exists() else 0
    if x_offset == 0 and have:                      # client restarts from scratch
        p.unlink()
        have = 0
    if x_offset != have:
        return JSONResponse({"recording_id": rid, "received": have, "error": "offset mismatch"}, status_code=409)
    n = await save_body(request, p, "ab", have)
    return {"recording_id": rid, "received": n, "job": None}


@app.post("/api/uploads/{rid}/complete", dependencies=[Depends(auth)], status_code=201)
def upload_complete(
    rid: str,
    language: Optional[str] = Query(None),
    summarize_: bool = Query(True, alias="summarize"),
    diarize: Optional[bool] = Query(None),
    title: Optional[str] = Query(None),
    filename: Optional[str] = Query(None),
    priority: str = Query(""),
    x_total_size: Optional[int] = Header(None, description="optional: full file size, checked before queueing"),
):
    check_id(rid)
    if (j := existing_for(rid)):
        return JSONResponse(job_out(j), status_code=200)
    p = part_path(rid)
    if not p.exists():
        raise HTTPException(404, "nothing uploaded for this recording id")
    if x_total_size is not None and p.stat().st_size != x_total_size:
        return JSONResponse({"recording_id": rid, "received": p.stat().st_size, "error": "size mismatch"}, status_code=409)
    return job_out(new_job(p, safe_name(filename or rid), options(language, summarize_, diarize, title), rid, priority))


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
}


def _result(j: dict, name: str):
    if name not in RESULT_FILES:
        raise HTTPException(404, "unknown result file")
    if j["status"] != "done":
        raise HTTPException(409, f"job is {j['status']}")
    fname, ctype = RESULT_FILES[name]
    p = worker.result_dir(j["id"]) / fname
    if not p.exists():
        raise HTTPException(404, f"{name} not available")
    stem = Path(j["filename"]).stem
    dl = f"{stem}.md" if name == "result.md" else f"{stem}_{name}"
    return FileResponse(p, media_type=ctype, filename=dl, content_disposition_type="inline")


@app.get("/api/jobs/{jid}/{name}", dependencies=[Depends(auth)])
def job_result(jid: str, name: str):
    return _result(_job(jid), name)


@app.get("/api/recordings/{rid}/{name}", dependencies=[Depends(auth)])
def recording_result(rid: str, name: str):
    j = db.get_by_recording(check_id(rid))
    if not j:
        raise HTTPException(404, "no job for this recording id")
    return _result(j, name)


@app.post("/api/jobs/{jid}/summarize", dependencies=[Depends(auth)])
def resummarize(jid: str):
    """Write the summary again (e.g. after changing the LLM). Runs in the background."""
    j = _job(jid)
    if j["status"] not in ("done",):
        raise HTTPException(409, f"job is {j['status']}")
    if not summarize.enabled():
        raise HTTPException(400, "SUMMARY_BACKEND is none")
    opts = dict(j["options"], summarize=True, resummarize=True)
    db.update(jid, status="queued", stage="summary queued", progress=0, options=opts)
    worker.wake()
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
    return {"deleted": jid}


# ---------------------------------------------------------------- chat (Ask AI on the device)
@app.post("/api/ask", dependencies=[Depends(auth)], status_code=202)
async def ask(request: Request):
    """{"messages": [{"role": "user"|"assistant", "content": "..."}]} -> {"id", "status": "pending"}.
    The answer can take a while on a CPU, so poll GET /api/ask/{id}."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "body must be JSON")
    msgs = data.get("messages") if isinstance(data, dict) else None
    if not msgs and isinstance(data, dict) and data.get("prompt"):
        msgs = [{"role": "user", "content": str(data["prompt"])}]
    if not msgs or not isinstance(msgs, list):
        raise HTTPException(400, "send {'messages': [...]} or {'prompt': '...'}")
    clean = [{"role": m.get("role") if m.get("role") in ("user", "assistant") else "user", "content": str(m.get("content", ""))[:8000]}
             for m in msgs[-20:] if isinstance(m, dict)]
    if not summarize.enabled():
        raise HTTPException(400, "SUMMARY_BACKEND is none - no LLM configured")
    return chat.start(clean)


@app.get("/api/ask/{aid}", dependencies=[Depends(auth)])
def ask_result(aid: str):
    r = chat.get(aid)
    if not r:
        raise HTTPException(404, "unknown or expired")
    return r


# ---------------------------------------------------------------- misc
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "version": VERSION,
        "model": settings.whisper_model,
        "model_loaded": transcribe.model_loaded() or settings.transcriber == "fake",
        "summary_backend": settings.summary_backend,
        "summary_model": settings.ollama_model if settings.summary_backend == "ollama" else settings.openai_model,
        "diarize_default": settings.diarize,
        "jobs": db.counts(),
        "running": worker.current_job,
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return PlainTextResponse("", status_code=204)
