"""HTTP API + small web page. Run: uvicorn app.main:app --host 0.0.0.0 --port 8000"""
import importlib.util
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response

from . import agenda, chat, db, redact, render, search, summarize, transcribe, voices, worker
from .config import settings

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
VERSION = "1.2.0"
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
        "template": (j.get("options") or {}).get("template", "general"),
        "best": bool((j.get("options") or {}).get("best")),
        "speakers": (j.get("options") or {}).get("speakers"),
        "speaker_names": (j.get("options") or {}).get("speaker_names") or {},
        "speaker_auto": (j.get("options") or {}).get("speaker_auto") or [],
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
            redact_: Optional[bool] = None, attendees: Optional[str] = None) -> dict:
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
    return o


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
    if rid and (live := worker.live_state_path(rid)).exists():
        d = worker.result_dir(j["id"])
        d.mkdir(parents=True, exist_ok=True)
        shutil.move(str(live), d / "live.json")      # parts already transcribed while it was recording
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
        opts = options(language, summarize_, diarize, title, template, speakers, best, redact_, attendees)
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
    if worker.live_state_path(rid).exists():
        worker.wake()
    return {"recording_id": rid, "received": n, "job": None}


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
    opts = apply_extra(options(language, summarize_, diarize, title, template, speakers, best, redact_, attendees), extra)
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


def _result(j: dict, name: str):
    if name == "result.docx":
        if j["status"] != "done":
            raise HTTPException(409, f"job is {j['status']}")
        p = worker.result_dir(j["id"]) / "result.md"
        if not p.exists():
            raise HTTPException(404, "result not available")
        data = render.docx(p.read_text(encoding="utf-8"))
        stem = Path(j.get("title") or Path(j["filename"]).stem).name
        stem = re.sub(r"[^A-Za-z0-9 ._-]", "", stem)[:80] or "minutes"
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
    stem = Path(j["filename"]).stem
    dl = f"{stem}.md" if name == "result.md" else f"{stem}_{name}"
    return FileResponse(p, media_type=ctype, filename=dl, content_disposition_type="inline")


@app.get("/api/jobs/{jid}/{name}", dependencies=[Depends(auth)])
def job_result(jid: str, name: str):
    if name == "speakers":
        return get_speakers(jid)
    return _result(_job(jid), name)


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


@app.post("/api/ask", dependencies=[Depends(auth)], status_code=202)
async def ask(request: Request):
    """{"messages": [...]} -> {"id", "status": "pending"}; poll GET /api/ask/{id}.
    Optional: "recording": <recording or job id> (ask about one meeting), "scope": "meetings" (search all
    meetings), "task": "email" (draft a follow-up email for "recording"), "best": true."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "body must be JSON")
    if not isinstance(data, dict):
        raise HTTPException(400, "body must be a JSON object")
    msgs = data.get("messages")
    if not msgs and data.get("prompt"):
        msgs = [{"role": "user", "content": str(data["prompt"])}]
    if data.get("task") == "email":
        if not data.get("recording"):
            raise HTTPException(400, "task=email needs 'recording'")
        msgs = [{"role": "user", "content": EMAIL_TASK}]
    if not msgs or not isinstance(msgs, list):
        raise HTTPException(400, "send {'messages': [...]} or {'prompt': '...'}")
    clean = [{"role": m.get("role") if m.get("role") in ("user", "assistant") else "user", "content": str(m.get("content", ""))[:8000]}
             for m in msgs[-20:] if isinstance(m, dict)]
    if not summarize.enabled():
        raise HTTPException(400, "SUMMARY_BACKEND is none - no LLM configured")
    context, system = "", ""
    if data.get("recording"):
        context, system = meeting_context(_job_for(str(data["recording"]))), MEETING_SYSTEM
    elif data.get("scope") == "meetings":
        q = " ".join(m["content"] for m in clean if m["role"] == "user")[-1000:]
        context, system = search_context(q), MEETING_SYSTEM
    return chat.start(clean, context, bool(data.get("best")), system)


@app.get("/api/ask/{aid}", dependencies=[Depends(auth)])
def ask_result(aid: str):
    r = chat.get(aid)
    if not r:
        raise HTTPException(404, "unknown or expired")
    return r

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
    for label, name in names.items():
        label, name = str(label)[:40], re.sub(r"\s+", " ", str(name or "")).strip()[:60]
        if not re.match(r"^Speaker \d+$", label):
            raise HTTPException(400, f"unknown speaker label {label!r}")
        if name:
            cur[label] = name
            if data.get("remember", True) and (spk.get(label) or {}).get("embedding"):
                voices.remember(name, spk[label]["embedding"])
        else:
            cur.pop(label, None)
        auto.discard(label)
    opts["speaker_names"], opts["speaker_auto"] = cur, sorted(auto)
    db.update(jid, options=opts)
    if j["status"] == "done":
        worker.build_outputs(jid)
    return job_out(db.get(jid))


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
    labels = sorted(set(spk) | {s.get("speaker") for s in tr.get("segments", []) if s.get("speaker")},
                    key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 99)
    return {"speakers": [{"label": lab, "name": names.get(lab, ""), "auto": lab in auto,
                          "seconds": (spk.get(lab) or {}).get("seconds", 0), "has_voice": bool((spk.get(lab) or {}).get("embedding")),
                          "sample": sample.get(lab, "")} for lab in labels]}


@app.get("/api/voices", dependencies=[Depends(auth)])
def get_voices():
    return {"voices": voices.list_voices(), "threshold": settings.voice_match}


@app.delete("/api/voices/{name}", dependencies=[Depends(auth)])
def delete_voice(name: str):
    if not voices.forget(name):
        raise HTTPException(404, "unknown voice")
    return {"deleted": name}


@app.get("/api/glossary", dependencies=[Depends(auth)])
def get_glossary():
    p = settings.glossary_path
    return {"text": p.read_text(encoding="utf-8") if p.exists() else "", "terms": voices.glossary()}


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
    return {"terms": voices.set_glossary(raw)}


@app.get("/api/search", dependencies=[Depends(auth)])
def search_meetings(q: str = Query(..., min_length=2), limit: int = Query(20, le=50)):
    return {"hits": search.search(q, limit)}


@app.get("/api/templates")
def templates():
    return {"templates": [{"id": k, "label": v[0]} for k, v in summarize.TEMPLATES.items()],
            "best_available": bool(summarize.best_model())}


# ---------------------------------------------------------------- calendar (device Agenda app)
@app.get("/api/agenda", dependencies=[Depends(auth)])
def get_agenda(days: int = Query(7, ge=1, le=60)):
    if not agenda.urls():
        raise HTTPException(503, "no calendars configured - set CALENDAR_ICS_URLS")
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
        "diarize_installed": importlib.util.find_spec("pyannote") is not None,
        "hf_token_set": bool(settings.hf_token),
        "calendars": len(agenda.urls()),
        "best_model": summarize.best_model(),
        "redact": settings.redact,
        "voices": len(voices.list_voices()),
        "jobs": db.counts(),
        "running": worker.current_job,
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return PlainTextResponse("", status_code=204)
