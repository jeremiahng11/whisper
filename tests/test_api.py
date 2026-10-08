"""End-to-end API tests with a fake transcriber and a fake LLM (no models needed).
Run:  pytest -q
"""
import io
import os
import struct
import tempfile
import time
import wave

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="whisper-test-")
os.environ["API_KEY"] = "test-key"
os.environ["TRANSCRIBER"] = "fake"
os.environ["SUMMARY_BACKEND"] = "openai"
os.environ["PRELOAD_MODEL"] = "0"

import pytest
from fastapi.testclient import TestClient

from app import summarize
from app.config import settings
from app.main import app

H = {"Authorization": "Bearer test-key"}
CALLS = []
FAIL = {"on": False}


def fake_chat(messages):
    CALLS.append(messages[-1]["content"])
    if FAIL["on"]:
        raise RuntimeError("LLM is down")
    if "part" in messages[-1]["content"][:60]:
        return "- notes for a part"
    return "Title: Weekly sync on the Idea Saver\n\n### Summary\nThe team talked.\n\n### Action items\n- [ ] Speaker 1 - ship V1 (Friday)"


summarize.chat = fake_chat


def wav_bytes(seconds=12, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def wait(client, jid, timeout=20):
    t = time.time()
    while time.time() - t < timeout:
        j = client.get(f"/api/jobs/{jid}", headers=H).json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_health_and_auth(client):
    assert client.get("/api/health").json()["ok"]
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/jobs", headers={"X-API-Key": "test-key"}).status_code == 200
    assert "Whisper Minutes" in client.get("/").text


def test_raw_upload_full_flow(client):
    r = client.post("/api/jobs?language=en", content=wav_bytes(12), headers={**H, "X-Filename": "2026-10-08_1535.wav",
                                                                           "Content-Type": "audio/wav"})
    assert r.status_code == 201, r.text
    j = wait(client, r.json()["id"])
    assert j["status"] == "done", j
    assert j["title"] == "Weekly sync on the Idea Saver"
    assert j["duration"] == pytest.approx(12, abs=0.01)
    md = client.get(j["results"]["markdown"], headers=H)
    assert md.status_code == 200
    body = md.text
    assert body.startswith("# Weekly sync on the Idea Saver")
    assert "2026-10-08 15:35" in body                     # recording time from the file name
    assert "### Action items" in body and "## Transcript" in body
    assert "test sentence number 3" in body
    assert "Title:" not in body
    srt = client.get(j["results"]["srt"], headers=H).text
    assert "00:00:05,000 --> 00:00:10,000" in srt
    assert client.get(j["results"]["json"], headers=H).json()["language"] == "en"
    assert "2026-10-08_1535.md" in md.headers["content-disposition"]


def test_multipart_upload(client):
    r = client.post("/api/jobs?summarize=false", files={"file": ("memo.wav", wav_bytes(3), "audio/wav")}, headers=H)
    assert r.status_code == 201, r.text
    j = wait(client, r.json()["id"])
    assert j["status"] == "done" and j["filename"] == "memo.wav" and j["title"] == ""
    assert "### Summary" not in client.get(j["results"]["markdown"], headers=H).text


def test_same_recording_id_returns_same_job(client):
    h = {**H, "X-Recording-Id": "dev1-2026-10-08_1600"}
    a = client.post("/api/jobs", content=wav_bytes(2), headers=h)
    b = client.post("/api/jobs", content=wav_bytes(2), headers=h)
    assert a.status_code == 201 and b.status_code == 200
    assert a.json()["id"] == b.json()["id"]
    wait(client, a.json()["id"])
    assert client.get("/api/recordings/dev1-2026-10-08_1600", headers=H).json()["status"] == "done"
    assert client.get("/api/recordings/dev1-2026-10-08_1600/result.md", headers=H).status_code == 200


def test_resumable_upload(client):
    data = wav_bytes(20)
    rid = "dev1-2026-10-08_1700"
    half = len(data) // 2
    assert client.get(f"/api/uploads/{rid}", headers=H).json()["received"] == 0
    r = client.put(f"/api/uploads/{rid}", content=data[:half], headers={**H, "X-Offset": "0"})
    assert r.json()["received"] == half
    # connection dropped, device asks where to continue
    assert client.get(f"/api/uploads/{rid}", headers=H).json()["received"] == half
    bad = client.put(f"/api/uploads/{rid}", content=data[half:], headers={**H, "X-Offset": "5"})
    assert bad.status_code == 409 and bad.json()["received"] == half
    r = client.put(f"/api/uploads/{rid}", content=data[half:], headers={**H, "X-Offset": str(half)})
    assert r.json()["received"] == len(data)
    wrong = client.post(f"/api/uploads/{rid}/complete", headers={**H, "X-Total-Size": str(len(data) + 1)})
    assert wrong.status_code == 409
    r = client.post(f"/api/uploads/{rid}/complete?filename=2026-10-08_1700.wav&language=en",
                    headers={**H, "X-Total-Size": str(len(data))})
    assert r.status_code == 201, r.text
    j = wait(client, r.json()["id"])
    assert j["status"] == "done" and j["recording_id"] == rid and j["duration"] == pytest.approx(20, abs=0.01)
    # uploading again after it's done just returns the job
    again = client.put(f"/api/uploads/{rid}", content=data[:10], headers={**H, "X-Offset": "0"}).json()
    assert again["job"]["id"] == j["id"]
    assert client.get(f"/api/uploads/{rid}", headers=H).json()["job"]["status"] == "done"


def test_long_transcript_is_summarised_in_parts(client):
    CALLS.clear()
    old = settings.summary_chunk_chars
    settings.summary_chunk_chars = 2000
    try:
        r = client.post("/api/jobs", content=wav_bytes(600), headers=H)   # 10 min -> ~120 fake sentences
        j = wait(client, r.json()["id"])
    finally:
        settings.summary_chunk_chars = old
    assert j["status"] == "done"
    parts = [c for c in CALLS if c.startswith("This is part")]
    assert len(parts) >= 2
    assert "Notes from the whole transcript" in CALLS[-1]


def test_summary_failure_keeps_transcript_then_redo(client):
    FAIL["on"] = True
    try:
        j = wait(client, client.post("/api/jobs", content=wav_bytes(4), headers=H).json()["id"])
    finally:
        FAIL["on"] = False
    assert j["status"] == "done" and "LLM is down" in j["summary_error"]
    md = client.get(j["results"]["markdown"], headers=H).text
    assert "Summary not available" in md and "test sentence number 1" in md
    r = client.post(f"/api/jobs/{j['id']}/summarize", headers=H)
    assert r.status_code == 200
    j2 = wait(client, j["id"])
    time.sleep(0.2)
    j2 = wait(client, j["id"])
    assert j2["summary_error"] == "" and j2["title"] == "Weekly sync on the Idea Saver"
    assert "### Summary" in client.get(j2["results"]["markdown"], headers=H).text


def test_bad_audio_errors_and_reupload_replaces(client):
    h = {**H, "X-Recording-Id": "broken-1"}
    j = wait(client, client.post("/api/jobs", content=b"not audio at all", headers=h).json()["id"])
    assert j["status"] == "error" and j["error"]
    assert client.get(f"/api/jobs/{j['id']}/result.md", headers=H).status_code == 409
    r = client.post("/api/jobs", content=wav_bytes(2), headers=h)       # device retries with a good file
    assert r.status_code == 201 and r.json()["id"] != j["id"]
    assert wait(client, r.json()["id"])["status"] == "done"
    assert client.get(f"/api/jobs/{j['id']}", headers=H).status_code == 404


def test_validation_and_delete(client):
    assert client.post("/api/jobs", content=b"", headers=H).status_code == 400
    assert client.get("/api/uploads/bad id!", headers=H).status_code in (400, 404)
    assert client.put("/api/uploads/..%2Fetc", content=b"x", headers=H).status_code in (400, 404)
    j = wait(client, client.post("/api/jobs", content=wav_bytes(1), headers=H).json()["id"])
    assert client.get(f"/api/jobs/{j['id']}/nope.exe", headers=H).status_code == 404
    assert client.delete(f"/api/jobs/{j['id']}", headers=H).status_code == 200
    assert client.get(f"/api/jobs/{j['id']}", headers=H).status_code == 404
    assert not (settings.result_dir / j["id"]).exists()
