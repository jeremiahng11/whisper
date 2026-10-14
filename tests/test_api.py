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
os.environ["DIARIZER"] = "fake"
os.environ["AUTOTAG_NEW"] = "0"          # tested on their own below (they'd change other tests' meetings)
os.environ["AUTOTAG_SWEEP"] = "0"

import pytest
from fastapi.testclient import TestClient

from app import summarize
from app.config import settings
from app.main import app

H = {"Authorization": "Bearer test-key"}
CALLS = []
FAIL = {"on": False}


BEST = []
SYSTEMS = []


def fake_chat(messages, best=False):
    CALLS.append(messages[-1]["content"])
    BEST.append(best)
    SYSTEMS.append(messages[0]["content"] if messages[0]["role"] == "system" else "")
    if FAIL["on"]:
        raise RuntimeError("LLM is down")
    if "as JSON" in messages[-1]["content"]:
        return '```json\n[{"task": "ship V1", "owner": "Speaker 1", "due": "2026-10-09"}]\n```'
    if "pinyin" in messages[-1]["content"].lower() or messages[0]["content"].startswith("You answer questions"):
        return "From the meetings: the rollback plan is owned by San [00:13]."
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
    raise AssertionError(f"job did not finish: {j}")


def test_health_and_auth(client):
    assert client.get("/api/health").json()["ok"]
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/jobs", headers={"X-API-Key": "test-key"}).status_code == 200
    assert "Oracle Minutes" in client.get("/").text


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
    assert "_8-Oct-2026.md" in md.headers["content-disposition"]   # title + date


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
    assert "Notes from the whole transcript" in CALLS[-2] and "as JSON" in CALLS[-1]


def test_summary_failure_keeps_transcript_then_redo(client):
    FAIL["on"] = True
    try:
        j = wait(client, client.post("/api/jobs", content=wav_bytes(20), headers=H).json()["id"])
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


def test_ask_ai(client):
    r = client.post("/api/ask", json={"messages": [{"role": "user", "content": "Name ideas?"}]}, headers=H)
    assert r.status_code == 202 and r.json()["status"] == "pending"
    aid = r.json()["id"]
    for _ in range(400):
        a = client.get(f"/api/ask/{aid}", headers=H).json()
        if a["status"] != "pending":
            break
        time.sleep(0.05)
    assert a["status"] == "done" and a["answer"]
    assert client.get("/api/ask/nope", headers=H).status_code == 404
    assert client.post("/api/ask", json={"x": 1}, headers=H).status_code == 400


def test_priority_jumps_queue(client):
    from app import db
    import app.worker as w
    w.stop(); time.sleep(0.3)                    # pause the worker so jobs stay queued
    try:
        a = client.post("/api/jobs", content=wav_bytes(1), headers=H).json()
        b = client.post("/api/jobs?priority=high", content=wav_bytes(1), headers=H).json()
        assert db.next_queued()["id"] == b["id"]
        assert client.get(f"/api/jobs/{b['id']}", headers=H).json()["queue_position"] == 1
        assert client.get(f"/api/jobs/{a['id']}", headers=H).json()["queue_position"] == 2
    finally:
        w.start()
    wait(client, a["id"]); wait(client, b["id"])


def test_meeting_notes_feed_summary_and_minutes(client):
    rid = "dev1-2026-10-08_1535"
    data = wav_bytes(6)
    client.put(f"/api/uploads/{rid}", content=data, headers={**H, "X-Offset": "0"})
    notes = "[0:02] Agreed to move go-live to Nov 3\n[0:13] San owns the rollback plan"
    r = client.post(f"/api/uploads/{rid}/complete?filename=2026-10-08_1535.wav", json={"notes": notes},
                    headers={**H, "X-Total-Size": str(len(data))})
    assert r.status_code == 201
    j = wait(client, r.json()["id"])
    assert j["status"] == "done"
    assert any("San owns the rollback plan" in c for c in CALLS)
    md = client.get(f"/api/recordings/{rid}/result.md", headers=H).text
    assert "## My notes" in md and "go-live to Nov 3" in md


ICS = b"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//test//EN
BEGIN:VEVENT
UID:a1
DTSTART:%(s1)s
DTEND:%(e1)s
SUMMARY:Coolify migration review
LOCATION:Zoom
ORGANIZER;CN=Jeremiah Ng:mailto:jeremiah@example.com
ATTENDEE;CN="San Tan";ROLE=REQ-PARTICIPANT:mailto:san@example.com
ATTENDEE;CUTYPE=ROOM;CN=Room 3:mailto:room3@example.com
ATTENDEE:mailto:komei.foong@example.com
END:VEVENT
BEGIN:VEVENT
UID:a2
DTSTART;VALUE=DATE:%(d2)s
SUMMARY:Public holiday
END:VEVENT
BEGIN:VEVENT
UID:a3
DTSTART:%(s3)s
DTEND:%(e3)s
RRULE:FREQ=DAILY;COUNT=3
SUMMARY:Standup
END:VEVENT
BEGIN:VEVENT
UID:a4
DTSTART:%(s1)s
DTEND:%(e1)s
STATUS:CANCELLED
SUMMARY:Cancelled thing
END:VEVENT
END:VCALENDAR
"""


def test_agenda(client, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from app import agenda
    assert client.get("/api/agenda", headers=H).status_code == 503       # nothing configured
    now = datetime.now(timezone.utc).replace(microsecond=0)
    f = lambda d: d.strftime("%Y%m%dT%H%M%SZ")
    body = ICS % {b"s1": f(now + timedelta(hours=1)).encode(), b"e1": f(now + timedelta(hours=2)).encode(),
                  b"d2": (now + timedelta(days=2)).strftime("%Y%m%d").encode(),
                  b"s3": f(now + timedelta(minutes=30)).encode(), b"e3": f(now + timedelta(minutes=45)).encode()}
    monkeypatch.setattr(settings, "calendar_ics_urls", "https://cal.example/a.ics, webcal://cal.example/b.ics")
    monkeypatch.setattr(agenda, "_fetch", lambda u: body)
    ev = client.get("/api/agenda?days=7", headers=H).json()["events"]
    titles = [e["title"] for e in ev]
    assert "Cancelled thing" not in titles
    assert titles.count("Standup") == 3 and titles.count("Coolify migration review") == 1   # deduped across 2 calendars
    m = next(e for e in ev if e["title"] == "Coolify migration review")
    assert m["location"] == "Zoom" and m["end"] - m["start"] == 3600 and not m["all_day"]
    assert m["attendees"] == ["Jeremiah Ng", "San Tan", "Komei Foong"]
    assert next(e for e in ev if e["title"] == "Public holiday")["all_day"]


def test_file_sync(client):
    from app.main import fnv1a
    assert fnv1a(b"") == "811c9dc5" and fnv1a(b"a") == "e40c292c"         # same as the device
    q = "device=dev1"
    r = client.put(f"/api/files/notes/Inbox.md?{q}&base_rev=0", content=b"- idea one\n", headers=H).json()
    assert r["rev"] == 1 and r["hash"] == fnv1a(b"- idea one\n")
    # same content again: unchanged, no new rev
    assert client.put(f"/api/files/notes/Inbox.md?{q}&base_rev=0", content=b"- idea one\n", headers=H).json()["rev"] == 1
    # edit from the web with the right base
    assert client.put(f"/api/files/notes/Inbox.md?{q}&base_rev=1&by=web", content=b"- idea one!\n", headers=H).json()["rev"] == 2
    # device still thinks rev 1 -> conflict
    c = client.put(f"/api/files/notes/Inbox.md?{q}&base_rev=1", content=b"- idea two\n", headers=H)
    assert c.status_code == 409 and c.json()["rev"] == 2
    g = client.get(f"/api/files/notes/Inbox.md?{q}", headers=H)
    assert g.text == "- idea one!\n" and g.headers["X-Rev"] == "2"
    lst = client.get(f"/api/files?{q}", headers=H).json()["files"]
    assert lst[0]["path"] == "notes/Inbox.md" and lst[0]["updated_by"] == "web"
    assert "dev1" in client.get("/api/files", headers=H).json()["devices"]
    # delete needs the current rev
    assert client.delete(f"/api/files/notes/Inbox.md?{q}&base_rev=1", headers=H).status_code == 409
    assert client.delete(f"/api/files/notes/Inbox.md?{q}&base_rev=2", headers=H).json()["deleted"]
    assert client.get(f"/api/files/notes/Inbox.md?{q}", headers=H).status_code == 404
    # re-create after delete with base 0
    assert client.put(f"/api/files/notes/Inbox.md?{q}&base_rev=0", content=b"new\n", headers=H).json()["rev"] == 4
    for bad in ["../x.md", "notes/x.exe", "notes//x.md"]:
        assert client.put(f"/api/files/{bad}?{q}", content=b"x", headers=H).status_code in (400, 404)


def upload(client, rid, seconds=30, query="", body=None):
    data = wav_bytes(seconds)
    client.put(f"/api/uploads/{rid}", content=data, headers={**H, "X-Offset": "0"})
    r = client.post(f"/api/uploads/{rid}/complete?filename=2026-10-09_1000.wav&{query}", json=body or {},
                    headers={**H, "X-Total-Size": str(len(data))})
    assert r.status_code == 201, r.text
    return wait(client, r.json()["id"])


def test_speaker_names_and_voice_memory(client):
    j = upload(client, "dev1-spk-a", 40, "diarize=true&speakers=2")
    assert j["status"] == "done" and j["speakers"] == 2
    sp = client.get(f"/api/jobs/{j['id']}/speakers", headers=H).json()["speakers"]
    assert [s["label"] for s in sp] == ["Speaker 1", "Speaker 2"] and all(s["has_voice"] for s in sp)
    r = client.post(f"/api/jobs/{j['id']}/speakers", json={"names": {"Speaker 1": "San"}, "remember": True}, headers=H)
    assert r.json()["speaker_names"] == {"Speaker 1": "San"}
    md = client.get(f"/api/jobs/{j['id']}/result.md", headers=H).text
    assert "**San:**" in md and "**Speaker 2:**" in md and "San - ship V1" in md       # summary renamed too
    acts = client.get(f"/api/jobs/{j['id']}/actions.json", headers=H).json()
    assert acts == [{"task": "ship V1", "owner": "San", "due": "2026-10-09"}]
    assert any(v["name"] == "San" for v in client.get("/api/voices", headers=H).json()["voices"])
    # next recording: San's voice is recognised automatically
    j2 = upload(client, "dev1-spk-b", 40, "diarize=true&speakers=2")
    assert j2["speaker_names"] == {"Speaker 1": "San"} and j2["speaker_auto"] == ["Speaker 1"]
    assert any("San:" in c for c in CALLS[-3:])                                         # the LLM saw the name
    assert client.delete("/api/voices/San", headers=H).status_code == 200
    assert client.post(f"/api/jobs/{j['id']}/speakers", json={"names": {"bogus": "x"}}, headers=H).status_code == 400


def test_templates_best_glossary_attendees_marks(client):
    assert client.put("/api/glossary", content="Aleta Planet\nSkenPay\n# comment", headers=H).json()["terms"] == ["Aleta Planet", "SkenPay"]
    t = client.get("/api/templates").json()
    assert {x["id"] for x in t["templates"]} >= {"general", "standup", "client", "one_on_one", "interview", "board"}
    assert client.post("/api/uploads/x/complete?template=nope", headers=H).status_code in (400, 404)
    BEST.clear()
    notes = "[0:12] !! budget for Q4\n[0:20] normal note"
    j = upload(client, "dev1-board", 30, "template=board&best=true", {"notes": notes, "attendees": ["San Tan", "Komei"]})
    assert j["template"] == "board" and j["best"] is True and True in BEST
    final = [c for c in CALLS if "RESOLVED THAT" in c][-1]
    assert "budget for Q4" in final and "test sentence number 3" in final          # flagged moment with context
    assert any("SkenPay" in s and "Komei" in s for s in SYSTEMS[-3:])
    md = client.get(f"/api/jobs/{j['id']}/result.md", headers=H).text
    assert "## Flagged moments" in md and "**budget for Q4**" in md
    # redo the minutes as a client meeting
    r = client.post(f"/api/jobs/{j['id']}/summarize", json={"template": "client", "best": False}, headers=H)
    assert r.json()["template"] == "client"
    wait(client, j["id"])
    assert "What the client needs" in [c for c in CALLS if "What the client needs" in c][-1]
    client.put("/api/glossary", content="", headers=H)


def test_redaction():
    from app.redact import redact
    assert redact("my NRIC is S1234567D ok") == "my NRIC is [NRIC] ok"
    assert redact("FIN g 123 4567 x") == "FIN [NRIC]"
    assert redact("card 4111 1111 1111 1111 expires") == "card [card number] expires"
    assert redact("order 1234 5678 9012 3456") == "order 1234 5678 9012 3456"           # fails Luhn: kept
    assert redact("account number is 123-456789-001 thanks") == "account number is [account number] thanks"
    assert redact("meet at 3pm on 2026-10-09, call 9123 4567") == "meet at 3pm on 2026-10-09, call 9123 4567"


def test_ask_meeting_search_email_docx(client):
    j = upload(client, "dev1-ask", 20)

    def ask(body):
        r = client.post("/api/ask", json=body, headers=H)
        assert r.status_code == 202, r.text
        for _ in range(400):
            a = client.get(f"/api/ask/{r.json()['id']}", headers=H).json()
            if a["status"] != "pending":
                return a
            time.sleep(0.05)

    a = ask({"recording": "dev1-ask", "messages": [{"role": "user", "content": "who owns rollback?"}]})
    assert a["status"] == "done" and "Meeting minutes and transcript" in SYSTEMS[-1]
    a = ask({"scope": "meetings", "messages": [{"role": "user", "content": "what was test sentence number 2?"}]})
    assert a["status"] == "done" and "test sentence number 2" in SYSTEMS[-1]
    ask({"recording": j["id"], "task": "email"})
    assert CALLS[-1].startswith("Draft a short follow-up email")
    hits = client.get("/api/search?q=sentence number", headers=H).json()["hits"]
    assert hits and hits[0]["title"]
    d = client.get(f"/api/jobs/{j['id']}/result.docx", headers=H)
    assert d.status_code == 200 and d.content[:2] == b"PK"


def test_live_upload_transcribes_while_recording(client, monkeypatch):
    from app import worker
    monkeypatch.setattr(settings, "live_min_new_s", 5)
    rid = "dev1-live"
    data = bytearray(wav_bytes(60))
    data[4:8] = b"\0\0\0\0"; data[40:44] = b"\0\0\0\0"             # header of a WAV still being recorded
    first = 44 + 40 * 16000 * 2
    r = client.put(f"/api/uploads/{rid}", content=bytes(data[:first]), headers={**H, "X-Offset": "0", "X-Live": "1"})
    assert r.status_code == 200
    st = worker.live_state_path(rid)
    for _ in range(400):
        if st.exists() and '"until": 0,' not in st.read_text() and '"until": 0}' not in st.read_text():
            break
        time.sleep(0.05)
    import json as _j
    assert _j.loads(st.read_text())["until"] == 15
    client.put(f"/api/uploads/{rid}", content=bytes(data[first:]), headers={**H, "X-Offset": str(first), "X-Live": "1"})
    r = client.post(f"/api/uploads/{rid}/complete?filename=2026-10-09_1100.wav", headers={**H, "X-Total-Size": str(len(data))})
    j = wait(client, r.json()["id"])
    assert j["status"] == "done" and abs(j["duration"] - 60) < 0.1
    segs = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"]
    nums = [int(s["text"].split()[-1].rstrip(".")) for s in segs]
    assert nums == list(range(1, 13))                                  # no gaps, no doubles
    assert not st.exists() and (worker.result_dir(j["id"]) / "live.json").exists()


# ---------------------------------------------------------------- round Idea Saver: voice
STEP = [7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107,
        118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
        1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894,
        6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767]
IDXT = [-1, -1, -1, -1, 2, 4, 6, 8] * 2


def adpcm_wav(samples, header_sizes=True):
    """Same IMA ADPCM WAV the round device writes (256-byte blocks, 505 samples each)."""
    data, pred, idx = bytearray(), 0, 0
    for i in range(0, len(samples) - 504, 505):
        blk = samples[i:i + 505]
        pred = int(blk[0])
        data += struct.pack("<hBB", pred, idx, 0)
        nib = []
        for s in blk[1:]:
            step, diff, code = STEP[idx], int(s) - pred, 0
            if diff < 0:
                code, diff = 8, -diff
            d = step >> 3
            if diff >= step: code |= 4; diff -= step; d += step
            step >>= 1
            if diff >= step: code |= 2; diff -= step; d += step
            step >>= 1
            if diff >= step: code |= 1; d += step
            pred = max(-32768, min(32767, pred - d if code & 8 else pred + d))
            idx = max(0, min(88, idx + IDXT[code]))
            nib.append(code)
        for j in range(0, len(nib), 2):
            data.append(nib[j] | (nib[j + 1] << 4))
    fmt = struct.pack("<HHIIHHHH", 0x11, 1, 16000, 8110, 256, 4, 2, 505)
    size = len(data) if header_sizes else 0
    return b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + size if header_sizes else 0) + b"WAVE" + \
        b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", size) + bytes(data)


def test_adpcm_recordings_decode(tmp_path):
    import numpy as np
    from app.transcribe import load_audio
    t = np.arange(16000 * 3) / 16000
    x = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    for full in (True, False):                                  # finished file, and one still being recorded
        p = tmp_path / f"a{full}.wav"
        p.write_bytes(adpcm_wav(x, full))
        y = load_audio(str(p)) * 32768
        n = len(y)
        assert abs(n - 16000 * 3) < 600
        snr = 10 * np.log10(np.mean(x[:n].astype(float) ** 2) / np.mean((y - x[:n]) ** 2))
        assert snr > 25


def test_adpcm_upload_full_flow(client):
    import numpy as np
    x = np.zeros(16000 * 12, dtype=np.int16)
    data = adpcm_wav(x)
    r = client.post("/api/jobs?summarize=false", content=data, headers={**H, "X-Filename": "memo_2026-10-11_093000.wav"})
    j = wait(client, r.json()["id"])
    assert j["status"] == "done" and abs(j["duration"] - 12) < 0.2


def test_stt(client):
    r = client.post("/api/stt", content=wav_bytes(7), headers=H)
    assert r.status_code == 200
    assert r.json()["text"].startswith("This is test sentence number 1.") and abs(r.json()["duration"] - 7) < 0.1
    assert client.post("/api/stt", content=b"", headers=H).status_code == 400
    r = client.post("/api/stt?languages=zh,en", content=wav_bytes(3), headers=H)     # pick from a list
    assert r.status_code == 200 and r.json()["language"] == "zh"


def test_intent_rules():
    from datetime import datetime, timedelta, timezone
    from app.speech import intent
    tz = timezone(timedelta(hours=8))
    now = int(datetime(2026, 10, 11, 13, 5, tzinfo=tz).timestamp())          # Sunday 1:05 pm
    at = lambda h, m, d=11: int(datetime(2026, 10, d, h, m, tzinfo=tz).timestamp())
    assert intent("Set a timer for 10 minutes", now)["seconds"] == 600
    assert intent("timer one hour and 30 minutes", now)["seconds"] == 5400
    r = intent("Remind me at 3 to call San", now)
    assert r["type"] == "reminder" and r["at"] == at(15, 0) and r["text"].lower() == "call san"
    r = intent("remind me tomorrow at 9:30 am to send the rollback plan", now)
    assert r["at"] == at(9, 30, 12) and "rollback" in r["text"]
    assert intent("remind me in 20 minutes to check the oven", now)["at"] == now + 1200
    r = intent("Add buy RTX riser to my tasks", now)
    assert r == {"type": "task", "text": "buy RTX riser", "source": "rules"}
    assert intent("take a note the Wi-Fi password is on the fridge", now)["type"] == "note"
    assert intent("start a meeting called Q4 planning", now) == {"type": "meeting", "title": "Q4 planning", "source": "rules"}
    r = intent("how do you say good morning in Japanese", now)
    assert r["type"] == "translate" and r["lang"] == "Japanese" and r["text"] == "good morning"


def test_intent_llm_fallback(client):
    r = client.post("/api/intent", json={"text": "what's the capital of France?", "now": 1791700000, "tz": 480}, headers=H)
    assert r.status_code == 200 and r.json()["type"] == "question"


def test_tts_not_installed(client):
    assert client.post("/api/tts", json={"text": "hi"}, headers=H).status_code == 503
    from app.speech import to_wav16k
    import numpy as np
    w = to_wav16k(np.zeros(22050, dtype=np.int16), 22050)
    assert w[:4] == b"RIFF" and abs(len(w) - 44 - 32000) < 4


def test_llm_error_explained(monkeypatch):
    from app import summarize
    monkeypatch.setattr(summarize.settings, "summary_backend", "ollama")
    msg = summarize.explain_llm_error(Exception("[Errno -2] Name or service not known"))
    assert "OLLAMA_URL" in msg and "host.docker.internal" in msg


def test_strip_thinking():
    from app.summarize import strip_thinking
    assert strip_thinking("<think>hmm</think>\n\nHello") == "Hello"
    assert strip_thinking("We are given a transcript...\nlots of reasoning\n</think>\n\nTitle: X") == "Title: X"
    assert strip_thinking("plain answer") == "plain answer"


def test_short_recording_skips_minutes():
    from app.summarize import summarize
    title, body = summarize("[00:00] Testing one, this is one. Testing two.", "en")
    assert "too little speech" in body and title.startswith("Testing")


def test_region_sg_hints(monkeypatch):
    from app import voices
    monkeypatch.setattr(voices.settings, "region", "sg")
    assert "Shenzhen" in voices.whisper_prompt() and "Shenzhen" in voices.region_rule()
    monkeypatch.setattr(voices.settings, "region", "")
    assert voices.region_rule() == ""


# ---------------------------------------------------------------- listen to a line and say who it was
def test_clip_and_tag_a_line(client):
    j = upload(client, "dev1-tag", 40, "diarize=true&speakers=2")
    assert j["status"] == "done"
    r = client.get(f"/api/jobs/{j['id']}/clip?start=10&end=14&enhance=false", headers=H)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert abs((len(r.content) - 44) / 32000 - 4) < 0.05                  # 4 s of 16 kHz PCM16
    r = client.get(f"/api/jobs/{j['id']}/clip?start=10&end=14", headers=H)   # clearer: + 0.3 s each side
    assert r.status_code == 200 and abs((len(r.content) - 44) / 32000 - 4.6) < 0.05
    segs = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"]
    i = next(k for k, s in enumerate(segs) if s.get("speaker") == "Speaker 1")
    # a new name -> a new speaker in this meeting, and the voice is learned
    r = client.post(f"/api/jobs/{j['id']}/segments/{i}", json={"name": "Priya", "learn": True}, headers=H)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["name"] == "Priya" and out["speaker"] not in ("Speaker 1", "Speaker 2") and out["learned"]
    segs2 = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"]
    assert segs2[i]["speaker"] == out["speaker"]
    assert "**Priya:**" in client.get(f"/api/jobs/{j['id']}/result.md", headers=H).text
    assert any(v["name"] == "Priya" for v in client.get("/api/voices", headers=H).json()["voices"])
    # the same name again maps to that speaker, not another new one
    k = next(n for n, s in enumerate(segs) if n != i and s.get("speaker") == "Speaker 2")
    assert client.post(f"/api/jobs/{j['id']}/segments/{k}", json={"name": "priya"}, headers=H).json()["speaker"] == out["speaker"]
    assert client.post(f"/api/jobs/{j['id']}/segments/9999", json={"name": "x"}, headers=H).status_code == 404
    assert client.post(f"/api/jobs/{j['id']}/segments/{i}", json={}, headers=H).status_code == 400


def test_storage_kind_detects_temporary_volume(client):
    from app import main as m
    d = os.path.realpath(str(settings.data_dir))
    root = "1 0 0:1 / / rw - overlay overlay rw\n"
    anon = root + f"2 1 8:1 /var/lib/docker/volumes/{'a3' * 32}/_data {d} rw - ext4 /dev/sda1 rw\n"
    named = root + f"2 1 8:1 /var/lib/docker/volumes/hg9f-whisper-data/_data {d} rw - ext4 /dev/sda1 rw\n"
    bind = root + f"2 1 8:1 /srv/whisper {d} rw - ext4 /dev/sda1 rw\n"
    assert m.storage_kind(root) == "temporary"
    assert m.storage_kind(anon) == "temporary"
    assert m.storage_kind(named) == "persistent"
    assert m.storage_kind(bind) == "persistent"
    assert client.get("/api/health").json()["storage"] in ("temporary", "persistent", "unknown")


def test_tts_adpcm_is_smaller_and_decodes():
    import numpy as np
    from app import speech, transcribe
    sr = 16000
    x = np.arange(sr * 3)
    pcm = (np.sin(2 * np.pi * 300 * x / sr) * 8000).astype(np.float32)
    wav = speech.to_wav16k(pcm, sr)
    ad = speech.wav_to_ima_adpcm(wav)
    assert len(ad) < len(wav) / 3.5 and ad[20:22] == b"\x11\x00"            # IMA ADPCM format tag
    from pathlib import Path
    p = Path(tempfile.mkdtemp()) / "t.wav"
    p.write_bytes(ad)
    dec = transcribe.load_audio(str(p))[:len(pcm)] * 32768
    err = np.sqrt(np.mean((pcm - dec) ** 2)) / np.sqrt(np.mean(pcm ** 2))
    assert len(dec) == len(pcm) and err < 0.05


def test_microsoft_graph_calendar(client, monkeypatch):
    from app import agenda
    import httpx as _h
    monkeypatch.setattr(settings, "ms_tenant_id", "t"); monkeypatch.setattr(settings, "ms_client_id", "c")
    monkeypatch.setattr(settings, "ms_client_secret", "s"); monkeypatch.setattr(settings, "ms_calendar_user", "me@aleta.test")
    monkeypatch.setattr(settings, "calendar_ics_urls", "")
    agenda._cache.clear(); agenda._token = (0.0, "")
    calls = []

    class R:
        def __init__(self, code, j): self.status_code, self._j, self.text = code, j, ""
        def json(self): return self._j

    def post(url, **kw):
        calls.append(("POST", url)); assert kw["data"]["grant_type"] == "client_credentials"
        return R(200, {"access_token": "tok", "expires_in": 3600})

    def get(url, params=None, headers=None, **kw):
        calls.append(("GET", url)); assert headers["Authorization"] == "Bearer tok" and "me@aleta.test" in url
        return R(200, {"value": [
            {"subject": "Shenzhen supplier call", "start": {"dateTime": "2030-01-07T02:00:00.0000000"},
             "end": {"dateTime": "2030-01-07T03:00:00.0000000"}, "isAllDay": False, "location": {"displayName": "Microsoft Teams Meeting"},
             "organizer": {"emailAddress": {"name": "Jeremiah Ng"}},
             "attendees": [{"type": "required", "emailAddress": {"name": "Alan Lim"}}, {"type": "resource", "emailAddress": {"name": "Room 3"}}]},
            {"subject": "Cancelled one", "isCancelled": True, "start": {"dateTime": "2030-01-07T05:00:00"}, "end": {"dateTime": "2030-01-07T06:00:00"}},
        ]})
    monkeypatch.setattr(_h, "post", post); monkeypatch.setattr(_h, "get", get)
    assert client.get("/api/health").json()["calendars"] == 1
    ev = client.get("/api/agenda?days=7", headers=H).json()["events"]
    assert len(ev) == 1 and ev[0]["title"] == "Shenzhen supplier call" and ev[0]["attendees"] == ["Jeremiah Ng", "Alan Lim"]
    assert ev[0]["start"] == 1893981600 and ev[0]["location"] == "Microsoft Teams Meeting"      # 2030-01-07 02:00 UTC
    client.get("/api/agenda?days=7", headers=H)
    assert [c[0] for c in calls] == ["POST", "GET"]                                            # token + events cached


def test_level_audio_lifts_quiet_voices_not_silence():
    import numpy as np
    from app import transcribe
    rng = np.random.default_rng(0)
    sr = 16000
    sil = rng.normal(0, 0.001, sr * 4).astype(np.float32)
    near = rng.normal(0, 0.08, sr * 4).astype(np.float32)
    far = rng.normal(0, 0.01, sr * 4).astype(np.float32)          # 18 dB quieter
    a = np.concatenate([sil, near, sil, far, sil])
    b = transcribe.level_audio(a)
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    assert rms(b[sr * 13:sr * 15]) > 4 * rms(a[sr * 13:sr * 15])     # far voice lifted > 12 dB
    assert abs(rms(b[sr * 5:sr * 7]) / rms(a[sr * 5:sr * 7]) - 1) < 0.2   # near voice ~unchanged
    assert rms(b[sr:sr * 3]) < 2 * rms(a[sr:sr * 3])                 # silence not pumped
    assert np.max(np.abs(b)) <= 1.0


def test_tagging_one_line_tags_the_same_voice_everywhere(client):
    j = upload(client, "dev1-autotag", 40, "diarize=true&speakers=2")
    segs = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"]
    first = next(i for i, s in enumerate(segs) if s["start"] < 10)                  # voice A (0-10 s, 20-30 s)
    r = client.post(f"/api/jobs/{j['id']}/segments/{first}", json={"name": "Wei Ling"}, headers=H).json()
    assert r["autotag"] == "started"
    for _ in range(400):
        st = client.get(f"/api/jobs/{j['id']}/autotag", headers=H).json()
        if st["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert st["state"] == "done", st
    segs2 = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"]
    lab = r["speaker"]
    for i, s in enumerate(segs2):
        voice_a = int(s["start"] // 10) % 2 == 0
        assert (s["speaker"] == lab) == voice_a, (i, s["start"], s["speaker"])         # A everywhere, B untouched
    assert first in st["manual"] and len(st["auto"]) >= 1 and first not in st["auto"]
    md = client.get(f"/api/jobs/{j['id']}/result.md", headers=H).text
    assert md.count("**Wei Ling:**") >= 2
    # a line you tag yourself is never overridden by auto-tagging
    b = next(i for i, s in enumerate(segs2) if int(s["start"] // 10) % 2 == 1)
    client.post(f"/api/jobs/{j['id']}/segments/{b}", json={"name": "Wei Ling", "spread": False}, headers=H)
    client.post(f"/api/jobs/{j['id']}/segments/{first}", json={"name": "Wei Ling"}, headers=H)
    for _ in range(400):
        if client.get(f"/api/jobs/{j['id']}/autotag", headers=H).json()["state"] == "done":
            break
        time.sleep(0.05)
    assert client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"][b]["speaker"] == lab


def test_split_line_at_speaker_change_and_mark_overlap(client):
    from app import transcribe as t
    words = [[0.0, 0.4, " Can"], [0.4, 0.8, " we"], [0.8, 1.2, " ship"], [1.2, 1.6, " Friday?"],
             [1.8, 2.2, " Yes,"], [2.2, 2.6, " if"], [2.6, 3.0, " QA"], [3.0, 3.6, " passes."]]
    seg = t.Segment(0.0, 3.6, "Can we ship Friday? Yes, if QA passes.", words=words)
    turns = [(0.0, 1.7, "A"), (1.7, 4.0, "B")]
    parts = t.split_by_speaker(seg, turns)
    assert [p.text for p in parts] == ["Can we ship Friday?", "Yes, if QA passes."]
    tr = t.Transcript("en", 1, 4, "fake", [seg])
    import numpy as np
    orig = t._diarize
    try:
        t._diarize = lambda a, n: ([(0.0, 1.7, "A"), (1.7, 4.0, "B"), (5.0, 9.0, "A"), (5.5, 8.5, "B")], {})
        tr.segments = [seg, t.Segment(5.0, 9.0, "talking over each other", words=[])]
        t.assign_speakers(np.zeros(16000 * 10, np.float32), tr, 2)
    finally:
        t._diarize = orig
    assert [s.speaker for s in tr.segments] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    assert tr.segments[2].also == ["Speaker 2"] and not tr.segments[0].also
    from app import render
    named = render.apply_names({"segments": [s.__dict__ for s in tr.segments]}, {"Speaker 1": "Alan", "Speaker 2": "Wei Ling"})
    assert named["segments"][2]["speaker"] == "Alan + Wei Ling"
    # API: add / clear a second person on a line
    j = upload(client, "dev1-also", 20, "diarize=true&speakers=2")
    r = client.post(f"/api/jobs/{j['id']}/segments/1", json={"also": "Rahul", "spread": False}, headers=H).json()
    assert r["also_names"] == ["Rahul"] and r["learned"] is False
    md = client.get(f"/api/jobs/{j['id']}/result.md", headers=H).text
    assert "+ Rahul:**" in md
    r = client.post(f"/api/jobs/{j['id']}/segments/1", json={"also": ""}, headers=H).json()
    assert r["also"] == []


def test_backup_and_restore_names_terms_voices(client):
    from app import voices
    client.put("/api/glossary", content="Aleta Planet\nShenzhen", headers=H)
    voices.remember("Backup Person", [0.1, 0.2, 0.3, 0.4])
    b = client.get("/api/backup", headers=H).json()
    assert b["kind"] == "whisper-minutes-backup" and "Shenzhen" in b["glossary"]
    assert any(v["name"] == "Backup Person" for v in b["voices"])
    # wipe (like a redeploy without persistent storage), then restore
    voices.forget("Backup Person")
    client.put("/api/glossary", content="Coolify", headers=H)
    r = client.post("/api/backup", json=b, headers=H).json()
    assert r["voices"] >= 1 and r["terms_added"] == 2
    terms = client.get("/api/glossary", headers=H).json()["terms"]
    assert {"Coolify", "Aleta Planet", "Shenzhen"} <= set(terms)
    assert any(v["name"] == "Backup Person" for v in client.get("/api/voices", headers=H).json()["voices"])
    assert client.post("/api/backup", json={"kind": "x"}, headers=H).status_code == 400


def test_tagging_a_line_names_its_whole_speaker_group_at_once(client):
    j = upload(client, "dev1-group", 40, "diarize=true&speakers=2")
    segs = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"]
    s2 = [i for i, s in enumerate(segs) if s["speaker"] == "Speaker 2"]
    r = client.post(f"/api/jobs/{j['id']}/segments/{s2[0]}", json={"name": "Ahmad"}, headers=H).json()
    assert r["moved"] == len(s2) - 1                                     # straight away, no waiting
    segs2 = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"]
    assert all(segs2[i]["speaker"] == r["speaker"] for i in s2)
    assert all(segs2[i]["speaker"] == "Speaker 1" for i, s in enumerate(segs) if s["speaker"] == "Speaker 1")
    labels = [s["label"] for s in client.get(f"/api/jobs/{j['id']}/speakers", headers=H).json()["speakers"]]
    assert "Speaker 2" not in labels and r["speaker"] in labels           # the empty old group isn't listed
    st = client.get(f"/api/jobs/{j['id']}/autotag", headers=H).json()
    assert set(s2[1:]) <= set(st["auto"]) and s2[0] in st["manual"]
    # "Me"
    assert client.put("/api/me", json={"name": "Jeremiah Ng"}, headers=H).json()["name"] == "Jeremiah Ng"
    assert client.get("/api/me", headers=H).json()["name"] == "Jeremiah Ng"


def test_redo_minutes_keeps_names_and_tags(client):
    j = upload(client, "dev1-redo", 40, "diarize=true&speakers=2")
    segs = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"]
    a = next(i for i, s in enumerate(segs) if s["speaker"] == "Speaker 1")
    b = next(i for i, s in enumerate(segs) if s["speaker"] == "Speaker 2")
    client.post(f"/api/jobs/{j['id']}/segments/{a}", json={"name": "Kai Wen"}, headers=H)
    client.post(f"/api/jobs/{j['id']}/segments/{b}", json={"name": "Nurul", "also": "Kai Wen"}, headers=H)
    _wait_autotag(client, j["id"])                         # tagging looks for the same voices in the background
    before = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"]
    st0 = client.get(f"/api/jobs/{j['id']}/autotag", headers=H).json()
    r = client.post(f"/api/jobs/{j['id']}/summarize", json={}, headers=H)
    assert r.status_code == 200
    j2 = wait(client, j["id"])
    assert j2["status"] == "done"
    names = set(j2["speaker_names"].values())
    assert {"Kai Wen", "Nurul"} <= names
    after = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"]
    assert [s["speaker"] for s in after] == [s["speaker"] for s in before] and after[b]["also"]
    st = client.get(f"/api/jobs/{j['id']}/autotag", headers=H).json()
    assert st["manual"] == st0["manual"]
    md = client.get(f"/api/jobs/{j['id']}/result.md", headers=H).text
    assert "**Kai Wen:**" in md and "Nurul + Kai Wen" in md


def test_second_person_only_does_not_mark_line_as_yours(client):
    j = upload(client, "dev1-also2", 20, "diarize=true&speakers=2")
    client.post(f"/api/jobs/{j['id']}/segments/2", json={"also": "Rahul"}, headers=H)
    st = client.get(f"/api/jobs/{j['id']}/autotag", headers=H).json()
    assert 2 not in st["manual"]
    client.post(f"/api/jobs/{j['id']}/segments/2", json={"name": "Siti"}, headers=H)
    assert 2 in client.get(f"/api/jobs/{j['id']}/autotag", headers=H).json()["manual"]


def test_autotag_takes_over_a_split_group_of_the_same_voice(client):
    import json
    from app import worker
    j = upload(client, "dev1-split", 40, "diarize=true&speakers=2")
    jid = j["id"]
    tr = worker.load_transcript(jid)
    # the speaker separation split voice A into two groups: lines at 20-30 s became "Speaker 3"
    for s in tr["segments"]:
        if 20 <= s["start"] < 30:
            s["speaker"] = "Speaker 3"
    worker._write(jid, "transcript.json", json.dumps(tr))
    spk = worker.load_json(jid, "speakers.json", {})
    spk["Speaker 3"] = {"seconds": 10, "embedding": [1.0, 0.0, 0.0, 0.0]}           # same voice as Speaker 1
    worker._write(jid, "speakers.json", json.dumps(spk))
    first = next(i for i, s in enumerate(tr["segments"]) if s["start"] < 10)
    r = client.post(f"/api/jobs/{jid}/segments/{first}", json={"name": "Priscilla"}, headers=H).json()
    for _ in range(400):
        st = client.get(f"/api/jobs/{jid}/autotag", headers=H).json()
        if st["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert st["state"] == "done", st
    segs = client.get(f"/api/jobs/{jid}/transcript.json", headers=H).json()["segments"]
    for s in segs:
        voice_a = int(s["start"] // 10) % 2 == 0
        assert (s["speaker"] == r["speaker"]) == voice_a, (s["start"], s["speaker"])


def _only_these_voices():
    from app import db
    with db._lock:                                         # earlier tests remembered the same two fake voices
        db._conn.execute("DELETE FROM voices")
        db._conn.commit()


def _wait_autotag(client, jid):
    for _ in range(400):
        st = client.get(f"/api/jobs/{jid}/autotag", headers=H).json()
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.05)
    return st


def test_remembered_voices_name_an_untagged_older_meeting(client):
    import json
    from app import autotag, db, worker
    _only_these_voices()
    # meeting 1: you name both voices (they're remembered for every meeting)
    a = upload(client, "dev1-known1", 40, "diarize=true&speakers=2")
    segs = client.get(f"/api/jobs/{a['id']}/transcript.json", headers=H).json()["segments"]
    client.post(f"/api/jobs/{a['id']}/segments/0", json={"name": "Wen Hui"}, headers=H)
    other = next(i for i, s in enumerate(segs) if int(s["start"] // 10) % 2 != int(segs[0]["start"] // 10) % 2)
    client.post(f"/api/jobs/{a['id']}/segments/{other}", json={"name": "Darius"}, headers=H)
    # meeting 2: recorded earlier, nobody named, the voices are only "Speaker 7/8" ("?" on the web)
    b = upload(client, "dev1-known2", 40, "diarize=true&speakers=2")
    jid = b["id"]
    tr = worker.load_transcript(jid)
    for s in tr["segments"]:
        s["speaker"] = "Speaker 7" if int(s["start"] // 10) % 2 == 0 else "Speaker 8"
    worker._write(jid, "transcript.json", json.dumps(tr))
    worker._write(jid, "speakers.json", "{}")
    o = dict(db.get(jid)["options"]); o.pop("speaker_names", None); o.pop("recognized_voices", None)
    o.pop("speaker_auto", None); o.pop("manual_lines", None); o.pop("auto_lines", None)
    db.update(jid, options=o)
    assert client.get(f"/api/jobs/{jid}/autotag", headers=H).json()["stale"]
    r = client.post(f"/api/jobs/{jid}/autotag", headers=H)
    assert r.status_code == 200, r.text
    st = _wait_autotag(client, jid)
    assert st["state"] == "done" and not st["stale"], st
    names = client.get(f"/api/jobs/{jid}", headers=H).json()["speaker_names"]
    for s in client.get(f"/api/jobs/{jid}/transcript.json", headers=H).json()["segments"]:
        want = "Wen Hui" if int(s["start"] // 10) % 2 == int(segs[0]["start"] // 10) % 2 else "Darius"
        assert names.get(s["speaker"]) == want, (s["start"], s["speaker"], names)
    # nothing new remembered since: the background pass leaves it alone
    assert not autotag.stale(db.get(jid))


def test_new_meeting_is_named_from_remembered_voices(client):
    from app import autotag
    from app.config import settings
    _only_these_voices()
    a = upload(client, "dev1-known3", 20, "diarize=true&speakers=2")
    client.post(f"/api/jobs/{a['id']}/segments/0", json={"name": "Wen Hui"}, headers=H)
    _wait_autotag(client, a["id"])
    settings.autotag_new = True
    try:
        b = upload(client, "dev1-known4", 20, "diarize=true&speakers=2")
        _wait_autotag(client, b["id"])
        assert not client.get(f"/api/jobs/{b['id']}/autotag", headers=H).json()["stale"]
        assert "Wen Hui" in client.get(f"/api/jobs/{b['id']}", headers=H).json()["speaker_names"].values()
    finally:
        settings.autotag_new = False


def test_voice_check_reports_why_lines_stay_unnamed(client):
    _only_these_voices()
    j = upload(client, "dev1-report", 40, "diarize=true&speakers=2")
    client.post(f"/api/jobs/{j['id']}/segments/0", json={"name": "Wen Hui"}, headers=H)
    st = _wait_autotag(client, j["id"])
    r = st["report"]
    assert r["lines"] == 8 and "Wen Hui" in r["people"] and not st["loose"]
    assert r["unnamed"] + r["named_now"] <= r["lines"]
    r2 = client.post(f"/api/jobs/{j['id']}/autotag?loose=true", headers=H)
    assert r2.status_code == 200
    st = _wait_autotag(client, j["id"])
    assert st["loose"] and st["report"]["loose"]


def test_voices_relearned_from_tags_after_they_were_lost(client):
    from app import autotag, db
    _only_these_voices()
    j = upload(client, "dev1-relearn", 40, "diarize=true&speakers=2")
    client.post(f"/api/jobs/{j['id']}/segments/0", json={"name": "Priscilla"}, headers=H)
    _wait_autotag(client, j["id"])
    _only_these_voices()                                   # a redeploy without persistent /data lost them
    assert not client.get("/api/voices", headers=H).json()["voices"]
    r = client.post("/api/voices/relearn", headers=H).json()
    assert "Priscilla" in r["learned"]
    assert "Priscilla" in [v["name"] for v in client.get("/api/voices", headers=H).json()["voices"]]
    assert client.post("/api/voices/relearn", headers=H).json()["learned"] == []   # only once


def test_names_and_terms_fix_misheard_words(client):
    from app import terms
    client.put("/api/glossary", content="CTRIP\nPriscilla\nAleta Planet", headers=H)
    t = terms.known_terms()
    assert terms.fix_text("So the C trip dashboard, Pricilla said", t) == "So the CTRIP dashboard, Priscilla said"
    assert terms.fix_text("the trip was fine, see you", t) == "the trip was fine, see you"
    assert terms.fix_text("Alita planet pays", t) == "Aleta Planet pays"
    assert terms.spelled("pricilla") == "Priscilla" and terms.spelled("Speaker 2") == "Speaker 2"
    tr = {"segments": [{"text": "ask Pricilla about c trip"}]}
    assert terms.fix_transcript(tr, t) == 1 and tr["segments"][0]["heard"] == "ask Pricilla about c trip"
    client.put("/api/glossary", content="", headers=H)


def test_meeting_without_speakers_gets_them_again_and_keeps_your_tags(client):
    import json
    from app import worker
    _only_these_voices()
    j = upload(client, "dev1-nospk", 40, "diarize=true&speakers=2")
    jid = j["id"]
    tr = worker.load_transcript(jid)
    for s in tr["segments"]:                               # speaker separation failed when it was transcribed
        s["speaker"], s["also"] = "", []
    worker._write(jid, "transcript.json", json.dumps(tr))
    worker._write(jid, "speakers.json", "{}")
    from app import db
    o = dict(db.get(jid)["options"]); o["speaker_names"] = {}; o.pop("speaker_auto", None)
    db.update(jid, options=o)
    client.post(f"/api/jobs/{jid}/segments/0", json={"name": "Priscilla"}, headers=H)
    _wait_autotag(client, jid)
    assert client.get(f"/api/jobs/{jid}/autotag", headers=H).json()["no_speakers"]
    r = client.post(f"/api/jobs/{jid}/speakers/redo", headers=H)
    assert r.status_code == 200
    st = _wait_autotag(client, jid)
    assert st["state"] == "done" and not st["no_speakers"], st
    segs = client.get(f"/api/jobs/{jid}/transcript.json", headers=H).json()["segments"]
    names = client.get(f"/api/jobs/{jid}", headers=H).json()["speaker_names"]
    assert all(s["speaker"] for s in segs)
    a = [names.get(s["speaker"]) for s in segs if int(s["start"] // 10) % 2 == 0]
    assert a and all(n == "Priscilla" for n in a), (a, names)


def test_no_undefined_names_in_app():
    import subprocess, sys, pathlib
    files = [str(p) for p in pathlib.Path(__file__).resolve().parent.parent.joinpath("app").glob("*.py")]
    out = subprocess.run([sys.executable, "-m", "pyflakes", *files], capture_output=True, text=True).stdout
    bad = [l for l in out.splitlines() if "undefined name" in l]
    assert not bad, bad


def test_confirm_auto_lines_marks_them_yours(client):
    _only_these_voices()
    j = upload(client, "dev1-accept", 40, "diarize=true&speakers=2")
    client.post(f"/api/jobs/{j['id']}/segments/0", json={"name": "Priscilla"}, headers=H)
    st = _wait_autotag(client, j["id"])
    auto = st["auto"]
    assert auto
    r = client.post(f"/api/jobs/{j['id']}/autotag/accept", json={"name": "Priscilla"}, headers=H).json()
    assert r["accepted"] >= 1
    st = client.get(f"/api/jobs/{j['id']}/autotag", headers=H).json()
    assert set(auto) - set(st["auto"]) <= set(st["manual"]) and 0 in st["manual"]


def test_same_name_on_two_speakers_joins_them(client):
    import json
    from app import worker
    _only_these_voices()
    j = upload(client, "dev1-join", 40, "diarize=true&speakers=2")
    jid = j["id"]
    tr = worker.load_transcript(jid)
    for s in tr["segments"]:                               # one person split into Speaker 1 and Speaker 3
        if 20 <= s["start"] < 30:
            s["speaker"] = "Speaker 3"
    worker._write(jid, "transcript.json", json.dumps(tr))
    labs = sorted({s["speaker"] for s in tr["segments"]})
    a = next(s["speaker"] for s in tr["segments"] if s["start"] < 10)
    r = client.post(f"/api/jobs/{jid}/speakers", json={"names": {a: "Priscilla", "Speaker 3": "priscilla"}}, headers=H).json()
    assert r["joined"] == ["Priscilla"]
    segs = client.get(f"/api/jobs/{jid}/transcript.json", headers=H).json()["segments"]
    assert "Speaker 3" not in {s["speaker"] for s in segs}
    assert a in {s["speaker"] for s in segs if 20 <= s["start"] < 30}   # the lower number stays
    names = client.get(f"/api/jobs/{jid}", headers=H).json()["speaker_names"]
    assert list(names.values()).count("Priscilla") == 1 and "Speaker 3" not in names
    sp = [s["label"] for s in client.get(f"/api/jobs/{jid}/speakers", headers=H).json()["speakers"]]
    assert "Speaker 3" not in sp, sp


def test_rename_meeting_title_kept_after_redo(client):
    j = upload(client, "dev1-title", 20)
    r = client.post(f"/api/jobs/{j['id']}/title", json={"title": "CTRIP dashboard review"}, headers=H).json()
    assert r["title"] == "CTRIP dashboard review"
    assert "CTRIP dashboard review" in client.get(f"/api/jobs/{j['id']}/result.md", headers=H).text
    client.post(f"/api/jobs/{j['id']}/summarize", json={}, headers=H)
    assert wait(client, j["id"])["title"] == "CTRIP dashboard review"
    d = client.get(f"/api/jobs/{j['id']}/result.docx", headers=H)
    assert d.status_code == 200 and "CTRIP_dashboard_review_9-Oct-2026.docx" in d.headers["content-disposition"], d.headers["content-disposition"]
    t = client.get(f"/api/jobs/{j['id']}/transcript.txt", headers=H)
    assert "CTRIP_dashboard_review_9-Oct-2026_transcript.txt" in t.headers["content-disposition"]


def test_fill_a_gap_adds_lines_and_keeps_tags(client):
    import json
    from app import worker, db
    j = upload(client, "dev1-gap", 40, "diarize=true&speakers=2")
    jid = j["id"]
    tr = worker.load_transcript(jid)
    tr["segments"] = [s for s in tr["segments"] if not (10 <= s["start"] < 20)]   # 10-20 s was "not heard"
    worker._write(jid, "transcript.json", json.dumps(tr))
    last = len(tr["segments"]) - 1
    o = dict(db.get(jid)["options"]); o["manual_lines"] = {str(last): tr["segments"][last]["speaker"]}
    db.update(jid, options=o)
    r = client.post(f"/api/jobs/{jid}/fill", json={"start": 10, "end": 20}, headers=H)
    assert r.status_code == 200
    st = _wait_autotag(client, jid)
    assert st["state"] == "done" and st.get("filled", 0) >= 1, st
    segs = client.get(f"/api/jobs/{jid}/transcript.json", headers=H).json()["segments"]
    assert any(10 <= s["start"] < 20 for s in segs)
    assert [s["start"] for s in segs] == sorted(s["start"] for s in segs)
    assert str(len(segs) - 1) in [str(i) for i in st["manual"]]      # your tag moved with its line


def test_minutes_title_carries_the_date():
    import datetime
    from app import render
    d = datetime.date(2026, 10, 14)
    assert render.titled("MANCO Meeting", d) == "MANCO Meeting – 14th Oct 2026"
    assert render.titled("MANCO - 14 Oct 2026", d) == "MANCO - 14 Oct 2026"
    assert [render.nice_date(datetime.date(2026, 10, n)) for n in (1, 2, 3, 11, 22)] == \
        ["1st Oct 2026", "2nd Oct 2026", "3rd Oct 2026", "11th Oct 2026", "22nd Oct 2026"]


def test_fixing_words_is_learned_for_every_meeting(client):
    from app import terms
    terms.save_corrections({})
    j = upload(client, "dev1-words", 20)
    r = client.post(f"/api/jobs/{j['id']}/segments/0/text", json={"text": "This is TAPIN sentence number 1."}, headers=H).json()
    r = client.post(f"/api/jobs/{j['id']}/segments/0/text", json={"text": "This is DABIN sentence number 1."}, headers=H).json()
    assert {"heard": "TAPIN", "right": "DABIN"} in r["learned"]
    seg = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"][0]
    assert seg["text"] == "This is DABIN sentence number 1." and seg["edited"]
    assert terms.corrections()["tapin"] == "DABIN"
    assert terms.apply_corrections("we went to tapin today, TAPIN-2 stays") == "we went to DABIN today, TAPIN-2 stays"
    assert "DABIN" in terms.known_terms()
    tr = {"segments": [{"text": "send it to tapin"}, {"text": "tapin again", "edited": True}]}
    terms.fix_transcript(tr)
    assert tr["segments"][0]["text"] == "send it to DABIN" and tr["segments"][1]["text"] == "tapin again"
    assert "DABIN" in client.get(f"/api/jobs/{j['id']}/result.md", headers=H).text
    assert client.get("/api/corrections", headers=H).json()["corrections"]["tapin"] == "DABIN"
    assert client.delete("/api/corrections/TAPIN", headers=H).status_code == 200
    terms.save_corrections({})


def test_word_fix_also_changes_the_minutes_summary(client):
    from app import terms, worker
    terms.save_corrections({})
    j = upload(client, "dev1-sumfix", 20)
    worker._write(j["id"], "summary.md", "### Summary\nThe team reviewed the TAPIN dashboard.\n")
    worker.build_outputs(j["id"])
    import json
    tr = worker.load_transcript(j["id"])
    tr["segments"][0]["text"] = "This is TAPIN sentence number 1."     # what the speech model heard
    worker._write(j["id"], "transcript.json", json.dumps(tr))
    client.post(f"/api/jobs/{j['id']}/segments/0/text", json={"text": "This is DABIN sentence number 1.", "learn": False}, headers=H)
    md = client.get(f"/api/jobs/{j['id']}/result.md", headers=H).text
    assert "reviewed the DABIN dashboard" in md, md[:400]
    terms.save_corrections({})


def test_speakers_renumbered_without_gaps(client):
    import json
    from app import autotag, worker, db
    j = upload(client, "dev1-renum", 40, "diarize=true&speakers=2")
    jid = j["id"]
    tr = worker.load_transcript(jid)
    for s in tr["segments"]:
        s["speaker"] = {"Speaker 1": "Speaker 5", "Speaker 2": "Speaker 15"}.get(s["speaker"], s["speaker"])
    worker._write(jid, "transcript.json", json.dumps(tr))
    worker._write(jid, "summary.md", "### Action items\n- [ ] Speaker 15 - send the plan\n")
    o = dict(db.get(jid)["options"]); o["speaker_names"] = {"Speaker 5": "James"}; o["manual_lines"] = {"0": "Speaker 5"}
    db.update(jid, options=o)
    m = autotag.renumber(jid)
    assert m["Speaker 5"] == "Speaker 1" and m["Speaker 15"] == "Speaker 2"   # old voice groups move after them
    segs = worker.load_transcript(jid)["segments"]
    assert {s["speaker"] for s in segs} == {"Speaker 1", "Speaker 2"}
    o = db.get(jid)["options"]
    assert o["speaker_names"] == {"Speaker 1": "James"} and o["manual_lines"] == {"0": "Speaker 1"}
    assert "Speaker 2 - send the plan" in (worker.result_dir(jid) / "summary.md").read_text()
    assert autotag.renumber(jid) == {}
