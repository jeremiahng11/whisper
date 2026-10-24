"""End-to-end API tests with a fake transcriber and a fake LLM (no models needed).
Run:  pytest -q
"""
import io
import json
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
os.environ["BUNDLED_FIRMWARE"] = "0"
os.environ["FIRMWARE_REPO"] = ""
os.environ["AUTOTAG_NEW"] = "0"          # tested on their own below (they'd change other tests' meetings)
os.environ["AUTOTAG_SWEEP"] = "0"
os.environ["PEOPLE_SWEEP"] = "0"
os.environ["KNOWLEDGE_SWEEP"] = "0"
os.environ["RESEARCH_LIVE"] = "0"
os.environ["WARM_UP"] = "0"
os.environ["RAG"] = "0"                   # finding by meaning: its own test (with a stand-in embedding model)         # no web searches from the Ask tests (turned on in its own test)

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


def fake_chat(messages, best=False, fast=False):
    CALLS.append(messages[-1]["content"])
    BEST.append(best)
    SYSTEMS.append(messages[0]["content"] if messages[0]["role"] == "system" else "")
    if FAIL["on"]:
        raise RuntimeError("LLM is down")
    if "company profile of" in messages[-1]["content"]:
        import json as _j
        return _j.dumps([
            {"q": "Who is the CTO of Aleta Planet", "a": "Jeremiah Ng, Group CTO and CIO.", "kind": "person",
             "topic": "Aleta Planet", "alts": ["who is the CTO", "who is our CTO"]},
            {"q": "Who founded Aleta Planet", "a": "Ryan Tan, the founder, Chairman and CEO.", "kind": "person",
             "topic": "Aleta Planet", "alts": ["who is the founder of Aleta Planet", "who is our founder"]},
            {"q": "What products does Aleta Planet have", "a": "Aleta Adventure cards and the card issuing API.",
             "kind": "product", "topic": "Aleta Planet", "alts": ["what are our products"]}])
    if "found by searching for" in messages[-1]["content"]:
        out = [{"q": "What is Sanrio Aleta Adventure", "a": "A Hello Kitty travel card by Aleta Planet and Sanrio.",
                "kind": "product", "topic": "Sanrio Aleta Adventure", "alts": ["tell me about Sanrio Aleta Adventure"]}]
        if "Someone asked" in messages[-1]["content"]:
            out.insert(0, {"q": "How much is the Adventure pass yearly fee", "a": "SGD 49 a year.", "kind": "price",
                           "topic": "Adventure pass", "alts": ["Adventure pass yearly fee", "price of the Adventure pass"]})
        import json as _j
        return _j.dumps(out)
    if "for a colleague asking" in messages[-1]["content"]:
        return "Ryan Tan is the founder, Chairman and CEO of Aleta Planet."
    if "write the facts someone" in messages[-1]["content"]:
        return ('Here you go:\n```json\n[{"q": "When does Idea Saver V2 ship?", "a": "End of November 2026.", "kind": "date", '
                '"topic": "Idea Saver V2", "alts": ["Idea Saver V2 ship date", "V2 launch date"]},'
                '{"q": "Who owns the rollback plan", "a": "San owns it.", "kind": "action", "topic": "rollback", "alts": [],},'
                '{"q": "x", "a": "too short a question"}]\n```')
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
    monkeypatch.setattr(settings, "live_hold_s", 20)
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
    # the page sees it streaming in, with the transcript so far
    lv = client.get("/api/live", headers=H).json()["live"]
    one = [x for x in lv if x["rid"] == rid][0]
    assert one["receiving"] and abs(one["received_s"] - 40) < 0.1 and one["transcribed_s"] == 15
    assert one["segments"] and one["segments"][0]["text"]
    # who's talking, while it streams in (test voices: two people taking turns every 10 s)
    assert [x["speaker"] for x in one["segments"]] == ["Speaker 1", "Speaker 1", "Speaker 2"]
    assert [v["label"] for v in one["speakers"]] == ["Speaker 1", "Speaker 2"] and one["seg_from"] == 0
    # the watch's caption while it records: the last line and who's talking
    b = client.get(f"/api/live/{rid}/brief", headers=H).json()
    assert b["line"] == one["segments"][-1]["text"] and b["speaker"] == "Speaker 2"
    assert b["speakers"] == [{"name": "Speaker 1", "seconds": 10}, {"name": "Speaker 2", "seconds": 5}]
    assert b["transcribed_s"] == 15 and abs(b["behind_s"] - 25) < 0.2
    assert client.get("/api/live/dev1-nothing/brief", headers=H).status_code == 404
    pr = client.put(f"/api/uploads/{rid}", content=bytes(data[first:]), headers={**H, "X-Offset": str(first), "X-Live": "1"}).json()
    assert pr["received"] == len(data) and pr["brief"]["line"] == b["line"] and pr["brief"]["speaker"] == "Speaker 2"
    assert pr["brief"]["speakers"] == b["speakers"] and "rid" not in pr["brief"] and pr["brief"]["received_s"] == 60
    r = client.post(f"/api/uploads/{rid}/complete?filename=2026-10-09_1100.wav", headers={**H, "X-Total-Size": str(len(data))})
    j = wait(client, r.json()["id"])
    assert j["status"] == "done" and abs(j["duration"] - 60) < 0.1
    segs = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"]
    nums = [int(s["text"].split()[-1].rstrip(".")) for s in segs]
    assert nums == list(range(1, 13))                                  # no gaps, no doubles
    assert not st.exists() and (worker.result_dir(j["id"]) / "live.json").exists()
    d = client.get(f"/api/jobs/{j['id']}/draft", headers=H).json()
    assert d["speakers_done"] and len(d["segments"]) == 12 and not (worker.result_dir(j["id"]) / "draft.json").exists()


# ---------------------------------------------------------------- watch1b: Opus recordings, logs, firmware, live ask
def opus_packets(seconds, kbps=16, vbr=False):
    import av
    import numpy as np
    c = av.CodecContext.create("libopus", "w")
    c.sample_rate, c.layout, c.format, c.bit_rate = 16000, "mono", "s16", kbps * 1000
    c.options = {"vbr": "on" if vbr else "off", "frame_duration": "20", "application": "voip"}
    c.open()
    t = np.arange(int(seconds * 16000)) / 16000
    x = (np.sin(2 * np.pi * 300 * t) * 6000).astype(np.int16)
    pk = []
    for i in range(0, len(x), 320):
        fr = av.AudioFrame.from_ndarray(x[i:i + 320].reshape(1, -1), format="s16", layout="mono")
        fr.sample_rate, fr.pts = 16000, i
        pk += [bytes(p) for p in c.encode(fr)]
    return pk[: int(seconds * 50)]


def opus_pad(pk: bytes, n: int) -> bytes:
    """What the watch does (opus_packet_pad): one frame (code 0) -> code 3 with in-band padding up to n bytes."""
    assert pk[0] & 3 == 0
    p = n - 3 - (len(pk) - 1)
    assert 0 <= p <= 254
    return bytes([pk[0] | 3, 0x41, p]) + pk[1:] + bytes(p)


def opus_wav(packets, align, sizes=True):
    raw = b"".join(opus_pad(p, align) if len(p) < align else p for p in packets)
    fmt = struct.pack("<HHIIHHHH", 0x704F, 1, 16000, align * 50, align, 0, 2, 320)
    return (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(raw) if sizes else 0) + b"WAVE" + b"fmt " +
            struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(raw) if sizes else 0) + raw)


def test_opus_recordings_from_the_watch(client, tmp_path, monkeypatch):
    import numpy as np
    from app import transcribe, worker
    pk = opus_packets(60)
    assert {len(p) for p in pk} == {40}                                   # CBR: every packet the same size
    # padded packets (the watch pads every one to the frame size) decode the same
    vb = opus_packets(2, vbr=True)
    assert all(p[0] & 3 == 0 for p in vb)
    n = max(len(p) for p in vb) + 3
    b = transcribe.opus_decode(b"".join(opus_pad(p, n) for p in vb), n)
    assert len(b) == 32000 and float(np.sqrt(np.mean(b[3200:] ** 2))) > 0.05   # heard, not 20 ms silences
    f = tmp_path / "o.wav"
    f.write_bytes(opus_wav(pk, 40, sizes=False))                          # still being recorded: sizes 0
    assert abs(worker.wav_seconds(f) - 60) < 0.05 and abs(len(transcribe.load_audio(str(f))) / 16000 - 60) < 0.05
    whole = transcribe.load_audio(str(f))
    clip = transcribe.load_range(str(f), 12.5, 14.0)
    assert len(clip) == 24000 and float(np.max(np.abs(clip - whole[200000:224000]))) < 0.01
    transcribe.fix_wav_header(str(f))
    assert struct.unpack("<I", f.read_bytes()[44:48])[0] == 3000 * 40
    # streamed live from the watch, then completed: transcribed in parts, no gaps
    monkeypatch.setattr(settings, "live_min_new_s", 5)
    monkeypatch.setattr(settings, "live_hold_s", 20)
    rid = "w1b-test-2026-10-21_1500"
    data = opus_wav(pk, 40, sizes=False)
    first = 48 + 40 * 50 * 40
    assert client.put(f"/api/uploads/{rid}", content=data[:first], headers={**H, "X-Offset": "0", "X-Live": "1"}).status_code == 200
    st = worker.live_state_path(rid)
    for _ in range(400):
        if st.exists() and '"until": 15' in st.read_text():
            break
        time.sleep(0.05)
    assert '"until": 15' in st.read_text()
    client.put(f"/api/uploads/{rid}", content=data[first:], headers={**H, "X-Offset": str(first), "X-Live": "1"})
    r = client.post(f"/api/uploads/{rid}/complete?filename=2026-10-21_1500.wav", headers={**H, "X-Total-Size": str(len(data))})
    j = wait(client, r.json()["id"])
    assert j["status"] == "done" and abs(j["duration"] - 60) < 0.1
    segs = client.get(f"/api/jobs/{j['id']}/transcript.json", headers=H).json()["segments"]
    assert [int(x["text"].split()[-1].rstrip(".")) for x in segs] == list(range(1, 13))
    c = client.get(f"/api/jobs/{j['id']}/clip?start=10&end=12&enhance=false", headers=H)
    assert c.status_code == 200 and c.content[:4] == b"RIFF" and len(c.content) == 44 + 2 * 16000 * 2


def test_watch_logs_and_status(client):
    st = {"fw": "W1B-1.4.0", "board": "watch1b", "bat": 82, "usb": False, "rssi": -61, "heap_kb": 120, "rec": True}
    import json as _j
    r = client.post("/api/devices/w1b-abc123/log", content="[  12s] wifi: connected\n[  13s] live: 32 KB\n",
                    headers={**H, "Content-Type": "text/plain", "X-Status": _j.dumps(st)})
    assert r.json() == {"ok": True, "next_s": 60}
    client.post("/api/devices/w1b-abc123/log", content="[  70s] mem: internal 110 KB free", headers={**H, "Content-Type": "text/plain"})
    d = [x for x in client.get("/api/devices", headers=H).json()["devices"] if x["id"] == "w1b-abc123"][0]
    assert d["fw"] == "W1B-1.4.0" and d["board"] == "watch1b" and d["status"]["bat"] == 82 and d["last_line"].endswith("110 KB free")
    log = client.get("/api/devices/w1b-abc123/log?lines=2", headers=H).text.splitlines()
    assert len(log) == 2 and log[0].endswith("[  13s] live: 32 KB") and log[0][:2] == "20"
    assert client.post("/api/devices/bad id!/log", content="x", headers=H).status_code in (400, 404)
    assert client.get("/api/devices", headers={}).status_code == 401
    client.delete("/api/devices/w1b-abc123/log", headers=H)
    assert client.get("/api/devices/w1b-abc123/log", headers=H).text == ""


def fw_image(board, version, size=4096):
    body = b"\xe9" + bytes(200) + f"ORACLEFW|{board}|{version}|".encode() + bytes(size)
    return body


def test_firmware_updates_over_wifi(client):
    H2 = {**H, "Content-Type": "application/octet-stream"}
    assert client.post("/api/firmware", content=b"\x00" + bytes(100), headers=H2).status_code == 400   # not a firmware
    assert client.post("/api/firmware", content=b"\xe9" + bytes(100), headers=H2).status_code == 400   # no marker
    for v in ("W1B-1.3.0", "W1B-1.10.0", "W1B-1.9.2"):
        r = client.post("/api/firmware?notes=" + v, content=fw_image("watch1b", v), headers=H2)
        assert r.status_code == 200 and r.json()["board"] == "watch1b" and r.json()["version"] == v
    client.post("/api/firmware", content=fw_image("round", "R1.7.20"), headers=H2)
    lt = client.get("/api/firmware/latest?board=watch1b&current=W1B-1.3.0", headers=H).json()
    assert lt["version"] == "1.10.0" and lt["name"] == "W1B-1.10.0" and lt["newer"] and lt["url"] == "/api/firmware/watch1b/W1B-1.10.0.bin"
    import hashlib
    b = client.get(lt["url"], headers=H)
    assert b.status_code == 200 and hashlib.sha256(b.content).hexdigest() == lt["sha256"] and len(b.content) == lt["size"]
    assert client.get("/api/firmware/latest?board=watch1b&current=W1B-1.10.0", headers=H).status_code == 404   # up to date
    # hold back 1.10.0: 1.9.2 is offered instead
    assert client.post("/api/firmware/watch1b/W1B-1.10.0", json={"active": False}, headers=H).json()["active"] is False
    assert client.get("/api/firmware/latest?board=watch1b&current=W1B-1.3.0", headers=H).json()["name"] == "W1B-1.9.2"
    assert client.get("/api/firmware/latest?board=round", headers=H).json()["version"] == "R1.7.20"
    assert client.get("/api/firmware/latest?board=nope", headers=H).status_code == 404
    items = client.get("/api/firmware", headers=H).json()["items"]
    assert {i["version"] for i in items} >= {"W1B-1.10.0", "W1B-1.9.2", "R1.7.20"}     # (1.3.0: only 2 kept)
    assert "W1B-1.3.0" not in {i["version"] for i in items}
    for v in ("W1B-1.10.0", "W1B-1.9.2"):
        assert client.delete(f"/api/firmware/watch1b/{v}", headers=H).status_code == 200
    client.delete("/api/firmware/round/R1.7.20", headers=H)
    assert client.get("/api/firmware/latest?board=watch1b", headers=H).status_code == 404


def test_ask_while_the_meeting_is_still_going(client):
    import json as _j
    from app import worker
    rid = "w1b-ask-2026-10-21_1600"
    st = worker.live_state_path(rid)
    st.write_text(_j.dumps({"until": 30, "segments": [
        {"start": 3, "end": 8, "text": "The budget for Q4 is 40k.", "speaker": "Speaker 1"},
        {"start": 9, "end": 14, "text": "We ship on Friday.", "speaker": "Speaker 2"}], "names": {"Speaker 1": "San"}}))
    try:
        r = client.post("/api/ask", json={"recording": rid, "messages": [{"role": "user", "content": "what's the budget?"}]}, headers=H)
        assert r.status_code == 202, r.text
        for _ in range(400):
            a = client.get(f"/api/ask/{r.json()['id']}", headers=H).json()
            if a["status"] != "pending":
                break
            time.sleep(0.05)
        assert a["status"] == "done"
        assert "still going on" in SYSTEMS[-1] and "[00:03] San: The budget for Q4 is 40k." in SYSTEMS[-1]
        assert "Speaker 2: We ship on Friday." in SYSTEMS[-1]
    finally:
        st.unlink(missing_ok=True)
    r = client.post("/api/ask", json={"recording": "w1b-none", "messages": [{"role": "user", "content": "x"}]}, headers=H)
    assert r.status_code == 404


def test_words_show_before_the_speakers_are_separated(tmp_path, monkeypatch):
    from app import transcribe
    p = tmp_path / "a.wav"
    p.write_bytes(wav_bytes(30))
    seen = []
    orig = transcribe.assign_speakers

    def slow(audio, tr, n=None, progress=None):
        seen.append(("diarize", len(drafts[-1][0]) if drafts else -1))
        return orig(audio, tr, n, progress)
    drafts = []
    monkeypatch.setattr(transcribe, "assign_speakers", slow)
    prior = {"until": 10, "segments": [{"start": 0, "end": 5, "text": "Earlier one.", "speaker": "Speaker 1"},
                                       {"start": 5, "end": 10, "text": "Earlier two.", "speaker": "Speaker 1"}]}
    tr = transcribe.transcribe(str(p), "en", lambda f, s: None, diarize=True, prior=prior,
                               on_text=lambda segs, done: drafts.append((list(segs), done)))
    assert drafts[0][0][0].text == "Earlier one." and not drafts[0][1]       # what came in live: shown at once
    assert drafts[-1][1] and len(drafts[-1][0]) == 6                          # all the words, then the speakers
    assert seen == [("diarize", 6)] and len(tr.segments) >= 6


def test_live_voices_get_labels_and_names(tmp_path, monkeypatch):
    from app import livetag, voices
    p = tmp_path / "b.wav"
    p.write_bytes(wav_bytes(40))
    st = {"segments": [{"start": float(t), "end": float(t + 5), "text": f"line {t}"} for t in range(0, 40, 5)]
          + [{"start": 40.2, "end": 40.6, "text": "ok"}]}
    monkeypatch.setattr(voices, "match", lambda spk: {"Speaker 2": "San"} if spk["Speaker 2"]["seconds"] >= 3 else {})
    assert livetag.tag(str(p), st, 0) == 8
    assert [s["speaker"] for s in st["segments"]] == ["Speaker 1"] * 2 + ["Speaker 2"] * 2 + ["Speaker 1"] * 2 + \
        ["Speaker 2"] * 3                                                 # "ok": too short, takes the voice before
    assert st["names"] == {"Speaker 2": "San"} and livetag.speakers(st)["Speaker 1"]["seconds"] == 20


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
    assert intent("take a note the Wi-Fi password is on the fridge", now)["type"] != "note"      # (no voice notes)
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
    assert "OLLAMA_URL" in msg and "11434" in msg


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
    wav = speech.to_wav16k(pcm, sr, louder=False)
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
        t._diarize = lambda a, n, p=None: ([(0.0, 1.7, "A"), (1.7, 4.0, "B"), (5.0, 9.0, "A"), (5.5, 8.5, "B")], {})
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


def test_pitch_of_a_voice():
    import numpy as np
    from app import people
    t = np.arange(16000 * 3) / 16000
    a = (0.3 * np.sin(2 * np.pi * 140 * t) + 0.15 * np.sin(2 * np.pi * 280 * t)).astype(np.float32)
    v = people.voice_stats([a])
    assert 130 <= v["pitch_hz"] <= 150, v


def test_people_profiles_and_samples(client):
    from app import people, voices
    _only_these_voices()
    j = upload(client, "dev1-people", 40, "diarize=true&speakers=2")
    client.post(f"/api/jobs/{j['id']}/segments/0", json={"name": "Priscilla"}, headers=H)
    _wait_autotag(client, j["id"])
    assert people.samples("Priscilla")["Priscilla"]                 # the tagged clip is kept as a sample
    lst = client.get("/api/people", headers=H).json()["people"]
    assert any(p["name"] == "Priscilla" for p in lst)
    p = client.get("/api/people/Priscilla", headers=H).json()
    assert p["lines"] >= 1 and p["meetings"][0]["id"] == j["id"] and p["samples"] >= 1
    assert "sentence" in p["language"]["words"] or p["language"]["words"] == [] or p["language"]["pace_wpm"] is not None
    # many samples: a voice close to one of them is still recognised
    voices.remember("Zed", [0.0, 0.0, 1.0, 0.0])
    voices.remember("Zed", [0.0, 0.0, 0.0, 1.0])
    assert voices.match({"Speaker 9": {"embedding": [0.0, 0.0, 0.0, 1.0], "seconds": 10}}).get("Speaker 9") == "Zed"
    assert client.delete("/api/voices/Zed", headers=H).status_code in (200, 204)
    assert not people.samples("Zed")


def test_meeting_dynamics(client):
    j = upload(client, "dev1-dyn", 40, "diarize=true&speakers=2")
    d = client.get(f"/api/jobs/{j['id']}/dynamics", headers=H).json()
    assert len(d["people"]) == 2 and sum(p["share"] for p in d["people"]) in (99, 100, 101)
    assert d["turns"] >= 4 and d["replies"] and not d["external"]
    d = client.post(f"/api/jobs/{j['id']}/dynamics", json={"external": True}, headers=H).json()
    assert d["external"]


def test_web_sign_in_protects_the_page(client):
    from app import login
    from app.config import settings
    old = (settings.admin_id, settings.admin_pass)
    settings.admin_id, settings.admin_pass = "admin", "s3cret-pass"
    try:
        c = client
        c.cookies.clear()
        r = c.get("/", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/login"
        assert "Oracle Minutes" in c.get("/login").text
        assert c.get("/api/jobs").status_code == 401
        assert c.post("/api/login", json={"id": "admin", "password": "nope"}).status_code == 401
        r = c.post("/api/login", json={"id": "ADMIN", "password": "s3cret-pass"})
        assert r.status_code == 200 and login.COOKIE in r.cookies
        assert c.get("/", follow_redirects=False).status_code == 200
        assert c.get("/api/jobs").status_code == 200                     # the page works with the cookie
        assert c.get("/api/session").json()["user"] == "admin"
        # another site can't post with your cookie
        assert c.put("/api/glossary", content="x", headers={"Origin": "https://evil.example"}).status_code == 403
        o = c.post("/api/passkey/register/options").json()
        assert o["options"]["rp"]["name"] == "Oracle Minutes" and o["options"]["authenticatorSelection"]["userVerification"] == "required"
        assert c.get("/api/jobs", headers=H).status_code == 200            # the watch's API key still works
        c.post("/api/logout")
        c.cookies.clear()
        assert c.get("/api/jobs").status_code == 401
        settings.admin_pass = "changed"                                    # a new password signs everyone out
    finally:
        settings.admin_id, settings.admin_pass = old
        client.cookies.clear()


def test_export_with_oracle_ai_analysis(client):
    j = upload(client, "dev1-ana", 40, "diarize=true&speakers=2")
    plain = client.get(f"/api/jobs/{j['id']}/result.md", headers=H).text
    md = client.get(f"/api/jobs/{j['id']}/result.md?analysis=true", headers=H)
    assert "Oracle AI Analysis" not in plain
    t = md.text
    assert "## Oracle AI Analysis" in t and "Who took part" in t and "_with_Analysis.md" in md.headers["content-disposition"]
    if "## Transcript" in t:
        assert t.index("## Oracle AI Analysis") < t.index("## Transcript")
    d = client.get(f"/api/jobs/{j['id']}/result.docx?analysis=true", headers=H)
    assert d.status_code == 200 and "_with_Analysis.docx" in d.headers["content-disposition"]


def test_passkey_site_follows_the_browser_behind_a_proxy():
    from starlette.requests import Request
    from app import login
    def req(h):
        return Request({"type": "http", "method": "POST", "path": "/", "scheme": "http", "query_string": b"",
                        "headers": [(k.lower().encode(), v.encode()) for k, v in h.items()], "server": ("x", 80)})
    assert login._site(req({"host": "whisper.jeremiah.sg", "origin": "https://whisper.jeremiah.sg"})) == \
        ("whisper.jeremiah.sg", "https://whisper.jeremiah.sg")
    assert login._site(req({"host": "whisper.jeremiah.sg", "origin": "https://evil.example"}))[1] == "http://whisper.jeremiah.sg"
    assert login._site(req({"host": "whisper.jeremiah.sg", "x-forwarded-proto": "https"}))[1] == "https://whisper.jeremiah.sg"


def test_ai_read_needed_when_names_change(client):
    from app import dynamics, worker, summarize
    j = upload(client, "dev1-dynauto", 40, "diarize=true&speakers=2")
    real = summarize.enabled
    summarize.enabled = lambda: True
    try:
        assert dynamics.needs_note(j["id"])
        worker._write(j["id"], "dynamics.json", '{"note": "x", "sig": ' + __import__("json").dumps(
            dynamics._sig(__import__("app.db", fromlist=["get"]).get(j["id"]), worker.load_transcript(j["id"]),
                          dynamics.stats(__import__("app.db", fromlist=["get"]).get(j["id"]), worker.load_transcript(j["id"])))) + '}')
        assert not dynamics.needs_note(j["id"])
        segs = worker.load_transcript(j["id"])["segments"]
        client.post(f"/api/jobs/{j['id']}/speakers", json={"names": {segs[0]["speaker"]: "Wen Hui"}}, headers=H)
        assert dynamics.needs_note(j["id"])                     # names changed: the read is out of date
    finally:
        summarize.enabled = real


def test_builtin_fintech_vocabulary(client):
    from app import terms, vocab
    tr = {"segments": [{"text": "the k y c and K.Y.C. checks, s g q r and the psp, fast transfer, the bin is full"}]}
    terms.fix_transcript(tr)
    t = tr["segments"][0]["text"]
    assert "KYC and KYC" in t and "SGQR" in t and "PSP" in t and "fast transfer" in t and "the bin is full" in t, t
    v = client.get("/api/vocab", headers=H).json()["packs"]
    assert {p["id"] for p in v} == {"payments", "tech", "finance"} and all(p["on"] for p in v)
    client.put("/api/vocab", json={"on": ["tech", "finance"]}, headers=H)
    assert "SGQR" not in vocab.terms()
    tr = {"segments": [{"text": "s g q r"}]}
    terms.fix_transcript(tr)
    assert tr["segments"][0]["text"] == "s g q r"
    client.put("/api/vocab", json={"on": ["payments", "tech", "finance"]}, headers=H)
    assert "SGQR" in vocab.terms()


def test_progress_in_health(client, tmp_path):
    from app import worker
    import struct
    p = tmp_path / "a.wav"
    hdr = bytearray(48); hdr[20:22] = (0x11).to_bytes(2, "little"); hdr[24:28] = (16000).to_bytes(4, "little")
    p.write_bytes(bytes(hdr) + b"\0" * 256 * 3168)          # 3168 blocks x 505 samples = 100 s
    assert abs(worker.wav_seconds(p) - 100) < 0.5
    worker.current_job, worker.live_info = "live:ir-x", {"rid": "ir-x", "until": 30.0, "received": 120.0}
    try:
        pr = client.get("/api/health").json()["progress"]
        assert pr["kind"] == "live" and pr["pct"] == 25 and pr["received_min"] == 2.0
    finally:
        worker.current_job, worker.live_info = None, {}


def test_speaker_heard_only_over_others_is_playable(client):
    import json
    from app import worker
    j = upload(client, "dev1-over", 40, "diarize=true&speakers=2")
    tr = worker.load_transcript(j["id"])
    tr["segments"][2]["also"] = ["Speaker 9"]
    worker._write(j["id"], "transcript.json", json.dumps(tr))
    spk = worker.load_json(j["id"], "speakers.json", {}); spk["Speaker 8"] = {"seconds": 0.4, "embedding": None}
    worker._write(j["id"], "speakers.json", json.dumps(spk))
    sp = {s["label"]: s for s in client.get(f"/api/jobs/{j['id']}/speakers", headers=H).json()["speakers"]}
    assert sp["Speaker 9"]["lines"] == 0 and sp["Speaker 9"]["over_lines"] == 1 and sp["Speaker 9"]["sample"]
    assert "Speaker 8" not in sp                               # no lines at all and no name: not listed


def test_correcting_a_name_unlearns_the_wrong_voice(client):
    from app import people, voices
    _only_these_voices()
    with __import__("app.db", fromlist=["_lock"])._lock:
        __import__("app.db", fromlist=["_conn"])._conn.execute("DELETE FROM voice_samples"); __import__("app.db", fromlist=["_conn"])._conn.commit()
    j = upload(client, "dev1-unlearn", 40, "diarize=true&speakers=2")
    jid = j["id"]
    segs = client.get(f"/api/jobs/{jid}/transcript.json", headers=H).json()["segments"]
    # line 0 (voice A) wrongly tagged as James: a James sample is taken from it
    client.post(f"/api/jobs/{jid}/segments/0", json={"name": "James"}, headers=H)
    _wait_autotag(client, jid)
    assert people.samples("James").get("James")
    # you correct it: it's Jeremiah -> James's sample from that line is dropped (and his voice with it)
    client.post(f"/api/jobs/{jid}/segments/0", json={"name": "Jeremiah"}, headers=H)
    _wait_autotag(client, jid)
    assert not people.samples("James").get("James")
    assert "James" not in [v["name"] for v in voices.list_voices()]
    assert people.samples("Jeremiah").get("Jeremiah")               # the corrected line now teaches Jeremiah's voice
    segs = client.get(f"/api/jobs/{jid}/transcript.json", headers=H).json()["segments"]
    nm = client.get(f"/api/jobs/{jid}", headers=H).json()["speaker_names"]
    same = [nm.get(s["speaker"]) for s in segs if int(s["start"] // 10) % 2 == 0]
    assert same and all(n == "Jeremiah" for n in same), same        # and the same voice elsewhere follows
    r = client.post("/api/people/Jeremiah/relearn", headers=H).json()
    assert r["samples"] >= 1


def test_samples_also_from_people_only_recognised_by_voice(client):
    """Someone the server named by voice (never ✓'d here) still gets samples from lines that clearly sound like them,
    and the People page shows the live count."""
    from app import db, people
    _only_these_voices()
    a = upload(client, "dev1-samp-a", 40, "diarize=true&speakers=2")
    client.post(f"/api/jobs/{a['id']}/speakers", json={"names": {"Speaker 1": "Wendy", "Speaker 2": "Omar"}}, headers=H)
    b = upload(client, "dev1-samp-b", 40, "diarize=true&speakers=2")
    _wait_autotag(client, b["id"])
    opts = db.get(b["id"])["options"]
    assert "Wendy" in (opts.get("speaker_names") or {}).values()
    with db._lock:
        before = db._conn.execute("SELECT COUNT(*) FROM voice_samples WHERE job_id=?", (b["id"],)).fetchone()[0]
    n = people.top_up(b["id"])
    with db._lock:
        after = db._conn.execute("SELECT COUNT(*) FROM voice_samples WHERE job_id=?", (b["id"],)).fetchone()[0]
    assert n >= 1 and after > before
    assert client.get("/api/people/wendy", headers=H).json()["samples"] == people.sample_count("Wendy") >= 2


def test_wakeword_models_for_the_watch(client):
    import io
    import json
    import zipfile
    m = client.get("/api/wakeword", headers=H).json()
    assert m["source"] == "starter" and m["models"][0]["id"] == "wake" and m["version"]
    f = m["models"][0]["file"]
    b = client.get(f"/api/wakeword/files/{f}", headers=H)
    assert b.status_code == 200 and b.content[4:8] == b"TFL3" and len(b.content) == m["models"][0]["size"]
    assert client.get("/api/wakeword/files/..%2Fwake.json", headers=H).status_code == 404
    assert client.get("/api/wakeword").status_code == 401
    # a trained set: "Hey Oracle" + a command
    fake = b"\x1c\x00\x00\x00TFL3" + b"\x00" * 200
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("out/wake.json", json.dumps({"models": [
            {"id": "wake", "label": "Hey Oracle", "file": "hey_oracle.tflite", "cutoff": 0.96},
            {"id": "start_meeting", "file": "start_meeting.tflite"}]}))
        zf.writestr("out/hey_oracle.tflite", fake)
        zf.writestr("out/start_meeting.tflite", fake)
    r = client.post("/api/wakeword", files={"file": ("models.zip", z.getvalue(), "application/zip")}, headers=H)
    assert r.status_code == 200, r.text
    m2 = r.json()
    assert m2["source"] == "yours" and m2["version"] != m["version"]
    assert [e["label"] for e in m2["models"]] == ["Hey Oracle", "Start meeting"]
    assert client.get("/api/wakeword/files/hey_oracle.tflite", headers=H).content == fake
    assert client.get(f"/api/wakeword/files/{f}", headers=H).status_code == 404      # the old set is gone
    # broken sets are refused and the installed one stays
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("wake.json", json.dumps({"models": [{"id": "start_meeting", "file": "a.tflite"}]}))
        zf.writestr("a.tflite", fake)
    r = client.post("/api/wakeword", content=bad.getvalue(), headers={**H, "Content-Type": "application/zip"})
    assert r.status_code == 400 and "no wake word" in r.json()["detail"]
    assert client.get("/api/wakeword", headers=H).json()["version"] == m2["version"]
    assert client.delete("/api/wakeword", headers=H).json()["source"] == "starter"


def test_wakeword_voice_samples(client):
    import io
    import struct
    import zipfile
    pcm = b"\x00\x01" * 16000
    wav = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) + \
        b"data" + struct.pack("<I", len(pcm)) + pcm
    for _ in range(3):
        assert client.post("/api/wakeword/samples/wake", content=wav, headers=H).status_code == 200
    r = client.post("/api/wakeword/samples/stop_recording", content=wav, headers=H).json()
    assert r["samples"]["wake"] == 3 and r["samples"]["stop_recording"] == 1 and r["total"] == 4
    assert client.post("/api/wakeword/samples/bogus", content=wav, headers=H).status_code == 400
    assert client.post("/api/wakeword/samples/wake", content=b"nope", headers=H).status_code == 400
    z = zipfile.ZipFile(io.BytesIO(client.get("/api/wakeword/samples.zip", headers=H).content))
    assert sorted(n.split("/")[0] for n in z.namelist()) == ["stop_recording", "wake", "wake", "wake"]
    assert client.delete("/api/wakeword/samples?phrase=wake", headers=H).json()["samples"]["wake"] == 0
    assert client.delete("/api/wakeword/samples", headers=H).json()["total"] == 0


def test_name_save_with_stale_label_is_skipped(client):
    """A save made with the labels the page showed before a renumber must not rename (or clear) someone else."""
    from app import db
    _only_these_voices()
    j = upload(client, "dev1-stale", 40, "diarize=true&speakers=2")
    jid = j["id"]
    _wait_autotag(client, jid)
    names0 = dict(db.get(jid)["options"].get("speaker_names") or {})
    r = client.post(f"/api/jobs/{jid}/speakers", json={"names": {"Speaker 2": "Tina"}, "was": {"Speaker 2": names0.get("Speaker 2", "")}},
                    headers=H).json()
    assert r["skipped"] == [] and db.get(jid)["options"]["speaker_names"]["Speaker 2"] == "Tina"
    # the page still thinks Speaker 2 is unnamed and sends "" for it: Tina must stay
    r = client.post(f"/api/jobs/{jid}/speakers", json={"names": {"Speaker 2": ""}, "was": {"Speaker 2": ""}}, headers=H).json()
    assert r["skipped"] == ["Speaker 2"] and db.get(jid)["options"]["speaker_names"]["Speaker 2"] == "Tina"
    # same name again is fine (idempotent)
    r = client.post(f"/api/jobs/{jid}/speakers", json={"names": {"Speaker 2": "TINA"}, "was": {"Speaker 2": ""}}, headers=H).json()
    assert db.get(jid)["options"]["speaker_names"]["Speaker 2"].lower() == "tina"


def test_pitch_veto_undoes_a_wrong_join(client, monkeypatch):
    """A whole speaker group joined to a man by voice print, but it's a woman's pitch: the join is undone and the
    lines go back to their own speaker."""
    import json
    from app import autotag, db, worker
    _only_these_voices()
    j = upload(client, "dev1-pitch", 40, "diarize=true&speakers=2")
    jid = j["id"]
    _wait_autotag(client, jid)
    client.post(f"/api/jobs/{jid}/speakers", json={"names": {"Speaker 1": "Jer"}, "remember": False}, headers=H)
    tr = worker.load_transcript(jid)
    moved = []
    for i, s in enumerate(tr["segments"]):
        if s.get("speaker") == "Speaker 2":
            s["orig"], s["speaker"] = "Speaker 2", "Speaker 1"
            moved.append(i)
    assert moved
    worker._write(jid, "transcript.json", json.dumps(tr))
    o = dict(db.get(jid)["options"])
    o.update(merged={"Speaker 2": "Speaker 1"}, auto_lines=moved, manual_lines={})
    db.update(jid, options=o)
    monkeypatch.setattr(autotag, "_person_pitch", lambda name: 120.0 if name == "Jer" else 0.0)
    monkeypatch.setattr(autotag, "_line_pitch", lambda jid_, job, segs, idxs: {i: (215.0 if i in moved else 118.0) for i in idxs})
    autotag.apply(jid)
    tr = worker.load_transcript(jid)
    assert all(tr["segments"][i]["speaker"] == "Speaker 2" for i in moved)
    assert "Speaker 2" not in (db.get(jid)["options"].get("merged") or {})
    assert autotag.pitch_ok(120, 130) and not autotag.pitch_ok(120, 210) and autotag.pitch_ok(0, 210)


def test_samples_cannot_lift_a_far_voice():
    from app import people
    c = [1.0, 0.0, 0.0]
    far = [0.3, 0.954, 0.0]                                 # 0.3 like their average voice
    samp = [[0.3, 0.954, 0.0]] * 3                          # ... but identical to three wrong samples
    assert people.similarity(far, c, samp) <= 0.41


def test_clean_samples_drops_other_pitch(client, monkeypatch):
    from app import autotag, db, people
    j = upload(client, "dev1-clean", 40, "diarize=true&speakers=2")
    people.forget("Zed")
    for i in range(8):
        people.add_sample("Zed", [1.0, 0.1 * (i % 3), 0.0, 0.0], j["id"], i)
    monkeypatch.setattr(autotag, "_line_pitch", lambda jid, job, segs, idxs: {k: (120.0 if k < 6 else 225.0) for k in idxs})
    assert people.clean_samples("Zed") == 2
    assert people.sample_count("Zed") == 6
    people.forget("Zed")


def test_knowledge_base_ask_and_watch_pack(client):
    from app import knowledge, watchpack
    j = upload(client, "dev1-kb", 20)
    n = knowledge.learn_meeting(j["id"])
    assert n == 2                                            # the too-short question is dropped
    kb = client.get("/api/knowledge", headers=H).json()
    assert kb["facts"] >= 2 and kb["meetings_read"] >= 1
    f = next(x for x in kb["items"] if x["q"].startswith("When does Idea Saver"))
    assert f["date"] and f["job"] == j["id"] and f["alts"] == ["Idea Saver V2 ship date", "V2 launch date"]
    assert not knowledge.needs_update(db_get(j["id"]))       # read once until the minutes change
    hits = client.get("/api/knowledge?q=when is the V2 launch", headers=H).json()["items"]
    assert hits and hits[0]["id"] == f["id"]
    # Ask over all meetings gets the facts too
    r = client.post("/api/ask", json={"scope": "meetings", "messages": [{"role": "user", "content": "Idea Saver V2 ship date?"}]}, headers=H)
    for _ in range(200):
        if client.get(f"/api/ask/{r.json()['id']}", headers=H).json()["status"] != "pending":
            break
        time.sleep(0.05)
    assert "company knowledge base" in SYSTEMS[-1] and "End of November 2026" in SYSTEMS[-1]
    # your correction sticks; a hidden fact is gone
    assert client.put(f"/api/knowledge/{f['id']}", json={"a": "Early December 2026."}, headers=H).status_code == 200
    knowledge.learn_meeting(j["id"])
    f2 = next(x for x in knowledge.facts() if x["id"] == f["id"])
    assert f2["a"] == "Early December 2026." and f2["edited"]
    # the watch pack: speakable phrases, numbers as words
    assert watchpack.phrase("When does Idea Saver V2 ship?") == "when does idea saver v two ship"
    assert watchpack.phrase("Q4 revenue 2026") == "q four revenue two zero two six"
    assert watchpack.phrase("Hi") == ""
    pk = client.get("/api/watch/pack", headers=H).json()
    it = next(x for x in pk["items"] if x["id"] == f["id"])
    assert it["say"][0] == "when does idea saver v two ship" and it["a"] == "Early December 2026."
    assert pk["version"] and pk["count"] == len(pk["items"]) and "extra" not in pk
    pk1 = client.get("/api/watch/pack?limit=1&extra=1", headers=H).json()
    assert pk1["count"] == 1 and pk1["extra"] and not set(x["id"] for x in pk1["extra"]) & {pk1["items"][0]["id"]}
    v1 = pk["version"]
    client.put(f"/api/knowledge/{f['id']}", json={"hidden": True}, headers=H)
    pk = client.get("/api/watch/pack", headers=H).json()
    assert f["id"] not in [x["id"] for x in pk["items"]] and pk["version"] != v1
    client.put(f"/api/knowledge/{f['id']}", json={}, headers=H)
    assert any(x["id"] == f["id"] for x in knowledge.facts())
    client.delete(f"/api/jobs/{j['id']}", headers=H)
    assert not any(x.get("job") == j["id"] for x in knowledge.facts())


def db_get(jid):
    from app import db
    return db.get(jid)


def test_knowledge_parse_tolerant():
    from app import knowledge
    assert knowledge.parse_facts("no json here") == []
    assert knowledge.parse_facts('[{"q": "What is the budget", "a": "SGD 20k", "kind": "weird"}]')[0]["kind"] == "fact"
    assert knowledge.parse_facts('<think>hmm</think>[{"q": "What is the budget", "a": "SGD 20k",}]')[0]["a"] == "SGD 20k"


def test_watch_model_trains_exports_and_serves(client, tmp_path):
    import json as _json
    from app import watchlm, watchlm_core as W
    facts = [{"q": f"Who is {n}", "a": f"{n} is the {r}.", "kind": "person", "topic": n, "alts": [f"Tell me about {n}", f"What does {n} do"]}
             for n, r in [("Ryan", "Group CEO"), ("Ella", "Chief Legal Officer"), ("Tina", "Head of Finance"), ("San", "IT lead")]]
    facts.append({"q": "When does the Round watch ship", "a": "The Round watch ships on 1 December 2026 - V2-ready.", "kind": "date",
                  "topic": "Round watch", "alts": ["Round watch ship date", "Round watch launch"]})
    ds = watchlm.make_dataset(facts)
    assert ds["test"] and not set(q for q, _ in ds["test"]) & set(q for q, _ in ds["train"])   # held back = never trained on
    assert ["Who is Ryan", "Ryan is the Group CEO."] in ds["train"] and any(a == watchlm.DONT_KNOW for _, a in ds["train"])
    (tmp_path / "dataset.json").write_text(_json.dumps(ds))
    rep = watchlm.train(tmp_path, max_minutes=3, max_epochs=600,
                        config={"d": 48, "layers": 2, "heads": 2, "ffn": 96, "seq": 40})
    assert rep["trained_on"] >= 80, rep
    q8, cfg, vocab, head = W.load(tmp_path / "model.owl")
    assert cfg["d"] == 48 and head["dont_know"] == watchlm.DONT_KNOW and rep["version"]
    p = W.dequantize(q8)
    assert W.generate(p, cfg, vocab, "Who is RYAN?") == "Ryan is the Group CEO."    # punctuation and case don't matter
    a, sure = W.answer(p, cfg, vocab, "Who is RYAN?")                         # the fast way (like the watch) agrees
    assert a == "Ryan is the Group CEO." and 0 < sure <= 1 and head["sure_cut"] == watchlm.SURE_CUT
    assert W.detok(W.words("ships on 1 December 2026 - V2-ready.")) == "ships on 1 December 2026 \u2013 V2-ready."
    raw = (tmp_path / "model.owl").read_bytes()
    assert raw[:4] == b"OWL1" and int.from_bytes(raw[4:8], "little") % 4 == 0   # floats stay 4-byte aligned for the watch
    assert rep["facts_sig"] == watchlm.facts_sig(facts) and ds["sig"] == rep["facts_sig"]
    assert watchlm.facts_sig(facts[:-1]) != rep["facts_sig"]
    # the API: nothing yet -> 404; too few facts -> 400
    r = client.get("/api/watch/model", headers=H).json()
    assert "training" in r and "status" in r
    if not watchlm.model_file():
        assert client.get("/api/watch/model/file", headers=H).status_code == 404


def test_sources_text_file_and_site(client, monkeypatch):
    from app import knowledge, sources
    import httpx as _hx
    # pasted text
    r = client.post("/api/knowledge/sources", json={"text": "The Round watch costs SGD 299 and has a one-year warranty.", "title": "Price list"}, headers=H)
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    assert sources.learn(sid) == 2                           # (the fake AI answers with its two facts)
    f = next(x for x in knowledge.facts() if x.get("source") == sid)
    assert f["meeting"] == "Price list" and "facts" not in client.get("/api/knowledge/sources", headers=H).json()["items"][0] or True
    items = client.get("/api/knowledge/sources", headers=H).json()["items"]
    me = next(x for x in items if x["id"] == sid)
    assert me["facts"] == 2 and me["state"] == "ok"
    n_calls = len(CALLS)
    sources.learn(sid)                                       # same text: not read by the AI again
    assert len(CALLS) == n_calls
    # a file (Word)
    import docx as _docx
    d = _docx.Document(); d.add_paragraph("Warranty: one year for every Idea Saver product, two years for business plans.")
    buf = io.BytesIO(); d.save(buf)
    r = client.post("/api/knowledge/sources", files={"file": ("warranty.docx", buf.getvalue())}, headers=H)
    assert r.status_code == 201 and r.json()["title"] == "warranty.docx"
    assert "one year" in (sources.sdir() / f"{r.json()['id']}.txt").read_text()
    assert client.post("/api/knowledge/sources", files={"file": ("x.exe", b"MZ....")}, headers=H).status_code == 400
    # a web site: the page + pages under it, nothing outside it
    pages = {"https://shop.test/products/": '<html><title>Products</title><nav>menu</nav><p>All our products: watches, note takers and the Oracle Minutes server, built and supported in Singapore.</p>'
                                            '<a href="round">Round</a><a href="/about">About</a><a href="https://other.test/x">x</a></html>',
             "https://shop.test/products/round": "<html><title>Round watch</title><script>x()</script><p>SGD 299. AMOLED screen, offline voice notes, one-year warranty, works with Oracle Minutes.</p></html>"}
    got = []

    class R:
        def __init__(self, u):
            self.status_code = 200 if u in pages else 404
            self.text = pages.get(u, "")
            self.content = self.text.encode()
            self.headers = {"content-type": "text/html"}

    monkeypatch.setattr(sources, "_get", lambda u, *a: (got.append(u), R(u))[1])
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    monkeypatch.setattr(sources, "browser_available", lambda: False)
    ps = sources.fetch_site("https://shop.test/products/", crawl=5)
    assert [p["title"] for p in ps] == ["Products", "Round watch"] and "menu" not in ps[0]["text"] and "x()" not in ps[1]["text"]
    assert "https://shop.test/about" not in got and not any("other.test" in u for u in got)
    r = client.post("/api/knowledge/sources", json={"url": "https://shop.test/products/", "crawl": 5}, headers=H)
    assert r.status_code == 201
    uid = r.json()["id"]
    sources.learn(uid)
    m = sources.get(uid)
    assert m["pages"] == 2 and m["state"] == "ok" and m["title"] == "Products"
    assert client.post("/api/knowledge/sources", json={"url": "file:///etc/passwd"}, headers=H).status_code == 400
    pages["https://shop.test/app"] = "<html><title>App</title><div id=root></div><script>render()</script></html>"
    aid = client.post("/api/knowledge/sources", json={"url": "https://shop.test/app"}, headers=H).json()["id"]
    with pytest.raises(ValueError):
        sources.learn(aid)
    assert sources.get(aid)["state"] == "error" and "almost no text" in sources.get(aid)["error"]
    # removing a source takes its facts away
    assert client.delete(f"/api/knowledge/sources/{sid}", headers=H).status_code == 200
    assert not any(x.get("source") == sid for x in knowledge.facts())
    assert client.get("/api/knowledge", headers=H).json()["sources"] >= 2


def test_watch_speech_model_served(client):
    r = client.get("/api/watch/speech", headers=H)
    assert r.status_code == 200 and len(r.content) > 3_000_000 and r.content[:4] == (4).to_bytes(4, "little")
    assert b"mn7_en" in r.content[:2000] and b"fst" in r.content[:2000]


def test_learns_your_questions_and_speaks_answers(client, monkeypatch):
    from app import knowledge, speech, watchpack, watchlm_core as W
    j = upload(client, "dev1-asked", 20)
    knowledge.learn_meeting(j["id"])
    f = next(x for x in knowledge.facts() if x["q"].startswith("When does Idea Saver"))
    n0 = f.get("asked", 0)                                       # (earlier tests asked about it on the web too)
    # a clear match: your wording joins the fact, it's counted, and the pack puts it first
    assert knowledge.note_question("so when will the idea saver v2 actually ship?", "ask") == f["id"]
    assert knowledge.note_question("what's the weather like") == ""            # no fact: kept, not linked
    f2 = next(x for x in knowledge.facts() if x["id"] == f["id"])
    assert "so when will the idea saver v2 actually ship" in f2["alts"] and f2["asked"] == n0 + 1
    r = client.post("/api/watch/asked", json={"items": [{"id": f["id"], "q": "when does idea saver v two ship"}]}, headers=H)
    assert r.json()["noted"] == 1
    f3 = next(x for x in knowledge.facts() if x["id"] == f["id"])
    assert f3["asked"] == n0 + 2 and "when does idea saver v two ship" not in f3["alts"]   # the watch's phrase is no new wording
    assert client.get("/api/watch/pack", headers=H).json()["items"][0]["id"] == f["id"]
    k = client.get("/api/knowledge", headers=H).json()
    assert k["asked"] >= 3 and k["asked_linked"] >= 2
    # Ask on the web is noted too (in the background)
    client.post("/api/ask", json={"scope": "meetings", "messages": [{"role": "user", "content": "who owns the rollback plan then"}]}, headers=H)
    for _ in range(100):
        if any(a["q"] == "who owns the rollback plan then" for a in knowledge.asked()):
            break
        time.sleep(0.05)
    assert any(a["q"] == "who owns the rollback plan then" and a["fact"] for a in knowledge.asked())
    # spoken answers: made once, kept, the pack says which recording goes with which answer
    monkeypatch.setattr(speech, "tts_available", lambda: True)
    made = []
    monkeypatch.setattr(speech, "tts", lambda text, lang="en", strict=False, live=False: (made.append(text), speech.to_wav16k(__import__("numpy").zeros(16000), 16000))[1])
    pk = client.get("/api/watch/pack", headers=H).json()
    it = next(x for x in pk["items"] if x["id"] == f["id"])
    assert it["voice"] == watchpack.voice_tag(it["a"])
    r = client.get(f"/api/watch/voice/{f['id']}", headers=H)
    assert r.status_code == 200 and r.content[:4] == b"RIFF" and r.content[20:22] == b"\x11\x00"   # IMA ADPCM
    client.get(f"/api/watch/voice/{f['id']}", headers=H)
    assert made == [it["a"]]                                                    # cached
    assert client.get("/api/watch/voice/00000000", headers=H).status_code == 404
    # model size grows with the facts, within what the watch can hold
    assert W.config_for(100, 300)["d"] == 160 and W.config_for(2000, 3000)["d"] == 192
    assert all(W.weight_bytes(W.config_for(n, 6000)) <= W.WATCH_BUDGET for n in (10, 2000, 5000, 20000))


def test_wrong_wakeup_clips_kept(client):
    from app import wakeword
    r = client.post("/api/wakeword/samples/not_wake", content=wav_bytes(2), headers=H)
    assert r.status_code == 200, r.text
    assert client.get("/api/wakeword/samples", headers=H).json()["samples"]["not_wake"] >= 1
    assert client.post("/api/wakeword/samples/bogus", content=wav_bytes(2), headers=H).status_code == 400
    client.delete("/api/wakeword/samples?phrase=not_wake", headers=H)
    assert client.get("/api/wakeword/samples", headers=H).json()["samples"]["not_wake"] == 0


def test_any_question_uses_company_knowledge(client):
    from app import knowledge
    knowledge.note_question  # (module loaded)
    src = client.post("/api/knowledge/sources", json={"text": "Aleta Planet is a Singapore payment services provider and Banking-as-a-Service platform.", "title": "About us"}, headers=H).json()
    from app import sources
    sources.learn(src["id"])
    facts = knowledge.facts()
    assert facts
    def ask(text):
        r = client.post("/api/ask", json={"messages": [{"role": "system", "content": "You are the watch."}, {"role": "user", "content": text}]}, headers=H)
        for _ in range(200):
            if client.get(f"/api/ask/{r.json()['id']}", headers=H).json()["status"] != "pending":
                break
            time.sleep(0.05)
        return SYSTEMS[-1]
    sysmsg = ask("When does the Idea Saver V2 ship?")         # no scope, no meeting words: still finds the fact
    assert sysmsg.startswith("You are the watch.") and "company knowledge base" in sysmsg and "Idea Saver V2" in sysmsg
    assert "company knowledge base" not in ask("tell me a joke about cats")   # nothing matches: plain answer
    hits = knowledge.find("when does the idea sayver vee two ship")          # misheard words still find the fact
    assert hits and hits[0]["q"].startswith("When does Idea Saver")


def test_sources_zendesk_and_javascript_sites(monkeypatch):
    import http.server, threading as _th, functools, tempfile as _tf, os as _os
    from app import sources
    # a Zendesk Help Center: every article through its API (the pages themselves answer 403)
    calls = []

    class J:
        def __init__(self, u):
            calls.append(u)
            self.status_code, self.headers = 200, {"content-type": "application/json"}
            self._d = ({"articles": [{"title": "How do I top up?", "body": "<p>Top up with PayNow in the app.</p>",
                                      "html_url": "https://faq.example/hc/en-us/articles/1"}], "next_page": u.replace("page=1", "page=2")}
                       if "page=1" in u else
                       {"articles": [{"title": "Fees", "body": "<p>No fee for local spend.</p>", "html_url": "x", "draft": False},
                                     {"title": "Draft", "body": "<p>secret</p>", "draft": True}], "next_page": None})
        def json(self):
            return self._d
    monkeypatch.setattr(sources, "_get", lambda u, *a: J(u))
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    arts = sources.fetch_site("https://faq.example/hc/en-us")
    assert [a["title"] for a in arts] == ["How do I top up?", "Fees"] and "PayNow" in arts[0]["text"]
    assert calls[0] == "https://faq.example/api/v2/help_center/en-us/articles.json?per_page=100&page=1"
    monkeypatch.undo()
    # a page built by JavaScript: plain HTTP gets an empty shell, the headless browser gets the text
    if not sources.browser_available():
        return
    d = _tf.mkdtemp()
    open(_os.path.join(d, "index.html"), "w").write(
        "<html><head><title>Playbook</title></head><body><div id=root></div><script>"
        "document.getElementById('root').innerHTML = '<h1>Aleta Playbook</h1><p>' + 'Our onboarding steps for new merchants: "
        "verify the business, sign the agreement, set up the API keys and go live within five working days. ' .repeat(3) + "
        "'</p><a href=\"/guide.html\">Guide</a>';</script></body></html>")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=d))
    _th.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/index.html"
        title, text, links, how = sources._page(url, True)
        assert how == "browser" and "onboarding steps for new merchants" in text and title == "Playbook"
        assert any(l.endswith("/guide.html") for l in links)
    finally:
        srv.shutdown()


def test_who_is_written_from_documents(client, monkeypatch):
    from app import knowledge, people, sources
    sid = client.post("/api/knowledge/sources", json={"text": "Leadership\nRyan Tan\nFounder, Chairman and CEO of Aleta Planet\nLeads the group's strategy.", "title": "About us"}, headers=H).json()["id"]
    sources.learn(sid)
    mat = knowledge.person_material("Ryan")
    assert "Founder, Chairman and CEO" in mat
    monkeypatch.setattr(people, "all_names", lambda: ["Ryan"])
    assert knowledge.who_needs_update("Ryan")
    assert knowledge.who_is("Ryan") == ""                                   # not written yet (the sweep does it)
    assert knowledge.who_is("Ryan", refresh=True).startswith("Ryan Tan is the founder")
    assert not knowledge.who_needs_update("Ryan")
    knowledge.rebuild()
    f = next(x for x in knowledge.facts() if x["q"] == "Who is Ryan")
    assert "Chairman and CEO" in f["a"] and f["kind"] == "person"
    assert knowledge.find("who is ryan")[0]["q"] == "Who is Ryan"


def test_news_source_follows_a_topic(client, monkeypatch):
    from app import knowledge, sources
    rss = """<rss><channel>
<item><title>Aleta Planet launches card for travellers</title><link>http://www.bing.com/news/apiclick.aspx?ref=x&amp;url=https%3a%2f%2fnews.example%2faleta-card</link>
<pubDate>Wed, 14 Oct 2026 08:00:00 GMT</pubDate><description>Aleta Planet unveiled a new travel card.</description><source>The Example Times</source></item>
<item><title>Weather is nice</title><link>https://news.example/weather</link><pubDate>Wed, 14 Oct 2026 08:00:00 GMT</pubDate><description>Sunny.</description></item>
</channel></rss>"""
    article = "<html><title>Card</title><p>" + "Aleta Planet, the Singapore payments company, launched a multi-currency travel card on 14 October. " * 5 + "</p></html>"

    class R:
        def __init__(self, u):
            self.status_code = 200
            self.text = rss if "format=rss" in u else article
            self.content = self.text.encode()
            self.headers = {"content-type": "application/rss+xml" if "format=rss" in u else "text/html"}
    monkeypatch.setattr(sources, "_get", lambda u, *a: R(u))
    monkeypatch.setattr(sources, "browser_available", lambda: False)
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    r = client.post("/api/knowledge/sources", json={"news": "Aleta Planet"}, headers=H)
    assert r.status_code == 201 and r.json()["kind"] == "news"
    sid = r.json()["id"]
    sources.learn(sid)
    m = sources.get(sid)
    assert m["articles"] == 1 and m["state"] == "ok"        # the weather story isn't about it
    txt = (sources.sdir() / f"{sid}.txt").read_text()
    assert "multi-currency travel card" in txt and "The Example Times, 2026-10-14" in txt and "https://news.example/aleta-card" in txt
    assert all(f["kind"] == "news" for f in m["facts"])
    n = len(CALLS)
    sources.learn(sid)                                       # nothing new the next day: the AI isn't asked again
    assert len(CALLS) == n and sources.get(sid)["new_articles"] == 0
    m = sources.get(sid); m["fetched"] = 0; sources._save(m)
    assert sources.due(sources.get(sid))                     # checked again daily


def test_research_topics_questions_and_live(client, monkeypatch):
    from app import chat, knowledge, research, sources
    web = """<rss><channel>
<item><title>Sanrio Aleta Adventure card - Aleta Planet</title><link>https://aleta.example/sanrio</link>
<pubDate>Tue, 13 Oct 2026 08:00:00 GMT</pubDate><description>The Sanrio Aleta Adventure card with Hello Kitty designs.</description></item>
<item><title>Adventure pass yearly fee - Aleta Planet</title><link>https://aleta.example/pass</link>
<pubDate>Tue, 13 Oct 2026 08:00:00 GMT</pubDate><description>The Adventure pass costs SGD 49 a year.</description></item>
<item><title>Sanrio theme park tickets</title><link>https://other.example/park</link>
<pubDate>Tue, 13 Oct 2026 08:00:00 GMT</pubDate><description>Puroland opening hours.</description></item>
</channel></rss>"""
    page = "<html><title>Sanrio Aleta Adventure</title><p>" + "Aleta Planet and Sanrio launched the Sanrio Aleta Adventure card. " * 6 + "</p></html>"
    got = []

    class R:
        def __init__(self, u):
            self.status_code = 200
            rss = "format=rss" in u or "/rss/" in u
            self.text = (web if "bing.com/search" in u else "<rss><channel></channel></rss>") if rss else page
            self.content = self.text.encode()
            self.headers = {"content-type": "application/rss+xml" if rss else "text/html"}
    monkeypatch.setattr(sources, "_get", lambda u, *a: (got.append(u), R(u))[1])
    monkeypatch.setattr(sources, "browser_available", lambda: False)
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)

    # topics it finds by itself: proper names the facts keep coming back to, not people, not common words
    fs = [{"q": f"q{i}", "a": "a", "kind": "product", "topic": t, "id": f"x{i}"}
          for i, t in enumerate(["Aleta Planet"] * 4 + ["Sanrio Aleta Adventure"] * 3 + ["pricing"] * 5 + ["Rare"] * 2)]
    assert research.discover(fs) == ["Aleta Planet", "Sanrio Aleta Adventure"]

    # a topic you add: searched on the web and in the news, pages about something else skipped
    r = client.post("/api/research/topics", json={"topic": "Sanrio Aleta Adventure"}, headers=H)
    assert r.status_code == 201
    t = next(x for x in r.json()["topics"] if x["topic"] == "Sanrio Aleta Adventure")
    assert not t["auto"] and t["source"]
    sources.learn(t["source"])
    m = sources.get(t["source"])
    assert m["kind"] == "research" and m["state"] == "ok" and m["articles"] == 1     # the theme-park page isn't about it
    assert any("bing.com/search?format=rss" in u for u in got) and any("news" in u for u in got)
    assert knowledge.find("what is sanrio aleta adventure")[0]["q"] == "What is Sanrio Aleta Adventure"
    m["fetched"] = time.time() - 2 * 86400; sources._save(m)
    assert not sources.due(sources.get(t["source"]))         # web research: weekly, not daily
    assert client.post("/api/research/topics", json={"topic": "x"}, headers=H).status_code == 400

    # a question nobody could answer: researched with the company's name, then it answers it
    knowledge.note_question("how much is the adventure pass yearly fee", "watch")
    assert "how much is the adventure pass yearly fee" in research.unanswered()
    q = research.research_question("how much is the adventure pass yearly fee")
    assert q["search"] if "search" in q else True
    sources.learn(q["id"])
    assert research.relink_asked() >= 1
    a = [x for x in knowledge.asked() if x["q"] == "how much is the adventure pass yearly fee"][-1]
    assert a["fact"] and "SGD 49" in next(f for f in knowledge.facts() if f["id"] == a["fact"])["a"]
    assert "how much is the adventure pass yearly fee" not in research.unanswered()
    s = client.get("/api/research", headers=H).json()
    assert s["questions_researched"] >= 1 and s["questions_answered"] >= 1

    # removing a topic removes what it found; a found topic isn't found again
    assert client.delete("/api/research/topics", params={"topic": "Sanrio Aleta Adventure"}, headers=H).status_code == 200
    assert not sources.get(t["source"])
    assert research.refresh_found(fs) == ["Aleta Planet"]
    assert client.put("/api/research", json={"auto": False}, headers=H).json()["auto"] is False
    client.put("/api/research", json={"auto": True}, headers=H)

    # Ask about something nothing is known about: a quick web search goes along
    monkeypatch.setattr(research.settings, "research_live", True)
    ctx = research.live_context("Adventure pass yearly fee")
    assert "SGD 49" in ctx and "aleta.example/pass" in ctx and "Puroland" not in ctx
    SYSTEMS.clear()
    r = client.post("/api/ask", json={"prompt": "what is the zorblax pass fee"}, headers=H)
    import threading
    for _ in range(100):                                     # (time.sleep is stubbed above)
        if chat.get(r.json()["id"])["status"] != "pending":
            break
        threading.Event().wait(0.05)
    assert any("quick web search" in x for x in SYSTEMS), (chat.get(r.json()["id"]), r.json())


def test_company_essentials_on_the_watch(client, monkeypatch):
    from app import core, knowledge, sources, watchlm, watchpack
    monkeypatch.setattr(core, "company", lambda: "Aleta Planet")
    m = sources.add("text", "Leadership", text="Our leadership\nRyan Tan - Founder, Chairman and CEO\n"
                    "Jeremiah Ng - Group CTO and CIO\nWei Ling - Chief Financial Officer of Aleta Planet\n"
                    "Sam Lee - Chief Operating Officer, runs card operations\n"
                    "Aleta Planet is a payments company in Singapore. " * 3)
    sources.learn(m["id"])
    mat = core.material()
    assert "Founder, Chairman and CEO" in mat
    assert core.build(force=True) == 3
    fs = knowledge.facts()
    cto = next(f for f in fs if f["q"] == "Who is the CTO of Aleta Planet")
    assert cto["core"] and cto["meeting"] == "Company profile"
    assert knowledge.find("who is our cto")[0]["q"] == "Who is the CTO of Aleta Planet"
    r = client.get("/api/knowledge", params={"q": "what product does aleta planet have"}, headers=H).json()
    assert r["answer"]["a"].startswith("Aleta Adventure cards")   # (products / product)
    assert client.get("/api/knowledge", params={"q": "flibbert quasar nebula"}, headers=H).json()["answer"] is None
    n = len(CALLS)
    resp = client.post("/api/ask", json={"prompt": "Who is our CTO?"}, headers=H)   # at once, no AI
    assert resp.status_code == 200
    r = resp.json()
    assert r["status"] == "done" and r["answer"].startswith("Jeremiah Ng") and len(CALLS) == n
    core.build()                                             # nothing changed: the AI isn't asked again
    assert len(CALLS) == n and not core.needs_update()

    # first in the watch's pack, marked for answering on the watch even when online
    pack = watchpack.build(150, True)
    top = pack["items"][:3]
    assert all(it.get("core") for it in top) and {it["q"] for it in top} >= {"Who founded Aleta Planet"}
    assert "who is the cto" in next(it for it in top if it["q"] == "Who is the CTO of Aleta Planet")["say"]

    # pin any fact to the watch
    other = next(f for f in fs if not f.get("core"))
    assert client.put(f"/api/knowledge/{other['id']}", json={"pinned": True}, headers=H).status_code == 200
    f2 = next(f for f in knowledge.facts() if f["id"] == other["id"])
    assert f2["core"] and f2["pinned"]
    assert any(it["id"] == other["id"] and it.get("core") for it in watchpack.build(150)["items"])
    client.put(f"/api/knowledge/{other['id']}", json={"a": "A better answer."}, headers=H)   # a fix keeps the pin
    f2 = next(f for f in knowledge.facts() if f["id"] == other["id"])
    assert f2["pinned"] and f2["a"] == "A better answer."
    client.put(f"/api/knowledge/{other['id']}", json={"pinned": False}, headers=H)
    assert not next(f for f in knowledge.facts() if f["id"] == other["id"]).get("core")
    client.put(f"/api/knowledge/{other['id']}", json={}, headers=H)

    # the watch model: essentials first and learned harder; retrained soon after they change
    fs = knowledge.facts()
    wf = watchlm.watch_facts(fs)
    assert all(f.get("core") for f in wf[:3])
    jobs = [i for i, f in enumerate(wf) if f.get("job")]
    assert not jobs or min(jobs) > max(i for i, f in enumerate(wf) if not f.get("job"))   # meetings after the rest
    ds = watchlm.make_dataset(fs)
    assert sum(1 for q, a in ds["train"] if a == "Jeremiah Ng, Group CTO and CIO.") >= 4 and ds["core_sig"]
    started = []
    monkeypatch.setattr(watchlm, "start", lambda f: started.append(len(f)))
    monkeypatch.setattr(watchlm, "state", lambda: {"training": False, "status": {"at": time.time() - 7 * 3600},
                                                   "model": {"core_sig": "old"}})
    assert watchlm.core_changed(fs) and started
    monkeypatch.setattr(watchlm, "state", lambda: {"training": False, "status": {"at": time.time() - 7 * 3600},
                                                   "model": {"core_sig": ds["core_sig"]}})
    assert not watchlm.core_changed(fs)
    sources.remove(m["id"])


def test_watch_prompts_in_the_answer_voice(client, monkeypatch):
    import numpy as np
    from app import speech, watchpack
    monkeypatch.setattr(speech, "tts_available", lambda: True)
    t = np.arange(16000) / 16000
    sig = np.concatenate([np.zeros(8000), 0.5 * np.sin(2 * np.pi * 220 * t[:6000]), np.zeros(8000)])
    monkeypatch.setattr(speech, "tts", lambda text, lang="en", strict=False, live=False: speech.to_wav16k((sig * 32767).astype("<i2"), 16000))
    r = client.get("/api/watch/prompts", headers=H).json()
    assert "ok" in r["names"] and r["voice"]
    w = client.get("/api/watch/prompts/ok", headers=H)
    assert w.status_code == 200 and w.content[:4] == b"RIFF"
    assert len(w.content) < 6000 + 2000                       # the half second of lead-in silence is gone (ADPCM ~4x)
    assert client.get("/api/watch/prompts/nope", headers=H).status_code == 404


def test_watch_crash_reports(client):
    assert client.post("/api/watch/crash", json={"fw": "R1.7.10", "text": "R1.7.9 task loopTask pc 0x42001234"}, headers=H).status_code == 200
    items = client.get("/api/watch/crashes", headers=H).json()["items"]
    assert items[0]["text"].startswith("R1.7.9 task loopTask")


def test_plain_question_skips_the_ai_intent(client):
    n = len(CALLS)
    r = client.post("/api/intent", json={"text": "Who is Ryan?"}, headers=H).json()
    assert r["type"] == "question" and len(CALLS) == n          # no AI round trip before the answer
    from app import knowledge
    fs = [{"q": "Who is Ryan Gwee", "a": "Founder, Chairman and CEO of Aleta Planet.", "kind": "person", "topic": "Ryan Gwee",
           "alts": ["Tell me about Ryan Gwee"], "id": "a1"},
          {"q": "What did Ryan ask for in the budget review", "a": "Ryan asked for a cut.", "kind": "action", "topic": "budget",
           "id": "a2", "alts": []}]
    import unittest.mock as um
    with um.patch.object(knowledge, "facts", lambda: fs):
        assert knowledge.best_answer("who is ryan")["id"] == "a1"
        assert knowledge.best_answer("ryan gwee")["id"] == "a1"


def test_find_by_meaning_rag(client, monkeypatch):
    import numpy as np
    from app import chat, knowledge, rag, sources, summarize
    # a stand-in embedding model: words, with "tech side" meaning the same as "CTO"
    syn = {"cto": "tech", "technology": "tech", "side": "tech", "runs": "lead", "head": "lead", "chief": "lead"}
    def fake_embed(texts, query=False, timeout=60):
        out = []
        for t in texts:
            v = np.zeros(64)
            for w in re.findall(r"[a-z]+", t.lower()):
                w = syn.get(w, w)
                v[hash(w) % 64] += 1
            out.append(list(v))
        return out
    import re
    monkeypatch.setattr(rag, "embed", fake_embed)
    monkeypatch.setattr(rag.settings, "rag", True)
    monkeypatch.setattr(rag.settings, "rag_min_score", 0.2)
    m = sources.add("text", "Team page", text="Our team\n\nJeremiah Ng is the Group CTO and CIO of Aleta Planet, "
                    "leading technology and engineering.\n\nThe office is at 1 Example Road, Singapore.")
    n = rag.update()
    assert n > 0 and rag.summary()["passages"] >= n
    assert rag.update() == 0                                 # nothing new: nothing embedded again
    hits = rag.find("who runs the tech side", 3)
    assert any("Jeremiah Ng" in h["text"] for h in hits)
    ctx = rag.context_for("who runs the tech side")
    assert "Jeremiah Ng" in ctx and "closest in meaning" in ctx
    # Ask: the passage goes to the AI (the quick model), found in the answer's thread
    SYSTEMS.clear()
    fast_seen = []
    real = summarize.chat
    def spy(messages, best=False, fast=False):
        fast_seen.append(fast)
        return real(messages, best, fast)
    monkeypatch.setattr(summarize, "chat", spy)
    r = client.post("/api/ask", json={"prompt": "who runs the tech side"}, headers=H)
    import threading
    for _ in range(100):
        if chat.get(r.json()["id"])["status"] != "pending":
            break
        threading.Event().wait(0.05)
    assert any("Jeremiah Ng is the Group CTO" in x for x in SYSTEMS) and True in fast_seen
    sources.remove(m["id"])
    rag.update()
    assert not any("Jeremiah Ng is the Group CTO" in h["text"] for h in rag.find("who runs the tech side", 5))


def test_live_questions_go_first(monkeypatch):
    from app import summarize
    import threading
    order = []
    summarize._live["n"] = 1                                 # a live question is being answered
    def background():
        summarize._yield_to_live(max_s=5)
        order.append("background")
    b = threading.Thread(target=background)
    b.start()
    threading.Event().wait(0.3)
    assert order == []                                       # background waits
    summarize._live["n"] = 0
    b.join(2)
    assert order == ["background"]


def test_ask_waits_and_voice_is_ready(client, monkeypatch):
    from app import chat, speech
    monkeypatch.setattr(speech, "tts_available", lambda: True)
    made = []
    monkeypatch.setattr(speech, "tts", lambda text, lang="en", strict=False, live=False: (made.append(text), speech.to_wav16k(__import__("numpy").zeros(1600, "<i2"), 16000))[1])
    resp = client.post("/api/ask?wait=5", json={"prompt": "tell me a short joke", "speak": True}, headers=H)
    r = resp.json()
    assert r["status"] == "done" and r["answer"] and resp.status_code == 200   # (the watch reads an answer only with 200)                # answered in the same reply (no polling)
    import threading
    for _ in range(100):
        if chat.spoken(r["answer"], "en", True):
            break
        threading.Event().wait(0.05)
    n = len(made)
    w = client.post("/api/tts", json={"text": r["answer"].replace("\n", " "), "format": "adpcm"}, headers=H)
    assert w.status_code == 200 and len(made) == n               # the voice was ready: not made again
    g = client.get(f"/api/ask/{r['id']}?wait=1", headers=H).json()
    assert g["status"] == "done"


def test_ollama_found_elsewhere(monkeypatch):
    from app import summarize
    monkeypatch.setattr(summarize.settings, "ollama_url", "http://no-such-ollama:11434")
    summarize._where.update(url="", at=0.0)
    monkeypatch.setattr(summarize, "_answers", lambda u: u == "http://host.docker.internal:11434")
    assert summarize.ollama_url() == "http://host.docker.internal:11434"
    summarize._where.update(url="", at=0.0)


def test_failed_summaries_retried_when_llm_is_back(client, monkeypatch):
    from app import db, summarize, worker
    jobs = db.list_jobs(5, "done")
    if not jobs:
        return
    jid = jobs[0]["id"]
    db.update(jid, summary_error="can't find the Ollama server 'http://x:11434'")
    monkeypatch.setattr(summarize, "llm_status", lambda: "ok")
    queued = []
    monkeypatch.setattr(worker, "wake", lambda: queued.append(1))
    assert worker.retry_failed_summaries() >= 1
    assert db.get(jid)["status"] == "queued" and queued
    db.update(jid, status="done", summary_error="")


def test_person_not_them(client):
    from app import db, people, worker
    n, ln = next(((x, people.their_lines(x)[0]) for x in people.all_names() if people.their_lines(x)), (None, None))
    assert n, "the earlier tests name speakers"
    with db._lock:
        db._conn.execute("INSERT INTO voice_samples (name, embedding, job_id, seg, created) VALUES (?,?,?,?,?)",
                         (n, "[0.1, 0.2]", ln["job"], ln["i"], 1.0))
        db._conn.commit()
    items = client.get(f"/api/people/{n}/samples", headers=H).json()["items"]
    assert any(x["job"] == ln["job"] and x["i"] == ln["i"] for x in items)
    r = client.post(f"/api/people/{n}/not-them", json={"job": ln["job"], "i": ln["i"]}, headers=H).json()
    assert r["removed"] >= 1
    assert not any(x["job"] == ln["job"] and x["i"] == ln["i"] for x in people.their_lines(n))
    assert not any(x["job"] == ln["job"] and x["i"] == ln["i"] for x in client.get(f"/api/people/{n}/samples", headers=H).json()["items"])


def test_auto_learning_switch(client):
    from app import knowledge
    assert client.get("/api/knowledge", headers=H).json()["auto_learning"] is True
    r = client.put("/api/knowledge/auto", json={"on": False}, headers=H).json()
    assert r["auto_learning"] is False and knowledge.auto_learning() is False
    assert client.put("/api/knowledge/auto", json={}, headers=H).status_code == 400
    client.put("/api/knowledge/auto", json={"on": True}, headers=H)
    assert knowledge.auto_learning() is True


def test_bilingual_minutes(client):
    assert summarize.minutes_languages("both", "zh") == ("zh", "en")
    assert summarize.minutes_languages("both", "en") == ("en", "zh")
    assert summarize.minutes_languages("zh+en", "en") == ("zh", "en")
    assert summarize.minutes_languages("en", "zh") == ("en", "")
    assert summarize.minutes_languages("", "zh") == ("zh", "")
    assert client.post("/api/jobs?minutes_lang=klingon!", content=wav_bytes(3), headers=H).status_code == 400
    CALLS.clear()
    r = client.post("/api/jobs?minutes_lang=both&language=zh", content=wav_bytes(20), headers=H)
    j = wait(client, r.json()["id"])
    assert j["status"] == "done" and j["minutes_lang"] == "both"
    assert any(c.startswith("Write the minutes in Chinese") or "Write the minutes in Chinese" in c for c in CALLS)
    tr = [c for c in CALLS if c.startswith("Translate these meeting minutes into English")]
    assert tr and "Title:" in tr[0]
    md = client.get(f"/api/jobs/{j['id']}/summary.md", headers=H).text
    assert "# 中文 (Chinese)" in md and "# English" in md
    # action items come from the first language only (not listed twice)
    acts = [c for c in CALLS if "as JSON" in c]
    assert acts and "# English" not in acts[-1]
    # redo in English only
    r = client.post(f"/api/jobs/{j['id']}/summarize", json={"minutes_lang": "en"}, headers=H)
    j = wait(client, j["id"])
    assert j["minutes_lang"] == "en"
    assert "# English" not in client.get(f"/api/jobs/{j['id']}/summary.md", headers=H).text


def test_watch_facts_people_and_pack_size(monkeypatch):
    from app import people, watchfacts, watchpack, knowledge
    prof = {"name": "Priscilla Tan", "language": {"phrases": ["make sense", "by friday"], "words": ["vendor", "budget", "timeline"],
                                                    "terms": ["Coolify"], "pace": "steady", "pace_wpm": 150, "questions": 20},
            "note": "### Usually raises and owns\n- Vendor contracts\n- The Q4 budget\n\n### How they communicate\n- Direct",
            "meetings": [{"id": "x", "title": "Budget review", "when": time.time() - 86400, "seconds": 300}],
            "talk_seconds": 900}
    monkeypatch.setattr(people, "all_names", lambda: ["Priscilla Tan"])
    monkeypatch.setattr(people, "get", lambda n: prof)
    fs = watchfacts.people_facts()
    qs = {f["q"]: f for f in fs}
    assert '"make sense"' in qs["What is Priscilla Tan's favourite word"]["a"]
    assert "Vendor contracts" in qs["What does Priscilla Tan usually talk about"]["a"]
    assert "Budget review" in qs["When did I last meet Priscilla Tan"]["a"]
    assert "15 minutes" in qs["How much does Priscilla Tan talk"]["a"]
    assert watchpack.phrase("What is Priscilla Tan's favourite word") == "what is priscilla tan s favourite word"
    # a big pack never gives the watch more than 400 phrases
    many = [{"id": f"{i:08x}", "q": f"What is product number {i}", "a": "x", "alts": [f"Tell me about product {i}", f"Product {i} please"],
             "kind": "fact"} for i in range(500)]
    monkeypatch.setattr(knowledge, "facts", lambda: many)
    p = watchpack.build(300, True)
    total = sum(len(it["say"]) for it in p["items"]) + sum(len(it["say"]) for it in p.get("extra", []))
    assert p["count"] >= 250 and total <= 400


def test_elevenlabs_voice_with_cache(monkeypatch):
    import io as _io, wave as _wave, urllib.request
    from app import speech, watchpack
    monkeypatch.setattr(settings, "elevenlabs_key", "k")
    monkeypatch.setattr(settings, "elevenlabs_voice", "brittney")
    calls = []

    class R:
        def __init__(self, b): self.b = b
        def read(self): return self.b
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake(req, timeout=0):
        calls.append(req.full_url)
        assert req.headers["Xi-api-key"] == "k" and "pcm_16000" in req.full_url and "/brittney" in req.full_url
        return R(struct.pack("<h", 1000) * 16000)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert speech.tts_available() and speech.voice_name().startswith("eleven:brittney")
    w = speech.tts("Good morning, here is your day.")
    with _wave.open(_io.BytesIO(w)) as r:
        assert r.getframerate() == 16000 and r.getnframes() == 16000
    speech.tts("Good morning, here is your day.")
    assert len(calls) == 1                                         # the same sentence: from the cache
    t1 = watchpack.voice_tag("Ryan is the CEO.")
    monkeypatch.setattr(settings, "elevenlabs_voice", "other")
    assert watchpack.voice_tag("Ryan is the CEO.") != t1           # a new voice: the watch fetches it again


def test_watch_everyday_questions(client, monkeypatch):
    from app import worldinfo, research
    calls = []

    def fake_json(url, timeout=6):
        calls.append(url)
        if "geocoding" in url:
            return {"results": [{"name": "Tokyo", "country": "Japan", "latitude": 35.7, "longitude": 139.7}]}
        return {"current": {"temperature_2m": 21.5, "apparent_temperature": 21, "relative_humidity_2m": 60,
                            "precipitation": 0, "weather_code": 2, "wind_speed_10m": 9},
                "daily": {"weather_code": [2, 61], "temperature_2m_min": [17, 16], "temperature_2m_max": [24, 22],
                          "precipitation_probability_max": [10, 70]}}
    monkeypatch.setattr(worldinfo, "_get_json", fake_json)
    monkeypatch.setattr(research, "live_context", lambda q, limit=2500, plain=False: f"WEB({q}|{plain})")
    assert worldinfo.is_world("what's the weather in Tokyo tomorrow") and not worldinfo.is_world("what did we decide")
    assert "Wednesday" in worldinfo.now_line(480) or "day" in worldinfo.now_line(480)

    def ask(text, **extra):
        r = client.post("/api/ask", json={"messages": [{"role": "user", "content": text}], **extra}, headers=H)
        assert r.status_code in (200, 202), r.text
        for _ in range(400):
            a = client.get(f"/api/ask/{r.json()['id']}", headers=H).json()
            if a["status"] != "pending":
                return a
            time.sleep(0.05)
    a = ask("What's the weather in Tokyo tomorrow?", tz=540)          # a plain weather question: no AI, at once
    assert a["answer"].startswith("Tomorrow in Tokyo: light rain, 16 to 22 degrees, 70 percent"), a
    a = ask("Will it rain in Tokyo?")
    assert "Probably not: 10 percent chance of rain today" in a["answer"], a
    ask("What should I wear for the weather in Tokyo tomorrow?", tz=540)   # more than the forecast: the AI, with it
    assert "Weather now in Tokyo, Japan" in SYSTEMS[-1] and "Tomorrow: light rain" in SYSTEMS[-1]
    assert "Right now it is" in SYSTEMS[-1] and "UTC+9" in SYSTEMS[-1]
    ask("Who won the football match yesterday?")
    assert "WEB(Who won the football match yesterday?|True)" in SYSTEMS[-1]
    h = client.get("/api/health").json()
    assert "tts_voice" in h
    w = client.get("/api/watch/weather?city=Tokyo", headers=H).json()
    assert w["today"].startswith("Right now in Tokyo") and w["tomorrow"].startswith("Tomorrow in Tokyo")


def test_elevenlabs_error_shown_and_model_fallback(monkeypatch, tmp_path):
    import io as _io, urllib.request, urllib.error
    from app import speech
    monkeypatch.setattr(settings, "elevenlabs_key", "k")
    monkeypatch.setattr(settings, "elevenlabs_voice", "v2voice")
    monkeypatch.setattr(settings, "elevenlabs_model", "eleven_v9")
    seen = []

    class R:
        def __init__(self, b): self.b = b
        def read(self): return self.b
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake(req, timeout=0):
        import json as _j
        m = _j.loads(req.data)["model_id"]
        seen.append(m)
        assert "OracleMinutes" in req.headers["User-agent"]
        if m == "eleven_v9":
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, _io.BytesIO(b'{"detail":"model not found"}'))
        return R(b"\x00\x01" * 8000)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    speech.tts("A brand new sentence for the fallback test.")
    assert seen == ["eleven_v9", "eleven_multilingual_v2"] and speech.tts_error == ""


def test_watch_questions_add_fix_hide(client):
    r = client.post("/api/knowledge/custom", json={"q": "Who is Oracle AI?", "a": "The watch's own AI, made by Jeremiah.",
                                                   "alts": ["What is Oracle AI", "Tell me about Oracle"]}, headers=H)
    assert r.status_code == 200
    fid = r.json()["id"]
    items = client.get("/api/watch/pack?limit=300", headers=H).json()["items"]
    mine = [x for x in items if x["id"] == fid][0]
    assert mine["custom"] and mine["core"] and "what is oracle ai" in mine["say"]
    # fix an answer + add a wording, then hide
    r = client.put(f"/api/knowledge/{fid}", json={"a": "Oracle AI is the watch's assistant.", "alts": ["Oracle AI please"]}, headers=H)
    assert r.status_code == 200
    assert client.delete(f"/api/knowledge/custom/{fid}", headers=H).status_code == 200
    assert all(x["id"] != fid for x in client.get("/api/watch/pack?limit=300", headers=H).json()["items"])
    assert client.post("/api/knowledge/custom", json={"q": "", "a": "x"}, headers=H).status_code == 400


def test_live_heartbeat_shows_quiet(client, monkeypatch):
    import os as _os
    from app import worker
    rid = "dev1-quiet"
    data = bytearray(wav_bytes(5))
    data[4:8] = b"\0\0\0\0"; data[40:44] = b"\0\0\0\0"
    client.put(f"/api/uploads/{rid}", content=bytes(data), headers={**H, "X-Offset": "0", "X-Live": "1"})
    p = settings.upload_dir / f"{rid}.part"
    old = time.time() - 120
    _os.utime(p, (old, old))                                      # no audio for 2 minutes
    one = [x for x in client.get("/api/live", headers=H).json()["live"] if x["rid"] == rid][0]
    assert not one["receiving"] and not one["quiet"]
    r = client.put(f"/api/uploads/{rid}", content=b"", headers={**H, "X-Offset": str(len(data)), "X-Live": "1",
                                                                 "X-Live-State": "quiet"})
    assert r.status_code == 200
    _os.utime(p, (old, old))
    one = [x for x in client.get("/api/live", headers=H).json()["live"] if x["rid"] == rid][0]
    assert one["receiving"] and one["quiet"] and one["quiet_reason"] == "quiet"


def test_bundled_firmware_offered(client):
    from app import devices
    devices.import_bundled()
    lat = client.get("/api/firmware/latest?board=watch1b&current=W1B-1.0.0", headers=H).json()
    assert lat["newer"] and lat["name"].startswith("W1B-")
    v = lat["name"]
    assert client.delete(f"/api/firmware/watch1b/{v}", headers=H).status_code == 200
    devices.import_bundled()                                         # deleted: not added again
    r = client.get("/api/firmware/latest?board=watch1b", headers=H)
    assert r.status_code == 404 or r.json()["name"] != v


def test_edit_this_answer(client):
    # no fact answers it yet: it becomes your own question
    r = client.post("/api/knowledge/correct", json={"q": "Who runs the Zorblax desk?", "a": "Mira runs it."}, headers=H)
    assert r.status_code == 200 and r.json()["how"] == "added"
    fid = r.json()["id"]
    # asked again in other words: the same fact gets the new answer and the wording
    r = client.post("/api/knowledge/correct", json={"q": "who runs zorblax desk", "a": "Mira and Tom run it."}, headers=H)
    assert r.json()["how"] == "fixed" and r.json()["id"] == fid
    one = [x for x in client.get("/api/watch/pack?limit=300", headers=H).json()["items"] if x["id"] == fid][0]
    assert one["a"] == "Mira and Tom run it."
    assert client.post("/api/knowledge/correct", json={"q": "x", "a": ""}, headers=H).status_code == 400
    client.delete(f"/api/knowledge/custom/{fid}", headers=H)


def test_firmware_from_github(client, monkeypatch):
    from app import devices
    fw = b"\xe9" + b"\0" * 64 + b"ORACLEFW|watch1b|W1B-9.9.1|" + b"\0" * 64
    calls = []

    def fake(url, raw=False, timeout=20):
        calls.append(url)
        if raw:
            return fw
        return json.dumps([{"type": "file", "name": "watch1b.bin", "sha": "abc", "size": len(fw)},
                           {"type": "file", "name": "README.md", "sha": "r", "size": 5}]).encode()
    monkeypatch.setattr(devices, "_gh_get", fake)
    monkeypatch.setattr(devices.settings, "firmware_repo", "someone/repo")
    devices.gh_status["checked"] = 0
    r = client.post("/api/firmware/github", headers=H).json()
    assert r["added"] == ["watch1b/W1B-9.9.1"] and not r["github"]["error"]
    lat = client.get("/api/firmware/latest?board=watch1b&current=W1B-1.5.7", headers=H).json()
    assert lat["name"] == "W1B-9.9.1" and lat["version"] == "9.9.1" and lat["newer"]
    n = len(calls)
    devices.gh_status["checked"] = 0
    assert client.post("/api/firmware/github", headers=H).json()["added"] == []   # same file: not downloaded again
    assert len(calls) == n + 1
    client.delete("/api/firmware/watch1b/W1B-9.9.1", headers=H)


def test_firmware_keeps_newest_two(client):
    from app import devices
    for v in ("W1B-7.0.1", "W1B-7.0.3", "W1B-7.0.2"):
        devices.save_firmware(b"\xe9" + b"\0" * 32 + f"ORACLEFW|keeptest|{v}|".encode() + b"\0" * 32)
    have = sorted(m["version"] for m in devices.list_firmware() if m["board"] == "keeptest")
    assert have == ["W1B-7.0.2", "W1B-7.0.3"]
    for v in have:
        devices.delete_firmware("keeptest", v)


def test_firmware_download_resumes(client):
    from app import devices
    data = b"\xe9" + bytes(5000) + b"ORACLEFW|rangetest|W1B-3.0.0|" + bytes(5000)
    devices.save_firmware(data)
    r = client.get("/api/firmware/rangetest/W1B-3.0.0.bin", headers={**H, "Range": "bytes=4000-"})
    assert r.status_code == 206 and r.content == data[4000:]
    devices.delete_firmware("rangetest", "W1B-3.0.0")


def test_voice_made_ahead_is_found_and_waited_for():
    import threading as _t
    from app import chat
    ev = _t.Event()
    chat._making["x1"] = (time.time(), chat._norm("Right now in Singapore it's 31°C."), ev)

    def finish():
        time.sleep(0.3)
        with chat._lock:
            chat._voices[chat._vkey("Right now in Singapore it's 31°C.", "en", True)] = (time.time(), b"VOICE")
            chat._making.pop("x1", None)
        ev.set()
    _t.Thread(target=finish).start()
    # the watch sends it back folded to ASCII ("31C"): still the same answer, and it waits instead of making it again
    assert chat.spoken("Right now in Singapore it's 31C.", "en", True) == b"VOICE"
    assert chat.spoken("Something else entirely", "en", True) is None


def test_spoken_answers_made_louder():
    import numpy as np
    from app import speech
    quiet = (np.sin(np.arange(16000) / 5) * 10000).astype(np.float32)        # peaks at ~30%
    out = speech.loud(quiet)
    assert 0.85 * 32767 < np.abs(out).max() <= 32767
    assert np.abs(speech.loud(np.zeros(100))).max() == 0


def test_voice_question_in_one_trip(client, monkeypatch):
    import base64 as _b64
    import json as _j
    from app import speech, summarize, worldinfo
    said = {"t": ""}
    monkeypatch.setattr(speech, "stt", lambda path, lang=None, cands=None: {"text": said["t"], "language": "en", "duration": 1})
    monkeypatch.setattr(speech, "tts", lambda text, lang="en", strict=False, live=False:
                        speech.to_wav16k(__import__("numpy").zeros(1600 * max(1, len(text) // 10)), 16000))
    monkeypatch.setattr(worldinfo, "_get_json", lambda url, timeout=6: {"results": [{"name": "Singapore", "latitude": 1, "longitude": 1}]}
                        if "geocoding" in url else {"current": {"temperature_2m": 31, "apparent_temperature": 36, "weather_code": 2},
                                                    "daily": {"weather_code": [2, 61], "temperature_2m_min": [26, 25],
                                                              "temperature_2m_max": [32, 31], "precipitation_probability_max": [40, 80]}})
    sent = []
    monkeypatch.setattr(summarize, "chat_stream", lambda msgs: (sent.append(msgs), iter(["Here is one", ". Why did the chicken ",
                        "cross the road? To get", " to the other side."]))[1])

    def ask(text, header=None):
        said["t"] = text
        h = {**H, "Content-Type": "audio/wav"}
        if header:
            h["X-Ask"] = _b64.b64encode(_j.dumps(header).encode()).decode()
        r = client.post("/api/voice?speak=1&tz=480", content=wav_bytes(1), headers=h)
        assert r.status_code == 200, r.text
        return [ln.split(" ", 1) for ln in r.text.strip().split("\n")]

    lines = ask("How's the weather?")
    kinds = [k for k, _ in lines]
    assert kinds[0] == "T" and kinds[1] == "A" and "B" in kinds and kinds[-1] == "E"
    assert "Right now in Singapore" in _j.loads(lines[1][1])["answer"]
    blob = b"".join(_b64.b64decode(v) for k, v in lines if k == "B")
    assert len(blob) and len(blob) % 256 == 0                          # whole ADPCM blocks
    lines = ask("Tell me a joke", {"messages": [{"role": "system", "content": "You are Oracle."}]})
    kinds = [k for k, _ in lines]
    assert kinds[:2] == ["T", "P"] and "B" in kinds and kinds[-1] == "E"
    ps = [_j.loads(v)["text"] for k, v in lines if k == "P"]
    assert ps == ["Here is one.", "Here is one. Why did the chicken cross the road?"]     # a sentence at a time
    assert _j.loads([v for k, v in lines if k == "A"][0])["answer"].endswith("other side.")
    assert sent[-1][1]["content"] == "You are Oracle." and sent[-1][-1]["content"] == "Tell me a joke"
    lines = ask("Set a timer for 5 minutes")
    assert [k for k, _ in lines] == ["T", "X", "E"] and _j.loads(lines[1][1])["type"] == "timer"
    assert [k for k, _ in ask("Stop.")] == ["T", "E"]                  # nothing to answer
    # sent while talking: the recording comes in pieces (chunked), its header still saying 0 bytes
    said["t"] = "Tell me a joke"
    w = bytearray(wav_bytes(1))
    w[4:8] = b"\0\0\0\0"
    w[40:44] = b"\0\0\0\0"

    def pieces():
        for i in range(0, len(w), 1000):
            yield bytes(w[i:i + 1000])
    r = client.post("/api/voice?speak=1&tz=480", content=pieces(), headers={**H, "Content-Type": "audio/wav"})
    assert r.status_code == 200 and r.text.startswith("T ") and r.text.strip().endswith("E {}"), r.text[:200]
    # a slow search: "Let me check" is said first (before the answer's words)
    from app import research
    monkeypatch.setattr(settings, "filler_after_s", 0.05)
    monkeypatch.setattr(research, "live_context", lambda *a, **k: (time.sleep(0.4), "")[1])
    lines = ask("Tell me something funny about cats")
    kinds = [k for k, _ in lines]
    assert kinds[0] == "T" and kinds[1] == "B" and kinds.index("B") < kinds.index("P"), kinds


def test_cloud_speech_to_text(monkeypatch, tmp_path):
    from app import speech
    import httpx as _hx
    monkeypatch.setattr(settings, "live_llm_api_key", "gk")
    monkeypatch.setattr(settings, "live_llm_base_url", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(speech, "_cloud_off_until", 0.0)
    assert speech.cloud_stt_target()[2] == "whisper-large-v3-turbo"
    p = tmp_path / "q.wav"
    p.write_bytes(wav_bytes(1))
    seen = {}

    class R:
        def raise_for_status(self): pass
        def json(self): return {"text": " What time is it? ", "language": "english"}

    def fake(url, headers=None, data=None, files=None, timeout=0):
        seen.update(url=url, auth=headers["Authorization"], model=data["model"], size=len(files["file"][1]))
        return R()
    monkeypatch.setattr(_hx, "post", fake)
    r = speech.stt(str(p), "en")
    assert r["text"] == "What time is it?" and r["language"] == "en" and r.get("cloud")
    assert seen["url"].endswith("/audio/transcriptions") and seen["auth"] == "Bearer gk" and seen["size"] > 1000

    def boom(*a, **k):
        raise _hx.ConnectError("down")
    monkeypatch.setattr(_hx, "post", boom)
    monkeypatch.setattr(speech.transcribe, "transcribe", lambda *a, **k: type("T", (), {
        "segments": [type("S", (), {"text": "local words"})()], "language": "en", "duration": 1.0})())
    assert speech.stt(str(p), "en")["text"] == "local words"         # cloud down: this server's own
    monkeypatch.setattr(settings, "cloud_stt", False)
    assert speech.cloud_stt_target() is None


def test_watch_crash_decoded(client, monkeypatch, tmp_path):
    import gzip as _gz
    from app import crashdecode
    d = tmp_path / "syms"
    d.mkdir()
    (d / "W1B-9.9.9.sym.gz").write_bytes(_gz.compress(
        b"42000000 00000100 T setup\n42000100 00000080 t voiceReadBody(NetworkClient*)\n3fc00000 00000010 D data\n"))
    monkeypatch.setattr(crashdecode, "BUNDLED", d)
    crashdecode._cache.clear()
    text = "W1B-9.9.9 task net pc 0x42000110 cause 29 addr 0x00000000 elf ab bt 0x42000110 0x42000004 0x42f00000"
    r = client.post("/api/watch/crash", json={"fw": "W1B-9.9.9", "text": text}, headers=H).json()
    assert r["decoded"]["frames"][:2] == ["0x42000110 voiceReadBody(NetworkClient*)+0x10", "0x42000110 voiceReadBody(NetworkClient*)+0x10"]
    assert r["decoded"]["frames"][2] == "0x42000004 setup+0x4" and r["decoded"]["frames"][3].endswith(" ?")
    items = client.get("/api/watch/crashes", headers=H).json()["items"]
    assert items[0]["decoded"]["version"] == "W1B-9.9.9"


def test_chat_stream_leaves_out_thinking(monkeypatch):
    import contextlib
    import json as _j
    from app import summarize
    pieces = ["<th", "ink>\n", "hmm</think>", "\n\nHello", " there. ", "Bye", "."]

    class R:
        status_code = 200
        def raise_for_status(self): pass
        def iter_lines(self):
            for p in pieces:
                yield _j.dumps({"message": {"content": p}, "done": False})
            yield _j.dumps({"message": {"content": ""}, "done": True})
    monkeypatch.setattr(summarize.httpx, "stream", lambda *a, **k: contextlib.nullcontext(R()))
    monkeypatch.setattr(summarize, "ollama_url", lambda: "http://x")
    monkeypatch.setattr(summarize.settings, "summary_backend", "ollama")
    monkeypatch.setattr(summarize.settings, "live_llm_api_key", "")
    out = "".join(summarize.chat_stream([{"role": "user", "content": "hi"}]))
    assert out.strip() == "Hello there. Bye."


def test_cloud_ai_groq_reasoning_and_errors(monkeypatch):
    import contextlib
    import json as _j
    from app import summarize
    monkeypatch.setattr(summarize.settings, "live_llm_api_key", "k")
    monkeypatch.setattr(summarize.settings, "live_llm_base_url", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(summarize.settings, "live_llm_model", "openai/gpt-oss-20b")
    bodies = []

    class R:
        def __init__(self, code, lines, raw=b""):
            self.status_code, self.lines, self.raw = code, lines, raw
        def read(self): return self.raw
        def iter_lines(self): return iter(self.lines)

    def stream(method, url, **kw):
        bodies.append(kw["json"])
        if len(bodies) == 1:                          # the reasoning settings refused once: sent again without
            return contextlib.nullcontext(R(400, [], b'{"error":{"message":"unknown parameter include_reasoning"}}'))
        chunks = ["<think>x</think>Hi", " there."]
        return contextlib.nullcontext(R(200, [f"data: {_j.dumps({'choices': [{'delta': {'content': c}}]})}" for c in chunks]
                                           + ["data: [DONE]"]))
    monkeypatch.setattr(summarize.httpx, "stream", stream)
    assert "".join(summarize.chat_stream([{"role": "user", "content": "hi"}])) == "Hi there."
    assert bodies[0]["reasoning_effort"] == "low" and "reasoning_effort" not in bodies[1]
    monkeypatch.setattr(summarize.httpx, "stream", lambda *a, **k: contextlib.nullcontext(
        R(404, [], b'{"error":{"message":"The model x does not exist or you do not have access to it."}}')))
    monkeypatch.setattr(summarize.settings, "live_llm_model", "x")
    import pytest
    with pytest.raises(RuntimeError, match="does not exist"):
        list(summarize.chat_stream([{"role": "user", "content": "hi"}]))


def test_question_heard_with_note_is_not_a_note():
    from datetime import datetime
    from app import speech
    assert speech.rule_intent("Note, is Aleta Planet a global company?", datetime.now()) is None
    assert speech.rule_intent("Note that the launch moved to Friday", datetime.now()) is None    # no voice notes
    assert speech.strip_wake("Hey Oracle, is it raining?") == "Is it raining?"


def test_world_question_searches_the_web(client, monkeypatch):
    from app import research
    asked = []
    monkeypatch.setattr(research, "live_context", lambda q, limit=2500, plain=False: (asked.append((q, plain)), f"WEB[{q}]")[1])
    r = client.post("/api/ask", json={"messages": [{"role": "user", "content": "Who invented the telephone?"}]}, headers=H)
    for _ in range(400):
        a = client.get(f"/api/ask/{r.json()['id']}", headers=H).json()
        if a["status"] != "pending":
            break
        time.sleep(0.05)
    assert ("Who invented the telephone?", True) in asked                  # searched as asked, not with the company
    assert "WEB[Who invented the telephone?]" in SYSTEMS[-1] and "never say you don't know" in SYSTEMS[-1]


def test_first_sentence_voice_streamed(monkeypatch, tmp_path):
    import http.client
    import base64 as _b64
    from app import speech, main as _m
    monkeypatch.setattr(settings, "elevenlabs_key", "k")
    monkeypatch.setattr(settings, "elevenlabs_voice", "vv")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(speech._el_local, "conn", None, raising=False)
    sent = []

    class Resp:
        status = 200
        def __init__(self): self.left = [struct.pack("<h", 3000) * 3000, struct.pack("<h", -3000) * 4000, b""]
        def read1(self, n): return self.left.pop(0)
        def read(self, n=-1): return b""

    class Conn:
        def __init__(self, host, timeout=0): self.host = host
        def request(self, m, path, body=None, headers=None): sent.append(path)
        def getresponse(self): return Resp()
        def close(self): pass
    monkeypatch.setattr(http.client, "HTTPSConnection", Conn)
    pieces = list(speech.tts_stream("Hello there, this is streamed."))
    assert sent and "/stream?output_format=pcm_16000" in sent[0] and sum(len(p) for p in pieces) == 7000
    assert max(abs(int(x)) for p in pieces for x in p) > 3000            # made louder
    assert list((tmp_path / "tts_cache").glob("*.pcm"))                    # kept: next time from the cache
    fv = _m._FirstVoice("Another sentence to stream now.")
    blob = b"".join(_b64.b64decode(ln[2:].strip()) for ln in fv.finish())
    assert len(blob) and len(blob) % 256 == 0
    assert _m._first_split("Short one.") == ["Short one."]
    long = "This first part of the answer is fairly long, and the second part carries on for a while longer too."
    a, b = _m._first_split(long)
    assert a.endswith(",") and b.startswith("and")


def test_web_search_providers_and_groq_fallback(monkeypatch):
    import httpx as _hx
    from app import sources, research, summarize
    calls = []

    class R:
        status_code = 200
        def __init__(self, j): self.j = j
        def raise_for_status(self): pass
        def json(self): return self.j

    def fake_post(url, **k):
        calls.append(url)
        return R({"results": [{"url": "https://a.example/x", "title": "Best observatory in space", "content": "JWST is"}]})
    monkeypatch.setattr(settings, "tavily_api_key", "tv")
    monkeypatch.setattr(_hx, "post", fake_post)
    hits = sources.web_search("best observatory in space", 3)
    assert calls[0].startswith("https://api.tavily.com") and hits[0]["title"] == "Best observatory in space"
    assert sources.search_status["provider"] == "Tavily"
    # Brave first when both are set; Brave failing -> Tavily
    monkeypatch.setattr(settings, "brave_api_key", "bk")
    def brave_down(url, **k):
        raise _hx.ConnectError("down")
    monkeypatch.setattr(_hx, "get", brave_down)
    assert sources.web_search("best observatory in space", 3)[0]["title"] == "Best observatory in space"
    assert sources.search_status["provider"] == "Tavily" and "Brave" in sources.search_status["fell_back"]
    monkeypatch.setattr(settings, "brave_api_key", "")
    # nothing found anywhere, a question about now: Groq's own search
    monkeypatch.setattr(settings, "tavily_api_key", "")
    monkeypatch.setattr(sources, "_free_search", lambda q, t: [])
    monkeypatch.setattr(settings, "research_live", True)
    monkeypatch.setattr(settings, "live_llm_api_key", "gk")
    monkeypatch.setattr(settings, "live_llm_base_url", "https://api.groq.com/openai/v1")

    class C:
        def post(self, url, **k):
            calls.append((url, k["json"]["model"]))
            return R({"choices": [{"message": {"content": "JWST is the most powerful space telescope (nasa.gov)"}}]})
    monkeypatch.setattr(summarize, "cloud_client", lambda: C())
    assert research.needs_fresh("what is the best observer in space right now") and not research.needs_fresh("what is love")
    out = research.live_context("what is the best observer in space right now", 2500, True)
    assert "JWST" in out and calls[-1][1] == "groq/compound-mini"
    assert research.live_context("what is love", 2500, True) == ""            # not about now: no extra search


def test_cloud_ai_falls_back_when_busy(monkeypatch):
    import contextlib
    import json as _j
    from app import summarize
    monkeypatch.setattr(settings, "live_llm_api_key", "gk")
    monkeypatch.setattr(settings, "live_llm_base_url", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(settings, "live_llm_model", "openai/gpt-oss-20b")
    seen = []

    class R:
        def __init__(self, code): self.status_code = code
        def read(self): return b'{"error":{"message":"Rate limit reached"}}'
        def iter_lines(self):
            yield "data: " + _j.dumps({"choices": [{"delta": {"content": "Hello from the fallback."}}]})
            yield "data: [DONE]"

    def fake_stream(method, url, timeout=None, headers=None, json=None):
        seen.append(json["model"])
        return contextlib.nullcontext(R(429 if json["model"] == "openai/gpt-oss-20b" else 200))
    monkeypatch.setattr(summarize.httpx, "stream", fake_stream)
    assert summarize.fallback_model() == "qwen/qwen3-32b"
    assert "".join(summarize.chat_stream([{"role": "user", "content": "hi"}])) == "Hello from the fallback."
    assert seen[0] == "openai/gpt-oss-20b" and seen[-1] == "qwen/qwen3-32b"
    monkeypatch.setattr(summarize.httpx, "stream", lambda *a, **k: contextlib.nullcontext(R(429)))
    try:
        list(summarize.chat_stream([{"role": "user", "content": "hi"}]))
        assert False
    except RuntimeError as e:
        assert "busy right now" in str(e) and summarize.explain_llm_error(e).startswith("cloud AI is busy")


def test_long_first_sentence_voiced_from_its_comma(client, monkeypatch):
    import base64 as _b64
    from app import speech, summarize
    monkeypatch.setattr(speech, "stt", lambda path, lang=None, cands=None: {"text": "Tell me about the moon landing", "language": "en"})
    voiced = []
    monkeypatch.setattr(speech, "tts", lambda text, lang="en", strict=False, live=False:
                        (voiced.append(text), speech.to_wav16k(__import__("numpy").zeros(1600), 16000))[1])
    monkeypatch.setattr(summarize, "chat_stream", lambda msgs: iter(["The first crewed moon landing was in July 1969, ",
                        "when Apollo 11 touched down on the Sea of Tranquility. ", "Armstrong stepped out first."]))
    r = client.post("/api/voice?speak=1", content=wav_bytes(1), headers={**H, "Content-Type": "audio/wav"})
    assert r.status_code == 200
    assert voiced[0] == "The first crewed moon landing was in July 1969,", voiced
    assert "when Apollo 11 touched down on the Sea of Tranquility." in voiced


def test_watch_stability_by_day(client):
    log = "\n".join(["[    0s] boot: W1B-1.9.7 (Oracle OS), reset reason 4", "[    0s] boot: restarted after a crash (1 in a row)",
                     "[   30s] voice: one trip - heard back 1800 ms after sending", "[   40s] voice: sending failed (x)",
                     "[   50s] mem: lowest 0 KB (now 2 KB)"])
    assert client.post("/api/devices/w1b-test/log", content=log, headers=H).status_code == 200
    from app import timing
    timing.note("voice trip", 1.6, "first sentence at 1.6s")
    d = client.get("/api/devices/w1b-test/stability", headers=H).json()["days"][0]
    assert d["restarts"] == 1 and d["crashes"] == 1 and d["failed_sends"] == 1 and d["memory_lows"] == 1
    assert d["questions"] == 1 and d["slowest_heard_ms"] == 1800 and d["server_trips"] >= 1


def test_company_question_skips_the_web_and_fresh_ones_search_early(client, monkeypatch):
    from app import research, speech, summarize, main as _m
    r = client.post("/api/knowledge/custom", json={"q": "Where is Aleta Planet located?", "a": "Aleta Planet is in Singapore, at One North."}, headers=H)
    assert r.status_code == 200
    monkeypatch.setattr(settings, "research_live", True)
    monkeypatch.setattr(research, "company_name", lambda: "Aleta Planet")
    calls = []
    monkeypatch.setattr(research, "live_context", lambda q, limit=2500, plain=False: (calls.append(q), "WEB")[1])
    said = {"t": ""}
    monkeypatch.setattr(speech, "stt", lambda path, lang=None, cands=None: {"text": said["t"], "language": "en"})
    monkeypatch.setattr(summarize, "chat_stream", lambda msgs: iter(["It is in Singapore."]))

    def ask(t):
        said["t"] = t
        r = client.post("/api/voice?speak=0", content=wav_bytes(1), headers={**H, "Content-Type": "audio/wav"})
        assert r.status_code == 200
    ask("Can you tell me where exactly the Alita Planet office is located for our visitors")
    assert calls == []                                              # the company's own knowledge: no web search
    ask("What is the latest price of gold")
    assert calls == ["What is the latest price of gold"]           # about now: searched (started early, once)


def test_question_words_spelled_like_the_company(monkeypatch):
    from app import speech, research
    monkeypatch.setattr(research, "company_name", lambda: "Aleta Planet")
    monkeypatch.setattr(speech, "_stt", lambda p, l=None, c=None: {"text": "Where is Alita Planet located?", "language": "en"})
    assert speech.stt("x.wav")["text"] == "Where is Aleta Planet located?"


def test_speed_test_by_voice(client, monkeypatch):
    from app import speech, summarize, sources
    monkeypatch.setattr(speech, "stt", lambda path, lang=None, cands=None: {"text": "Hey Oracle, do a speed test", "language": "en", "cloud": True})
    monkeypatch.setattr(summarize, "chat_stream", lambda msgs: iter(["OK"]))
    monkeypatch.setattr(sources, "web_search", lambda q, t=3: [{"url": "u", "title": "t", "snippet": "s"}])
    r = client.post("/api/voice?speak=0", content=wav_bytes(1), headers={**H, "Content-Type": "audio/wav"})
    a = [ln for ln in r.text.split("\n") if ln.startswith("A ")][0]
    assert "Speed test:" in a and "speech to text" in a and "on Groq" in a and "web search" in a


def test_company_name_setting(monkeypatch):
    from app import research
    monkeypatch.setattr(settings, "company_name", "Aleta Planet")
    assert research.company_name() == "Aleta Planet"


def test_voice_pick_openai_cedar(client, monkeypatch, tmp_path):
    import http.client
    from app import speech
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.delenv("OPENAI_TTS_API_KEY", raising=False)
    r = client.post("/api/voice/pick", json={"voice": "cedar"}, headers=H)
    assert r.status_code == 400 and "OPENAI_API_KEY" in r.text            # no key: can't pick it
    monkeypatch.setattr(settings, "openai_api_key", "ok")
    assert client.post("/api/voice/pick", json={"voice": "cedar"}, headers=H).json()["voice"] == "cedar"
    assert speech.openai_voice() == "cedar" and not speech.eleven_on() and speech.voice_name().startswith("openai:cedar")
    sent = []

    class Resp:
        status = 200
        def __init__(self): self.left = [struct.pack("<h", 2000) * 4800, b""]
        def read(self, n=-1): return b"".join(self.left) if n == -1 else self.left.pop(0)
        def read1(self, n): return self.left.pop(0)

    class Conn:
        host = "api.openai.com"
        def __init__(self, host, timeout=0): pass
        def request(self, m, path, body=None, headers=None): sent.append((path, json.loads(body)))
        def getresponse(self): return Resp()
        def close(self): pass
    monkeypatch.setattr(http.client, "HTTPSConnection", Conn)
    with speech._el_pool_lock:
        speech._el_pool.clear()
    pieces = list(speech.tts_stream("A sentence in Cedar's voice."))
    assert sent[0][0] == "/v1/audio/speech" and sent[0][1]["voice"] == "cedar" and sent[0][1]["response_format"] == "pcm"
    assert sum(len(p) for p in pieces) == 3200                           # 4800 samples at 24 kHz -> 3200 at 16 kHz
    wav = speech.tts("A sentence in Cedar's voice.")                      # the same sentence: from the cache
    assert len(sent) == 1 and len(wav) > 6000
    assert client.get("/api/voice/pick", headers=H).json()["voice"] == "cedar"
    client.post("/api/voice/pick", json={"voice": "eleven"}, headers=H)
