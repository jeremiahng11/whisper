"""Full-text search over all transcripts and minutes (SQLite FTS5), used by "ask across meetings"."""
import re

from . import db, render

STOP = set("""a an and are as at be been but by can could did do does for from had has have how i if in into is it its
me my no not of on or our so than that the their them then there these they this to was we were what when where which
who why will with would you your about any all also just like more most some such us very shall should may might
tell show give find said say says meeting meetings did does""".split())


def index(jid: str, summary: str, paragraphs: list[dict]) -> None:
    with db._lock:
        db._conn.execute("DELETE FROM para_fts WHERE job_id=?", (jid,))
        rows = [(jid, -1.0, "", summary)] if summary else []
        rows += [(jid, float(p["start"]), p.get("speaker", ""), p["text"]) for p in paragraphs if p.get("text")]
        db._conn.executemany("INSERT INTO para_fts (job_id, start, speaker, text) VALUES (?,?,?,?)", rows)
        db._conn.commit()


def remove(jid: str) -> None:
    with db._lock:
        db._conn.execute("DELETE FROM para_fts WHERE job_id=?", (jid,))
        db._conn.commit()


def indexed_jobs() -> set[str]:
    with db._lock:
        return {r[0] for r in db._conn.execute("SELECT DISTINCT job_id FROM para_fts").fetchall()}


def query_terms(q: str) -> list[str]:
    words = re.findall(r"[\w']+", q.lower())
    return [w.strip("'") for w in words if len(w) > 1 and w not in STOP][:12]


def search(q: str, limit: int = 12) -> list[dict]:
    terms = query_terms(q)
    if not terms:
        return []
    fts = " OR ".join('"' + t.replace('"', "") + '"*' for t in terms)
    with db._lock:
        rows = db._conn.execute(
            "SELECT job_id, start, speaker, text, bm25(para_fts) AS score FROM para_fts WHERE para_fts MATCH ? "
            "ORDER BY score LIMIT ?", (fts, limit * 3)).fetchall()
    out, per_job = [], {}
    for r in rows:                                  # at most 4 hits per meeting, so others get a turn
        if per_job.get(r["job_id"], 0) >= 4:
            continue
        j = db.get(r["job_id"])
        if not j or j["status"] != "done":
            continue
        per_job[r["job_id"]] = per_job.get(r["job_id"], 0) + 1
        out.append({"job_id": r["job_id"], "recording_id": j.get("recording_id"), "title": j.get("title") or j["filename"],
                    "date": render.recorded_at(j["filename"], j["created_at"]).strftime("%Y-%m-%d"),
                    "start": r["start"], "speaker": r["speaker"], "text": r["text"]})
        if len(out) >= limit:
            break
    return out
