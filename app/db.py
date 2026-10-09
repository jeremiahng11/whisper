"""Tiny SQLite job store. One worker thread + API threads share it behind a lock."""
import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Optional

from .config import settings

_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None

STATUSES = ("queued", "transcribing", "summarizing", "done", "error")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id            TEXT PRIMARY KEY,
  recording_id  TEXT UNIQUE,
  filename      TEXT NOT NULL,
  audio_path    TEXT,
  status        TEXT NOT NULL,
  stage         TEXT DEFAULT '',
  progress      REAL DEFAULT 0,
  error         TEXT DEFAULT '',
  summary_error TEXT DEFAULT '',
  title         TEXT DEFAULT '',
  language      TEXT DEFAULT '',
  duration      REAL DEFAULT 0,
  options       TEXT DEFAULT '{}',
  priority      INTEGER DEFAULT 0,
  created_at    REAL NOT NULL,
  updated_at    REAL NOT NULL,
  started_at    REAL,
  finished_at   REAL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS files (
  device      TEXT NOT NULL,
  path        TEXT NOT NULL,
  rev         INTEGER NOT NULL,
  hash        TEXT NOT NULL,
  size        INTEGER NOT NULL,
  deleted     INTEGER DEFAULT 0,
  content     BLOB,
  updated_at  REAL NOT NULL,
  updated_by  TEXT DEFAULT '',
  PRIMARY KEY (device, path)
);
CREATE TABLE IF NOT EXISTS voices (
  name TEXT PRIMARY KEY, embedding TEXT NOT NULL, samples INTEGER DEFAULT 1, updated_at REAL
);
CREATE VIRTUAL TABLE IF NOT EXISTS para_fts USING fts5(job_id UNINDEXED, start UNINDEXED, speaker UNINDEXED, text);
CREATE TABLE IF NOT EXISTS file_history (
  device TEXT, path TEXT, rev INTEGER, content BLOB, updated_at REAL
);
"""


def init() -> None:
    global _conn
    settings.ensure_dirs()
    with _lock:
        _conn = sqlite3.connect(settings.db_path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
        cols = [r[1] for r in _conn.execute("PRAGMA table_info(jobs)").fetchall()]
        if "priority" not in cols:                       # upgrade older databases
            _conn.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER DEFAULT 0")
        # jobs interrupted by a restart go back to the queue
        _conn.execute(
            "UPDATE jobs SET status='queued', stage='requeued after restart', progress=0 "
            "WHERE status IN ('transcribing','summarizing')"
        )
        _conn.commit()


def _row(r: Optional[sqlite3.Row]) -> Optional[dict]:
    if r is None:
        return None
    d = dict(r)
    d["options"] = json.loads(d.get("options") or "{}")
    return d


def create(filename: str, audio_path: str, options: dict, recording_id: Optional[str] = None,
           priority: int = 0) -> dict:
    now = time.time()
    jid = uuid.uuid4().hex[:12]
    with _lock:
        _conn.execute(
            "INSERT INTO jobs (id, recording_id, filename, audio_path, status, options, priority, created_at, updated_at) "
            "VALUES (?,?,?,?, 'queued', ?, ?, ?, ?)",
            (jid, recording_id, filename, audio_path, json.dumps(options), priority, now, now),
        )
        _conn.commit()
    return get(jid)


def get(jid: str) -> Optional[dict]:
    with _lock:
        return _row(_conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone())


def get_by_recording(rid: str) -> Optional[dict]:
    with _lock:
        return _row(_conn.execute("SELECT * FROM jobs WHERE recording_id=?", (rid,)).fetchone())


def list_jobs(limit: int = 100, status: Optional[str] = None) -> list[dict]:
    with _lock:
        if status:
            rows = _conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)
            ).fetchall()
        else:
            rows = _conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def update(jid: str, **fields: Any) -> None:
    if not fields:
        return
    if "options" in fields:
        fields["options"] = json.dumps(fields["options"])
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    with _lock:
        _conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), jid))
        _conn.commit()


def next_queued() -> Optional[dict]:
    with _lock:
        return _row(
            _conn.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY priority DESC, created_at LIMIT 1").fetchone()
        )


def queue_position(jid: str) -> int:
    """0 = running or not queued, 1 = next, ..."""
    with _lock:
        j = _conn.execute("SELECT status, created_at, priority FROM jobs WHERE id=?", (jid,)).fetchone()
        if not j or j["status"] != "queued":
            return 0
        return _conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='queued' AND (priority > ? OR (priority = ? AND created_at <= ?))",
            (j["priority"], j["priority"], j["created_at"]),
        ).fetchone()[0]


def counts() -> dict:
    with _lock:
        rows = _conn.execute("SELECT status, COUNT(*) c FROM jobs GROUP BY status").fetchall()
    return {r["status"]: r["c"] for r in rows}


def delete(jid: str) -> None:
    with _lock:
        _conn.execute("DELETE FROM jobs WHERE id=?", (jid,))
        _conn.commit()


def older_than(ts: float, statuses=("done", "error")) -> list[dict]:
    q = ",".join("?" * len(statuses))
    with _lock:
        rows = _conn.execute(
            f"SELECT * FROM jobs WHERE status IN ({q}) AND COALESCE(finished_at, updated_at) < ?", (*statuses, ts)
        ).fetchall()
    return [_row(r) for r in rows]


# ---------------------------------------------------------------- synced text files (notes backup)
def _frow(r) -> Optional[dict]:
    if r is None:
        return None
    d = dict(r)
    d.pop("content", None)
    d["deleted"] = bool(d["deleted"])
    return d


def file_list(device: str) -> list[dict]:
    with _lock:
        rows = _conn.execute("SELECT * FROM files WHERE device=? ORDER BY path", (device,)).fetchall()
    return [_frow(r) for r in rows]


def file_devices() -> list[str]:
    with _lock:
        return [r[0] for r in _conn.execute("SELECT DISTINCT device FROM files ORDER BY device").fetchall()]


def file_get(device: str, path: str) -> tuple[Optional[dict], Optional[bytes]]:
    with _lock:
        r = _conn.execute("SELECT * FROM files WHERE device=? AND path=?", (device, path)).fetchone()
    if r is None:
        return None, None
    return _frow(r), (None if r["deleted"] else bytes(r["content"] or b""))


def file_put(device: str, path: str, content: Optional[bytes], hsh: str, by: str, keep_history: int = 20) -> dict:
    """Write (content) or delete (None) a file; returns the new row. Caller checks base_rev first."""
    now = time.time()
    with _lock:
        old = _conn.execute("SELECT rev, content, deleted FROM files WHERE device=? AND path=?", (device, path)).fetchone()
        rev = (old["rev"] if old else 0) + 1
        if old and not old["deleted"]:
            _conn.execute("INSERT INTO file_history VALUES (?,?,?,?,?)", (device, path, old["rev"], old["content"], now))
            _conn.execute(
                "DELETE FROM file_history WHERE device=? AND path=? AND rev NOT IN "
                "(SELECT rev FROM file_history WHERE device=? AND path=? ORDER BY rev DESC LIMIT ?)",
                (device, path, device, path, keep_history))
        _conn.execute(
            "INSERT OR REPLACE INTO files (device, path, rev, hash, size, deleted, content, updated_at, updated_by) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (device, path, rev, hsh if content is not None else "", len(content or b""), 0 if content is not None else 1,
             content, now, by))
        _conn.commit()
    return file_get(device, path)[0]
