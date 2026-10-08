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
