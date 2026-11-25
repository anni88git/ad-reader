"""Operational database (jobs, provider usage, worker heartbeats) - separate from the master data.

SQLite in WAL mode: several processes (Streamlit app + one or more workers) can use it safely.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import DATA_DIR

OPS_DB = DATA_DIR / "db" / "ops.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,                 -- queued | processing | retrying | completed | failed
    stage TEXT,                           -- ocr | extract | verify | validate | done
    name TEXT NOT NULL,
    input_path TEXT NOT NULL,             -- preserved copy of the uploaded image
    dedupe_key TEXT,                      -- same image + options -> same job (refresh-safe)
    options TEXT NOT NULL DEFAULT '{}',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 4,
    next_attempt_at REAL,                 -- unix time; NULL = now
    last_error TEXT,
    error_kind TEXT,                      -- transient | permanent | all_providers_failed
    worker TEXT,
    heartbeat_at REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_jobs_dedupe ON jobs(dedupe_key);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    role TEXT,                            -- extract | verify | other
    job_id INTEGER,
    ok INTEGER NOT NULL,
    error_type TEXT,
    http_status INTEGER,
    latency_s REAL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL
);
CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage(provider, ts);
CREATE TABLE IF NOT EXISTS workers (name TEXT PRIMARY KEY, pid INTEGER, last_seen REAL);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = Path(path or OPS_DB)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=30, isolation_level=None)   # autocommit; explicit BEGIN where needed
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    return conn
