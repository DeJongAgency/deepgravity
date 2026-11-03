"""store.py — dev-scoped SQLite helper (the pre-flip bootstrap store).

Per the bootstrap-store decision (2026-09-07): before the flip imports the
real braid, the engine's tool_registry + telemetry live in a dev-scoped
SQLite store under data/devstore/. The seam: when the real braid lands at
~/.local/share/deepgrav/user/{occupant}/data/braid/ (B13), the engine points
at it and this store retires.

Operational rule honored: one connection per call, close fast, busy_timeout
set — never a lingering handle on a live store.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional, Union

DEFAULT_DEVSTORE = Path(os.environ.get(
    "DG_DEVSTORE",
    "/srv/dg-dev/data/devstore/devstore.db",
))


def _conn(db: Union[str, Path] = DEFAULT_DEVSTORE) -> sqlite3.Connection:
    p = Path(db)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=5.0)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_schema(db: Union[str, Path] = DEFAULT_DEVSTORE) -> None:
    """Create the dev-store tables if absent. Idempotent; called at engine start."""
    conn = _conn(db)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tool_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                layer TEXT NOT NULL DEFAULT 'dev',
                domain TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                argv_json TEXT NOT NULL,
                param_types_json TEXT NOT NULL DEFAULT '{}',
                cwd TEXT,
                timeout REAL NOT NULL DEFAULT 60.0
            );
            CREATE TABLE IF NOT EXISTS milestones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                event TEXT NOT NULL,
                detail_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_milestones_ts ON milestones(ts_utc);
            """
        )
        conn.commit()
    finally:
        conn.close()


def query_one(sql: str, params: tuple = (), db: Union[str, Path] = DEFAULT_DEVSTORE):
    """Run a SELECT returning the first row (or None). Connection opened and
    closed within the call — small bites, close fast."""
    conn = _conn(db)
    try:
        cur = conn.execute(sql, params)
        return cur.fetchone()
    finally:
        conn.close()


def query_all(sql: str, params: tuple = (), db: Union[str, Path] = DEFAULT_DEVSTORE):
    conn = _conn(db)
    try:
        cur = conn.execute(sql, params)
        return cur.fetchall()
    finally:
        conn.close()


def execute(sql: str, params: tuple = (), db: Union[str, Path] = DEFAULT_DEVSTORE) -> int:
    """Run a single write statement in its own transaction; returns rowcount."""
    conn = _conn(db)
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
