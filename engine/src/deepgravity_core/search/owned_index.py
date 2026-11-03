"""owned_index.py — DoD 7 core: the OWNED index (braid + vectors) is the real
engine; adapters are periphery. Single-writer, append-only, no DELETE+re-INSERT.

Pre-flip (bootstrap store): records live in the dev-scoped SQLite store. The
seam: when the real braid lands at ~/.local/share/deepgrav/user/{occupant}/data/braid/
(B13) via the flip, this module's store pointer switches and the dev store
retires. Vector search hooks in here (embedder :8207) — keyword fallback is
operational today.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

from .. import store
from ..time import now_utc


def _ensure_records_table(db) -> None:
    store.init_schema(db)
    store.execute(
        """
        CREATE TABLE IF NOT EXISTS owned_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            source_type TEXT NOT NULL,        -- chat | web_page | doc | note
            content TEXT NOT NULL,
            meta_json TEXT NOT NULL DEFAULT '{}'
        )
        """,
        db=db,
    )


def add(source_type: str, content: str, meta: Optional[Dict[str, Any]] = None, db=None) -> int:
    """Append a record. THE single writer into the owned index — append-only,
    no DELETE+re-INSERT (single-writer doctrine). Returns the new id."""
    _ensure_records_table(db)
    store.execute(
        "INSERT INTO owned_records (ts_utc, source_type, content, meta_json) VALUES (?, ?, ?, ?)",
        (now_utc(), source_type, content, json.dumps(meta or {})),
        db=db,
    )
    row = store.query_one("SELECT last_insert_rowid()", db=db)
    return row[0]


def search(term: str, limit: int = 10, source_type: str = "", db=None) -> List[dict]:
    """Keyword retrieval over the owned index (vector path lands here later).
    Bounded, close-fast."""
    _ensure_records_table(db)
    rows = store.query_all(
        """
        SELECT id, ts_utc, source_type, content, meta_json FROM owned_records
        WHERE content LIKE ?
          AND (? = '' OR source_type = ?)
        ORDER BY id DESC LIMIT ?
        """,
        (f"%{term}%", source_type, source_type, limit),
        db=db,
    )
    return [
        {"id": r[0], "ts_utc": r[1], "source_type": r[2],
         "content": r[3], "meta": json.loads(r[4])}
        for r in rows
    ]


def count(db=None) -> int:
    _ensure_records_table(db)
    return store.query_one("SELECT COUNT(*) FROM owned_records", db=db)[0]
