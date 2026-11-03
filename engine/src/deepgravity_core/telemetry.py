"""telemetry.py — Perf doctrine 1 + DoD 6: milestones at rest, cost counters.

Telemetry is DATA AT REST. Written at milestones (tool_start, tool_end,
control, error, phase change) — never per-token. Fetched when something needs
analysis; never live-streamed as part of inference. A live view is derived
from the milestone stream, not from per-token logging.

DoD 6: cost telemetry AND a syntax-failure counter (quote/encode/time-class
errors) are first-class from birth. The syntax tax is measurable and trends
to zero.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import store
from .time import now_utc


def record_milestone(event: str, detail: Optional[Dict[str, Any]] = None, db=None) -> None:
    """Append a milestone event (tool_start/end, control, error, phase)."""
    store.init_schema(db)
    store.execute(
        "INSERT INTO milestones (ts_utc, event, detail_json) VALUES (?, ?, ?)",
        (now_utc(), event, json.dumps(detail or {})),
        db=db,
    )


def record_cost(tokens_in: int, tokens_out: int, cost_usd: float, model: str = "", db=None) -> None:
    """DoD 6 cost telemetry. Append-only; the ledger is never rewritten."""
    record_milestone("cost", {
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "cost_usd": round(cost_usd, 6), "model": model,
    }, db=db)


def record_syntax_failure(kind: str, detail: str = "", db=None) -> None:
    """DoD 6 syntax-tax counter: quote / encode / time-class errors."""
    record_milestone("syntax_failure", {"kind": kind, "detail": detail}, db=db)


def recent(limit: int = 50, db=None) -> List[dict]:
    """Fetch the most recent milestone stream (oldest->newest of the last N)."""
    store.init_schema(db)
    rows = store.query_all(
        "SELECT ts_utc, event, detail_json FROM milestones ORDER BY id DESC LIMIT ?",
        (limit,), db=db,
    )
    return [
        {"ts_utc": r[0], "event": r[1], "detail": json.loads(r[2])}
        for r in reversed(rows)
    ]


def counts(db=None) -> Dict[str, int]:
    """Quick aggregates for a status bar (not a firehose)."""
    store.init_schema(db)
    total = store.query_one("SELECT COUNT(*) FROM milestones", db=db)[0]
    syntax = store.query_one(
        "SELECT COUNT(*) FROM milestones WHERE event = 'syntax_failure'", db=db)[0]
    return {"milestones": total, "syntax_failures": syntax}
