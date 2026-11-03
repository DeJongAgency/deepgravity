"""cloud_ledger.py — the per-turn cloud cost & cache ledger (DG dash data layer).

The soul of the cost/cache pane: exactly what a cloud turn cost — tokens in/out,
cache hits vs misses, and cash when there's API access. Append-only, per-turn,
per-conversation. Feeds DoD 5 (cache anchor: hit ratio measured, not assumed)
and DoD 6 (per-turn token cost + cash on the launch dashboard).

Provider adapters at the gateway seam normalize usage fields into this shape:
  deepseek/openai-compatible: prompt_cache_hit_tokens / prompt_cache_miss_tokens
  gemini: cached-content token counts (adapter maps them)
  cost_usd is what the provider billed (or the engine computed from rates).

The "pretty picture tax": the dash shows the last turn's cost — what THIS
response cost to produce.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import store
from .time import now_utc


def _ensure_table(db=None) -> None:
    store.init_schema(db)
    store.execute(
        """
        CREATE TABLE IF NOT EXISTS cloud_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            conv_label TEXT NOT NULL,
            turn_seq INTEGER NOT NULL DEFAULT 0,
            tokens_in INTEGER NOT NULL DEFAULT 0,
            tokens_out INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0.0,
            status_traffic INTEGER NOT NULL DEFAULT 0,
            meta_json TEXT NOT NULL DEFAULT '{}'
        )
        """,
        db=db,
    )


def record(
    *,
    provider: str,
    model: str,
    conv_label: str,
    turn_seq: int = 0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    cost_usd: float = 0.0,
    status_traffic: int = 0,
    meta: Optional[Dict[str, Any]] = None,
    db=None,
) -> int:
    """Append one cloud turn to the ledger. Append-only — never rewritten."""
    _ensure_table(db)
    store.execute(
        """
        INSERT INTO cloud_turns
            (ts_utc, provider, model, conv_label, turn_seq, tokens_in, tokens_out,
             cache_read_tokens, cache_write_tokens, cost_usd, status_traffic, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (now_utc(), provider, model, conv_label, turn_seq, tokens_in, tokens_out,
         cache_read, cache_write, round(cost_usd, 6), status_traffic,
         __import__("json").dumps(meta or {})),
        db=db,
    )
    row = store.query_one("SELECT last_insert_rowid()", db=db)
    return row[0]


def conversation(conv_label: str, db=None) -> List[dict]:
    """Per-turn breakdown of one conversation (the 'was it cached?' view)."""
    _ensure_table(db)
    rows = store.query_all(
        """
        SELECT ts_utc, provider, model, turn_seq, tokens_in, tokens_out,
               cache_read_tokens, cache_write_tokens, cost_usd, status_traffic
        FROM cloud_turns WHERE conv_label = ? ORDER BY turn_seq ASC, id ASC
        """,
        (conv_label,), db=db,
    )
    return [dict(zip(
        ["ts_utc", "provider", "model", "turn_seq", "tokens_in", "tokens_out",
         "cache_read", "cache_write", "cost_usd", "status_traffic"], r)) for r in rows]


def conversation_summary(conv_label: str, db=None) -> dict:
    """Totals + cache-hit ratio for one conversation."""
    turns = conversation(conv_label, db=db)
    if not turns:
        return {"conv_label": conv_label, "turns": 0}
    tot_in = sum(t["tokens_in"] for t in turns)
    tot_read = sum(t["cache_read"] for t in turns)
    tot_cost = sum(t["cost_usd"] for t in turns)
    billed = tot_read + sum(t["cache_write"] for t in turns)  # the billable prompt base
    return {
        "conv_label": conv_label,
        "turns": len(turns),
        "total_tokens_in": tot_in,
        "cache_read_total": tot_read,
        "cache_hit_ratio": (tot_read / tot_in) if tot_in else 0.0,
        "billable_prompt_tokens": billed,
        "total_cost_usd": round(tot_cost, 4),
        "avg_cost_per_turn": round(tot_cost / len(turns), 6),
    }


def totals(db=None) -> dict:
    """Grand totals across the ledger (by provider). The dashboard's top line."""
    _ensure_table(db)
    rows = store.query_all(
        """
        SELECT provider,
               COUNT(*), SUM(tokens_in), SUM(tokens_out),
               SUM(cache_read_tokens), SUM(cache_write_tokens), SUM(cost_usd)
        FROM cloud_turns GROUP BY provider ORDER BY SUM(cost_usd) DESC
        """, db=db)
    return {
        "by_provider": [
            {"provider": r[0], "turns": r[1], "tokens_in": r[2], "tokens_out": r[3],
             "cache_read": r[4], "cache_write": r[5], "cost_usd": round(r[6], 4)}
            for r in rows
        ],
    }


if __name__ == "__main__":
    import sys
    conv = sys.argv[1] if len(sys.argv) > 1 else None
    if conv:
        print("=== conversation ===")
        for t in conversation(conv):
            ratio = (t["cache_read"] / t["tokens_in"]) if t["tokens_in"] else 0
            print(f"  turn {t['turn_seq']}: in={t['tokens_in']} out={t['tokens_out']} "
                  f"cache_read={t['cache_read']} ({ratio:.0%}) cost=${t['cost_usd']:.4f}")
        print(conversation_summary(conv))
    else:
        print(totals())
