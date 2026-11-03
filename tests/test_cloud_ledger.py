"""test_cloud_ledger.py — per-turn cost & cache ledger acceptance.

Models the real Meta specimen shape: a 3-turn conversation where turn 1 is a
cold start (big uncached block) and turns 2-3 are cache hits. Asserts the
summary answers the DoD 5/6 question ('was it cached?'). Plain asserts.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core import cloud_ledger as cl


def _tmp_db():
    d = tempfile.TemporaryDirectory()
    return str(Path(d.name) / "devstore.db"), d


def _record_meta_shaped(db):
    """3-turn convo: turn1 cold $0.28, turns 2-3 cached $0.06 each = $0.40 total."""
    cl.record(provider="meta", model="gemini-class", conv_label="JH-Meta #1", turn_seq=1,
              tokens_in=60000, tokens_out=2000, cache_read=0, cache_write=58000,
              cost_usd=0.28, db=db)
    cl.record(provider="meta", model="gemini-class", conv_label="JH-Meta #1", turn_seq=2,
              tokens_in=59000, tokens_out=1500, cache_read=57000, cache_write=0,
              cost_usd=0.06, db=db)
    cl.record(provider="meta", model="gemini-class", conv_label="JH-Meta #1", turn_seq=3,
              tokens_in=59000, tokens_out=1200, cache_read=57000, cache_write=0,
              cost_usd=0.06, db=db)


def test_conversation_breakdown_and_summary():
    db, d = _tmp_db()
    try:
        _record_meta_shaped(db)
        turns = cl.conversation("JH-Meta #1", db=db)
        assert len(turns) == 3
        assert turns[0]["cache_read"] == 0 and turns[0]["cost_usd"] == 0.28  # cold
        assert turns[1]["cache_read"] > 0 and turns[1]["cost_usd"] == 0.06  # cached

        s = cl.conversation_summary("JH-Meta #1", db=db)
        assert s["turns"] == 3
        assert abs(s["total_cost_usd"] - 0.40) < 1e-6, "the 40-cent shape"
        assert s["avg_cost_per_turn"] == round(0.40 / 3, 6)
        # cache hit ratio over all prompt tokens (57k+57k read / 60k+59k+59k)
        assert abs(s["cache_hit_ratio"] - (114000 / 178000)) < 1e-9
    finally:
        d.cleanup()


def test_mutating_anchor_shape_shows_no_cache_benefit():
    """The disease shape: same cost every turn (no cache hits) -> ratio 0."""
    db, d = _tmp_db()
    try:
        for i in (1, 2, 3):
            cl.record(provider="meta", model="x", conv_label="no-cache", turn_seq=i,
                      tokens_in=60000, tokens_out=1000, cache_read=0, cache_write=0,
                      cost_usd=0.13, db=db)
        s = cl.conversation_summary("no-cache", db=db)
        assert s["cache_hit_ratio"] == 0.0
        assert s["total_cost_usd"] == round(0.39, 4)
    finally:
        d.cleanup()


def test_totals_by_provider():
    db, d = _tmp_db()
    try:
        _record_meta_shaped(db)
        cl.record(provider="deepseek", model="v4-flash", conv_label="us", turn_seq=1,
                  tokens_in=5000, tokens_out=800, cache_read=4000, cache_write=0,
                  cost_usd=0.002, db=db)
        t = cl.totals(db=db)
        provs = {p["provider"]: p for p in t["by_provider"]}
        assert set(provs) == {"meta", "deepseek"}
        assert abs(provs["meta"]["cost_usd"] - 0.40) < 1e-6
        assert provs["deepseek"]["turns"] == 1
    finally:
        d.cleanup()


def test_append_only_no_update_delete():
    import inspect
    callables = {n for n, _ in inspect.getmembers(cl, inspect.isfunction) if not n.startswith("_")}
    banned = callables & {"delete", "remove", "clear", "update", "drop"}
    assert not banned, f"ledger must be append-only, got: {banned}"
    assert {"record", "conversation", "conversation_summary", "totals"} <= callables


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
