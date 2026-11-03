"""test_time.py — DoD 2 acceptance: single time rep, Z=UTC, engine-side compare.

Runnable without pytest: python tests/test_time.py  (plain asserts, PASS/FAIL).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core import time as t


def test_now_utc_shape():
    s = t.now_utc()
    assert s.endswith("Z"), f"must end with Z: {s}"
    parsed = t.parse_utc(s)
    assert parsed.tzinfo is not None and str(parsed.tzinfo) == "UTC"


def test_parse_naive_treated_as_utc():
    dt = t.parse_utc("2026-09-07T22:00:00")
    assert str(dt.tzinfo) == "UTC", "naive stored time must be treated as UTC, never local"


def test_round_trip_epoch():
    s = t.now_utc()
    assert abs(t.to_epoch(s) - t.parse_utc(s).timestamp()) < 1e-6


def test_cmp_engine_side():
    earlier = "2026-09-07T21:00:00Z"
    later = "2026-09-07T22:00:00Z"
    assert t.cmp(earlier, later) == -1
    assert t.cmp(later, earlier) == 1
    assert t.cmp(earlier, earlier) == 0


def test_render_local_display_only():
    s = "2026-09-07T22:00:00Z"
    out = t.render_local(s)
    assert isinstance(out, str) and "T" in out, "render must produce a local ISO string"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
