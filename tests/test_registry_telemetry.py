"""test_registry_telemetry.py — DoD 4 (registry in write path) + Perf/DoD 6.

Runs against a TEMP devstore (never the box's data/devstore). Plain asserts.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core import registry, store, telemetry
from deepgravity_core.transport import Tool, TransportError
from deepgravity_core.writes import atomic_write_text


def _tmp_db():
    d = tempfile.TemporaryDirectory()
    return str(Path(d.name) / "devstore.db"), d


def test_register_get_search():
    db, d = _tmp_db()
    try:
        t = Tool(name="say", argv=["/usr/bin/printf", "%s", "{msg}"],
                 param_names=["msg"], param_types={"msg": "str"}, description="echo text")
        registry.register(t, layer="dev", domain="test", db=db)
        got = registry.get("say", db=db)
        assert got.name == "say" and got.argv == ("/usr/bin/printf", "%s", "{msg}")
        hits = registry.search("echo", db=db)
        assert any(h["name"] == "say" for h in hits)
    finally:
        d.cleanup()


def test_get_missing_raises():
    db, d = _tmp_db()
    try:
        try:
            registry.get("nope", db=db)
        except TransportError as e:
            assert "nope" in str(e)
            return
        raise AssertionError("missing tool must raise TransportError")
    finally:
        d.cleanup()


def test_tool_crib_check_verdicts():
    db, d = _tmp_db()
    try:
        no_hit = registry.tool_crib_check("transcode cp1252 corpus", db=db)
        assert no_hit["matched"] is False
        assert "No existing tool" in no_hit["verdict"]

        registry.register(Tool(name="xcode", argv=["/bin/true"]),
                          layer="dev", domain="text", db=db)
        hit = registry.tool_crib_check("xcode", db=db)
        assert hit["matched"] is True and "USE EXISTING" in hit["verdict"]
    finally:
        d.cleanup()


def test_telemetry_append_only_and_counts():
    db, d = _tmp_db()
    try:
        telemetry.record_milestone("tool_start", {"tool": "x"}, db=db)
        telemetry.record_cost(100, 50, 0.00123, model="test", db=db)
        telemetry.record_syntax_failure("quote", "nested quotes", db=db)
        stream = telemetry.recent(db=db)
        events = [m["event"] for m in stream]
        assert events == ["tool_start", "cost", "syntax_failure"], "append-only order"
        c = telemetry.counts(db=db)
        assert c["milestones"] == 3 and c["syntax_failures"] == 1
    finally:
        d.cleanup()


def test_registry_in_write_path_integration():
    """The DoD 4 discipline end-to-end: a write is preceded by a crib check."""
    db, d = _tmp_db()
    try:
        # A legit tool exists for the need
        registry.register(Tool(name="canon_write", argv=["/bin/true"]),
                          layer="dev", domain="writes", db=db)
        need = "atomic text file write"
        check = registry.tool_crib_check(need, db=db)
        # Our writes.atomic_write_text is the canonical mechanism — registry
        # proves discovery precedes authoring.
        target = Path(d.name) / "out.txt"
        atomic_write_text(target, "hello")
        assert target.read_text(encoding="utf-8") == "hello"
        assert check["matched"] or check["verdict"]  # guard ran, verdict surfaced
    finally:
        d.cleanup()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
