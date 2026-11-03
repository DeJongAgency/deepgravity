"""test_engine_registry.py — per-artifact registry over the REAL card shape.

Runnable without pytest: python tests/test_engine_registry.py (plain asserts,
PASS/FAIL). Uses temp-dir fixtures — never touches the live model_cards/.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core.engine_registry import EngineRegistry, classify_dialect

# ---- fixtures: real on-card shape (as the probe machinery writes it) ----

STANDARD = {
    "model": "qwen2.5:3b",
    "digest": "aaaa1111",
    "family": "qwen2.5",
    "dialect": {"api": "chat", "template_dependent": False,
                "thinking_field": "none", "raw_output_markers": [],
                "peg_parser_hazard": False, "content_only": True},
    "behavioral_notes": [],
}

THINKING = {
    "model": "gurubot/gpt-oss-derestricted:20b",
    "digest": "bbbb2222",
    "family": "gpt-oss-20b-moe",
    "dialect": {"api": "chat", "template_dependent": False,
                "thinking_field": "thinking", "has_thinking": True,
                "raw_output_markers": [], "peg_parser_hazard": False,
                "content_only": False},
    "behavioral_notes": [],
}

HAZARDOUS = {
    "model": "muse-glimmer-30b-ablit:q4",
    "digest": "cccc3333",
    "family": "muse-glimmer",
    "aliases": [{"tag": "glimmer:latest", "first_seen": "2026-09-08T00:20Z", "last_seen": None}],
    "dialect": {"api": "chat", "template_dependent": True,
                "thinking_field": "thinking", "has_thinking": True,
                "raw_output_markers": ["<|start|>", "<|message|>", "<|eot|>"],
                "peg_parser_hazard": True, "content_only": False},
    "behavioral_notes": [],
}

LATEST_TAG = {
    "model": "dora-14b-4k:latest",
    "digest": "dddd4444",
    "family": "qwen2-14b-ablit",
    "dialect": {"api": "chat", "template_dependent": False,
                "thinking_field": "none", "raw_output_markers": [],
                "peg_parser_hazard": False, "content_only": True},
    "behavioral_notes": [],
}


def _write_card(dirpath: Path, rec: dict) -> Path:
    p = dirpath / (rec["model"].split(":")[0].replace("/", "-") + ".json")
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return p


def _make_registry():
    tmp = tempfile.TemporaryDirectory()
    d = Path(tmp.name)
    for rec in (STANDARD, THINKING, HAZARDOUS, LATEST_TAG):
        _write_card(d, rec)
    reg = EngineRegistry(d)
    return tmp, d, reg


def test_loads_all_cards():
    tmp, d, reg = _make_registry()
    try:
        assert len(reg.records()) == 4
    finally:
        tmp.cleanup()


def test_dialect_classes():
    assert classify_dialect(STANDARD) == "standard"
    assert classify_dialect(THINKING) == "thinking"
    assert classify_dialect(HAZARDOUS) == "hazardous_template"


def test_query_by_class():
    tmp, d, reg = _make_registry()
    try:
        assert {r["family"] for r in reg.query(dialect_class="standard")} == {"qwen2.5", "qwen2-14b-ablit"}
        assert {r["family"] for r in reg.query(dialect_class="thinking")} == {"gpt-oss-20b-moe"}
        assert {r["family"] for r in reg.query(dialect_class="hazardous_template")} == {"muse-glimmer"}
    finally:
        tmp.cleanup()


def test_by_model_exact_and_normalized():
    tmp, d, reg = _make_registry()
    try:
        assert reg.by_model("qwen2.5:3b")["digest"] == "aaaa1111"
        # 'name' and 'name:latest' are the same artifact
        assert reg.by_model("dora-14b-4k")["digest"] == "dddd4444"
        assert reg.by_model("dora-14b-4k:latest")["digest"] == "dddd4444"
    finally:
        tmp.cleanup()


def test_alias_resolves_to_current_record():
    tmp, d, reg = _make_registry()
    try:
        assert reg.by_model("glimmer:latest")["digest"] == "cccc3333"
    finally:
        tmp.cleanup()


def test_get_by_digest_and_unknown():
    tmp, d, reg = _make_registry()
    try:
        assert reg.get("cccc3333")["family"] == "muse-glimmer"
        assert reg.get("no-such-tag") is None
    finally:
        tmp.cleanup()


def test_validate_all_clean_for_fixtures():
    tmp, d, reg = _make_registry()
    try:
        problems = reg.validate_all()
        assert all(v == [] for v in problems.values()), problems
    finally:
        tmp.cleanup()


def test_validate_catches_missing_fields():
    tmp, d, reg = _make_registry()
    try:
        bad = {"digest": "eeee5555", "family": "x", "dialect": {"api": "chat",
               "thinking_field": "none", "content_only": True}}
        reg.put_record(bad)
        problems = reg.validate_all()
        assert "missing top-level 'model'" in problems["eeee5555"]
    finally:
        tmp.cleanup()


def test_put_record_duplicate_guard():
    tmp, d, reg = _make_registry()
    try:
        try:
            reg.put_record(dict(STANDARD))
            raise AssertionError("expected ValueError on duplicate digest")
        except ValueError:
            pass
        # overwrite path works
        rec = dict(STANDARD)
        rec["behavioral_notes"] = [{"ts": "2026-09-08T00:00Z", "note": "x", "observed_by": "t"}]
        reg.put_record(rec, overwrite=True)
        assert len(reg.records()) == 4
    finally:
        tmp.cleanup()


def test_append_note_writes_through_atomically():
    tmp, d, reg = _make_registry()
    try:
        reg.append_note("qwen2.5:3b", "new observation", observed_by="test")
        rec = reg.by_model("qwen2.5:3b")
        assert rec["behavioral_notes"][-1]["note"] == "new observation"
        # survives a fresh load from disk (file rewritten, still valid json)
        reg2 = EngineRegistry(d)
        assert reg2.by_model("qwen2.5:3b")["behavioral_notes"][-1]["note"] == "new observation"
    finally:
        tmp.cleanup()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed.append((name, e))
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
