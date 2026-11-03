"""test_search.py — DoD 7 acceptance: retrieval first-class, three substrates.

Runs against a TEMP devstore; web_adapter tested for graceful offline
degradation (never a live-SearXNG dependency in tests). Plain asserts.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core.search import files_tools, owned_index, web_adapter


def test_owned_index_add_and_search():
    db, d = _tmp_db()
    try:
        owned_index.add("chat", "the quick brown fox jumped", {"who": "jh"}, db=db)
        owned_index.add("note", "unrelated grocery list", {}, db=db)
        hits = owned_index.search("fox", db=db)
        assert len(hits) == 1 and hits[0]["content"].startswith("the quick")
        assert hits[0]["meta"]["who"] == "jh"
        assert owned_index.count(db=db) == 2
    finally:
        d.cleanup()


def test_owned_index_append_only_no_delete_api():
    """Single-writer doctrine: the module exposes add/search/count — there is
    NO delete/replace callable (no DELETE+re-INSERT disease by construction).
    (Checks the API surface, not docstring prose — 'DELETE' legitimately
    appears in the module docstring as the disease being refused.)"""
    import inspect
    callables = {
        name for name, _ in inspect.getmembers(owned_index, inspect.isfunction)
        if not name.startswith("_")
    }
    banned = callables & {"delete", "remove", "clear", "drop", "replace", "update"}
    assert not banned, f"owned index must not expose delete/replace callables: {banned}"
    assert {"add", "search", "count"} <= callables


def test_web_adapter_graceful_offline():
    """SearXNG down must degrade, not crash: learn() returns ok=False."""
    db, d = _tmp_db()
    try:
        # Port 1 is never listening; adapter must raise -> learn() degrades.
        out = web_adapter.learn("anything", base="http://127.0.0.1:1", db=db)
        assert out["ok"] is False and "error" in out and out["learned"] == 0
    finally:
        d.cleanup()


def test_files_read_find_search():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (root / "b.md").write_text("# note\nfoo called here\n", encoding="utf-8")
        (root / ".venv").mkdir()
        (root / ".venv" / "c.py").write_text("foo inside venv\n", encoding="utf-8")

        txt = files_tools.read(root / "a.py")
        assert "def foo" in txt

        hits = files_tools.find_in_file(root / "b.md", "foo")
        assert len(hits) == 1 and hits[0]["line"] == 2

        found = files_tools.search_files(root, "foo")
        paths = {f["path"] for f in found}
        assert any(str(root / "a.py") in p for p in paths)
        assert not any(".venv" in p for p in paths), "exclusions must hold"


def _tmp_db():
    d = tempfile.TemporaryDirectory()
    return str(Path(d.name) / "devstore.db"), d


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
