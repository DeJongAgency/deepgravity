"""test_writes.py — DoD 8 acceptance: atomic writes, before-or-after never partial.

Runnable without pytest: python tests/test_writes.py (plain asserts).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core import writes as w


def test_atomic_write_text_content_and_no_litter():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sub" / "file.md"
        w.atomic_write_text(p, "caf\u00e9\nline2\r\n")
        assert p.read_text(encoding="utf-8") == "caf\u00e9\nline2\n"  # LF-normalized via text.encode
        leftovers = list(Path(d).rglob("*.dg-tmp"))
        assert leftovers == [], "no temp files may remain after a successful write"


def test_overwrite_is_complete():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "f.txt"
        w.atomic_write_text(p, "old-content")
        w.atomic_write_text(p, "new-content-" + "x" * 100000)
        assert p.read_text(encoding="utf-8") == "new-content-" + "x" * 100000


def test_target_never_partial_on_failure():
    """Simulate a failure after temp creation: the target must remain the old
    complete file (never a fragment), and the temp is cleaned up."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "f.txt"
        w.atomic_write_text(p, "ORIGINAL")
        # Force os.replace to fail by making the parent read-only (POSIX).
        os.chmod(d, 0o500)
        try:
            try:
                w.atomic_write_bytes(p, b"NEW DATA")
            except (PermissionError, OSError):
                pass  # expected
            finally:
                os.chmod(d, 0o700)
            assert p.read_text(encoding="utf-8") == "ORIGINAL", "target must stay the old complete file"
            leftovers = list(Path(d).glob("*.dg-tmp"))
            assert leftovers == [], "failed write must clean its temp"
        finally:
            os.chmod(d, 0o700)


def test_sweep_orphan_temps():
    with tempfile.TemporaryDirectory() as d:
        orphan = Path(d) / "junk.md.dg-tmp"
        orphan.write_text("partial", encoding="utf-8")
        good = Path(d) / "keep.txt"
        good.write_text("fine", encoding="utf-8")
        removed = w.sweep_orphan_temps(d)
        assert removed == 1
        assert not orphan.exists() and good.exists()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
