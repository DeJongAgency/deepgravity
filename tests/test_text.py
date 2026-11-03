"""test_text.py — DoD 3 acceptance: UTF-8 NFC at every boundary, mojibake impossible.

Runnable without pytest: python tests/test_text.py  (plain asserts, PASS/FAIL).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core import text as tx


def test_normalize_nfc():
    # e + combining acute (NFD) -> é (NFC)
    nfd = "cafe\u0301"
    assert tx.normalize_nfc(nfd) == "caf\u00e9"


def test_strips_bom():
    assert tx.normalize_nfc("\ufeffhello") == "hello"


def test_lf_only():
    assert tx.to_lf("a\r\nb\rc") == "a\nb\nc"


def test_encode_no_bom():
    raw = tx.encode("héllo")
    assert raw == "h\u00e9llo".encode("utf-8")
    assert not raw.startswith(b"\xef\xbb\xbf")


def test_decode_strict_rejects_invalid():
    bad = b"\xff\xfe\x00invalid"
    try:
        tx.decode(bad)
    except UnicodeDecodeError:
        return  # expected: strict decode, never silent U+FFFD replacement
    raise AssertionError("invalid UTF-8 must raise, not silently replace")


def test_round_trip_read_write():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.md"
        tx.write_text(p, "caf\u00e9\r\nline2")
        raw = p.read_bytes()
        assert b"\r" not in raw, "canonical zone must be LF on disk"
        assert tx.read_text(p) == "caf\u00e9\nline2"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
