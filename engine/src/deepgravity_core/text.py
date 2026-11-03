"""text.py — DoD 3: UTF-8 NFC at every boundary.

Filesystem, subprocess, sockets, DB, logs: all declare and enforce UTF-8. No
codepage inheritance. Canonical zone contract (Bookmarks B8): UTF-8, NFC,
no BOM, LF. Mojibake glyphs (U+FFFD etc.) cannot appear in any store.

Acceptance: an encoding test fixture exists and is green; file opens specify
encoding.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

UTF8 = "utf-8"
BOM = "\ufeff"


def normalize_nfc(text: str) -> str:
    """NFC-normalize + strip BOM. The canonical stored form for all text."""
    return unicodedata.normalize("NFC", text).lstrip(BOM)


def to_lf(text: str) -> str:
    """Canonical-zone line endings: LF only (no CRLF, no lone CR)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def encode(text: str) -> bytes:
    """Encode canonical text to UTF-8 bytes (no BOM)."""
    return normalize_nfc(text).encode(UTF8)


def decode(data: bytes) -> str:
    """Decode bytes as UTF-8 (strict). Raises UnicodeDecodeError on invalid input —
    never silently replaces with U+FFFD. Callers at the perimeter transcode with
    validation and log (B8 layer 2); the canonical store never mangles."""
    return data.decode(UTF8)


def read_text(path: str | Path) -> str:
    """Read a canonical-zone file: UTF-8, strict, BOM-stripped, NFC, LF."""
    return normalize_nfc(Path(path).read_text(encoding=UTF8))


def write_text(path: str | Path, text: str) -> None:
    """Write a canonical-zone file: UTF-8, no BOM, LF, NFC.

    Non-atomic by itself — callers in the write path use writes.atomic() (DoD 8)
    so an interrupt leaves before-or-after, never partial.
    """
    Path(path).write_text(to_lf(normalize_nfc(text)), encoding=UTF8, newline="\n")
