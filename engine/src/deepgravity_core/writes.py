"""writes.py — DoD 8: interrupt-safe atomic writes, single choke point.

No user interrupt may leave a partial file or partial DB state. All file
writes go temp-file-in-same-dir -> flush -> fsync -> atomic rename (os.replace).
A kill before the rename leaves the old complete file; after, the new complete
file. Never a fragment.

One writer, append-only (Perf doctrine). The tool runner sweeps orphan temps
at next start (sweep_orphan_temps).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Union

from . import text as _text

_TMP_SUFFIX = ".dg-tmp"

class WriteError(RuntimeError):
    """Raised on write-path failures."""


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    """Atomically write bytes to `path` (temp in same dir -> fsync -> os.replace).

    Guarantees: the target is always either the old complete file or the new
    complete file. A temp is never left behind on success; on failure it is
    unlinked before re-raising.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=_TMP_SUFFIX, dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, p)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    """Atomic write of canonical text (UTF-8, NFC, LF, no BOM via text.py)."""
    atomic_write_bytes(path, _text.encode(text))


def sweep_orphan_temps(root: Union[str, Path]) -> int:
    """Remove stale *.dg-tmp files under `root` (run once at engine start).

    A temp that survives a kill is garbage; the target was never renamed, so
    the previous complete file (if any) is intact. Returns count removed.
    """
    root_p = Path(root)
    if not root_p.exists():
        return 0
    removed = 0
    for tmp in root_p.rglob(f"*{_TMP_SUFFIX}"):
        try:
            if tmp.is_file():
                tmp.unlink()
                removed += 1
        except OSError:
            pass
    return removed
