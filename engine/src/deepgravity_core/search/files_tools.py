"""files_tools.py — DoD 7 third substrate: files/tools search.

Hierarchy (coding.md §2.5, restored invariant): view_file → Select-String →
_db.py → discovery search. Grep is LAST RESORT. This module encodes that:
- read(): view a file (bounded)
- find_in_file(): Select-String equivalent — ONE file, no recursion
- search_files(): recursive discovery ONLY when asked explicitly, bounded,
  with exclusions (.venv/.git/node_modules/binary). Never whole-tree by default.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

from .. import text as _text

EXCLUDE_DIRS = {".venv", ".git", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", "site-packages"}
_TEXT_EXTS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".conf", ".csv", ".log", ".ce"}
DEFAULT_MAX_BYTES = 256 * 1024  # 256KB


def read(path: Union[str, Path], start: int = 1, end: Optional[int] = None, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """view_file equivalent: read a bounded slice of a file (line numbers 1-based).
    Raises FileNotFoundError / IsADirectoryError naturally."""
    p = Path(path)
    with open(p, "rb") as fh:
        head = fh.read(max_bytes)
    text = _text.decode(head)
    lines = text.splitlines()
    if end is None:
        end = len(lines)
    return "\n".join(lines[start - 1:end])


def find_in_file(path: Union[str, Path], pattern: str, max_hits: int = 20) -> List[dict]:
    """Select-String equivalent: line hits in ONE file. Case-sensitive by default;
    prefix (?i) for case-insensitive. Bounded. No recursion, ever."""
    p = Path(path)
    rx = re.compile(pattern)
    hits = []
    with open(p, "r", encoding="utf-8", errors="strict") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if rx.search(line):
                hits.append({"line": lineno, "text": line[:500]})
                if len(hits) >= max_hits:
                    break
    return hits


def _iter_text_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in _TEXT_EXTS:
                yield p


def search_files(root: Union[str, Path], pattern: str, max_results: int = 20) -> List[dict]:
    """LAST-RESORT recursive discovery. Caller must explicitly ask for
    recursion; never the default. Bounded, exclusions applied, binary skipped
    by extension allow-list."""
    rx = re.compile(pattern)
    out = []
    for p in _iter_text_files(Path(root)):
        try:
            with open(p, "r", encoding="utf-8", errors="strict") as fh:
                for lineno, line in enumerate(fh, 1):
                    if rx.search(line):
                        out.append({"path": str(p), "line": lineno, "text": line.rstrip("\n")[:300]})
                        if len(out) >= max_results:
                            return out
                        break  # one hit per file is enough for discovery
        except (OSError, UnicodeDecodeError):
            continue
    return out
