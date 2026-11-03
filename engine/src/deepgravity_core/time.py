"""time.py — DoD 2: single time representation.

All timestamps are stored/recorded as UTC ISO-8601 with Z suffix (or epoch) at
the recording layer, per turn. Display offset is applied only at render. The
model never converts or compares timestamps — the engine does.

Acceptance: no code path asks the model to compare two timestamps.
"""

from __future__ import annotations

from datetime import datetime, timezone


def now_utc() -> str:
    """Current UTC time as ISO-8601 with Z suffix (the canonical stored form).

    Example: 2026-09-07T22:30:00.123456Z
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_utc(iso: str) -> datetime:
    """Parse a stored timestamp (Z or +00:00) into an aware UTC datetime."""
    normalized = iso.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        # Defensive: a naive timestamp is treated as UTC (never local) per DoD 2.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_epoch(iso: str) -> float:
    """Stored ISO -> Unix epoch (UTC). Comparisons happen here, in the engine."""
    return parse_utc(iso).timestamp()


def render_local(iso: str) -> str:
    """Stored ISO -> local wall-clock string for DISPLAY ONLY. Never re-stored."""
    return parse_utc(iso).astimezone().isoformat(timespec="seconds")


def cmp(a: str, b: str) -> int:
    """Compare two stored timestamps. Returns -1/0/1. Engine-side, never model-side."""
    ea, eb = to_epoch(a), to_epoch(b)
    return (ea > eb) - (ea < eb)
