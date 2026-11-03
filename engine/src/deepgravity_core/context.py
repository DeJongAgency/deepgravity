"""context.py — DoD 5 + identity-in-context: the static session prompt.

The static prompt block is immutable per session — it is the provider cache
anchor; mutating it mid-session re-bills the whole block uncached (DoD 5).
Identity rides in context every turn (bundle, not weights, until reliteration):
the bundle is loaded ONCE at session start into a stable system prompt.
Until the flip seeds the real bundle on .24 (Bookmarks B10/B13), callers may
pass any bundle dir; the default is a minimal engine identity so the gateway
runs before flip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from . import text as _text

DEFAULT_MINIMAL = (
    "You are running inside the DeepGravity engine core (deepgravity_core) on "
    "the appliance. This is a development build; the full occupant identity "
    "bundle is seeded at flip."
)


def load_bundle(bundle_dir: Optional[Path]) -> str:
    """Assemble the immutable system prompt from an identity bundle directory
    (sorted .md files, canonical text rules). Empty/missing -> minimal default."""
    if bundle_dir is None or not Path(bundle_dir).exists():
        return DEFAULT_MINIMAL
    parts: list[str] = []
    for p in sorted(Path(bundle_dir).glob("*.md")):
        content = _text.read_text(p).strip()
        if content:
            parts.append(content)
    if not parts:
        return DEFAULT_MINIMAL
    return "\n\n---\n\n".join(parts)


@dataclass(frozen=True)
class SessionContext:
    """One immutable context per session. Constructed once; never mutated."""
    session_id: str
    system_prompt: str
    model: str = "deepgravity-core-14b:latest"

    def with_user_turn(self, raw_input: str) -> str:
        """Assemble the full prompt for a turn: static system prompt + user text.
        The static block is NEVER rebuilt per turn (cache anchor preserved)."""
        return f"{self.system_prompt}\n\nUser: {raw_input}\n\nAssistant:"


class ContextStore:
    """session_id -> SessionContext; created once, immutable thereafter (DoD 5)."""

    def __init__(self, bundle_dir: Optional[Path] = None,
                 default_model: str = "deepgravity-core-14b:latest"):
        self._bundle_dir = bundle_dir
        self._default_model = default_model
        self._cache: Dict[str, SessionContext] = {}

    def get(self, session_id: str) -> SessionContext:
        if session_id not in self._cache:
            self._cache[session_id] = SessionContext(
                session_id=session_id,
                system_prompt=load_bundle(self._bundle_dir),
                model=self._default_model,
            )
        return self._cache[session_id]

    def __len__(self) -> int:
        return len(self._cache)
