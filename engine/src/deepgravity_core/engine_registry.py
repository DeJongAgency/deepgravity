"""engine_registry.py — per-artifact model registry (facts only).

Loads model_cards/*.json into a digest-keyed store with tag->digest alias
resolution. The canonical record shape is the on-card format written by the
probe machinery, NOT the aspirational schema v0.1 file — implementing v0.1
against the real cards revealed the schema had drifted from what probes
actually emit (00:39 cards vs 00:45 schema). The cards are the truth.

FACTS ONLY. No routing decisions, no occupant values, no content classes:
the router reads these records and applies the loaded standard on top
(04:16 doctrine — facts in the record, values in the occupant).

Write path: append_note() / put_record() — probes accrue observations
through here. Atomic temp+rename via writes.atomic_write_bytes (DoD 8).
One writer, append-only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import writes

CARD_VERSION = 1
_LATEST = ":latest"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _norm_tag(tag: str) -> str:
    """Ollama naming: 'name' and 'name:latest' are the same artifact."""
    return tag[: -len(_LATEST)] if tag.endswith(_LATEST) else tag


def classify_dialect(rec: dict) -> str:
    """Probe-observed dialect class, derived live (never stored, never stale).

    standard | thinking | hazardous_template | unknown
    """
    d = rec.get("dialect") or {}
    if d.get("peg_parser_hazard") or d.get("raw_output_markers"):
        return "hazardous_template"
    if d.get("thinking_field") not in (None, "", "none") or d.get("has_thinking"):
        return "thinking"
    if d.get("content_only") is True:
        return "standard"
    return "unknown"


def _card_filename(rec: dict) -> str:
    """Stable human-readable filename from the model tag. Never the digest —
    readable diffs matter; digest stays the key inside the file."""
    base = rec.get("model") or rec.get("family") or "unnamed"
    slug = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in _norm_tag(base))
    slug = slug.strip(".-").lower()
    return (slug or "unnamed") + ".json"


class EngineRegistry:
    """Digest-keyed store over a model_cards/ directory. One card per artifact."""

    def __init__(self, cards_dir):
        self.cards_dir = Path(cards_dir)
        self.cards_dir.mkdir(parents=True, exist_ok=True)
        self.by_digest: dict[str, dict] = {}
        self._paths: dict[str, Path] = {}
        self.by_tag: dict[str, str] = {}  # normalized tag -> digest (current aliases only)
        self.reload()

    # ---------------- loading ----------------

    def reload(self) -> None:
        self.by_digest = {}
        self._paths = {}
        self.by_tag = {}
        for p in sorted(self.cards_dir.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"[engine_registry] skip {p.name}: {e}")
                continue
            digest = rec.get("digest")
            if not digest:
                print(f"[engine_registry] skip {p.name}: no digest key")
                continue
            if digest in self.by_digest:
                print(f"[engine_registry] DUPLICATE digest {digest}: {p.name} ignored")
                continue
            self.by_digest[digest] = rec
            self._paths[digest] = p
            self._index_tag(rec.get("model"), digest)
            for alias in rec.get("aliases") or []:
                if alias.get("last_seen") is None:  # current pointer only
                    self._index_tag(alias.get("tag"), digest)

    def _index_tag(self, tag, digest) -> None:
        if not tag:
            return
        self.by_tag[tag] = digest
        norm = _norm_tag(tag)
        if norm != tag:
            self.by_tag[norm] = digest

    # ---------------- queries ----------------

    def records(self) -> list[dict]:
        return list(self.by_digest.values())

    def by_digest(self, digest: str) -> dict | None:
        return self.by_digest.get(digest)

    def by_model(self, tag: str) -> dict | None:
        """Resolve a tag/model name to its current record. ':latest' re-points
        by moving the alias's last_seen; old records keep their notes."""
        digest = self.by_tag.get(tag)
        if digest is None and _norm_tag(tag) != tag:
            digest = self.by_tag.get(_norm_tag(tag))
        return self.by_digest.get(digest) if digest else None

    def get(self, digest_or_tag: str) -> dict | None:
        rec = self.by_digest.get(digest_or_tag)
        if rec is None:
            rec = self.by_model(digest_or_tag)
        return rec

    def query(self, dialect_class: str | None = None, family: str | None = None) -> list[dict]:
        out = []
        for rec in self.by_digest.values():
            if dialect_class and classify_dialect(rec) != dialect_class:
                continue
            if family and rec.get("family") != family:
                continue
            out.append(rec)
        return out

    # ---------------- validation ----------------

    def validate_all(self) -> dict[str, list[str]]:
        """Returns {digest: [problems]}. Empty list = conforms."""
        out = {}
        for digest, rec in self.by_digest.items():
            problems = []
            for k in ("model", "family", "digest", "dialect"):
                if k not in rec:
                    problems.append(f"missing top-level '{k}'")
            d = rec.get("dialect") or {}
            if d.get("api") not in (None, "chat", "generate", "custom"):
                problems.append(f"dialect.api unknown: {d.get('api')!r}")
            if classify_dialect(rec) == "unknown":
                problems.append("cannot derive dialect_class: set content_only/has_thinking/thinking_field")
            notes = rec.get("behavioral_notes")
            if notes is not None and not isinstance(notes, list):
                problems.append("behavioral_notes must be a list")
            out[digest] = problems
        return out

    # ---------------- write path (probes accrue here) ----------------

    def append_note(self, tag: str, note: str, observed_by: str = "Dora (probe)",
                    ts: str | None = None) -> dict:
        """Append a behavioral observation to the record for tag; atomic write-back.

        This is the write-through probe path: every touch that reveals
        something appends here. Tag resolution happens at the moment of
        touch, so a note is attributed to the artifact actually talked to.
        """
        rec = self.by_model(tag)
        if rec is None:
            raise KeyError(f"no record for tag/model {tag!r} — put_record first")
        notes = rec.setdefault("behavioral_notes", [])
        notes.append({"ts": ts or _utc_now(), "note": note, "observed_by": observed_by})
        self._save(rec)
        return rec

    def put_record(self, rec: dict, overwrite: bool = False) -> dict:
        """Insert or replace a full record (new card, or corrected burn)."""
        digest = rec.get("digest")
        if not digest:
            raise ValueError("record has no digest key")
        existing = digest in self.by_digest
        if existing and not overwrite:
            raise ValueError(f"digest {digest} already registered (overwrite=True to replace)")
        self._save(rec)
        self.reload()
        return rec

    def _save(self, rec: dict) -> None:
        digest = rec.get("digest")
        path = self._paths.get(digest) or (self.cards_dir / _card_filename(rec))
        payload = json.dumps(rec, indent=2, ensure_ascii=False) + "\n"
        writes.atomic_write_bytes(path, payload.encode("utf-8"))
        self._paths[digest] = path
