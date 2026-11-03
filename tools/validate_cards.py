"""Validate the live model_cards/ store against the schema-driven rules in engine_registry.

Run from /srv/dg-dev:  ./.venv/bin/python tools/validate_cards.py
Prints per-card dialect class + validation issues. Read-only; makes no writes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core.engine_registry import EngineRegistry, classify_dialect

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "engine/src/deepgravity_core/model_cards"

reg = EngineRegistry(CARDS)
records = reg.records()
print(f"Loaded {len(records)} card(s) from {CARDS}")
print(f"Registered tags: {sorted(reg.by_tag)}")
print()

problems = reg.validate_all()
for digest, rec in sorted(reg.by_digest.items()):
    print(f"- {rec.get('model')}  [{digest}]  class={classify_dialect(rec)}")
    issues = problems.get(digest, [])
    if issues:
        for i in issues:
            print(f"    ISSUE: {i}")
    else:
        print("    clean")

bad = {d for d, v in problems.items() if v}
print()
print(f"RESULT: {len(records) - len(bad)}/{len(records)} cards conform" if records else "no cards")
sys.exit(1 if bad else 0)
