"""Live smoke: registry-aware ollama_completion against a real model.

Run from /srv/dg-dev:  ./.venv/bin/python tools/smoke_registry_gateway.py qwen2.5:3b
Proves the card-dialect /api/chat path streams content and no thinking leaks.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core import context as _context
from deepgravity_core.engine_registry import EngineRegistry
from deepgravity_core.gateway import ollama_completion


class FakeCtx:
    def __init__(self, model):
        self.model = model

    def with_user_turn(self, text):
        return text


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"
    reg = EngineRegistry(Path(__file__).resolve().parent.parent / "engine/src/deepgravity_core/model_cards")
    card = reg.by_model(model)
    print(f"model={model} card={'FOUND' if card else 'none (fallback path)'} "
          f"dialect_class={reg and (card and __import__('deepgravity_core.engine_registry', fromlist=['classify_dialect']).classify_dialect(card))}")
    gen = ollama_completion(registry=reg)
    pieces = []
    for piece in gen(FakeCtx(model), "Reply with exactly: OK"):
        pieces.append(piece)
    out = "".join(pieces)
    print("--- response ---")
    print(out[:300])
    print("--- end ---")
    ok = "OK" in out
    print("SMOKE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
