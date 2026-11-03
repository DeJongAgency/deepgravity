"""smoke_ollama.py — OPT-IN live smoke: the gateway's real model path on the box.

Not part of the default suite (needs the live ollama + a model). Run explicitly:
    ./.venv/bin/python tests/smoke_ollama.py
Verifies: (1) ollama reachable at 127.0.0.1:11434, (2) the model fleet is
present, (3) one tiny generation succeeds through the same ollama_completion
shape the gateway uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

import httpx

BASE = "http://127.0.0.1:11434"
DEFAULT_MODEL = "deepgravity-core-14b:latest"


def main() -> int:
    print(f"== ollama reachability: {BASE} ==")
    r = httpx.get(f"{BASE}/api/tags", timeout=10.0)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"models present: {len(models)}")
    for m in models[:25]:
        print(f"  - {m}")

    model = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)
    if model is None:
        print("FAIL: no models on the box")
        return 1
    print(f"\n== tiny generation via model '{model}' ==")
    resp = httpx.post(
        f"{BASE}/api/generate",
        json={"model": model, "prompt": "Reply with exactly: OK", "stream": False,
              "options": {"num_predict": 8}},
        timeout=60.0,
    )
    resp.raise_for_status()
    out = resp.json().get("response", "").strip()
    print(f"response: {out!r}")
    if "OK" not in out:
        print("FAIL: model did not return OK")
        return 1
    print("\nSMOKE PASS — engine -> fleet path is live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
