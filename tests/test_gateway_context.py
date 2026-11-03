"""test_gateway_context.py — DoD 5 (static session context) + gateway isolation.

Gateway tested with an injected fake completion — never a live ollama dep in
tests. Plain asserts; uses fastapi TestClient (httpx present in the rack).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from fastapi.testclient import TestClient

from deepgravity_core import context as ctx
from deepgravity_core.gateway import create_app


def test_bundle_loads_sorted_immutable_prompt():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "b_identity.md").write_text("# Identity\nI am a test occupant.", encoding="utf-8")
        (p / "a_stance.md").write_text("Stand firmly.", encoding="utf-8")
        prompt = ctx.load_bundle(p)
        assert "I am a test occupant." in prompt and "Stand firmly." in prompt
        # sorted: a_stance first, b_identity second
        assert prompt.index("Stand firmly.") < prompt.index("I am a test occupant.")


def test_missing_bundle_gives_minimal():
    assert "DeepGravity engine core" in ctx.load_bundle(Path("/nonexistent/bundle"))


def test_context_store_immutable_per_session():
    store = ctx.ContextStore()
    c1 = store.get("sess-1")
    c2 = store.get("sess-1")
    assert c1 is c2, "same session must return the SAME immutable context (DoD 5)"
    assert len(store) == 1
    store.get("sess-2")
    assert len(store) == 2


def _fake_completion():
    def gen(sess_ctx, user_text):
        yield "hello"
        yield " "
        yield "world"

    return gen


def test_gateway_chat_sse_structured():
    app = create_app(completion=_fake_completion())
    client = TestClient(app)
    r = client.post("/v1/chat",
                    json={"session_id": "s1", "thread_id": "t1", "raw_input": "hi"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    body = r.text
    assert '"hello"' in body and '"world"' in body
    assert '"done": true' in body


def test_gateway_rejects_bad_payload():
    app = create_app()
    client = TestClient(app)
    r = client.post("/v1/chat", json={"session_id": "s1"})  # missing raw_input
    assert r.status_code == 422


def test_gateway_health():
    app = create_app()
    client = TestClient(app)
    r = client.get("/v1/health")
    assert r.status_code == 200 and r.json()["ok"] is True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
