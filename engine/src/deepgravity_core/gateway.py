"""gateway.py — the v1 chat gateway (Main Engine Core boundary).

Per gateway-spec-v2 + ui-layout-spec (both LOCKED): the frontend is stateless
and emits ONLY structured event packets ({session_id, thread_id, raw_input})
to /v1; it never calculates context, slices history, or touches stores. This
ASGI app is that boundary. It assembles context (context.py), calls a model
provider over structured transport (never a shell), and streams tokens back
as Server-Sent Events (ui-layout: SSE stream renderer).

Isolation: this module does NOT import store/braid/registry — state lives
behind the service layer. The completion function is injectable for tests.
"""

from __future__ import annotations

import json
from typing import Callable, Iterator, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import context as _context
from .time import now_utc


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    thread_id: str = Field("default", min_length=1)
    raw_input: str = Field(..., min_length=1)


CompletionFn = Callable[[_context.SessionContext, str], Iterator[str]]


def ollama_completion(base: str = "http://127.0.0.1:11434",
                      registry=None) -> CompletionFn:
    """Default completion: local ollama, dialect-aware (registry consumer #1).

    If `registry` (an EngineRegistry) holds a card for ctx.model, frame per
    the card's dialect: /api/chat with messages (ollama applies the model's
    chat template — required for template_dependent models like Glimmer),
    decode message.content only, never stream the thinking field into the
    answer. Unknown model (no card — no telepathy) -> legacy /api/generate.

    httpx only — no shell, no subprocess. Structured transport holds here too.
    """
    import httpx

    def _gen(ctx: _context.SessionContext, user_text: str) -> Iterator[str]:
        card = registry.by_model(ctx.model) if registry is not None else None
        try:
            if card is not None:
                payload = {
                    "model": ctx.model,
                    "messages": [{"role": "user", "content": ctx.with_user_turn(user_text)}],
                    "stream": True,
                }
                with httpx.stream("POST", f"{base}/api/chat",
                                  json=payload, timeout=120.0) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        msg = data.get("message") or {}
                        piece = msg.get("content") or ""
                        if piece:
                            yield piece
                        if data.get("done"):
                            break
            else:
                payload = {
                    "model": ctx.model,
                    "prompt": ctx.with_user_turn(user_text),
                    "stream": True,
                }
                with httpx.stream("POST", f"{base}/api/generate",
                                  json=payload, timeout=120.0) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        piece = data.get("response", "")
                        if piece:
                            yield piece
                        if data.get("done"):
                            break
        except (httpx.HTTPError, ValueError) as e:
            yield f"\n[engine: provider error — {e}]"

    return _gen


def create_app(
    *,
    contexts: Optional[_context.ContextStore] = None,
    completion: Optional[CompletionFn] = None,
    model: str = "deepgravity-core-14b:latest",
) -> FastAPI:
    app = FastAPI(title="deepgravity-core gateway", version="0.1.0")
    store_ctx = contexts or _context.ContextStore(default_model=model)
    complete = completion or ollama_completion()

    @app.get("/v1/health")
    def health() -> dict:
        return {"ok": True, "ts_utc": now_utc(), "model": model}

    @app.post("/v1/chat")
    def chat(req: ChatRequest) -> StreamingResponse:
        ctx = store_ctx.get(req.session_id)

        def _events():
            for piece in complete(ctx, req.raw_input):
                yield f"data: {json.dumps({'token': piece})}\n\n"
            yield f"data: {json.dumps({'done': True, 'session_id': req.session_id})}\n\n"

        return StreamingResponse(_events(), media_type="text/event-stream")

    return app
