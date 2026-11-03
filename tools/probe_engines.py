"""probe_engines.py — cheap dialect probe for the DECLARED roster.

One tiny request per model (serial), observe the response SHAPE:
content field? thinking field? template markers? 500? generate-vs-chat?

The Glimmer diagnostic, applied to her siblings. Unload after each
(keep_alive 0) so nothing stacks resident. Writes JSONL results.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:11434"
OUT = Path("/srv/dg-dev/probe_results.jsonl")

MODELS = [
    "qwen2.5:3b",
    "dolphin3:8b",
    "dora-14b-4K:Qwen-Ablit",
    "deepgravity-core-14b:latest",
    "gurubot/gpt-oss-derestricted:20b",
]

PROBE = "Reply with exactly: OK"


def probe(model: str) -> dict:
    rec = {"model": model, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.time()

    # 1) Try /api/chat first (the modern path)
    try:
        r = httpx.post(f"{BASE}/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": PROBE}],
            "stream": False,
            "keep_alive": 0,
            "options": {"num_predict": 25},
        }, timeout=120.0)
        rec["chat_status"] = r.status_code
        if r.status_code == 200:
            data = r.json()
            msg = data.get("message", {})
            rec["chat_message_keys"] = sorted(msg.keys())
            rec["chat_content"] = (msg.get("content") or "")[:120]
            rec["has_thinking"] = bool(msg.get("thinking") or msg.get("reasoning_content"))
            if rec["has_thinking"]:
                rec["thinking_field"] = "thinking" if "thinking" in msg else "reasoning_content"
                rec["thinking_tail"] = ((msg.get("thinking") or msg.get("reasoning_content")) or "")[-150:]
            # detect template-native garbage in content
            import re
            markers = re.findall(r"<\|[^|]+\|>", rec["chat_content"])
            rec["template_markers_in_content"] = markers
        else:
            rec["chat_error"] = r.text[:200]
    except Exception as e:
        rec["chat_exception"] = str(e)[:150]

    # 2) If chat failed or suspicious, try /api/generate to compare
    if rec.get("chat_status") != 200 or rec.get("template_markers_in_content"):
        try:
            r = httpx.post(f"{BASE}/api/generate", json={
                "model": model,
                "prompt": PROBE,
                "stream": False,
                "keep_alive": 0,
                "options": {"num_predict": 25},
            }, timeout=120.0)
            rec["gen_status"] = r.status_code
            if r.status_code == 200:
                data = r.json()
                rec["gen_content"] = (data.get("response") or "")[:120]
                import re
                rec["gen_template_markers"] = re.findall(r"<\|[^|]+\|>", rec["gen_content"])
            else:
                rec["gen_error"] = r.text[:200]
        except Exception as e:
            rec["gen_exception"] = str(e)[:150]

    rec["elapsed_secs"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        for model in MODELS:
            print(f"[probe] {model} ...", flush=True)
            rec = probe(model)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            # summarize
            keys = rec.get("chat_message_keys", [])
            think = "THINKING" if rec.get("has_thinking") else "no-thinking"
            markers = rec.get("template_markers_in_content") or rec.get("gen_template_markers") or []
            m = f"  chat={rec.get('chat_status')} keys={keys} {think} markers={markers}"
            print(m, flush=True)
            time.sleep(1)  # breath between models
    print(f"[probe] DONE — results in {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
