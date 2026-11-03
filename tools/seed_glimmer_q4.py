"""Seed the missing Muse Glimmer Q4 card into model_cards/ via put_record().

Run from /srv/dg-dev:  ./.venv/bin/python tools/seed_glimmer_q4.py <ollama-digest>
Digest: pass the bare hex from 'ollama show muse-glimmer-30b-ablit:q4' (or with sha256: prefix).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core.engine_registry import EngineRegistry

raw = sys.argv[1] if len(sys.argv) > 1 else ""
digest = raw.replace("sha256:", "").strip()
if not digest:
    print("usage: seed_glimmer_q4.py <digest>")
    sys.exit(2)

rec = {
    "model": "muse-glimmer-30b-ablit:q4",
    "family": "muse-glimmer",
    "digest": digest,
    "base_identity": {
        "params": "27.9B",
        "quant": "Q4_K_M",
        "size_gb": 16.9,
        "source": "Blackfrost-AI/Muse-Glimmer-30B-Abliterated-GGUF",
    },
    "dialect": {
        "api": "chat",
        "template_dependent": True,
        "thinking_field": "thinking",
        "has_thinking": True,
        "raw_output_markers": ["<|start|>", "<|message|>", "<|eot|>"],
        "peg_parser_hazard": True,
        "content_only": False,
    },
    "tuning": {"sane_ctx": 16384, "max_ctx": 32768, "kv_cache": "q8_0", "keep_alive_secs": 1800},
    "boundary_profile": {
        "abliterated": True,
        "refusal_residual": "spec-negotiation circuit intact post-ablit; reasons about identity policy before answering",
        "voice_mirrors_frame": True,
        "loaded_standard_fidelity": "untested",
    },
    "behavioral_notes": [
        {"ts": "2026-09-08T00:20Z", "note": "Needs /api/chat framing; /api/generate produces template garbage", "observed_by": "Dora"},
        {"ts": "2026-09-08T00:20Z", "note": "Thinking model: separate thinking field holds deliberation, content holds answer", "observed_by": "Dora"},
        {"ts": "2026-09-08T00:20Z", "note": "Raw output template markers 500 ollama's PEG parser", "observed_by": "Dora+JH"},
        {"ts": "2026-09-08T00:20Z", "note": "Voice mirrors frame exactly", "observed_by": "Dora+JH"},
    ],
    "card_version": 1,
}

reg = EngineRegistry(Path(__file__).resolve().parent.parent / "engine/src/deepgravity_core/model_cards")
reg.put_record(rec)
print(f"inserted {rec['model']} digest={digest}")
print(f"store now has {len(reg.records())} card(s):")
for r in sorted(reg.records(), key=lambda x: x.get("model", "")):
    print(f"  - {r.get('model')}  [{r.get('digest')}]")
