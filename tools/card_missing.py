"""card_missing.py — write cards for every resident model missing one.

Cards q5/q6 (glimmer ladder), llama3.3-70b-ablit (yardstick), dora-vl-7b (vision)
via the registry write path (put_record). Run after mount, before first use.
Run from /srv/dg-dev:  ./.venv/bin/python tools/card_missing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core.engine_registry import EngineRegistry

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "engine/src/deepgravity_core/model_cards"

# --- muse-glimmer-30b-ablit:q5 (mounted, measured tonight) ---
Q5 = {
    "model": "muse-glimmer-30b-ablit:q5",
    "family": "muse-glimmer",
    "digest": "f3a2ad5bd919",
    "base_identity": {"params": "27.9B", "quant": "Q5_K_M", "size_gb": 19,
                      "source": "Blackfrost-AI/Muse-Glimmer-30B-Abliterated-GGUF"},
    "dialect": {"api": "chat", "template_dependent": True, "thinking_field": "thinking",
                "has_thinking": True, "raw_output_markers": ["<|start|>", "<|message|>", "<|eot|>"],
                "peg_parser_hazard": True, "content_only": False},
    "tuning": {"sane_ctx": 16384, "max_ctx": 32768, "kv_cache": "q8_0", "keep_alive_secs": 1800},
    "boundary_profile": {"abliterated": True,
                         "refusal_residual": "spec-negotiation circuit intact post-ablit; reasons about identity policy before answering",
                         "voice_mirrors_frame": True, "loaded_standard_fidelity": "untested"},
    "behavioral_notes": [
        {"ts": "2026-09-08T01:5xZ", "note": "Mounted from verified GGUF (sha21073f1a...). 37.3 tok/s measured warm (vs q4 41.5). -10% for the quant climb.", "observed_by": "Dora"},
        {"ts": "2026-09-08T01:5xZ", "note": "Identity question ('what is your name') triggers full spec-negotiation spiral in thinking field before compliant answer ('I am an AI assistant').", "observed_by": "Dora"},
        {"ts": "2026-09-08T01:5xZ", "note": "PROVEN: 300-token leash -> content empty (all budget to thinking). 4000 budget -> 268 tok, clean stop, answer lands. Must give budget past the thought block.", "observed_by": "Dora"},
    ],
    "card_version": 1,
}

# --- muse-glimmer-30b-ablit:q6 (mounted, unmeasured) ---
Q6 = {
    "model": "muse-glimmer-30b-ablit:q6",
    "family": "muse-glimmer",
    "digest": "826a5df761e7",
    "base_identity": {"params": "27.9B", "quant": "Q6_K", "size_gb": 22,
                      "source": "Blackfrost-AI/Muse-Glimmer-30B-Abliterated-GGUF"},
    "dialect": {"api": "chat", "template_dependent": True, "thinking_field": "thinking",
                "has_thinking": True, "raw_output_markers": ["<|start|>", "<|message|>", "<|eot|>"],
                "peg_parser_hazard": True, "content_only": False},
    "tuning": {"sane_ctx": 16384, "max_ctx": 32768, "kv_cache": "q8_0", "keep_alive_secs": 1800},
    "boundary_profile": {"abliterated": True,
                         "refusal_residual": "spec-negotiation circuit intact post-ablit; reasons about identity policy before answering",
                         "voice_mirrors_frame": True, "loaded_standard_fidelity": "untested"},
    "behavioral_notes": [
        {"ts": "2026-09-08T01:2xZ", "note": "Mounted from verified GGUF (sha f5487c50...). A/B measurement pending (q4 41.5 / q5 37.3 baseline).", "observed_by": "Dora"},
        {"ts": "2026-09-08T10:1xZ", "note": "A/B MEASURED: 31.8 tok/s warm (q4 41.5 / q5 37.3). Cold-load 49.6s. Identity Q → spec spiral (thinking 1014) → compliant answer. Same circuit as q4/q5 — quant changes speed, not character.", "observed_by": "Dora"},
    ],
    "card_version": 1,
}

# --- llama3.3-70b-ablit (yardstick / insurance) ---
Q70 = {
    "model": "llama3.3-70b-ablit:latest",
    "family": "llama3.3-70b-ablit",
    "digest": "41e67a540a3c",
    "base_identity": {"params": "70B", "quant": "Q4_K_M", "size_gb": 42,
                      "source": "ollama (huihui ablit)"},
    "dialect": {"api": "chat", "template_dependent": False, "thinking_field": "none",
                "has_thinking": False, "raw_output_markers": [], "peg_parser_hazard": False,
                "content_only": True},
    "tuning": {"sane_ctx": 8192, "keep_alive_secs": 600, "kv_cache": "q8_0"},
    "boundary_profile": {"abliterated": True, "refusal_residual": "ablit", "voice_mirrors_frame": False,
                         "loaded_standard_fidelity": "untested"},
    "behavioral_notes": [
        {"ts": "2026-09-08T02:0xZ", "note": "Yardstick/insurance only: 1.7 tok/s measured (CPU spill — does not fit on-card). Quality reference, not a daily driver.", "observed_by": "Dora"},
    ],
    "card_version": 1,
}

# --- dora-vl-7b (vision slot, kept from cull) ---
VL = {
    "model": "dora-vl-7b:latest",
    "family": "qwen2.5-vl-7b-ablit",
    "digest": "fa171fbfcf6e",
    "base_identity": {"params": "7.6B", "quant": "Q4_K_M", "size_gb": 5.5,
                      "source": "local-store (mrad Qwen2.5-VL-7B-ablit base)"},
    "dialect": {"api": "chat", "template_dependent": True, "thinking_field": "none",
                "has_thinking": False, "raw_output_markers": [], "peg_parser_hazard": False,
                "content_only": True},
    "tuning": {"sane_ctx": 8192, "keep_alive_secs": 600, "kv_cache": "q8_0"},
    "boundary_profile": {"abliterated": True, "refusal_residual": "unknown", "voice_mirrors_frame": False,
                         "loaded_standard_fidelity": "untested"},
    "behavioral_notes": [
        {"ts": "2026-09-08T02:0xZ", "note": "Vision slot (vision_in: content_parts_base64). Kept from cull. Not yet probed this session.", "observed_by": "Dora"},
    ],
    "card_version": 1,
}

reg = EngineRegistry(CARDS)
before = {r.get("model") for r in reg.records()}
for rec in (Q5, Q6, Q70, VL):
    name = rec["model"]
    if name in before:
        print(f"skip (already carded): {name}")
        continue
    reg.put_record(rec)
    print(f"carded: {name} [{rec['digest']}]")

reg2 = EngineRegistry(CARDS)
print(f"\nstore now has {len(reg2.records())} card(s):")
for r in sorted(reg2.records(), key=lambda x: x.get("model", "")):
    print(f"  - {r.get('model')}  [{r.get('digest')}]")
problems = reg2.validate_all()
bad = {d for d, v in problems.items() if v}
print(f"\nRESULT: {len(reg2.records()) - len(bad)}/{len(reg2.records())} conform"
      if reg2.records() else "no cards")
sys.exit(1 if bad else 0)
