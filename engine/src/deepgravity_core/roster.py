"""roster.py — engine mounts & roster maintenance (v1: declare + reconcile).

The fleet's service layer. The canonical declared roster (the ladder, per
2026.09.07.0529 Fleet Status) lives in roster.json beside this module.
reconcile() compares DECLARED vs LIVE (ollama /api/tags) and classifies every
live tag:

  KEPT     — declared and live (healthy)
  MISSING  — declared KEEP but absent live (drift)
  QUEUED   — declared, not yet mounted (expected)
  STALE    — declared but marked for replacement
  UNLISTED — live but not in the declared roster (cull targets, raw imports,
             or models awaiting a roster decision)

READ-ONLY in v1: reconcile reports drift; it never deletes or mounts.
Pull, don't poll: run on demand / at tie-off / after any fleet op.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import httpx

OLLAMA_BASE = "http://127.0.0.1:11434"
ROSTER_PATH = Path(__file__).resolve().parent / "roster.json"
ARCHIVE_DIR = "/mnt/1thd01/models"


def load_roster(path: Path = ROSTER_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def live_tags(base: str = OLLAMA_BASE, timeout: float = 10.0) -> List[str]:
    resp = httpx.get(f"{base}/api/tags", timeout=timeout)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def archive_ggufs(archive_dir: str = ARCHIVE_DIR) -> List[str]:
    p = Path(archive_dir)
    if not p.exists():
        return []
    return sorted(str(x.name) for x in p.glob("*.gguf"))


def reconcile(base: str = OLLAMA_BASE, archive_dir: str = ARCHIVE_DIR,
              roster_path: Path = ROSTER_PATH) -> Dict[str, object]:
    """Compare declared roster vs live ollama tags. Read-only classification."""
    roster = load_roster(roster_path)
    declared = roster["rungs"]
    live = set(live_tags(base))
    archive = archive_ggufs(archive_dir)

    by_status: Dict[str, List[dict]] = {
        "KEPT": [], "MISSING": [], "QUEUED": [], "STALE": [], "UNLISTED": []}
    declared_names = set()
    for r in declared:
        declared_names.add(r["model"])
        st = r.get("status", "KEEP").upper()
        if st == "KEEP":
            (by_status["KEPT"] if r["model"] in live else by_status["MISSING"]).append(r)
        elif st.startswith("STALE"):
            by_status["STALE"].append(r)
        elif st == "QUEUED":
            by_status["QUEUED"].append(r)
        else:  # TEST / other declared-but-present
            (by_status["KEPT"] if r["model"] in live else by_status["MISSING"]).append(r)

    for tag in sorted(live):
        if tag not in declared_names:
            by_status["UNLISTED"].append({"model": tag})

    return {
        "as_of": roster.get("as_of"),
        "declared": len(declared),
        "live": len(live),
        "archive_ggufs": len(archive),
        "status": by_status,
        "healthy": not by_status["MISSING"] and not by_status["UNLISTED"],
    }


def report(result: Dict[str, object]) -> str:
    """Human-readable status table (status-bar-shaped, not a firehose)."""
    s = result["status"]
    lines = []
    lines.append(f"ROSTER — declared {result['declared']} · live {result['live']} · archive GGUFs {result['archive_ggufs']} · healthy={result['healthy']}")
    lines.append("")
    lines.append(f"  KEPT ({len(s['KEPT'])}):      " + ", ".join(r["model"] for r in s["KEPT"]) or "—")
    lines.append(f"  MISSING ({len(s['MISSING'])}):  " + ", ".join(r["model"] for r in s["MISSING"]) or "—")
    lines.append(f"  QUEUED ({len(s['QUEUED'])}):   " + ", ".join(r["model"] for r in s["QUEUED"]) or "—")
    lines.append(f"  STALE ({len(s['STALE'])}):     " + ", ".join(r["model"] for r in s["STALE"]) or "—")
    lines.append(f"  UNLISTED ({len(s['UNLISTED'])}): ")
    for r in s["UNLISTED"]:
        lines.append(f"      ? {r['model']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(reconcile()))
