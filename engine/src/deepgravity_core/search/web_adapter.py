"""web_adapter.py — DoD 7 periphery: SearXNG is an ADAPTER, never the core.

The owned index (owned_index.py) is the real engine; web fetch is periphery.
This module speaks to the SearXNG standalone instance (native, systemd,
0.0.0.0:8888 per System Topology). Results write THROUGH to the owned index
(source_type=web_page) via /learn — the seam the engine owns.

Graceful degradation is mandatory: if SearXNG is down, return a structured
error result — never crash the turn, never fake results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from . import owned_index

DEFAULT_BASE = "http://127.0.0.1:8888"
TIMEOUT = 10.0


class WebAdapterError(RuntimeError):
    pass


def fetch(query: str, base: str = DEFAULT_BASE, max_results: int = 5) -> List[dict]:
    """Query SearXNG (JSON). Raises WebAdapterError on unreachable/non-JSON —
    callers catch and degrade, never crash."""
    try:
        resp = httpx.get(
            f"{base}/search",
            params={"q": query, "format": "json"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        raise WebAdapterError(f"searxng unreachable at {base}: {e}") from e

    results = []
    for item in (data.get("results") or [])[:max_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", "") or item.get("snippet", ""),
        })
    return results


def learn(query: str, base: str = DEFAULT_BASE, db=None) -> dict:
    """fetch() + write results through to the owned index (source_type=web_page).
    The /learn seam: web results are not a separate silo — they join the owned
    index, tagged by source."""
    try:
        results = fetch(query, base=base)
    except WebAdapterError as e:
        return {"ok": False, "error": str(e), "learned": 0}
    n = 0
    for r in results:
        owned_index.add("web_page", r["content"], {"title": r["title"], "url": r["url"], "query": query}, db=db)
        n += 1
    return {"ok": True, "learned": n}
