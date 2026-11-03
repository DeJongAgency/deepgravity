"""registry.py — DoD 4: tool_registry in the write path.

Before any file write or script creation, the engine checks tool_registry and
surfaces the result. The temp-script reflex must be impossible, not
discouraged. tool_registry is the ONLY discovery path for tools.

This module wraps the dev-scoped SQLite store (store.py) and the structured
Tool type (transport.py). The write-path guard (tool_crib_check) is the
surface-visible declaration the engine makes before any write.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Union

from . import store
from .transport import Tool, TransportError

def register(tool: Tool, *, layer: str = "dev", domain: str = "", db=None) -> None:
    """Register (or replace) a tool in the registry."""
    store.init_schema(db)
    store.execute(
        """
        INSERT INTO tool_registry
            (name, layer, domain, description, argv_json, param_types_json, cwd, timeout)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            layer=excluded.layer, domain=excluded.domain,
            description=excluded.description, argv_json=excluded.argv_json,
            param_types_json=excluded.param_types_json,
            cwd=excluded.cwd, timeout=excluded.timeout
        """,
        (tool.name, layer, domain, tool.description,
         json.dumps(list(tool.argv)), json.dumps(tool.param_types),
         tool.cwd, tool.timeout),
        db=db,
    )


def _row_to_tool(row) -> Tool:
    return Tool(
        name=row[1],
        argv=tuple(json.loads(row[5])),
        description=row[4],
        param_types=json.loads(row[6]),
        cwd=row[7],
        timeout=row[8],
    )


def get(name: str, db=None) -> Tool:
    """Fetch a tool by name. Raises TransportError if absent (never silently
    falls back to a temp script)."""
    store.init_schema(db)
    row = store.query_one(
        "SELECT * FROM tool_registry WHERE name = ?", (name,), db=db)
    if row is None:
        raise TransportError(f"tool_registry: no tool named '{name}' — write nothing until this resolves")
    return _row_to_tool(row)


def search(term: str = "", domain: str = "", db=None) -> List[dict]:
    """List/search the registry (name, layer, domain, description)."""
    store.init_schema(db)
    rows = store.query_all(
        """
        SELECT name, layer, domain, description FROM tool_registry
        WHERE (? = '' OR name LIKE ? OR description LIKE ?)
          AND (? = '' OR domain = ?)
        ORDER BY domain, name
        """,
        (term, f"%{term}%", f"%{term}%", domain, domain),
        db=db,
    )
    return [
        {"name": r[0], "layer": r[1], "domain": r[2], "description": r[3]}
        for r in rows
    ]


def tool_crib_check(need: str, db=None) -> dict:
    """THE write-path guard (DoD 4). Call before ANY file write / script
    creation. Surfaces the result: either an existing tool matches the need,
    or the caller reports 'no existing tool found for [need]' before writing
    anything new."""
    hits = search(need, db=db)
    return {
        "need": need,
        "matched": bool(hits),
        "tools": hits,
        "verdict": (
            f"USE EXISTING: {hits[0]['name']}" if hits
            else f"No existing tool or Braid method found for [{need}]."
        ),
    }
