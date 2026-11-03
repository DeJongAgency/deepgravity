#!/usr/bin/env python3
"""run_compact.py — one compaction pass over closed sessions (dg-compact).

Thin runner for the dg-compact systemd service. It scans a directory of
session records (the archiver's flat-JSON shape), compacts each finished,
not-yet-compacted session through the LOCAL llama-server on 127.0.0.1:8209
(one small slot, low temperature), writes each distilled entry as JSON to
an output directory, and records the session id in the state file.

What this deliberately does NOT do:
- It never touches the real braid (/home/john/.local/share/deepgrav/...)
  and never touches the keystore. Output goes to --out-dir (default under
  /srv/dg-dev/data/) so the entries can be reviewed before anything lands
  in the braid. Wiring the real (encrypted, john-owned) session store in
  as --sessions-dir and the braid as the sink is a later step, on the
  coordinator's go.
- It never runs while a conversation is live: before calling the engine
  it checks llama-server's /slots, and if any slot is processing it exits
  0 having done nothing. The timer fires every 6 hours; the idle check is
  the real guard, the schedule is just the cadence.
- It never compacts a live session: only records whose status is "closed"
  or "finished" pass should_compact(). A bare list-of-turns file has no
  status, so it is skipped with a warning, not guessed at.

Usage:
    run_compact.py [--sessions-dir DIR] [--out-dir DIR] [--state FILE]
                   [--engine-url URL] [--model MODEL] [--min-turns N]
                   [--dry-run]

Exit codes: 0 = pass done (including "engine busy, skipped" and "nothing
to do"); 1 = engine unreachable or another hard failure (systemd logs it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core import dg_compact as dc  # noqa: E402
from deepgravity_core.writes import atomic_write_text  # noqa: E402

DEFAULT_SESSIONS_DIR = "/srv/dg-dev/data/sessions"
DEFAULT_OUT_DIR = "/srv/dg-dev/data/compacted"
DEFAULT_ENGINE_URL = "http://127.0.0.1:8209"


def engine_is_idle(engine_url: str, timeout: float = 10.0) -> bool:
    """True when no llama-server slot is processing a request.

    Raises httpx.HTTPError if the engine cannot be reached at all — that
    is a hard failure (exit 1), distinct from "busy" (exit 0, try later).
    """
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(f"{engine_url.rstrip('/')}/slots")
        resp.raise_for_status()
        slots = resp.json()
    return not any(slot.get("is_processing") for slot in slots)


def local_engine_fn(engine_url: str, model: str, timeout: float = 300.0):
    """Injected engine function: one small, non-streaming chat completion.

    Local llama-server only (127.0.0.1:8209, alias 5q30b). Low temperature:
    distillation should be faithful, not creative.
    """
    url = f"{engine_url.rstrip('/')}/v1/chat/completions"

    def call(prompt: str) -> str:
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.2, "max_tokens": 2048, "stream": False}
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    return call


def load_session_record(path: Path) -> dict | None:
    """Read one session file. Bare turn-lists have no status/source, so
    they are skipped (return None) rather than guessed at."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"WARN {path.name}: unreadable, skipping ({exc})")
        return None
    if isinstance(data, list):
        print(f"WARN {path.name}: bare turn list, no status/source — skipping "
              "(archiver dict records only)")
        return None
    if not isinstance(data, dict) or "session_id" not in data:
        print(f"WARN {path.name}: no session_id, skipping")
        return None
    return data


def run_pass(sessions_dir: Path, out_dir: Path, state_path: Path,
             engine_fn, *, min_turns: int, dry_run: bool) -> dict:
    """One pass. Returns counts: scanned, compacted, skipped, failed."""
    counts = {"scanned": 0, "compacted": 0, "skipped": 0, "failed": 0}
    compacted_ids = dc.load_compacted_ids(state_path)
    for path in sorted(sessions_dir.glob("*.json")):
        counts["scanned"] += 1
        record = load_session_record(path)
        if record is None:
            counts["skipped"] += 1
            continue
        session_id = str(record["session_id"])
        if not dc.should_compact(record, min_turns=min_turns,
                                 compacted_ids=compacted_ids):
            counts["skipped"] += 1
            continue
        try:
            entry = dc.compact_session(record, engine_fn, engine_local=True)
        except ValueError as exc:
            print(f"FAIL {session_id}: {exc}")
            counts["failed"] += 1
            continue
        if dry_run:
            print(f"DRY-RUN {session_id}: would write entry "
                  f"(ccf={entry['ccf']}, decisions={len(entry['decisions'])})")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                out_dir / f"{session_id}.compact.json",
                json.dumps(entry, indent=2, sort_keys=True) + "\n")
            compacted_ids = dc.mark_compacted(state_path, session_id)
            print(f"OK {session_id}: ccf={entry['ccf']} source={entry['source']} "
                  f"decisions={len(entry['decisions'])} "
                  f"open={len(entry['open_items'])}")
        counts["compacted"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions-dir", default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--state", default=dc.DEFAULT_STATE_PATH)
    parser.add_argument("--engine-url", default=DEFAULT_ENGINE_URL)
    parser.add_argument("--model", default=dc.DEFAULT_MODEL)
    parser.add_argument("--min-turns", type=int, default=dc.DEFAULT_MIN_TURNS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    sessions_dir = Path(args.sessions_dir)
    if not sessions_dir.is_dir():
        print(f"No sessions dir {sessions_dir} — nothing to do.")
        return 0
    try:
        if not engine_is_idle(args.engine_url):
            print("Engine busy (a slot is processing) — skipping this pass, "
                  "the timer will retry in 6 hours.")
            return 0
    except httpx.HTTPError as exc:
        print(f"ERROR: engine at {args.engine_url} unreachable: {exc}")
        return 1

    engine_fn = local_engine_fn(args.engine_url, args.model)
    counts = run_pass(sessions_dir, Path(args.out_dir), Path(args.state),
                      engine_fn, min_turns=args.min_turns, dry_run=args.dry_run)
    print(f"Pass done: scanned={counts['scanned']} "
          f"compacted={counts['compacted']} skipped={counts['skipped']} "
          f"failed={counts['failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
