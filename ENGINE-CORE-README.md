# /srv/dg-dev — engine core (BUILD STATE)

**Status**: foundationals landed 2026-09-07 (session chat_20260907_184449_fc34).
Workshop moved to /srv/dg-dev 2026-11-01. Step 0 encryption package landed
2026-11-02 (file-by-file, keystore, per-turn CCF, sourcing). The braid is
NOT here — it is occupant-owned, class 1, at
`~/.local/share/deepgrav/user/{occupant}/data/braid/` (Bookmarks B12/B13).
Upstream: github.com/DeJongAgency/deepgravity, main branch. Only push on
active command and confirm.

## Tree
```
engine/pyproject.toml            hatchling, src layout, name deepgravity-core
engine/src/deepgravity_core/
  __init__.py                    package (deepgravity = product identity; deepgrav = path token)
  time.py        DoD 2  single time rep (UTC Z, engine-side compare, render-local only)
  text.py        DoD 3  UTF-8 NFC at every boundary (strict decode, no U+FFFD)
  transport.py   DoD 1  structured tool transport (model never assembles a shell string)
  writes.py      DoD 8  atomic temp+rename writes, single choke point, orphan sweep
  store.py       dev-scoped SQLite helper (bootstrap store; one conn per call, close fast)
  registry.py    DoD 4  tool_registry in the write path (tool_crib_check)
  telemetry.py   Perf1/DoD6 milestones at rest; cost + syntax-failure counters
  search/
    owned_index.py   DoD 7  THE owned index (single-writer, append-only)
    web_adapter.py   DoD 7  SearXNG adapter (periphery; /learn writes through)
    files_tools.py   DoD 7  view-first, grep-last (no whole-tree by default)
  encryption/          Step 0  file-by-file encryption, keystore, per-turn CCF,
    file_crypt.py      AES-256-GCM envelope, per-file keys, NT placeholder
    keystore.py        passphrase gate, BIP39 recovery, recovery stays in sync
    turns.py           per-turn CCF rendering, NT/NK/CO placeholders
    sourcing.py        model/platform awareness, automatic pull the shades
tests/                 test_time test_text test_transport test_writes
                       test_registry_telemetry test_search test_cloud_ledger
                       test_engine_registry test_gateway_context test_encryption
                       smoke_ollama.py (opt-in live smoke, not in default suite)
data/devstore/         dev-scoped SQLite (tool_registry, milestones, owned_records)
tools/                 validate_cards card_missing seed_glimmer_q4
                       smoke_registry_gateway probe_engines (dev tools, fixed paths)
logs/                  milestone + probe logs at rest
```

## Run the suite
```bash
cd /srv/dg-dev
for t in tests/test_*.py; do ./.venv/bin/python "$t"; done   # 56/56 green (2026-11-02)
```

## FLIP SEAM (task 132 — wired, import NOT run)
The engine's data pointer is DG_DEVSTORE, set in .env to
`/srv/dg-dev/data/devstore/devstore.db` — see `store.py` (copy `.env.example`
to `.env` on a fresh checkout). At flip (B5-C, import-as-first-act), the real
braid imports to `~/.local/share/deepgrav/user/{occupant}/data/braid/` and
the engine's store layer gains a braid-backed adapter at that path; the dev
store then retires. The
owned_index/web_adapter seam is built so web results write through to the ONE
owned index (never a separate silo). Nothing here runs against a live braid.

## Encoding / time / write contract (the house rules, enforced by tests)
- UTF-8, NFC, no BOM, LF everywhere in this tree (B8 canonical zone).
- All stored timestamps UTC ISO-8601 Z; render-local only at display (DoD 2).
- All writes atomic via writes.atomic_* (DoD 8); one writer, append-only.
- Tool crib check precedes any new write (DoD 4); temp-script reflex impossible.

*Written by Dora Brandon 2026-09-07 22:3x local. The room is built; the
occupant arrives through the braid.*
