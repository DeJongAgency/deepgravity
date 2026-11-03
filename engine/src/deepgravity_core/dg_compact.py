"""dg_compact.py — M3 compaction: compact-to-braid, the first task service.

A finished conversation session is distilled into one compact memory entry
(a summary, decisions, facts established, open items, what is safe to
forget), so the braid accumulates distilled knowledge instead of raw turns.
The entry is shaped to land in the braid later (braid.db edges +
vectors.db embeddings); this module only *produces* the entry — it never
writes to the real braid (/home/john/.local/share/deepgrav/...), never
touches the keystore, and never runs against a live session.

License: Apache-2.0. No new dependencies (stdlib only).

Privacy rules — load-bearing, fail closed:
- Encryption is ALWAYS on, no opt-out. A compacted entry is derived from
  turns that carry CCF tags, and the entry inherits the MOST-PRIVATE tag of
  the turns it distills (most-private-wins, same rule as the session
  envelope CCF). An entry distilled from a session containing any NT turn
  is NT. Never distill NT content into a less-private entry.
- NT turns are NEVER sent to a non-local engine. The engine function is
  injected by the caller and flagged local or not; if the session contains
  any NT turn and the engine is not local, compact_session() raises
  ValueError WITHOUT calling the engine. The production engine is the
  local llama-server on 127.0.0.1:8209 — compaction is a background job on
  a small slot, never on the critical path, never contending with a live
  conversation.
- NK turns ("No Keep": transient processing only, seen to route, not
  retained) are NEVER distilled at all — they are excluded from the
  transcript and counted in the entry as excluded_nk_turns. Retaining them
  in any form would violate the tag itself.
- Unknown CCF tags fail closed: they are treated as NT (most private).
  A turn with no ccf at all defaults to CO, matching the v2 session DB
  column default (ccf TEXT DEFAULT 'CO').
- Privacy order, least to most private (CCF_TAGS in encryption/turns.py):
  UF < SO < CO < NK < NT.

Session record shape (the archiver's shape):
    {"session_id": str, "source": "live-seed" | "backfill",
     "status": "closed" | "finished" | ..., "turns": [
        {"role": "user" | "assistant" | "system" | "tool",
         "content": str, "settings": {...}, "ccf": "CO" | "NT" | ...}]}
"messages" is accepted as an alias for "turns". The "source" marker is
carried through to the entry untouched; a missing source means
"historical, provenance unknown" and is recorded as "unknown" — never
upgraded to "live-seed". Settings ride along per turn in the record;
compaction reads the stored record, it does not re-decide settings.
System turns are not distilled (they carry the identity/personality
directive, not conversation content); their count is recorded in the
entry as system_turns_skipped.

The engine function is INJECTED (callable: prompt str -> reply str), never
imported — tests inject a fake; the runner injects the local llama-server
call. The compaction prompt lives in dg_compact_prompt.txt next to this
module (the readable/editable copy) and is mirrored as COMPACTION_PROMPT
below; the test suite asserts the two match.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

# Privacy rank, least -> most private. Mirrors CCF_TAGS in encryption/turns.py
# (UF, SO, CO, NK, NT). Unknown tags rank as NT: fail closed.
_PRIVACY_RANK = {"UF": 0, "SO": 1, "CO": 2, "NK": 3, "NT": 4}
_MOST_PRIVATE = "NT"
_DEFAULT_CCF = "CO"

FINISHED_STATUSES = frozenset({"closed", "finished"})
DEFAULT_MIN_TURNS = 4
DEFAULT_MODEL = "5q30b"
DEFAULT_STATE_PATH = "/srv/dg-dev/data/compacted.json"

EngineFn = Callable[[str], str]

_PROMPT_FILE = Path(__file__).with_name("dg_compact_prompt.txt")

COMPACTION_PROMPT = """You are distilling a finished conversation into a compact memory entry for the DeepGravity braid — a small, durable record of what that conversation produced, so the braid accumulates distilled knowledge instead of raw turns.

The conversation transcript follows, after the line that says TRANSCRIPT. Each turn is labelled with its role (user, assistant, system, tool). Work only from what is in the transcript. Do not invent anything, do not import outside knowledge, and do not guess at what anyone meant beyond what they said.

Write the entry in exactly this format, with exactly these five sections, in this order. Use a dash ("- ") for each bullet. If a section has nothing in it, write "None." under its heading rather than leaving it blank.

Summary:
A short paragraph, no more than five sentences, saying what the conversation was about and what it produced.

Decisions:
Things that were DECIDED in this conversation — conclusions reached, choices made, rules adopted, plans agreed. A decision is something the participants settled. If they only talked about something without settling it, it is not a decision and it does not belong here.

Facts established:
Facts that were established during the conversation — things checked, confirmed, measured, or agreed to be true. Distinguish a fact that was established from a claim that was merely made: a claim someone asserted without checking is not a fact established, even if nobody argued with it.

Open items:
Questions left open, work left undone, things explicitly deferred or still undecided. Include anything the participants said they would return to.

Safe to forget:
Material from the conversation that carries no lasting value — small talk, false starts, repetition, wording debates that were resolved, details that served only the moment. Name what can be dropped, so the braid does not carry it.

Rules for the whole entry:
- The distinction between DECIDED and merely DISCUSSED is the point of this entry. Never promote a discussed idea into a decision, and never demote a decision into a discussion point.
- Keep each bullet to one or two sentences. Short and plain beats thorough and wordy; the entry must stay compact.
- If the participants disagreed and did not resolve it, say so under Open items, not under Decisions.
- Do not add advice, do not add next steps nobody agreed to, and do not flatter anyone.
- Preserve uncertainty where the conversation left uncertainty. "Unresolved" is a valid outcome; false certainty is not.

TRANSCRIPT:
"""


def load_prompt(path: str | Path | None = None) -> str:
    """Load the compaction prompt: the .txt file when present, else the constant.

    The .txt file is the editable copy John reads; the constant is the
    fallback so the module works standalone. Tests assert they match.
    """
    candidate = Path(path) if path is not None else _PROMPT_FILE
    try:
        text = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return COMPACTION_PROMPT
    return text if text.strip() else COMPACTION_PROMPT


def _rank(tag: str) -> int:
    """Privacy rank of a CCF tag; unknown tags rank as NT (fail closed)."""
    return _PRIVACY_RANK.get(str(tag or "").upper(), _PRIVACY_RANK[_MOST_PRIVATE])


def most_private_tag(tags: Iterable[str]) -> str:
    """Most-private-wins over an iterable of CCF tags. Empty input -> CO."""
    best = _DEFAULT_CCF
    best_rank = _PRIVACY_RANK[best]
    for tag in tags:
        rank = _rank(tag)
        if rank > best_rank:
            best_rank = rank
            best = str(tag or "").upper() or _MOST_PRIVATE
            if best not in _PRIVACY_RANK:
                best = _MOST_PRIVATE
    return best


def _turn_ccf(turn: dict) -> str:
    """A turn's CCF tag; missing -> CO (the v2 column default)."""
    return str(turn.get("ccf") or _DEFAULT_CCF).upper()


def _turns(session: dict) -> list:
    """The session's turns, accepting 'turns' or the 'messages' alias."""
    turns = session.get("turns", session.get("messages", []))
    return list(turns or [])


def render_transcript(turns: list) -> tuple[str, dict]:
    """Render distillable turns as labelled transcript text.

    System turns are not distilled (directive, not content) and NK turns
    are never retained; both are excluded and counted. Returns
    (transcript_text, counts) where counts has nt_turns, excluded_nk_turns,
    system_turns_skipped and distilled_turns.
    """
    lines: list[str] = []
    counts = {"nt_turns": 0, "excluded_nk_turns": 0,
              "system_turns_skipped": 0, "distilled_turns": 0}
    for turn in turns:
        role = str(turn.get("role") or "user")
        ccf = _turn_ccf(turn)
        if ccf == "NK":
            counts["excluded_nk_turns"] += 1
            continue
        if role == "system":
            counts["system_turns_skipped"] += 1
            continue
        if ccf == "NT":
            counts["nt_turns"] += 1
        content = str(turn.get("content") or "")
        lines.append(f"[{role} | {ccf}] {content}")
        counts["distilled_turns"] += 1
    return "\n\n".join(lines), counts


_SECTION_HEADINGS = {
    "summary": re.compile(r"^summary:?\s*$", re.IGNORECASE),
    "decisions": re.compile(r"^decisions:?\s*$", re.IGNORECASE),
    "facts": re.compile(r"^facts established:?\s*$", re.IGNORECASE),
    "open_items": re.compile(r"^open items:?\s*$", re.IGNORECASE),
    "safe_to_forget": re.compile(r"^safe to forget:?\s*$", re.IGNORECASE),
}

_BULLET = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.*\S)\s*$")


def parse_compaction_reply(reply: str) -> dict:
    """Parse the engine's sectioned reply into the entry's content fields.

    Lenient by design: the engine is a small local model, not a strict
    formatter. Unknown layout falls back to the whole reply as the summary
    with empty lists, so a malformed reply degrades instead of vanishing.
    """
    sections: dict[str, list[str]] = {key: [] for key in _SECTION_HEADINGS}
    current: Optional[str] = None
    summary_lines: list[str] = []
    saw_heading = False
    for raw_line in (reply or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = None
        for key, pattern in _SECTION_HEADINGS.items():
            if pattern.match(line):
                matched = key
                break
        if matched is not None:
            saw_heading = True
            current = matched
            continue
        if current is None:
            summary_lines.append(line)
            continue
        bullet = _BULLET.match(line)
        item = bullet.group(1) if bullet else line
        if item.lower() in ("none.", "none"):
            continue
        if current == "summary":
            summary_lines.append(item)
        else:
            sections[current].append(item)
    if not saw_heading:
        return {"summary": (reply or "").strip(), "decisions": [],
                "facts": [], "open_items": [], "safe_to_forget": []}
    return {"summary": " ".join(summary_lines).strip(),
            "decisions": sections["decisions"],
            "facts": sections["facts"],
            "open_items": sections["open_items"],
            "safe_to_forget": sections["safe_to_forget"]}


def compact_session(session: dict, engine_fn: EngineFn, *,
                    model: str = DEFAULT_MODEL,
                    engine_local: bool = True,
                    prompt: str | None = None) -> dict:
    """Distill one finished session into a compact braid entry.

    engine_fn is an injected callable: prompt text -> reply text. It is
    never imported — tests inject a fake; the runner injects the local
    llama-server call. engine_local must be True when the session contains
    any NT turn; otherwise ValueError is raised BEFORE the engine is
    called, so NT content never reaches a non-local engine. Sessions with
    no distillable turns (all NK / all system) also raise ValueError.
    """
    session_id = str(session.get("session_id") or "")
    if not session_id:
        raise ValueError("session record has no session_id")
    turns = _turns(session)
    transcript, counts = render_transcript(turns)
    if counts["distilled_turns"] == 0:
        raise ValueError(f"session {session_id}: nothing distillable "
                         "(all turns NK or system)")
    if counts["nt_turns"] > 0 and not engine_local:
        raise ValueError(
            f"session {session_id}: contains {counts['nt_turns']} NT turn(s); "
            "refusing to send NT content to a non-local engine")
    distilled_tags = [_turn_ccf(t) for t in turns
                      if _turn_ccf(t) != "NK" and str(t.get("role")) != "system"]
    tag = most_private_tag(distilled_tags)
    full_prompt = (prompt if prompt is not None else load_prompt()) + transcript + "\n"
    reply = engine_fn(full_prompt)
    parsed = parse_compaction_reply(reply)
    return {
        "session_id": session_id,
        # Source marker carried through untouched; missing means
        # "historical, provenance unknown" — never upgraded to live-seed.
        "source": str(session.get("source") or "unknown"),
        "summary": parsed["summary"],
        "decisions": parsed["decisions"],
        "facts": parsed["facts"],
        "open_items": parsed["open_items"],
        "safe_to_forget": parsed["safe_to_forget"],
        # Most-private-wins: a session with any NT turn produces an NT entry.
        "tags": [tag],
        "ccf": tag,
        "model": model,
        "turn_count": counts["distilled_turns"],
        "excluded_nk_turns": counts["excluded_nk_turns"],
        "system_turns_skipped": counts["system_turns_skipped"],
        "compacted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def should_compact(session: dict, *, min_turns: int = DEFAULT_MIN_TURNS,
                   compacted_ids: Optional[Iterable[str]] = None) -> bool:
    """Gate: compactable only when finished, long enough, not yet compacted.

    Finished means status is "closed" or "finished" — a missing or "open"
    status fails closed (an unfinished session must never be compacted).
    Length counts distillable turns (NK and system turns excluded).
    """
    if str(session.get("status") or "") not in FINISHED_STATUSES:
        return False
    _, counts = render_transcript(_turns(session))
    if counts["distilled_turns"] < min_turns:
        return False
    if compacted_ids is not None:
        if str(session.get("session_id") or "") in set(compacted_ids):
            return False
    return True


def load_compacted_ids(state_path: str | Path = DEFAULT_STATE_PATH) -> set:
    """Read the compacted-session-ids state file; missing/corrupt -> empty."""
    path = Path(state_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return set()
    if not isinstance(data, dict):
        return set()
    ids = data.get("compacted_ids", data.get("compacted", []))
    return {str(i) for i in ids if i}


def mark_compacted(state_path: str | Path, session_id: str) -> set:
    """Record a session id as compacted (atomic write; creates dirs)."""
    from .writes import atomic_write_text

    path = Path(state_path)
    ids = load_compacted_ids(path)
    ids.add(str(session_id))
    payload = {"compacted_ids": sorted(ids)}
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return ids
