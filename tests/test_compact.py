"""test_compact.py — M3 compaction acceptance tests.

Runnable without pytest: python tests/test_compact.py (plain asserts,
PASS/FAIL). No live engine, no network, no real sessions: every test uses
fabricated session records and an injected fake engine function.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core import dg_compact as dc

CANNED_REPLY = """Summary:
A planning session about the workshop move.

Decisions:
- The workshop moves to the new room on Friday.
- Dora owns the checklist.

Facts established:
- The new room has two network drops.
- The move needs four hours.

Open items:
- Who brings the spare monitor?

Safe to forget:
- The joke about the broken kettle.
- Repeated wording debate about the checklist title.
"""


def _fake_engine_factory(reply=CANNED_REPLY):
    """A recording fake engine: prompt -> canned reply; keeps every prompt."""
    calls: list[str] = []

    def fake(prompt: str) -> str:
        calls.append(prompt)
        return reply

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


def _session(session_id="chat_20261102_120000_ab12", source="live-seed",
             status="closed", turns=None):
    if turns is None:
        turns = [
            {"role": "user", "content": "We need to move the workshop.",
             "settings": {"preset": "default", "temperature": 0.8}, "ccf": "CO"},
            {"role": "assistant", "content": "Friday works. I will own the checklist.",
             "settings": {"preset": "default", "temperature": 0.8}, "ccf": "CO"},
            {"role": "user", "content": "Agreed, Friday it is.",
             "settings": {"preset": "default", "temperature": 0.8}, "ccf": "CO"},
            {"role": "assistant", "content": "Done. Two network drops confirmed in the new room.",
             "settings": {"preset": "default", "temperature": 0.8}, "ccf": "CO"},
        ]
    return {"session_id": session_id, "source": source, "status": status,
            "turns": turns}


def _nt_session():
    turns = [
        {"role": "user", "content": "SECRET-NT-PLAN: the private thing we discussed.",
         "settings": {"preset": "soma", "temperature": 0.8}, "ccf": "NT"},
        {"role": "assistant", "content": "Understood, keeping it local.",
         "settings": {"preset": "soma", "temperature": 0.8}, "ccf": "NT"},
        {"role": "user", "content": "Also the ordinary workshop move stands.",
         "settings": {"preset": "default", "temperature": 0.8}, "ccf": "CO"},
        {"role": "assistant", "content": "Noted.",
         "settings": {"preset": "default", "temperature": 0.8}, "ccf": "CO"},
    ]
    return _session(session_id="chat_20261102_130000_cd34", turns=turns)


def test_most_private_wins_nt():
    entry = dc.compact_session(_nt_session(), _fake_engine_factory(),
                               engine_local=True)
    assert entry["tags"] == ["NT"], f"NT session must produce NT entry, got {entry['tags']}"
    assert entry["ccf"] == "NT"


def test_most_private_wins_co():
    entry = dc.compact_session(_session(), _fake_engine_factory(),
                               engine_local=True)
    assert entry["tags"] == ["CO"], f"CO-only session must produce CO entry, got {entry['tags']}"
    assert entry["ccf"] == "CO"


def test_unknown_tag_fails_closed_to_nt():
    turns = _session()["turns"][:]
    turns[0] = dict(turns[0], ccf="BOGUS-TAG")
    entry = dc.compact_session(_session(turns=turns), _fake_engine_factory(),
                               engine_local=True)
    assert entry["tags"] == ["NT"], "unknown CCF tag must rank as NT"


def test_nt_never_sent_to_nonlocal_engine():
    fake = _fake_engine_factory()
    try:
        dc.compact_session(_nt_session(), fake, engine_local=False)
    except ValueError:
        pass
    else:
        raise AssertionError("NT session to a non-local engine must raise")
    assert fake.calls == [], "non-local engine must receive zero calls"
    assert not any("SECRET-NT-PLAN" in prompt for prompt in fake.calls), \
        "NT content must never reach a non-local engine"


def test_co_session_allowed_to_nonlocal_engine():
    fake = _fake_engine_factory()
    entry = dc.compact_session(_session(), fake, engine_local=False)
    assert len(fake.calls) == 1, "CO-only session may go to a non-local engine"
    assert entry["tags"] == ["CO"]


def test_local_engine_receives_nt_turns():
    fake = _fake_engine_factory()
    dc.compact_session(_nt_session(), fake, engine_local=True)
    assert len(fake.calls) == 1
    assert "SECRET-NT-PLAN" in fake.calls[0], \
        "local engine must receive the full transcript including NT turns"


def test_should_compact_too_few_turns():
    short = _session(turns=_session()["turns"][:2])
    assert dc.should_compact(short) is False, "2 turns < default min 4"
    assert dc.should_compact(short, min_turns=2) is True, "min_turns is a parameter"


def test_should_compact_already_compacted():
    session = _session()
    assert dc.should_compact(session, compacted_ids={session["session_id"]}) is False


def test_should_compact_unfinished():
    assert dc.should_compact(_session(status="open")) is False
    assert dc.should_compact(_session(status="")) is False
    record = _session()
    del record["status"]
    assert dc.should_compact(record) is False, "missing status fails closed"
    assert dc.should_compact(_session(status="finished")) is True
    assert dc.should_compact(_session(status="closed")) is True


def test_source_marker_carried_through():
    live = dc.compact_session(_session(source="live-seed"), _fake_engine_factory())
    assert live["source"] == "live-seed"
    back = dc.compact_session(_session(source="backfill"), _fake_engine_factory())
    assert back["source"] == "backfill"
    record = _session()
    del record["source"]
    unknown = dc.compact_session(record, _fake_engine_factory())
    assert unknown["source"] == "unknown", \
        "missing source must stay 'unknown', never upgraded to live-seed"


def test_distilled_entry_shape():
    entry = dc.compact_session(_session(), _fake_engine_factory(), model="5q30b")
    assert entry["session_id"] == "chat_20261102_120000_ab12"
    assert entry["model"] == "5q30b"
    assert entry["summary"].startswith("A planning session")
    assert entry["decisions"] == [
        "The workshop moves to the new room on Friday.",
        "Dora owns the checklist."]
    assert entry["open_items"] == ["Who brings the spare monitor?"]
    assert len(entry["facts"]) == 2
    assert len(entry["safe_to_forget"]) == 2
    assert entry["turn_count"] == 4
    assert "compacted_at" in entry


def test_system_turns_excluded_nk_never_distilled():
    turns = [
        {"role": "system", "content": "SYSTEM-SECRET-DIRECTIVE: you are Dora.",
         "settings": {}, "ccf": "CO"},
        {"role": "user", "content": "NK-TRANSIENT: route this and drop it.",
         "settings": {}, "ccf": "NK"},
        {"role": "user", "content": "Keep this one.", "settings": {}, "ccf": "CO"},
        {"role": "assistant", "content": "Kept.", "settings": {}, "ccf": "CO"},
        {"role": "user", "content": "And this.", "settings": {}, "ccf": "CO"},
        {"role": "assistant", "content": "Noted.", "settings": {}, "ccf": "CO"},
    ]
    fake = _fake_engine_factory()
    entry = dc.compact_session(_session(turns=turns), fake, engine_local=True)
    prompt = fake.calls[0]
    assert "SYSTEM-SECRET-DIRECTIVE" not in prompt, "system turns are not distilled"
    assert "NK-TRANSIENT" not in prompt, "NK turns are never retained, not even in the prompt"
    assert entry["excluded_nk_turns"] == 1
    assert entry["system_turns_skipped"] == 1
    assert entry["turn_count"] == 4
    assert entry["tags"] == ["CO"], "excluded NK turn must not raise the entry tag"


def test_nothing_distillable_raises():
    turns = [{"role": "system", "content": "x", "settings": {}, "ccf": "CO"},
             {"role": "user", "content": "y", "settings": {}, "ccf": "NK"}]
    fake = _fake_engine_factory()
    try:
        dc.compact_session(_session(turns=turns), fake, engine_local=True)
    except ValueError:
        pass
    else:
        raise AssertionError("all-NK/system session must raise, not call the engine")
    assert fake.calls == []


def test_prompt_preserves_decided_vs_discussed():
    prompt = dc.load_prompt()
    assert "DECIDED" in prompt and "DISCUSSED" in prompt, \
        "prompt must instruct the decided-vs-discussed distinction"
    assert "Decisions:" in prompt and "Open items:" in prompt
    assert "Safe to forget:" in prompt and "Facts established:" in prompt


def test_prompt_file_matches_constant():
    prompt_path = Path(dc.__file__).with_name("dg_compact_prompt.txt")
    assert prompt_path.exists(), "standalone prompt file must exist next to the module"
    on_disk = prompt_path.read_text(encoding="utf-8")
    assert on_disk == dc.COMPACTION_PROMPT, \
        "dg_compact_prompt.txt and COMPACTION_PROMPT must match exactly"
    assert dc.load_prompt() == dc.COMPACTION_PROMPT


def test_state_file_round_trip():
    with tempfile.TemporaryDirectory() as d:
        state = Path(d) / "compacted.json"
        assert dc.load_compacted_ids(state) == set()
        dc.mark_compacted(state, "chat_aaa")
        dc.mark_compacted(state, "chat_bbb")
        assert dc.load_compacted_ids(state) == {"chat_aaa", "chat_bbb"}
        session = _session(session_id="chat_aaa")
        assert dc.should_compact(session, compacted_ids=dc.load_compacted_ids(state)) is False
        assert dc.should_compact(_session(session_id="chat_ccc"),
                                 compacted_ids=dc.load_compacted_ids(state)) is True


def test_parse_fallback_keeps_malformed_reply():
    parsed = dc.parse_compaction_reply("just some unstructured prose, no headings")
    assert parsed["summary"] == "just some unstructured prose, no headings"
    assert parsed["decisions"] == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
