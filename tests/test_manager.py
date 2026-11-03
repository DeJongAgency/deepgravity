"""Tests for deepgravity_core.dg_manager (M1: session and routing logic).

The manager is tested with a fake engine (an injected callable) and a
fake default model id — never the shipped 5q30b, so a test that passes
proves the default comes from configuration, not from a hardcoded id.
The master key here is a fixed test key; it never touches the real
keystore or the real braid.
"""
import pytest

from deepgravity_core.dg_manager import (
    EngineConfig,
    Manager,
    Overlay,
    load_engine_config,
)
from deepgravity_core.encryption.sourcing import Engine
from deepgravity_core.encryption.turns import render_log

TEST_KEY = bytes(range(32))

CODE = Overlay(id="dg-code", system_prompt="You are a coding helper.",
               params={"temperature": 0.7, "top_p": 0.9}, default_ccf="CO")
SOMA = Overlay(id="dg-soma", system_prompt="Loose, liminal, somatic.",
               params={"temperature": 1.0, "top_p": 0.95}, default_ccf="NT")


def make_config():
    return EngineConfig(
        default_model_id="local-test",
        engines={
            "local-test": Engine(kind="local", provider="dora-box",
                                 model="local-test", endpoint="http://127.0.0.1:8209/v1"),
            "cloud-test": Engine(kind="cloud", provider="meta-api",
                                 model="cloud-test", endpoint="https://example.invalid/v1"),
        },
    )


def make_manager(engine_fn=None, classifier_fn=None, overlays=None):
    calls = []

    def fake_engine(rendered, engine, settings):
        calls.append({"rendered": rendered, "engine": engine, "settings": settings})
        return "fake reply"

    return Manager(TEST_KEY, make_config(),
                   overlays=overlays if overlays is not None else {"dg-code": CODE, "dg-soma": SOMA},
                   default_overlay=CODE,
                   engine_fn=engine_fn if engine_fn is not None else fake_engine,
                   classifier_fn=classifier_fn), calls


def test_default_comes_from_config_not_hardcoded():
    mgr, _ = make_manager()
    assert mgr.default_engine().model == "local-test"
    assert mgr.default_engine().kind == "local"
    session = mgr.create_session("s1")
    assert session.engine.model == "local-test"
    # The shipped model id never appears in the manager's behavior.
    assert "5q30b" not in mgr.default_engine().model


def test_unknown_default_model_raises():
    config = EngineConfig(default_model_id="no-such-model", engines={})
    with pytest.raises(ValueError, match="not in the engine configuration"):
        Manager(TEST_KEY, config)


def test_set_default_model_and_rejects_unconfigured():
    mgr, _ = make_manager()
    mgr.set_default_model("cloud-test")
    assert mgr.default_engine().model == "cloud-test"
    with pytest.raises(ValueError, match="not in the engine configuration"):
        mgr.set_default_model("no-such-model")


def test_settings_cascade_and_prospective_only():
    mgr, _ = make_manager()
    mgr.create_session("s1")
    t1 = mgr.add_turn("s1", "user", "first")
    assert t1["settings"] == {"temperature": 0.7, "top_p": 0.9}
    assert t1["ccf"] == "CO"
    mgr.set_conversation_settings("s1", {"temperature": 0.2})
    t2 = mgr.add_turn("s1", "user", "second")
    assert t2["settings"] == {"temperature": 0.2, "top_p": 0.9}
    # The first turn keeps the settings it was made with.
    assert mgr.get_session("s1").turns[0]["settings"] == {"temperature": 0.7, "top_p": 0.9}
    t3 = mgr.add_turn("s1", "user", "third", settings_override={"top_p": 0.5})
    assert t3["settings"] == {"temperature": 0.2, "top_p": 0.5}
    assert mgr.get_session("s1").turns[1]["settings"] == {"temperature": 0.2, "top_p": 0.9}


def test_overlay_change_is_prospective_and_does_not_retag():
    mgr, _ = make_manager()
    mgr.create_session("s1")
    mgr.add_turn("s1", "user", "plain code question")
    mgr.set_overlay("s1", "dg-soma")
    t2 = mgr.add_turn("s1", "user", "somatic turn")
    assert t2["ccf"] == "NT"
    assert t2["settings"]["temperature"] == 1.0
    # The earlier turn is untouched: still CO, still the code settings.
    earlier = mgr.get_session("s1").turns[0]
    assert earlier["ccf"] == "CO"
    assert earlier["settings"]["temperature"] == 0.7


def test_unknown_overlay_raises():
    mgr, _ = make_manager()
    mgr.create_session("s1")
    with pytest.raises(ValueError, match="unknown overlay"):
        mgr.set_overlay("s1", "no-such-overlay")


def test_engine_change_rerenders_immediately_and_stored_turns_untouched():
    mgr, _ = make_manager()
    mgr.create_session("s1")
    mgr.add_turn("s1", "user", "somatic content", ccf="NT")
    mgr.add_turn("s1", "user", "plain content", ccf="CO")
    report = mgr.set_engine("s1", "cloud-test")
    assert report["turns_withheld"] == 1
    assert report["withheld_by_tag"] == {"NT": 1}
    rendered, _ = mgr.render_for_engine("s1", mgr.get_session("s1").engine)
    # The rendered log the cloud engine sees: system message (first turn
    # is NT, so the system prompt is withheld too) + NT placeholder + CO.
    assert rendered[0].get("placeholder", "").startswith("[NT")
    assert rendered[1].get("placeholder", "").startswith("[NT")
    assert rendered[2]["content"] == "plain content"
    # The stored turns are untouched: the NT turn still decrypts.
    stored = render_log(mgr.get_session("s1").turns, TEST_KEY, reader="local")
    assert stored[0]["content"] == "somatic content"
    assert stored[0]["ccf"] == "NT"


def test_retro_load_withholds_co_from_cloud_after_conversation():
    mgr, _ = make_manager()
    mgr.create_session("s1")
    mgr.add_turn("s1", "user", "plain content", ccf="CO")
    cloud = mgr.resolve_engine("cloud-test")
    rendered, report = mgr.render_for_engine("s1", cloud, in_conversation=False)
    assert report["withheld_by_tag"] == {"CO": 1}
    assert rendered[1].get("placeholder", "").startswith("[CO")


def test_complete_turn_appends_reply_with_most_private_wins_tag():
    mgr, calls = make_manager()
    mgr.create_session("s1")
    reply, report = mgr.complete_turn("s1", "somatic content", ccf="NT")
    assert reply == "fake reply"
    assert report["turns_withheld"] == 0  # local engine sees everything
    turns = mgr.get_session("s1").turns
    assert len(turns) == 2
    assert turns[0]["ccf"] == "NT"
    # Reply to an NT user turn is NT even under the CO code overlay.
    assert turns[1]["role"] == "assistant"
    assert turns[1]["ccf"] == "NT"
    # The engine was handed the rendered log, the local engine, and the
    # resolved settings — the pluggable boundary.
    assert calls[0]["engine"].model == "local-test"
    assert calls[0]["settings"] == {"temperature": 0.7, "top_p": 0.9}
    assert calls[0]["rendered"][1]["content"] == "somatic content"


def test_complete_turn_reply_under_soma_is_nt():
    mgr, _ = make_manager()
    mgr.create_session("s1")
    mgr.set_overlay("s1", "dg-soma")
    mgr.complete_turn("s1", "plain words, soma overlay")
    turns = mgr.get_session("s1").turns
    assert turns[0]["ccf"] == "NT"
    assert turns[1]["ccf"] == "NT"


def test_complete_turn_without_engine_fn_raises():
    # Build a manager with no engine function at all: it must raise
    # rather than call a network itself.
    mgr2 = Manager(TEST_KEY, make_config(), overlays={"dg-code": CODE},
                   default_overlay=CODE, engine_fn=None)
    mgr2.create_session("s1")
    with pytest.raises(ValueError, match="no engine function"):
        mgr2.complete_turn("s1", "hello")


def test_classifier_suggestion_only_escalates():
    def classifier(text):
        return "NT" if "sensitive" in text else "CO"

    mgr, _ = make_manager(classifier_fn=classifier)
    mgr.create_session("s1")
    # Suggestion NT escalates a CO-default turn to NT.
    t1 = mgr.add_turn("s1", "user", "something sensitive here")
    assert t1["ccf"] == "NT"
    # Suggestion CO does not downgrade an NT turn (soma overlay).
    mgr.set_overlay("s1", "dg-soma")
    t2 = mgr.add_turn("s1", "user", "ordinary words")
    assert t2["ccf"] == "NT"
    # A user tag overrides the suggestion entirely.
    t3 = mgr.add_turn("s1", "user", "something sensitive here", ccf="CO")
    assert t3["ccf"] == "CO"


def test_classifier_failure_never_blocks_the_turn():
    def broken_classifier(text):
        raise RuntimeError("classifier exploded")

    mgr, _ = make_manager(classifier_fn=broken_classifier)
    mgr.create_session("s1")
    turn = mgr.add_turn("s1", "user", "hello")
    assert turn["ccf"] == "CO"


def test_system_message_takes_first_turn_tag_and_is_withheld_from_cloud():
    mgr, _ = make_manager()
    mgr.create_session("s1")
    mgr.add_turn("s1", "user", "somatic content", ccf="NT")
    cloud = mgr.resolve_engine("cloud-test")
    rendered, _ = mgr.render_for_engine("s1", cloud)
    assert rendered[0]["role"] == "system"
    assert rendered[0]["ccf"] == "NT"
    assert "placeholder" in rendered[0]
    assert "coding helper" not in str(rendered[0])
    # The local engine sees the system prompt in full.
    local = mgr.resolve_engine("local-test")
    rendered_local, _ = mgr.render_for_engine("s1", local)
    assert rendered_local[0]["content"] == "You are a coding helper."


def test_load_engine_config_reads_default_and_engines(tmp_path):
    path = tmp_path / "engine_config.json"
    path.write_text(
        '{"default_model": "my-hf-model", "engines": {'
        '"my-hf-model": {"kind": "local", "provider": "dora-box", '
        '"model": "my-hf-model", "endpoint": "http://127.0.0.1:8209/v1"}}}'
    )
    config = load_engine_config(str(path))
    assert config.default_model_id == "my-hf-model"
    assert config.default_engine().kind == "local"
    # A user-downloaded HF model works as the default without
    # special-casing: it is just another entry in the config.
    mgr = Manager(TEST_KEY, config, overlays={"dg-code": CODE},
                  default_overlay=CODE)
    assert mgr.create_session("s1").engine.model == "my-hf-model"
