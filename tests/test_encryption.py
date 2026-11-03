"""test_encryption.py — Step 0 file-by-file encryption as a discrete package.

Runnable without pytest: python tests/test_encryption.py  (plain asserts, PASS/FAIL).
"""

from __future__ import annotations

import os
import secrets
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core.encryption import Engine, Keystore
from deepgravity_core.encryption.file_crypt import decrypt_bytes, encrypt_bytes
from deepgravity_core.encryption.keystore import _check_phrase, generate_recovery_phrase
from deepgravity_core.encryption.sourcing import prepare_for_engine
from deepgravity_core.encryption.turns import encrypt_turn, render_log


def _make_key():
    from deepgravity_core.encryption.file_crypt import derive_master_key
    salt = secrets.token_bytes(16)
    return derive_master_key("test-passphrase-not-real", salt)


def test_file_crypt_round_trip():
    key = _make_key()
    env = encrypt_bytes(b"hello deepgravity", key, "CO")
    assert decrypt_bytes(env, key, reader="local") == b"hello deepgravity"


def test_file_crypt_nt_cloud_placeholder():
    key = _make_key()
    env = encrypt_bytes(b"secret turn content", key, "NT")
    assert decrypt_bytes(env, key, reader="local") == b"secret turn content"
    out = decrypt_bytes(env, key, reader="cloud")
    assert isinstance(out, dict)
    assert out["placeholder"] == "[NT — no transmission, local only]"
    assert "secret" not in str(out)


def test_file_crypt_wrong_key_fails():
    from deepgravity_core.encryption.file_crypt import derive_master_key
    salt = secrets.token_bytes(16)
    key = derive_master_key("test-passphrase-not-real", salt)
    wrong = derive_master_key("wrong-passphrase", salt)
    env = encrypt_bytes(b"hello", key, "CO")
    assert decrypt_bytes(env, wrong, reader="local") is None


def test_keystore_create_unlock_recovery():
    phrase = generate_recovery_phrase()
    assert len(phrase.split()) == 12
    assert _check_phrase(phrase)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "keystore.enc")
        ks = Keystore(path)
        rec = ks.create("test-passphrase-not-real")
        assert _check_phrase(rec)
        assert os.path.exists(path)
        assert os.path.exists(path.replace("keystore.enc", "keystore.recovery"))
        ks.add_session("sess-1", secrets.token_bytes(32))
        ks.lock()
        assert ks.master_key is None
        ks2 = Keystore(path)
        assert ks2.unlock("wrong-passphrase") is False
        assert ks2.unlock("test-passphrase-not-real") is True
        assert ks2.get_session("sess-1") is not None
        ks2.add_session("sess-2", secrets.token_bytes(32))
        ks3 = Keystore(path)
        assert ks3.unlock_recovery(rec) is True
        assert "sess-1" in ks3.sessions and "sess-2" in ks3.sessions
        ks3.set_passphrase("new-passphrase-not-real")
        ks4 = Keystore(path)
        assert ks4.unlock("new-passphrase-not-real") is True
        assert "sess-1" in ks4.sessions and "sess-2" in ks4.sessions
        assert Keystore(path).unlock("test-passphrase-not-real") is False


def test_turns_mixed_log():
    key = _make_key()
    log = [
        encrypt_turn("user", "hello, public question", key, "UF"),
        encrypt_turn("assistant", "service improvement note", key, "SO"),
        encrypt_turn("user", "help me right now with this", key, "CO"),
        encrypt_turn("user", "routing hint, do not keep", key, "NK"),
        encrypt_turn("user", "the most private thing", key, "NT"),
    ]
    local = render_log(log, key, reader="local", in_conversation=True)
    assert all("content" in t for t in local)
    cloud_in = render_log(log, key, reader="cloud", in_conversation=True)
    assert cloud_in[0].get("content") and cloud_in[1].get("content") and cloud_in[2].get("content")
    assert "placeholder" in cloud_in[3] and "placeholder" in cloud_in[4]
    cloud_out = render_log(log, key, reader="cloud", in_conversation=False)
    assert cloud_out[0].get("content") and cloud_out[1].get("content")
    assert "placeholder" in cloud_out[2] and "placeholder" in cloud_out[3] and "placeholder" in cloud_out[4]
    assert "private" not in str(cloud_in) and "private" not in str(cloud_out)


def test_sourcing_pull_shades():
    key = _make_key()
    log = [
        encrypt_turn("user", "hello, public question", key, "UF"),
        encrypt_turn("assistant", "service improvement note", key, "SO"),
        encrypt_turn("user", "help me right now with this", key, "CO"),
        encrypt_turn("user", "routing hint, do not keep", key, "NK"),
        encrypt_turn("user", "the most private thing", key, "NT"),
    ]
    local = Engine(kind="local", provider="dora-box", model="muse-glimmer-30b-ablit:q5")
    cloud = Engine(kind="cloud", provider="meta-api", model="muse-spark")
    unknown = Engine(kind="mystery", provider="unknown", model="unknown")
    _, rep_local = prepare_for_engine(log, key, local, True)
    assert rep_local["turns_withheld"] == 0
    _, rep_cloud = prepare_for_engine(log, key, cloud, True)
    assert rep_cloud["turns_withheld"] == 2
    assert rep_cloud["withheld_by_tag"] == {"NK": 1, "NT": 1}
    _, rep_unknown = prepare_for_engine(log, key, unknown, True)
    assert rep_unknown["engine_reader"] == "cloud"
    assert rep_unknown["turns_withheld"] == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
