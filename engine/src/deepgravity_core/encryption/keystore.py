"""
deepgravity_core.encryption.keystore — passphrase-gated keystore for DeepGravity, Step 0.

Dev-environment package version. Same logic as the prototype tooling, packaged
for the new DeepGravity engine core in /srv/dg-dev.

License: Apache-2.0. Dependency: Python `cryptography` 46.0.5,
License-Expression Apache-2.0 OR BSD-3-Clause, verified on dora-box.
Wordlist: BIP39 English, 2048 words, from bitcoin/bips bip-0039/english.txt,
BIPs repository MIT licensed. No other dependencies.

What this is: the passphrase gate. A keystore file holds the master key's
material encrypted at rest. Unlock with the passphrase, the master key is
in memory and can be passed to dg_file_crypt. Lock, and it is cleared.
No password reset. No recovery without the recovery phrase.

Fixes over the April keystore.py, which this replaces as prior art:
1. Recovery phrase is proper BIP39: 128-bit entropy plus 4-bit checksum
   from SHA256(entropy), split into 12 11-bit indices into the full 2048-word
   list. April used a 496-word A-D-only list and `idx % len`, biasing words
   and exhausting entropy before the 12th word.
2. Recovery file stays in sync. The recovery key is stored inside the
   encrypted payload, so a passphrase unlock can re-encrypt the recovery
   file when sessions change. April could not update the recovery file
   without the phrase, so it went stale and recovery lost new sessions.
3. No module-global key held implicitly across calls. A Keystore instance
   holds the unlocked keys, and lock() clears them explicitly.
"""

import base64
import hashlib
import json
import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

FORMAT = "dg-keystore-v2"
WORDLIST_PATH = Path(__file__).with_name("bip39-english.txt")

def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def _load_words():
    return WORDLIST_PATH.read_text(encoding="utf-8").split()

def _derive_key(secret: str, salt: bytes) -> bytes:
    kdf = Argon2id(salt=salt, length=32, memory_cost=64 * 1024, iterations=3, lanes=4)
    return kdf.derive(secret.encode("utf-8"))

def generate_recovery_phrase() -> str:
    """Proper BIP39: 128-bit entropy + 4-bit checksum, 12 words."""
    words = _load_words()
    assert len(words) == 2048
    entropy = secrets.token_bytes(16)
    checksum = hashlib.sha256(entropy).digest()[0] >> 4
    bits = int.from_bytes(entropy, "big") << 4 | checksum
    phrase = []
    for i in range(12):
        shift = (11 - i) * 11
        idx = (bits >> shift) & 0x7FF
        phrase.append(words[idx])
    return " ".join(phrase)

def _phrase_to_indices(phrase: str):
    words = _load_words()
    index = {w: i for i, w in enumerate(words)}
    parts = phrase.strip().split()
    if len(parts) != 12:
        raise ValueError("recovery phrase must be 12 words")
    return [index[w] for w in parts]

def _check_phrase(phrase: str) -> bool:
    """Validate the BIP39 checksum. Returns False instead of raising on a bad phrase."""
    try:
        indices = _phrase_to_indices(phrase)
    except Exception:
        return False
    bits = 0
    for idx in indices:
        bits = (bits << 11) | idx
    entropy = (bits >> 4).to_bytes(16, "big")
    checksum = bits & 0xF
    return (hashlib.sha256(entropy).digest()[0] >> 4) == checksum

def _encrypt_payload(payload: dict, key: bytes) -> dict:
    raw = json.dumps(payload).encode("utf-8")
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(key).encrypt(nonce, raw, None)
    return {"format": FORMAT, "nonce": _b64e(nonce), "ciphertext": _b64e(ciphertext)}

def _decrypt_payload(store: dict, key: bytes):
    if store.get("format") != FORMAT:
        raise ValueError("not a dg-keystore-v2 file")
    raw = AESGCM(key).decrypt(_b64d(store["nonce"]), _b64d(store["ciphertext"]), None)
    return json.loads(raw.decode("utf-8"))

def _atomic_write(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)

class Keystore:
    """A passphrase-gated keystore. One instance, one keystore file pair."""

    def __init__(self, keystore_path: str):
        self.path = keystore_path
        self.recovery_path = keystore_path.replace("keystore.enc", "keystore.recovery")
        if self.recovery_path == keystore_path:
            self.recovery_path = keystore_path + ".recovery"
        self.master_key = None
        self.recovery_key = None
        self.salt = None
        self.recovery_salt = None
        self.sessions = {}

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def create(self, passphrase: str) -> str:
        """Create a new keystore. Returns the recovery phrase. Save it, there is no reset."""
        self.salt = secrets.token_bytes(16)
        self.recovery_salt = secrets.token_bytes(16)
        phrase = generate_recovery_phrase()
        self.recovery_key = _derive_key(phrase, self.recovery_salt)
        self.master_key = _derive_key(passphrase, self.salt)
        self.sessions = {}
        self._save()
        return phrase

    def _payload(self) -> dict:
        return {
            "version": 2,
            "salt": _b64e(self.salt),
            "recovery_salt": _b64e(self.recovery_salt),
            "recovery_key_b64": _b64e(self.recovery_key),
            "sessions": self.sessions,
        }

    def _save(self):
        if self.master_key is None or self.recovery_key is None:
            raise RuntimeError("keystore is locked, cannot save")
        # Salt lives in the outer store in plaintext, so unlock can derive
        # the key before decrypting the payload. The payload itself stays encrypted.
        payload = self._payload()
        main_store = _encrypt_payload(payload, self.master_key)
        main_store["salt"] = _b64e(self.salt)
        _atomic_write(self.path, main_store)
        rec_store = _encrypt_payload(payload, self.recovery_key)
        rec_store["salt"] = _b64e(self.recovery_salt)
        _atomic_write(self.recovery_path, rec_store)

    def unlock(self, passphrase: str) -> bool:
        """Unlock with the passphrase. Returns True on success, False on a wrong passphrase."""
        if not self.exists():
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                store = json.load(f)
            # Salt is in the outer store in plaintext, so we can derive the key
            # before decrypting the payload.
            salt_b64 = store.get("salt")
            if salt_b64 is None:
                return False
            self.salt = _b64d(salt_b64)
            self.master_key = _derive_key(passphrase, self.salt)
            payload = _decrypt_payload(store, self.master_key)
            self.recovery_salt = _b64d(payload["recovery_salt"])
            self.recovery_key = _b64d(payload["recovery_key_b64"])
            self.sessions = payload.get("sessions", {})
            return True
        except Exception:
            self.lock()
            return False

    def unlock_recovery(self, phrase: str) -> bool:
        """Unlock with the recovery phrase. Sets recovery_key and sessions, master_key stays None until a new passphrase is set."""
        if not _check_phrase(phrase):
            return False
        if not os.path.exists(self.recovery_path):
            return False
        try:
            with open(self.recovery_path, "r", encoding="utf-8") as f:
                store = json.load(f)
            salt_b64 = store.get("salt")
            if salt_b64 is None:
                return False
            self.recovery_salt = _b64d(salt_b64)
            self.recovery_key = _derive_key(phrase, self.recovery_salt)
            payload = _decrypt_payload(store, self.recovery_key)
            self.salt = _b64d(payload["salt"])
            self.sessions = payload.get("sessions", {})
            self.master_key = None
            return True
        except Exception:
            self.lock()
            return False

    def set_passphrase(self, new_passphrase: str):
        """Set a new passphrase, re-encrypting both files. Requires an unlock first, by passphrase or by recovery."""
        if self.recovery_key is None:
            raise RuntimeError("keystore must be unlocked first")
        self.salt = secrets.token_bytes(16)
        self.master_key = _derive_key(new_passphrase, self.salt)
        self._save()

    def add_session(self, session_id: str, session_key: bytes):
        """Add a session key and re-encrypt both files, so the recovery file stays in sync."""
        if self.master_key is None:
            raise RuntimeError("keystore must be unlocked with the passphrase to add a session")
        self.sessions[session_id] = _b64e(session_key)
        self._save()

    def get_session(self, session_id: str):
        v = self.sessions.get(session_id)
        return _b64d(v) if v is not None else None

    def lock(self):
        self.master_key = None
        self.recovery_key = None
        self.salt = None
        self.recovery_salt = None
        self.sessions = {}
