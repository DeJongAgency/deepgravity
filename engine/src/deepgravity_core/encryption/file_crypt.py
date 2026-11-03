"""
deepgravity_core.encryption.file_crypt — file-by-file encryption for DeepGravity, Step 0.

Dev-environment package version. Same logic as the prototype tooling, packaged
for the new DeepGravity engine core in /srv/dg-dev.

License: Apache-2.0. Dependency: Python `cryptography` 46.0.5, License-Expression
Apache-2.0 OR BSD-3-Clause, verified on dora-box via pip show. No other dependencies.

What this is: the capability to encrypt a file individually, so the file carries its
own protection when it leaves the drive. Default is ciphertext. A file is only
readable when explicitly unlocked with the passphrase-derived master key.

What this is not: universal encryption on, a service, a keystore, or live braid
wiring. Master key is passed in, never stored here. No auto-unlock.

Design:
- Master key: Argon2id(passphrase, salt, 32 bytes, memory 64 MiB, 3 iterations,
  4 lanes). Same parameters as the April keystore.py, for continuity.
- Per-file key: random 32-byte data key, generated per file. The data key is
  wrapped (encrypted) with the master key using AES-256-GCM. The file payload
  is encrypted with the data key using AES-256-GCM. This is envelope encryption:
  the master key never touches the payload directly, and each file has its own key.
- Envelope: JSON, format dg-file-crypt-v1, with ccf tag, salt, wrapped key,
  nonces, and ciphertext, all base64. Transparent and auditable for a prototype.
- CCF tag travels with the payload. Reader kind matters: if reader is "cloud"
  and tag is "NT", decrypt refuses and returns a placeholder, without touching
  the key. That is what NT means. Local reader with the master key can decrypt.
"""

import base64
import json
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

FORMAT = "dg-file-crypt-v1"
NT_PLACEHOLDER = "[NT — no transmission, local only]"

def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def derive_master_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 256-bit master key from a passphrase with Argon2id."""
    kdf = Argon2id(
        salt=salt,
        length=32,
        memory_cost=64 * 1024,
        iterations=3,
        lanes=4,
    )
    return kdf.derive(passphrase.encode("utf-8"))

def encrypt_bytes(plaintext: bytes, master_key: bytes, ccf: str) -> dict:
    """Encrypt bytes with a fresh per-file data key. Returns the JSON envelope dict."""
    data_key = AESGCM.generate_key(256)
    data_nonce = secrets.token_bytes(12)
    data_ciphertext = AESGCM(data_key).encrypt(data_nonce, plaintext, None)
    wrap_nonce = secrets.token_bytes(12)
    wrapped_key = AESGCM(master_key).encrypt(wrap_nonce, data_key, None)
    return {
        "format": FORMAT,
        "ccf": ccf,
        "wrap_nonce": _b64e(wrap_nonce),
        "wrapped_key": _b64e(wrapped_key),
        "data_nonce": _b64e(data_nonce),
        "ciphertext": _b64e(data_ciphertext),
    }

def decrypt_bytes(envelope: dict, master_key: bytes, reader: str = "local"):
    """
    Decrypt an envelope. reader is "local" or "cloud".
    Returns bytes on success, or a placeholder dict if the reader is not allowed
    to see this CCF tag. Returns None on decryption failure.
    Raises ValueError on a bad envelope format.
    """
    if envelope.get("format") != FORMAT:
        raise ValueError("not a dg-file-crypt-v1 envelope")
    ccf = envelope.get("ccf", "")
    if reader == "cloud" and ccf == "NT":
        return {"_encrypted": True, "ccf": "NT", "placeholder": NT_PLACEHOLDER}
    # Unwrap the per-file data key, then decrypt the payload.
    try:
        wrap_nonce = _b64d(envelope["wrap_nonce"])
        wrapped_key = _b64d(envelope["wrapped_key"])
        data_key = AESGCM(master_key).decrypt(wrap_nonce, wrapped_key, None)
        data_nonce = _b64d(envelope["data_nonce"])
        ciphertext = _b64d(envelope["ciphertext"])
        return AESGCM(data_key).decrypt(data_nonce, ciphertext, None)
    except Exception:
        return None

def encrypt_file(plaintext_path: str, encrypted_path: str, master_key: bytes, ccf: str) -> dict:
    """Read a plaintext file, write the encrypted envelope to encrypted_path. Returns the envelope."""
    with open(plaintext_path, "rb") as f:
        plaintext = f.read()
    envelope = encrypt_bytes(plaintext, master_key, ccf)
    with open(encrypted_path, "w", encoding="utf-8") as f:
        json.dump(envelope, f)
    return envelope

def decrypt_file(encrypted_path: str, master_key: bytes, reader: str = "local"):
    """Read an encrypted envelope file. Returns bytes, or a placeholder dict for NT/cloud."""
    with open(encrypted_path, "r", encoding="utf-8") as f:
        envelope = json.load(f)
    return decrypt_bytes(envelope, master_key, reader=reader)
