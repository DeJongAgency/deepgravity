"""deepgravity_core.encryption — Step 0 file-by-file encryption, CCF rendering, and sourcing awareness.

Default is ciphertext. A file or turn is only readable when explicitly unlocked
with the passphrase-derived master key. NT turns render as placeholders to a
cloud reader. Unknown engine kinds fail closed to cloud.
"""

from .file_crypt import decrypt_bytes, decrypt_file, encrypt_bytes, encrypt_file, derive_master_key
from .keystore import Keystore, generate_recovery_phrase
from .turns import CCF_TAGS, encrypt_turn, render_log, render_turn
from .sourcing import Engine, prepare_for_engine

__all__ = [
    "decrypt_bytes", "decrypt_file", "encrypt_bytes", "encrypt_file", "derive_master_key",
    "Keystore", "generate_recovery_phrase",
    "CCF_TAGS", "encrypt_turn", "render_log", "render_turn",
    "Engine", "prepare_for_engine",
]
