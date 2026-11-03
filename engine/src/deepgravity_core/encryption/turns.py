"""
deepgravity_core.encryption.turns — per-turn CCF rendering for DeepGravity, Step 0.

Dev-environment package version.

License: Apache-2.0. Depends only on .file_crypt, which depends on
Python `cryptography` Apache-2.0 OR BSD-3-Clause.

What this is: a discussion history where each individual turn has its own
CCF tag, and rendering it depends on who is reading. A cloud API reader
is not allowed to see NT, that is what NT means, so NT turns render as
a placeholder. NK turns were never to be retained, so in a retained log
they render as a placeholder too. CO turns are conversation-scoped, so
they are visible inside the conversation and placeholder outside it.

CCF tags, from CCRFC-001:
- UF: Use Freely
- SO: Service Only, improve the service, keep it out of advertising
- CO: Conversation-Scoped, help right now, then forget, grant expires at the conversation edge
- NK: No Keep, transient processing only, seen to route, not retained
- NT: No Transmission, never leaves the device, local inference only

Reader kinds: "local" is the device owner with the keystore unlocked.
"cloud" is any cloud API or remote engine. Default-deny, most-private wins.
"""

from .file_crypt import decrypt_bytes, encrypt_bytes

CCF_TAGS = ["UF", "SO", "CO", "NK", "NT"]

PLACEHOLDERS = {
    "NT": "[NT — no transmission, local only]",
    "NK": "[NK — no keep, transient only, not retained]",
    "CO": "[CO — conversation-scoped, grant expired]",
}

def _placeholder(ccf: str) -> dict:
    return {"_encrypted": True, "ccf": ccf, "placeholder": PLACEHOLDERS[ccf]}

def encrypt_turn(role: str, plaintext: str, master_key: bytes, ccf: str) -> dict:
    """Encrypt one turn. The CCF tag travels with the payload."""
    if ccf not in CCF_TAGS:
        raise ValueError(f"unknown CCF tag: {ccf}")
    return {
        "role": role,
        "ccf": ccf,
        "envelope": encrypt_bytes(plaintext.encode("utf-8"), master_key, ccf),
    }

def render_turn(turn: dict, master_key: bytes, reader: str = "local", in_conversation: bool = True):
    """Render one turn for a reader. Returns a dict with role, ccf, and content or placeholder."""
    ccf = turn.get("ccf", "")
    role = turn.get("role", "")
    if reader == "cloud":
        if ccf == "NT":
            return {"role": role, "ccf": ccf, **_placeholder("NT")}
        if ccf == "NK":
            return {"role": role, "ccf": ccf, **_placeholder("NK")}
        if ccf == "CO" and not in_conversation:
            return {"role": role, "ccf": ccf, **_placeholder("CO")}
    result = decrypt_bytes(turn["envelope"], master_key, reader=reader)
    if isinstance(result, dict):
        return {"role": role, "ccf": ccf, **result}
    if result is None:
        return {"role": role, "ccf": ccf, "_encrypted": True, "placeholder": "[decrypt failed]"}
    return {"role": role, "ccf": ccf, "content": result.decode("utf-8")}

def render_log(log: list, master_key: bytes, reader: str = "local", in_conversation: bool = True) -> list:
    """Render a whole log, turn by turn, each with its own tag."""
    return [render_turn(t, master_key, reader=reader, in_conversation=in_conversation) for t in log]
