"""
deepgravity_core.encryption.sourcing — model and platform sourcing awareness, Step 0.

Dev-environment package version.

License: Apache-2.0. Depends only on .turns, which depends on .file_crypt
and Python `cryptography` Apache-2.0 OR BSD-3-Clause.

What this is: on-the-fly awareness of where the engine is and what platform
it is on, and automatic pull the shades when using a cloud API. Before any
log is sent to an engine, the system asks: is this engine local or cloud?
If cloud, the log is rendered for a cloud reader first, so NT turns are
placeholders, NK turns are placeholders, and CO turns are placeholders
outside the conversation. The engine never sees the plaintext of a turn
it is not allowed to see, because it never receives it.

Fail closed: an engine whose kind is unknown, missing, or anything other
than the explicit string "local" is treated as cloud. Default-deny, most
private wins. No engine defaults to seeing everything.
"""

from dataclasses import dataclass

from .turns import render_log

@dataclass
class Engine:
    kind: str  # "local" or "cloud", anything else is treated as cloud
    provider: str  # e.g. "dora-box", "meta-api", free text, recorded for audit
    model: str  # e.g. "muse-glimmer-30b-ablit:q5", free text
    endpoint: str = ""  # e.g. "http://192.168.0.24:11434/v1", free text

    def reader(self) -> str:
        return "local" if self.kind == "local" else "cloud"

    def is_local(self) -> bool:
        return self.kind == "local"

def prepare_for_engine(log: list, master_key: bytes, engine: Engine, in_conversation: bool = True):
    """
    Pull the shades automatically. Returns (rendered_log, report).
    rendered_log is what the engine is allowed to receive.
    report counts what was withheld, by CCF tag, with no content.
    """
    rendered = render_log(log, master_key, reader=engine.reader(), in_conversation=in_conversation)
    withheld = {}
    for original, out in zip(log, rendered):
        if "placeholder" in out:
            tag = out.get("ccf", "?")
            withheld[tag] = withheld.get(tag, 0) + 1
    report = {
        "engine_kind": engine.kind,
        "engine_reader": engine.reader(),
        "provider": engine.provider,
        "model": engine.model,
        "in_conversation": in_conversation,
        "turns_total": len(log),
        "turns_withheld": sum(withheld.values()),
        "withheld_by_tag": withheld,
    }
    return rendered, report
