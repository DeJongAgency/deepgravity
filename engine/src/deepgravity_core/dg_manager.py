"""deepgravity_core.dg_manager — the manager (M1): session and routing logic.

DeepGravity step two. License: Apache-2.0. Depends only on
.encryption (turns, sourcing), which depend on Python `cryptography`
(Apache-2.0 OR BSD-3-Clause).

What this is: the orchestrator. One model hot, one main session-template
holding the front of house and surface work and orchestration, and
mini-sessions spun up with smaller contexts and different parameter
values for focused jobs. The manager holds the sessions and routes each
turn to the engine it belongs to. It is session and routing logic ONLY.

What it is not (deliberation, not directives — do not build them here):
the gate's two checks (content gate + capability gate), the symbolic
evaluator, the embedder twin, the 2K spinoff slot. Those are separate
decisions; the manager must not block them, but it does not build them.

Rules this module enforces (John's rulings, load-bearing):
- Encryption is ALWAYS. Turns are stored encrypted (encrypt_turn) even
  in the live session, in memory. The manager holds the master key of
  the unlocked keystore; it never invents a passphrase and never stores
  one.
- The engine never decides whether it is allowed to think about what is
  put in front of it. All enforcement lives upstream, in the router and
  the renderer: what the engine is shown. prepare_for_engine renders the
  log for the engine first; a placeholder is an absence, not a refusal.
- Tag precedence: user tag > preset (overlay) default > any automatic
  suggestion, and an automatic suggestion may only escalate toward NT,
  never downgrade, never override a user tag. Uncertain means NT.
- Soma is presumed NT. That lives in the OVERLAY data (dg-soma's
  default_ccf is NT), not in this module — the module never hardcodes
  "soma". Entering or leaving an overlay does not retag earlier turns.
- When the engine changes mid-conversation, or a conversation is loaded
  under a different engine, the system immediately re-renders the entire
  history for the new engine (set_engine / render_for_engine). Settings
  changes are prospective-only, but tag enforcement is retroactive on
  engine change or load. The stored record is untouched — only what the
  new engine is shown changes.
- Settings cascade: engine (overlay) defaults -> conversation settings
  -> per-turn override. A change only affects turns AFTER the change.
  Each turn carries the settings in effect when it was made.
- The default model is a USER CHOICE, read from configuration, never
  hardcoded. The shipped model (5q30b) is the default in the deployment
  config, not in this module. A user-downloaded HF model works as the
  default without special-casing: put it in the config's engines and set
  default_model to its id. If a registry is supplied, a model id with no
  card cannot become the default — first contact must build its card
  first (the model-card rule).
- The engine boundary is clean and pluggable. The engine is an injected
  callable (engine_fn): it takes the rendered log, the Engine, and the
  resolved settings, and returns the reply text. That callable is the
  slot stream drivers land in later (a driver is an engine_fn with a
  specific stream shape). The manager never imports a real client and
  never calls a network.
- A reply's tag is the MORE PRIVATE of the generating overlay's default
  and the user turn it answers (most-private-wins): a reply to an NT
  turn is NT even under a CO overlay, and a reply under an NT overlay is
  NT regardless.
- The system prompt (the overlay's prompt) is prepended to the rendered
  log and takes the tag of the session's first turn (the filter's rule),
  or the overlay default if the session is empty.
- Unknown engine kind fails closed to cloud (sourcing.Engine.reader()).
  An unknown overlay id is a configuration error and raises. An unknown
  default model id raises. Fail closed, never guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .encryption.sourcing import Engine, prepare_for_engine
from .encryption.turns import CCF_TAGS, PLACEHOLDERS, encrypt_turn

_PRIVACY_RANK = {tag: i for i, tag in enumerate(CCF_TAGS)}  # UF < SO < CO < NK < NT
_MOST_PRIVATE = "NT"


def _more_private(a: str, b: str) -> str:
    """The more private of two tags (most-private-wins). Unknown tags rank as NT."""
    return a if _PRIVACY_RANK.get(a, 99) >= _PRIVACY_RANK.get(b, 99) else b


def _validate_tag(tag: str) -> str:
    if tag not in CCF_TAGS:
        raise ValueError(f"unknown CCF tag: {tag}")
    return tag


@dataclass
class Overlay:
    """One overlay: a system prompt plus parameter values together, laid
    over an engine. This is what OpenWebUI calls a "model"; what
    DeepGravity calls a "model" (the weights) is what OpenWebUI calls an
    engine. DeepGravity terms internally; map to OpenWebUI terms only at
    the OpenWebUI boundary."""
    id: str
    system_prompt: str = ""
    params: dict = field(default_factory=dict)
    default_ccf: str = "CO"  # the tag for turns under this overlay when the user has not set one


@dataclass
class EngineConfig:
    """The engine configuration: which engines exist and which one is the
    default. Read from configuration (a JSON file), never hardcoded.
    The shipped model is the default in the deployment config, not here."""
    default_model_id: str
    engines: dict  # model id -> Engine (kind, provider, model, endpoint)

    def default_engine(self) -> Engine:
        engine = self.engines.get(self.default_model_id)
        if engine is None:
            raise ValueError(
                f"default model {self.default_model_id!r} is not in the engine "
                "configuration — set it to a configured engine, do not assume one"
            )
        return engine


def load_engine_config(path: str) -> EngineConfig:
    """Read an engine configuration from a JSON file of the shape:
    {"default_model": "5q30b",
     "engines": {"5q30b": {"kind": "local", "provider": "dora-box",
                          "model": "muse-glimmer-30b-ablit:q5",
                          "endpoint": "http://127.0.0.1:8209/v1"}}}
    The default_model value is the user's choice; the file is the place
    that choice lives. A user-downloaded HF model is added as another
    entry in "engines" and named in "default_model" — no special-casing."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    default_model_id = raw.get("default_model")
    if not default_model_id:
        raise ValueError("engine config has no 'default_model'")
    engines = {}
    for model_id, spec in (raw.get("engines") or {}).items():
        engines[model_id] = Engine(
            kind=spec.get("kind", ""),
            provider=spec.get("provider", ""),
            model=spec.get("model", model_id),
            endpoint=spec.get("endpoint", ""),
        )
    return EngineConfig(default_model_id=default_model_id, engines=engines)


class Session:
    """One live session: an ordered list of encrypted turns, the engine
    and overlay in effect, and the conversation-level settings.
    Turns are stored encrypted even in memory; the plaintext lives only
    in the envelope, opened by rendering for a reader."""

    def __init__(self, session_id: str, engine: Engine, overlay: Overlay):
        self.id = session_id
        self.engine = engine
        self.overlay = overlay
        self.turns: list[dict] = []  # encrypted turn + "settings" + "preset"
        self.conversation_settings: dict = {}
        self.rendered_for_engine: str = engine.model
        self.last_report: dict = {}

    def current_settings(self) -> dict:
        """The settings in effect for the NEXT turn: overlay (engine)
        defaults, overridden by the conversation settings."""
        merged = dict(self.overlay.params)
        merged.update(self.conversation_settings)
        return merged


class Manager:
    """The orchestrator: holds sessions, routes turns to engines."""

    def __init__(self, master_key: bytes, config: EngineConfig, *,
                 overlays: dict | None = None,
                 default_overlay: Overlay | None = None,
                 registry=None,
                 engine_fn=None,
                 classifier_fn=None):
        if not master_key:
            raise ValueError("manager needs the master key of the unlocked keystore")
        if config is None:
            raise ValueError("manager needs an engine configuration (the default is a user choice)")
        self._master_key = master_key
        self.config = config
        # Fail fast: an unconfigured default is a configuration error,
        # and it must surface at construction, not at the first turn.
        self.config.default_engine()
        self.overlays = dict(overlays or {})
        # The default overlay is a deployment choice, supplied here, not
        # hardcoded — a session with no overlay given uses this one.
        self.default_overlay = default_overlay
        self.registry = registry  # optional EngineRegistry (model cards); None skips the card check
        self.engine_fn = engine_fn  # injected callable: (rendered_log, engine, settings) -> reply text
        self.classifier_fn = classifier_fn  # optional: (plaintext) -> suggested tag or None; escalates only
        self.sessions: dict[str, Session] = {}

    # -- configuration -------------------------------------------------

    def resolve_engine(self, model_id: str) -> Engine:
        """Resolve a model id to an Engine. From the configuration first.
        If a registry is supplied, a model with no card cannot be used —
        first contact must build its card first."""
        engine = self.config.engines.get(model_id)
        if engine is None:
            raise ValueError(
                f"model {model_id!r} is not in the engine configuration"
            )
        if self.registry is not None:
            card = self.registry.by_model(model_id)
            if card is None:
                raise ValueError(
                    f"model {model_id!r} has no card in the registry — "
                    "first contact must poll it and build its card before it can run"
                )
        return engine

    def default_engine(self) -> Engine:
        return self.config.default_engine()

    def set_default_model(self, model_id: str) -> None:
        """Change the default model (a user choice). The id must be a
        configured engine, and — when a registry is supplied — must have
        a card. The shipped model is the default, not the only option."""
        self.resolve_engine(model_id)  # raises unless configured and (when a registry is present) carded
        self.config.default_model_id = model_id

    def get_overlay(self, overlay_id: str) -> Overlay:
        overlay = self.overlays.get(overlay_id)
        if overlay is None:
            raise ValueError(f"unknown overlay {overlay_id!r}")
        return overlay

    # -- sessions ------------------------------------------------------

    def create_session(self, session_id: str, *,
                       engine: Engine | None = None,
                       overlay: Overlay | None = None) -> Session:
        if session_id in self.sessions:
            raise ValueError(f"session {session_id!r} already exists")
        session = Session(
            session_id,
            engine or self.default_engine(),
            overlay or self.default_overlay or Overlay(id="default"),
        )
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"no session {session_id!r}")
        return session

    def set_overlay(self, session_id: str, overlay_id: str) -> None:
        """Change the overlay for subsequent turns only (prospective).
        Entering or leaving an overlay does not retag earlier turns."""
        session = self.get_session(session_id)
        session.overlay = self.get_overlay(overlay_id)

    def set_conversation_settings(self, session_id: str, settings: dict) -> None:
        """Change the conversation-level settings. Prospective-only: only
        turns made after the change carry the new settings."""
        session = self.get_session(session_id)
        session.conversation_settings.update(settings)

    # -- tags ----------------------------------------------------------

    def _resolve_tag(self, plaintext: str, overlay: Overlay, user_tag: str | None) -> str:
        """Tag precedence: user tag > overlay default > automatic
        suggestion, and a suggestion may only escalate toward NT."""
        if user_tag is not None:
            return _validate_tag(user_tag)
        tag = _validate_tag(overlay.default_ccf)
        if self.classifier_fn is not None:
            try:
                suggestion = self.classifier_fn(plaintext)
            except Exception:
                suggestion = None  # a classifier failure never blocks the turn; it just has no opinion
            if suggestion is not None and suggestion in CCF_TAGS:
                tag = _more_private(tag, suggestion)  # escalate only, never downgrade
        return tag

    # -- turns ---------------------------------------------------------

    def add_turn(self, session_id: str, role: str, plaintext: str, *,
                 ccf: str | None = None,
                 settings_override: dict | None = None) -> dict:
        """Append a turn, encrypted, carrying the settings in effect when
        it was made. Returns the stored turn dict."""
        session = self.get_session(session_id)
        tag = self._resolve_tag(plaintext, session.overlay, ccf)
        settings = session.current_settings()
        if settings_override:
            settings = {**settings, **settings_override}
        turn = encrypt_turn(role, plaintext, self._master_key, tag)
        turn["settings"] = dict(settings)
        turn["preset"] = session.overlay.id
        session.turns.append(turn)
        return turn

    def _system_message(self, session: Session, engine: Engine, *,
                        in_conversation: bool = True) -> dict | None:
        """The overlay's system prompt, taking the tag of the session's
        first turn (the filter's rule), or the overlay default if the
        session is empty. Rendered through the SAME rule as the turns: a
        cloud engine that may not see the tag gets the placeholder, not
        the prompt — the system message is prepended after
        prepare_for_engine, so it must be withheld here, or it would leak
        past the render."""
        if not session.overlay.system_prompt:
            return None
        tag = session.turns[0]["ccf"] if session.turns else session.overlay.default_ccf
        if engine.reader() == "cloud":
            if tag in ("NT", "NK") or (tag == "CO" and not in_conversation):
                return {"role": "system", "ccf": tag, "_encrypted": True,
                        "placeholder": PLACEHOLDERS[tag]}
        return {"role": "system", "ccf": tag, "content": session.overlay.system_prompt}

    def render_for_engine(self, session_id: str, engine: Engine, *,
                          in_conversation: bool = True) -> tuple[list, dict]:
        """Render the whole session for an engine (the retro-load path).
        The stored turns are untouched — only what the engine is shown
        changes. Returns (rendered_log, report)."""
        session = self.get_session(session_id)
        rendered, report = prepare_for_engine(
            session.turns, self._master_key, engine, in_conversation=in_conversation
        )
        system = self._system_message(session, engine, in_conversation=in_conversation)
        if system is not None:
            rendered = [system, *rendered]
        return rendered, report

    def set_engine(self, session_id: str, model_id: str) -> dict:
        """Change the engine mid-conversation. The entire history is
        immediately re-rendered for the new engine, and anything the new
        engine may not see becomes a placeholder at once. Returns the
        re-render report (counts withheld, by tag, no content)."""
        session = self.get_session(session_id)
        engine = self.resolve_engine(model_id)
        session.engine = engine
        rendered, report = self.render_for_engine(session_id, engine)
        session.rendered_for_engine = engine.model
        session.last_report = report
        return report

    def complete_turn(self, session_id: str, user_text: str, *,
                      ccf: str | None = None,
                      settings_override: dict | None = None) -> tuple[str, dict]:
        """One full turn: append the user turn, render the history for the
        current engine, call the engine, append the reply. Returns
        (reply_text, report). The engine function is injected; without
        one, this raises — the manager never calls a network itself."""
        if self.engine_fn is None:
            raise ValueError("no engine function configured — the manager never calls an engine directly")
        session = self.get_session(session_id)
        user_turn = self.add_turn(session_id, "user", user_text,
                                  ccf=ccf, settings_override=settings_override)
        rendered, report = self.render_for_engine(session_id, session.engine)
        settings = session.turns[-1]["settings"]
        reply_text = self.engine_fn(rendered, session.engine, settings)
        # The reply's tag is the more private of the generating overlay's
        # default and the user turn it answers (most-private-wins).
        reply_tag = _more_private(session.overlay.default_ccf, user_turn["ccf"])
        reply_turn = encrypt_turn("assistant", reply_text, self._master_key, reply_tag)
        reply_turn["settings"] = dict(settings)
        reply_turn["preset"] = session.overlay.id
        session.turns.append(reply_turn)
        session.last_report = report
        return reply_text, report
