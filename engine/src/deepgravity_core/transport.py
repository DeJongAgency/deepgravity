"""transport.py — DoD 1: structured tool transport.

The model NEVER assembles a shell command string. Tools accept structured
parameters (typed args / JSON); the engine performs escaping, argv assembly,
and exec. Subprocess calls use exec-form arrays (no shell interpolation)
unless a shell is explicitly required — and then escaping is the engine's
job, never the model's.

Acceptance: no tool in the registry takes a raw shell string composed by the
model. Quoting error rate = 0.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

class TransportError(RuntimeError):
    """Raised on transport-level failures (validation, exec, timeout)."""


@dataclass(frozen=True)
class Tool:
    """A registered tool. `argv` is an exec-form template; a token of the form
    {param} is replaced with the (string-coerced) structured parameter value.

    The engine substitutes and executes — the model supplies only a typed
    parameter dict, never a command string. Tokens that are not {param} are
    literal argv elements (no shell re-parsing ever occurs).
    """
    name: str
    argv: Sequence[str]
    description: str = ""
    param_names: Sequence[str] = ()          # ordered names, for validation/help
    param_types: Dict[str, str] = field(default_factory=dict)  # name -> "str"|"int"|"float"
    cwd: Optional[str] = None
    timeout: float = 60.0

    def build_argv(self, params: Dict[str, Any]) -> List[str]:
        """Validate + substitute structured params into the argv template.

        Raises TransportError on missing params / wrong types. Never shells.
        """
        argv: List[str] = []
        for part in self.argv:
            if part.startswith("{") and part.endswith("}"):
                key = part[1:-1]
                if key not in params or params[key] is None:
                    raise TransportError(
                        f"tool '{self.name}': missing required param '{key}'")
                val = params[key]
                t = self.param_types.get(key, "str")
                if t == "int" and not isinstance(val, int):
                    raise TransportError(
                        f"tool '{self.name}': param '{key}' expects int, got {type(val).__name__}")
                if t == "float" and not isinstance(val, (int, float)):
                    raise TransportError(
                        f"tool '{self.name}': param '{key}' expects float, got {type(val).__name__}")
                if t == "str" and not isinstance(val, str):
                    raise TransportError(
                        f"tool '{self.name}': param '{key}' expects str, got {type(val).__name__}")
                argv.append(str(val))
            else:
                argv.append(part)
        # Reject un-substituted '{' tokens (a template bug, never model input)
        for a in argv:
            if "{" in a or "}" in a:
                raise TransportError(f"tool '{self.name}': unresolved template token in argv: {a!r}")
        return argv


def run_argv(
    argv: Sequence[str],
    *,
    cwd: Optional[str] = None,
    timeout: float = 60.0,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Execute an argv array directly (exec-form, no shell). stdin fed as UTF-8
    text when provided. Returns the CompletedProcess; raises TransportError on
    failure/timeout/missing executable."""
    try:
        return subprocess.run(
            list(argv),
            cwd=cwd,
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            input=input_text,
        )
    except subprocess.TimeoutExpired as e:
        raise TransportError(f"command timed out after {timeout}s: {argv[0]}") from e
    except FileNotFoundError as e:
        raise TransportError(f"executable not found: {argv[0]}") from e


def run_tool(tool: Tool, params: Dict[str, Any]) -> subprocess.CompletedProcess:
    """Validate params against a tool, build argv, execute. The single entry
    point the registry hands the model; no other path reaches subprocess."""
    argv = tool.build_argv(params)
    return run_argv(argv, cwd=tool.cwd, timeout=tool.timeout)


def run_shell_internal(command: str, *, cwd: Optional[str] = None, timeout: float = 60.0):
    """ENGINE-INTERNAL escape hatch ONLY (pipes/redirects in engine code).

    Never exposed to the model; never fed model-assembled strings. Escaping is
    the engine's responsibility at the call site (DoD 1 keeps the model out of
    this path entirely).
    """
    try:
        return subprocess.run(
            command, shell=True, cwd=cwd, timeout=timeout, check=False,
            capture_output=True, text=True, encoding="utf-8")
    except subprocess.TimeoutExpired as e:
        raise TransportError(f"internal shell command timed out after {timeout}s") from e
