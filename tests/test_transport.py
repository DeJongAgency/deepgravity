"""test_transport.py — DoD 1 acceptance: structured transport, quoting error rate 0.

Runnable without pytest: python tests/test_transport.py (plain asserts).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core.transport import Tool, TransportError, run_tool, run_argv


def test_quotes_and_spaces_survive_exec_form():
    """The DoD case: a model param full of quotes/spaces must pass through
    untouched because no shell ever re-parses it."""
    t = Tool(
        name="echo_arg",
        argv=[sys.executable, "-c", "import sys; print(sys.argv[1])", "{msg}"],
        param_names=["msg"],
        param_types={"msg": "str"},
    )
    nasty = "hello world 'single' \"double\" $HOME `tick` * glob ?"
    proc = run_tool(t, {"msg": nasty})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == nasty, "exec-form must preserve the exact value"


def test_missing_required_param_raises():
    t = Tool(name="need_msg", argv=[sys.executable, "-c", "pass", "{msg}"], param_names=["msg"])
    try:
        run_tool(t, {})
    except TransportError as e:
        assert "missing" in str(e)
        return
    raise AssertionError("missing param must raise TransportError")


def test_wrong_type_raises():
    t = Tool(name="need_int", argv=[sys.executable, "-c", "pass", "{n}"],
             param_names=["n"], param_types={"n": "int"})
    try:
        run_tool(t, {"n": "not-an-int"})
    except TransportError as e:
        assert "int" in str(e)
        return
    raise AssertionError("wrong type must raise TransportError")


def test_unresolved_template_token_raises():
    """A template bug (literal '{' left in argv) is caught, never executed."""
    t = Tool(name="bad", argv=[sys.executable, "-c", "pass", "{unbound}"], param_names=[])
    try:
        run_tool(t, {})
    except TransportError:
        return
    raise AssertionError("unresolved template token must raise")


def test_run_argv_no_shell():
    proc = run_argv([sys.executable, "-c", "print('ok')"])
    assert proc.returncode == 0 and proc.stdout.strip() == "ok"


def test_stdin_text_utf8():
    proc = run_argv([sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
                    input_text="caf\u00e9 \u2603")
    assert proc.stdout == "caf\u00e9 \u2603"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
