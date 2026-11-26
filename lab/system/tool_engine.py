"""Tool Execution Engine: Isolated Python Computation Sandbox & Stateful Multi-Turn API Environment."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from typing import Any

from lab.trackb.eval_capabilities import parse_tool_calls


def execute_python_computation(script: str, timeout_sec: float = 3.0) -> tuple[bool, str]:
    """Run an isolated Python verification/solver snippet and return (ok, stdout_or_error)."""
    cleaned = script.strip()
    if "```" in cleaned:
        import re
        m = re.search(r"```(?:python)?\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if m:
            cleaned = m.group(1).strip()
    if not cleaned:
        return False, "empty_script"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", cleaned],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"
    if proc.returncode == 0:
        out = proc.stdout.strip()
        return (True, out[:4000]) if out else (False, "no_stdout_printed")
    err_lines = (proc.stderr or proc.stdout).strip().splitlines()
    return False, (err_lines[-1][:200] if err_lines else f"exit_{proc.returncode}")


class StatefulToolEnvironment:
    """Multi-step API environment with real state transitions, chained IDs, and live error modes."""

    def __init__(self, schemas: list[dict[str, Any]], initial_state: dict[str, Any], handlers: dict[str, Any]):
        self.schemas = schemas
        self.state = copy.deepcopy(initial_state)
        self.handlers = handlers
        self.call_history: list[dict[str, Any]] = []

    def step(self, model_output: str) -> tuple[bool, list[dict[str, Any]], str]:
        """Parse and execute tool call(s) against live state.

        Returns `(done_no_tool_call, observations, parse_status)`.
        When the model outputs `{"tool_calls": [], "response": "..."}` or plain text final answer
        after completing tool calls, `done_no_tool_call=True`.
        """
        calls, status = parse_tool_calls(model_output)
        if calls is None:
            return False, [{"error": "MalformedToolJSON", "message": f"Failed to parse tool call JSON ({status})."}], status
        if len(calls) == 0:
            return True, [], status

        observations: list[dict[str, Any]] = []
        for c in calls:
            name = c.get("name", "")
            args = c.get("arguments", {})
            if not isinstance(args, dict):
                obs = {"tool": name, "ok": False, "error": "InvalidArgumentsType", "message": "`arguments` must be a JSON object."}
            elif name not in self.handlers:
                obs = {"tool": name, "ok": False, "error": "UnknownTool", "message": f"Tool `{name}` is not in available schemas."}
            else:
                try:
                    res = self.handlers[name](self.state, args)
                    obs = {"tool": name, **res}
                except Exception as exc:
                    obs = {"tool": name, "ok": False, "error": type(exc).__name__, "message": str(exc)}
            self.call_history.append({"name": name, "arguments": args, "observation": obs})
            observations.append(obs)
        return False, observations, "ok"
