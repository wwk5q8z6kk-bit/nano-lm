"""Coding-agent tool foundation — typed tools with path sandbox and structured results."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoscribe.tool_calling import ToolDefinition

READ_FILE_TOOL = "read_file"
LIST_DIRECTORY_TOOL = "list_directory"
SEARCH_CODE_TOOL = "search_code"
APPLY_PATCH_TOOL = "apply_patch"
RUN_COMMAND_TOOL = "run_command"

DEFAULT_COMMAND_TIMEOUT_S = 30
DEFAULT_READ_MAX_BYTES = 256_000


@dataclass(frozen=True, slots=True)
class CodingToolResult:
    ok: bool
    tool: str
    data: Mapping[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok, "tool": self.tool, "data": dict(self.data)}
        if self.error:
            payload["error"] = self.error
        return payload


def _resolve_sandbox_path(path: str, *, sandbox_root: Path) -> Path:
    root = sandbox_root.resolve()
    candidate = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if candidate == root or root in candidate.parents:
        return candidate
    raise ValueError(f"path escapes sandbox: {path}")


def coding_tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name=READ_FILE_TOOL,
            description="Read a UTF-8 text file within the repository sandbox.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from repo root"},
                    "offset": {"type": "integer", "description": "Start line (1-based)", "default": 1},
                    "limit": {"type": "integer", "description": "Max lines to return", "default": 200},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name=LIST_DIRECTORY_TOOL,
            description="List entries in a directory within the repository sandbox.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative directory path"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name=SEARCH_CODE_TOOL,
            description="Search code with ripgrep within the repository sandbox.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "glob": {"type": "string"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name=APPLY_PATCH_TOOL,
            description=(
                "Apply a unified diff patch to files in the sandbox. "
                "Dry-run by default; set apply=true to write."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "patch": {"type": "string", "description": "Unified diff text"},
                    "apply": {"type": "boolean", "default": False},
                },
                "required": ["patch"],
                "additionalProperties": False,
            },
        ),
    ]


def coding_tools_openai() -> list[dict[str, Any]]:
    return [tool.to_openai_tool() for tool in coding_tool_definitions()]


class CodingToolExecutor:
    """Execute coding tools with path confinement — no unrestricted shell."""

    def __init__(self, sandbox_root: str | Path | None = None) -> None:
        self.sandbox_root = Path(sandbox_root or os.getcwd()).resolve()

    def execute(self, name: str, arguments: Mapping[str, Any]) -> CodingToolResult:
        try:
            if name == READ_FILE_TOOL:
                return self._read_file(arguments)
            if name == LIST_DIRECTORY_TOOL:
                return self._list_directory(arguments)
            if name == SEARCH_CODE_TOOL:
                return self._search_code(arguments)
            if name == APPLY_PATCH_TOOL:
                return self._apply_patch(arguments)
            if name == RUN_COMMAND_TOOL:
                return CodingToolResult(False, name, {}, "host command execution is disabled")
            return CodingToolResult(
                ok=False,
                tool=name,
                data={},
                error=f"unknown coding tool: {name}",
            )
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            return CodingToolResult(ok=False, tool=name, data={}, error=str(exc))

    def _read_file(self, arguments: Mapping[str, Any]) -> CodingToolResult:
        path = _resolve_sandbox_path(str(arguments["path"]), sandbox_root=self.sandbox_root)
        offset = max(1, int(arguments.get("offset", 1)))
        limit = max(1, int(arguments.get("limit", 200)))
        raw = path.read_bytes()[:DEFAULT_READ_MAX_BYTES]
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        slice_lines = lines[offset - 1 : offset - 1 + limit]
        return CodingToolResult(
            ok=True,
            tool=READ_FILE_TOOL,
            data={
                "path": str(path.relative_to(self.sandbox_root)),
                "offset": offset,
                "limit": limit,
                "content": "\n".join(slice_lines),
                "total_lines": len(lines),
            },
        )

    def _list_directory(self, arguments: Mapping[str, Any]) -> CodingToolResult:
        path = _resolve_sandbox_path(str(arguments["path"]), sandbox_root=self.sandbox_root)
        if not path.is_dir():
            raise ValueError(f"not a directory: {path}")
        entries = sorted(item.name for item in path.iterdir())
        return CodingToolResult(
            ok=True,
            tool=LIST_DIRECTORY_TOOL,
            data={"path": str(path.relative_to(self.sandbox_root)), "entries": entries},
        )

    def _search_code(self, arguments: Mapping[str, Any]) -> CodingToolResult:
        pattern = str(arguments["pattern"])
        rel = str(arguments.get("path", "."))
        search_path = _resolve_sandbox_path(rel, sandbox_root=self.sandbox_root)
        cmd = ["rg", "--no-config", "--json", "--regexp", pattern]
        glob = arguments.get("glob")
        if glob:
            cmd.extend(["--glob", str(glob)])
        cmd.extend(["--", str(search_path)])
        proc = subprocess.run(
            cmd,
            cwd=self.sandbox_root,
            capture_output=True,
            text=True,
            timeout=DEFAULT_COMMAND_TIMEOUT_S,
            check=False,
        )
        matches: list[dict[str, Any]] = []
        for line in proc.stdout.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "match":
                data = payload.get("data") or {}
                path = data.get("path") or {}
                matches.append(
                    {
                        "path": path.get("text"),
                        "line_number": (data.get("line_number")),
                        "text": ((data.get("lines") or {}).get("text")),
                    }
                )
        return CodingToolResult(
            ok=proc.returncode in {0, 1},
            tool=SEARCH_CODE_TOOL,
            data={"pattern": pattern, "matches": matches[:50]},
            error=None if proc.returncode in {0, 1} else proc.stderr.strip() or "rg failed",
        )

    def _apply_patch(self, arguments: Mapping[str, Any]) -> CodingToolResult:
        patch = str(arguments["patch"])
        apply = bool(arguments.get("apply", False))
        if not apply:
            return CodingToolResult(
                ok=True,
                tool=APPLY_PATCH_TOOL,
                data={"dry_run": True, "patch_bytes": len(patch.encode("utf-8"))},
            )
        proc = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=self.sandbox_root,
            input=patch,
            capture_output=True,
            text=True,
            timeout=DEFAULT_COMMAND_TIMEOUT_S,
            check=False,
        )
        if proc.returncode != 0:
            return CodingToolResult(
                ok=False,
                tool=APPLY_PATCH_TOOL,
                data={},
                error=proc.stderr.strip() or "patch check failed",
            )
        proc = subprocess.run(
            ["git", "apply", "-"],
            cwd=self.sandbox_root,
            input=patch,
            capture_output=True,
            text=True,
            timeout=DEFAULT_COMMAND_TIMEOUT_S,
            check=False,
        )
        return CodingToolResult(
            ok=proc.returncode == 0,
            tool=APPLY_PATCH_TOOL,
            data={"applied": proc.returncode == 0},
            error=None if proc.returncode == 0 else proc.stderr.strip(),
        )

def execute_coding_tool_call(
    name: str,
    arguments: str | Mapping[str, Any],
    *,
    sandbox_root: str | Path | None = None,
) -> CodingToolResult:
    if isinstance(arguments, str):
        payload = json.loads(arguments)
    else:
        payload = dict(arguments)
    return CodingToolExecutor(sandbox_root=sandbox_root).execute(name, payload)
