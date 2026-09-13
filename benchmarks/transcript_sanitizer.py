"""Structure-preserving sanitizer for Claude Code session transcripts.

Purpose
-------
The transcript → action-log emitter (`benchmarks/transcript_emitter.py`) must be
developed and tested against **checked-in fixtures**, never by reading
`~/.claude/projects/` directly. Raw transcripts contain file bodies, prompts,
model prose, and potentially credentials. This module strips all of that while
preserving every field the emitter reads, so that

    emit(raw)  and  emit(sanitize(raw))

agree on `tool`, `action_class`, `targets`, `result_status`, `seq`, and token
accounting. That agreement is an invariant the validation experiment checks on
every real transcript; it is what makes the fixtures trustworthy stand-ins.

What is preserved (the emitter's inputs)
----------------------------------------
- record `type`, `uuid`, `parentUuid`, `timestamp`, `cwd`, `gitBranch`, `isSidechain`
- `message.role`, `message.model`, the four token counters in `message.usage`
- `tool_use` blocks: `id`, `name`, and a per-key-sanitized `input` — paths and
  shell commands are kept (they are the classification signal), free text is not
- `tool_result` blocks: `tool_use_id`, `is_error`, and the **first line only** of
  the content when it is an error header (``<tool_use_error>``, ``Exit code N``,
  ``File does not exist`` ...). Everything else is replaced.
- `toolUseResult`: structural keys (`interrupted`, `filePath`, `type`,
  `replaceAll`, patch hunk geometry) kept; content keys replaced.

What is replaced
----------------
Every free-text string becomes a **redaction marker** that preserves two
properties the emitter relies on:

    <redacted sha256=<16 hex> len=<N>>  followed by (lines - 1) newlines

1. Equality: identical real strings map to identical markers, so digests
   computed over them (`result_sha256`, `post_state_sha256` for Write) still
   detect repeats and reversions.
2. Line count: ``"\\n"`` is preserved, so `lines_changed` for a created file is
   the same on the fixture as on the raw transcript.

Shell commands additionally pass through a secret-pattern redactor. The
patterns are conservative and listed in `SECRET_PATTERNS`; a fixture is only as
clean as this list, which is why the fixture README requires a manual read of
every checked-in `command` string.

Not a general anonymizer. It is fit for one purpose: producing emitter fixtures
from this project's own sessions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

SANITIZER_VERSION = "benchmark_lab.transcript_sanitizer.v1"

#: Record types the emitter reads. Everything else (attachments, queue
#: operations, titles, mode switches) is dropped.
KEPT_RECORD_TYPES = frozenset({"user", "assistant"})

#: Envelope keys kept verbatim (after home-directory normalisation of `cwd`).
KEPT_ENVELOPE_KEYS = ("type", "uuid", "parentUuid", "timestamp", "cwd", "gitBranch", "isSidechain", "version")

#: Token counters kept from `message.usage`.
KEPT_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")

#: `tool_use.input` keys whose values are structural (paths, flags, small enums)
#: and are kept. Any string under a key NOT listed here is redacted.
KEPT_INPUT_KEYS = frozenset({
    "file_path", "path", "notebook_path", "pattern", "glob", "type", "output_mode",
    "replace_all", "description", "timeout", "run_in_background", "limit", "offset",
    "pages", "head_limit", "-n", "-i", "-A", "-B", "-C", "multiline", "cell_id",
    "cell_type", "edit_mode", "subagent_type", "skill", "model", "isolation", "to",
    "max_results", "scope", "name", "mode", "channel",
})

#: Keys whose values are shell commands: kept, but secret-redacted.
COMMAND_KEYS = frozenset({"command"})

#: `toolUseResult` keys kept verbatim.
KEPT_RESULT_KEYS = frozenset({
    "interrupted", "isImage", "noOutputExpected", "filePath", "type", "replaceAll",
    "userModified", "numFiles", "numLines", "durationMs", "truncated", "mode",
    "startLine", "totalLines", "returnCodeInterpretation", "exitCode", "filenames",
})

#: Error-header patterns. When a `tool_result.content` string starts with one of
#: these, its first line is kept so the emitter can classify `tool_error` vs
#: `error` identically on raw and sanitized input.
ERROR_HEADER = re.compile(
    r"^(<tool_use_error>|Exit code \d+|Error:|EISDIR|File does not exist|"
    r"Output does not match|Working directory|Failed to|InputValidationError)"
)

#: Conservative credential shapes. Group 1 (if any) is a keep-prefix; the
#: remainder of the match is replaced.
SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "<secret>"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "<secret>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<secret>"),
    (re.compile(r"xox[bap]-[A-Za-z0-9-]{10,}"), "<secret>"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._-]{16,}"), r"\1<secret>"),
    (re.compile(r"(\b(?:api[_-]?key|token|password|passwd|secret)\s*[=:]\s*)['\"]?[^\s'\"]{6,}", re.I), r"\1<secret>"),
)


def _home() -> str:
    return os.path.expanduser("~")


def normalize_home(value: str) -> str:
    """Replace the literal home directory with ``~`` so fixtures are user-agnostic."""
    home = _home()
    return value.replace(home, "~") if home and home != "/" else value


def redact_secrets(command: str) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        command = pattern.sub(replacement, command)
    return command


def redaction_marker(text: str) -> str:
    """The marker that stands in for free text. See module docstring for the two
    properties it preserves (equality, line count)."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    lines = text.count("\n")
    return f"<redacted sha256={digest} len={len(text)}>" + ("\n" * lines)


def _redact_result_content(content: Any) -> Any:
    if isinstance(content, str):
        first_line = content.split("\n", 1)[0]
        if ERROR_HEADER.match(content):
            # The kept header line can name a path ("Working directory "/Users/..."
            # does not exist"), so it gets the same home normalisation as paths.
            rest = content[len(first_line):]
            return normalize_home(redact_secrets(first_line)) + ("\n" + redaction_marker(rest[1:]) if rest else "")
        return redaction_marker(content)
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append({"type": "text", "text": redaction_marker(str(block.get("text", "")))})
            elif isinstance(block, dict):
                out.append({"type": block.get("type", "unknown"), "redacted": True})
            else:
                out.append({"redacted": True})
        return out
    return content


def _sanitize_value(key: str, value: Any) -> Any:
    """Per-key policy for `tool_use.input` and nested structures."""
    if key in COMMAND_KEYS and isinstance(value, str):
        return normalize_home(redact_secrets(value))
    if isinstance(value, str):
        if key in KEPT_INPUT_KEYS:
            return normalize_home(value)
        return redaction_marker(value)
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return [_sanitize_value(key, v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_value(k, v) for k, v in value.items()}
    return redaction_marker(str(value))


def sanitize_tool_use_input(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {k: _sanitize_value(k, v) for k, v in (tool_input or {}).items()}


def sanitize_tool_use_result(result: Any) -> Any:
    """`toolUseResult` envelope field: structural keys kept, content replaced."""
    if isinstance(result, str):
        return _redact_result_content(result)
    if isinstance(result, list):
        return {"_redacted": "list", "len": len(result)}
    if not isinstance(result, dict):
        return result
    out: dict[str, Any] = {}
    for key, value in result.items():
        if key == "structuredPatch" and isinstance(value, list):
            out[key] = [
                {
                    "oldStart": h.get("oldStart"), "oldLines": h.get("oldLines"),
                    "newStart": h.get("newStart"), "newLines": h.get("newLines"),
                    # keep only the diff prefix character per line: enough for
                    # lines_changed, nothing of the content
                    "lines": [(line[:1] if isinstance(line, str) and line else " ") for line in h.get("lines", [])],
                }
                for h in value if isinstance(h, dict)
            ]
        elif key == "file" and isinstance(value, dict):
            out[key] = {
                k: (normalize_home(v) if k == "filePath" and isinstance(v, str)
                    else redaction_marker(v) if isinstance(v, str) else v)
                for k, v in value.items()
            }
        elif key == "filenames" and isinstance(value, list):
            out[key] = [normalize_home(v) if isinstance(v, str) else v for v in value]
        elif key in KEPT_RESULT_KEYS:
            out[key] = normalize_home(value) if isinstance(value, str) else value
        elif isinstance(value, str):
            out[key] = redaction_marker(value)
        elif isinstance(value, (bool, int, float)) or value is None:
            out[key] = value
        else:
            out[key] = sanitize_tool_use_result(value)
    return out


def _sanitize_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": redaction_marker(content)}]
    if not isinstance(content, list):
        return []
    out: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "tool_use":
            out.append({
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": sanitize_tool_use_input(str(block.get("name")), block.get("input") or {}),
            })
        elif kind == "tool_result":
            entry: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id"),
                "content": _redact_result_content(block.get("content")),
            }
            if "is_error" in block:
                entry["is_error"] = block["is_error"]
            out.append(entry)
        elif kind == "text":
            out.append({"type": "text", "text": redaction_marker(str(block.get("text", "")))})
        elif kind in ("thinking", "redacted_thinking"):
            continue  # no structural value; dropped entirely
        else:
            out.append({"type": kind or "unknown", "redacted": True})
    return out


def sanitize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """One transcript line → one fixture line, or None if the record is dropped."""
    if record.get("type") not in KEPT_RECORD_TYPES:
        return None
    out: dict[str, Any] = {}
    for key in KEPT_ENVELOPE_KEYS:
        if key in record:
            value = record[key]
            out[key] = normalize_home(value) if isinstance(value, str) else value
    if "sessionId" in record:
        out["sessionId"] = "sha256:" + hashlib.sha256(str(record["sessionId"]).encode()).hexdigest()[:16]
    message = record.get("message") or {}
    sanitized_message: dict[str, Any] = {"role": message.get("role", record.get("type"))}
    if "model" in message:
        sanitized_message["model"] = message["model"]
    usage = message.get("usage")
    if isinstance(usage, dict):
        sanitized_message["usage"] = {k: usage[k] for k in KEPT_USAGE_KEYS if k in usage}
    sanitized_message["content"] = _sanitize_content_blocks(message.get("content"))
    out["message"] = sanitized_message
    if "toolUseResult" in record:
        out["toolUseResult"] = sanitize_tool_use_result(record["toolUseResult"])
    return out


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Parse a JSONL transcript, skipping unparseable lines (they are counted by
    the emitter's stats, not here)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def sanitize_records(records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for record in records:
        cleaned = sanitize_record(record)
        if cleaned is not None:
            yield cleaned


def sanitize_file(src: str | Path, dst: str | Path) -> dict[str, int]:
    """Write a sanitized fixture. Returns counts for the fixture README."""
    kept = dropped = tool_uses = 0
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as out:
        for record in iter_records(src):
            cleaned = sanitize_record(record)
            if cleaned is None:
                dropped += 1
                continue
            kept += 1
            tool_uses += sum(1 for b in cleaned["message"]["content"] if b.get("type") == "tool_use")
            out.write(json.dumps(cleaned, sort_keys=True, ensure_ascii=False) + "\n")
    return {"records_kept": kept, "records_dropped": dropped, "tool_uses": tool_uses}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Sanitize a Claude Code transcript into an emitter fixture.")
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args(argv)
    counts = sanitize_file(args.src, args.dst)
    print(json.dumps({"src": normalize_home(args.src), "dst": args.dst, **counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
