"""Transcript → action log, contract v1.1.

A separate implementation path (owner decision D1). v1.0's emitter
(`benchmarks/transcript_emitter.py`) is frozen and untouched; this module
*imports* its segment-level machinery — tokenizer, control-flow normaliser,
wrappers, read/write/test detection, path relativisation, result-status
mapping, post-state reconstruction — rather than copying it, so the two
emitters classify a single segment identically by construction and differ only
where v1.1 says they differ:

    C2  pre_state_sha256 on mutations            (this stage)
    C3  declared_scope / source_scope verbatim   (next stage)
    C1  one call -> ordered sub-actions, call_id  (final stage)

Stage discipline: each change lands as its own commit with its own fixtures
and tests (D15: C2 -> C3 -> C1). At the C2 stage this emitter still produces
exactly one action per call, computed by v1.0's `classify_bash`, so that the
backward-compatibility identity (spec T3) is trivially true and C2 can be
tested in isolation.

Importing private (`_`-prefixed) names across modules is a deliberate smell:
the alternative is a 900-line copy that would drift. v1.0 is frozen
permanently, so the imported surface cannot change under this module.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmarks.transcript_emitter import (  # frozen v1.0 machinery
    DEFAULT_VERIFY_PATTERNS,
    PATH_INPUT_KEYS,
    TOOL_CLASS,
    _canonical,
    _lines_changed_from_patch,
    _parse_ts,
    _post_state,
    _result_text,
    _sha256,
    classify_bash,
    relativize,
    result_status,
)
from benchmarks.transcript_sanitizer import iter_records

LOG_SCHEMA_VERSION = "benchmark_lab.agent_action_log.v1.1"
EMITTER_VERSION = "benchmark_lab.transcript_emitter.v1.1"


@dataclass
class ClassificationTrace:
    seq: int
    call_id: str
    sub_index: int
    tool: str
    action_class: str
    rule: str
    evidence: str = ""
    compound: bool = False
    segments: int = 1
    notes: list[str] = field(default_factory=list)
    preview: str = ""


@dataclass
class EmitResult:
    log: dict[str, Any]
    trace: list[ClassificationTrace]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"log": self.log, "trace": [asdict(t) for t in self.trace], "stats": self.stats}


@dataclass
class Effect:
    """One atomic effect of a tool call, in command order."""

    action_class: str
    rule: str
    evidence: str = ""
    targets: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _pre_state(tool: str, tur: Any, status: str, targets: Sequence[str]) -> str | None:
    """C2. Digest of the single target's content before the mutation.

    Non-null exactly when the transcript carries it: an ok Edit/MultiEdit or an
    ok Write-update has `originalFile`. Write-create has no prior content and
    is not distinguished from unknown (D8: null, no sentinel). Non-ok results
    are null (no mutation happened). More than one target is null (D9). Bash is
    always null — the transcript has no file content for shell writes.
    """
    if status != "ok" or len(targets) != 1 or not isinstance(tur, dict):
        return None
    if tool in ("Edit", "MultiEdit", "Write"):
        original = tur.get("originalFile")
        if isinstance(original, str):
            return _sha256(original)
    return None


def _effects_for_call(tool: str, tool_input: dict[str, Any], cwd: str | None, *,
                      verify_patterns: Sequence[re.Pattern[str]], read_only_patterns: Sequence[re.Pattern[str]]) -> tuple[list[Effect], bool, int, str]:
    """Effects of one call, in order. Returns (effects, compound, segments, preview).

    C2/C3 stage: exactly one effect per call, from v1.0's single-winner
    classifier, so the log is v1.0-shaped plus the new fields. C1 replaces the
    Bash branch with a decomposition.
    """
    if tool == "Bash":
        cls = classify_bash(str(tool_input.get("command", "")), verify_patterns=verify_patterns, read_only_patterns=read_only_patterns)
        effect = Effect(cls["action_class"], cls["rule"], cls["evidence"],
                        [relativize(t, cwd) for t in cls["targets"]], list(cls["tests_run"]), list(cls["notes"]))
        preview = str(tool_input.get("command", ""))[:120].replace("\n", "⏎")
        return [effect], cls["compound"], cls["segments"], preview
    action_class, rule, note = TOOL_CLASS.get(tool, ("other", "tool:unknown", "unmapped tool"))
    targets: list[str] = []
    for k in PATH_INPUT_KEYS:
        v = tool_input.get(k)
        if isinstance(v, str) and v:
            targets.append(relativize(v, cwd))
            break
    preview = (targets[0] if targets else str(tool_input.get("pattern", "")))[:120]
    return [Effect(action_class, rule, tool, targets, [], [note] if note else [])], False, 1, preview


def emit(
    records: Iterable[dict[str, Any]],
    *,
    verify_patterns: Sequence[re.Pattern[str]] = DEFAULT_VERIFY_PATTERNS,
    read_only_patterns: Sequence[re.Pattern[str]] = (),
) -> EmitResult:
    """Transcript records → v1.1 action log + trace + stats. Pure over its input."""
    pending: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    actions: list[dict[str, Any]] = []
    trace: list[ClassificationTrace] = []
    stats: dict[str, Any] = {
        "emitter_version": EMITTER_VERSION,
        "records": 0, "tool_uses": 0, "calls": 0, "actions": 0, "orphan_tool_use": 0, "orphan_result": 0,
        "by_tool": Counter(), "by_class": Counter(), "by_rule": Counter(), "by_status": Counter(),
        "notes": Counter(), "effects_per_call": Counter(), "non_monotonic_timestamps": 0, "usage_attributed": 0,
        "mutate_without_targets": 0, "read_without_targets": 0, "unknown_tools": Counter(),
        "pre_state_present": 0,
    }
    seq = 0
    last_ts = None

    for record in records:
        stats["records"] += 1
        message = record.get("message") or {}
        content = message.get("content")
        if not isinstance(content, list):
            continue
        ts = _parse_ts(record.get("timestamp"))
        if ts and last_ts and ts < last_ts:
            stats["non_monotonic_timestamps"] += 1
        if ts:
            last_ts = ts

        if record.get("type") == "assistant":
            first_in_message = True
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                    stats["tool_uses"] += 1
                    pending[block["id"]] = (block, record, {"usage": message.get("usage") if first_in_message else None, "attributed": first_in_message})
                    first_in_message = False
            continue
        if record.get("type") != "user":
            continue

        tur = record.get("toolUseResult")
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            call_id = block.get("tool_use_id")
            if call_id not in pending:
                stats["orphan_result"] += 1
                continue
            use_block, use_record, meta = pending.pop(call_id)
            stats["calls"] += 1
            tool = str(use_block.get("name"))
            tool_input = use_block.get("input") or {}
            cwd = use_record.get("cwd")
            if tool not in TOOL_CLASS and tool != "Bash":
                stats["unknown_tools"][tool] += 1

            effects, compound, segments, preview = _effects_for_call(
                tool, tool_input, cwd, verify_patterns=verify_patterns, read_only_patterns=read_only_patterns)
            status = result_status(block, tur)
            use_ts, res_ts = _parse_ts(use_record.get("timestamp")), _parse_ts(record.get("timestamp"))
            wall = int((res_ts - use_ts).total_seconds() * 1000) if use_ts and res_ts and res_ts >= use_ts else None
            usage = meta.get("usage") or {}
            if meta.get("attributed") and usage:
                stats["usage_attributed"] += 1
            tokens_in = (usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)) if usage else None
            tokens_out = usage.get("output_tokens") if usage else None
            normalized_input = {k: (relativize(v, cwd) if k in PATH_INPUT_KEYS and isinstance(v, str) else v) for k, v in tool_input.items()}
            args_sha = _sha256(_canonical(normalized_input))
            result_sha = _sha256(_result_text(block))
            stats["effects_per_call"][len(effects)] += 1

            for sub_index, eff in enumerate(effects):
                notes = list(eff.notes)
                if len(effects) > 1:
                    notes.append("status:inherited-from-call")
                targets = list(dict.fromkeys(eff.targets))
                lines_changed = None
                post_state = pre_state = None
                if eff.action_class == "mutate":
                    if tool != "Bash":
                        lines_changed = _lines_changed_from_patch(tur.get("structuredPatch") if isinstance(tur, dict) else None)
                        if lines_changed is None and tool == "Write" and isinstance(tool_input.get("content"), str) and status == "ok":
                            c = tool_input["content"]
                            lines_changed = c.count("\n") + (0 if c.endswith("\n") else 1)
                    if status == "ok":
                        post_state = _post_state(tool, tool_input, tur, notes) if len(targets) == 1 else None
                        pre_state = _pre_state(tool, tur, status, targets)
                        if pre_state is not None:
                            stats["pre_state_present"] += 1
                    if not targets:
                        stats["mutate_without_targets"] += 1
                if eff.action_class == "read" and not targets:
                    stats["read_without_targets"] += 1

                first = sub_index == 0
                action: dict[str, Any] = {
                    "seq": seq,
                    "call_id": str(call_id),
                    "action_class": eff.action_class,
                    "tool": tool,
                    "args_sha256": args_sha,
                    "targets": targets,
                    "result_status": status,
                    "result_sha256": result_sha,
                    "pre_state_sha256": pre_state,
                    "post_state_sha256": post_state,
                    "tokens_in": tokens_in if first else None,
                    "tokens_out": tokens_out if first else None,
                    "cost_usd": None,
                    "wall_clock_ms": wall if first else None,
                }
                if lines_changed is not None:
                    action["lines_changed"] = lines_changed
                if eff.tests_run:
                    action["tests_run"] = sorted(set(eff.tests_run))
                actions.append(action)
                trace.append(ClassificationTrace(seq=seq, call_id=str(call_id), sub_index=sub_index, tool=tool,
                                                 action_class=eff.action_class, rule=eff.rule, evidence=eff.evidence,
                                                 compound=compound, segments=segments, notes=notes, preview=preview))
                stats["actions"] += 1
                stats["by_tool"][tool] += 1
                stats["by_class"][eff.action_class] += 1
                stats["by_rule"][eff.rule] += 1
                stats["by_status"][status] += 1
                for n in notes:
                    stats["notes"][n] += 1
                seq += 1

    stats["orphan_tool_use"] = len(pending)
    for k in ("by_tool", "by_class", "by_rule", "by_status", "notes", "unknown_tools", "effects_per_call"):
        stats[k] = dict(stats[k])
    log: dict[str, Any] = {
        "schema_version": LOG_SCHEMA_VERSION,
        "agent_run_id": None,
        "actions": actions,
        "pre_run_state_sha256": None,
        "post_run_state_sha256": None,
    }
    return EmitResult(log=log, trace=trace, stats=stats)


def emit_file(path: str | Path, **kwargs: Any) -> EmitResult:
    return emit(iter_records(path), **kwargs)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Emit a v1.1 action log (+ trace, stats) from a transcript.")
    parser.add_argument("transcript")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    result = emit_file(args.transcript)
    if args.out:
        Path(args.out).write_text(json.dumps(result.to_dict(), indent=1, ensure_ascii=False))
    print(json.dumps(result.stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
