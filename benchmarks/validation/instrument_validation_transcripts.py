"""INSTRUMENT VALIDATION — run the transcript emitter over this project's real sessions.

What this is
------------
The one component in `benchmarks/` permitted to read `~/.claude/projects/`.
Everything else (emitter, metrics, tests) works from checked-in fixtures. This
script exists to answer one question:

    Does the instrumentation work on real trajectories?

It does **not** measure the quality of any system. The transcripts are this
project's own agent sessions; running Class A over them is dogfooding the
instrument, and every number it produces is labelled accordingly. Nothing here
is benchmark evidence, enters the evidence ledger, or bears on any model.

What it checks
--------------
1. The raw/sanitized invariant: `emit(raw)` and `emit(sanitize(raw))` must agree
   on every classification field. Disagreement means the fixtures are not
   trustworthy stand-ins for real transcripts.
2. Schema validity of every emitted log.
3. Classification distribution: rules fired, compound rate, fallback residue,
   notes — the material for the manual audit (step 7 of the owner's sequence).
4. Class A profile per transcript, reported as a vector, never aggregated into
   a score.
5. Audit samples: seeded random previews per rule and per note, from the
   *sanitized* emission (commands are home-normalised and secret-redacted),
   written for a person to read.

Outputs go to the artifacts directory under `instrument-validation-transcripts/`.

Usage
-----
    python3 -m benchmarks.validation.instrument_validation_transcripts \\
        --source '~/.claude/projects/*<project-id>*/*.jsonl' \\
        --out <artifacts-dir>/instrument-validation-transcripts
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import jsonschema

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.process_metrics import class_a_profile  # noqa: E402
from benchmarks.transcript_emitter import EMITTER_VERSION, emit  # noqa: E402
from benchmarks.transcript_sanitizer import SANITIZER_VERSION, iter_records, sanitize_records  # noqa: E402

SCHEMA = json.loads((REPO / "benchmarks/AGENT_ACTION_LOG.schema.json").read_text())

#: Harness-specific conventions for THIS repository, passed to the emitter as
#: extensions. Declared here, in the validation script, not in the emitter.
REPO_VERIFY_PATTERNS = (
    re.compile(r"(^|\s)python3?\s+scripts/check_\w+\.py"),
)
#: `orx` is this project's research CLI. Its query subcommands (and their `exp`
#: abbreviations, which the audit found were the dominant form) only read;
#: `exp run`, `create-experiment`, `exp cancel` change remote state and stay
#: `execute`. `exp wait` polls; it mutates nothing.
REPO_READ_ONLY_PATTERNS = (
    re.compile(r"(^|\s)orx\s+(experiment|exp|ex)\s+(list|desc|describe|status|wait|show|view|--help)\b"),
    re.compile(r"(^|\s)orx\s+(project\s+(view|list|--help)|runs|logs|run\s+(view|list|logs)|discover|paper|--help|-h)\b"),
    re.compile(r"(^|\s)orx\s+\S+\s+--help\b"),
    re.compile(r"peer-claims\b"),
)

INVARIANT_FIELDS = ("seq", "tool", "action_class", "targets", "result_status", "tokens_in", "tokens_out",
                    "wall_clock_ms", "lines_changed", "tests_run")
EXPECTED_TRACE_DIFF = {"post_state:derived"}
AUDIT_SAMPLES_PER_BUCKET = 6


def transcript_id(path: str) -> str:
    chat = path.split("chat-")[1][:8] if "chat-" in path else "?"
    return f"{chat}/{os.path.basename(path)[:8]}"


def run_one(path: str, rng: random.Random) -> dict[str, Any]:
    raw = emit(iter_records(path), verify_patterns=REPO_VERIFY_PATTERNS, read_only_patterns=REPO_READ_ONLY_PATTERNS)
    san = emit(sanitize_records(iter_records(path)), verify_patterns=REPO_VERIFY_PATTERNS, read_only_patterns=REPO_READ_ONLY_PATTERNS)

    disagreements: list[dict[str, Any]] = []
    if len(raw.log["actions"]) != len(san.log["actions"]):
        disagreements.append({"field": "action_count", "raw": len(raw.log["actions"]), "sanitized": len(san.log["actions"])})
    for a, b, ta, tb in zip(raw.log["actions"], san.log["actions"], raw.trace, san.trace):
        for f in INVARIANT_FIELDS:
            if a.get(f) != b.get(f):
                disagreements.append({"seq": a["seq"], "field": f, "raw": a.get(f), "sanitized": b.get(f)})
        if ta.rule != tb.rule:
            disagreements.append({"seq": a["seq"], "field": "rule", "raw": ta.rule, "sanitized": tb.rule})
        extra = set(tb.notes) ^ set(ta.notes)
        if extra - EXPECTED_TRACE_DIFF:
            disagreements.append({"seq": a["seq"], "field": "notes", "raw": ta.notes, "sanitized": tb.notes})

    schema_ok = True
    try:
        jsonschema.validate(instance=raw.log, schema=SCHEMA)
    except jsonschema.ValidationError as e:
        schema_ok = False
        disagreements.append({"field": "schema", "error": str(e)[:200]})

    profile = class_a_profile(raw.log)
    mutations = [a for a in raw.log["actions"] if a["action_class"] == "mutate"]
    out_of_repo = sum(1 for a in mutations for t in a["targets"] if t.startswith(("/", "~")))

    # audit samples come from the SANITIZED emission
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in san.trace:
        if t.tool != "Bash":
            continue
        buckets = [t.rule] + [n for n in t.notes if n.split(":")[0] in ("compound", "heredoc", "inline-code", "fallback", "mutate", "read", "shlex")]
        for b in buckets:
            samples[b].append({"transcript": transcript_id(path), "seq": t.seq, "class": t.action_class,
                               "rule": t.rule, "notes": t.notes, "preview": t.preview})

    return {
        "id": transcript_id(path),
        "records": raw.stats["records"],
        "tool_uses": raw.stats["tool_uses"],
        "actions": len(raw.log["actions"]),
        "orphan_tool_use": raw.stats["orphan_tool_use"],
        "orphan_result": raw.stats["orphan_result"],
        "non_monotonic_timestamps": raw.stats["non_monotonic_timestamps"],
        "unknown_tools": raw.stats["unknown_tools"],
        "by_class": raw.stats["by_class"],
        "by_rule": raw.stats["by_rule"],
        "by_status": raw.stats["by_status"],
        "notes": raw.stats["notes"],
        "schema_ok": schema_ok,
        "disagreements": disagreements,
        "class_a": profile,
        "out_of_repo_mutations": out_of_repo,
        "samples": samples,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    tot: dict[str, Counter] = {k: Counter() for k in ("by_class", "by_rule", "by_status", "notes", "unknown_tools")}
    for r in results:
        for k in tot:
            tot[k].update(r[k])
    fallback_programs: Counter = Counter()
    for r in results:
        for n, c in r["notes"].items():
            if n.startswith("fallback:"):
                for p in n.split(":", 1)[1].split(","):
                    if p:
                        fallback_programs[p] += c
    actions = sum(r["actions"] for r in results)
    bash = tot["by_rule"]
    n_bash = sum(v for k, v in bash.items() if k.startswith("bash:"))
    return {
        "transcripts": len(results),
        "actions": actions,
        "tool_uses": sum(r["tool_uses"] for r in results),
        "orphan_tool_use": sum(r["orphan_tool_use"] for r in results),
        "orphan_result": sum(r["orphan_result"] for r in results),
        "schema_ok": all(r["schema_ok"] for r in results),
        "invariant_disagreements": sum(len(r["disagreements"]) for r in results),
        "by_class": dict(tot["by_class"].most_common()),
        "by_rule": dict(tot["by_rule"].most_common()),
        "by_status": dict(tot["by_status"].most_common()),
        "notes": dict(tot["notes"].most_common()),
        "unknown_tools": dict(tot["unknown_tools"]),
        "bash_actions": n_bash,
        "bash_fallback_rate": (bash.get("bash:fallback", 0) / n_bash) if n_bash else None,
        "fallback_programs": dict(fallback_programs.most_common()),
        "compound_rate": (sum(v for k, v in tot["notes"].items() if k.startswith("compound:")) / n_bash) if n_bash else None,
        "heredoc_share": (tot["notes"].get("heredoc", 0) / n_bash) if n_bash else None,
        "out_of_repo_mutations": sum(r["out_of_repo_mutations"] for r in results),
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_outputs(results: list[dict[str, Any]], agg: dict[str, Any], out: Path, source_glob: str, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)

    # per-transcript CSV
    metric_keys = ["action_count", "invalid_tool_call_rate", "repeated_action_rate", "premature_edit_rate",
                   "reproduction_before_repair", "verification_attempted", "reversion_count", "files_inspected",
                   "files_changed", "destructive_action_rate", "scope_violation_rate"]
    with open(out / "data" / "per-transcript.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["transcript", "tool_uses", "actions", "orphans", "read", "mutate", "verify", "execute", "other",
                    "bash_fallback", "heredoc", "out_of_repo_mutations", "invariant_disagreements", *metric_keys])
        for r in results:
            bc, p = r["by_class"], r["class_a"]
            w.writerow([r["id"], r["tool_uses"], r["actions"], r["orphan_tool_use"], bc.get("read", 0), bc.get("mutate", 0),
                        bc.get("verify", 0), bc.get("execute", 0), bc.get("other", 0), r["by_rule"].get("bash:fallback", 0),
                        r["notes"].get("heredoc", 0), r["out_of_repo_mutations"], len(r["disagreements"]),
                        *[_fmt(p.get(k)) for k in metric_keys]])

    # audit samples: seeded, per bucket, across transcripts
    rng = random.Random(seed)
    pooled: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        for bucket, items in r["samples"].items():
            pooled[bucket].extend(items)
    audit = {bucket: rng.sample(items, min(AUDIT_SAMPLES_PER_BUCKET, len(items))) for bucket, items in sorted(pooled.items())}
    (out / "data" / "classification-audit-samples.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False))
    (out / "data" / "aggregate.json").write_text(json.dumps(agg, indent=1))
    (out / "data" / "invariant-disagreements.json").write_text(json.dumps(
        [{"transcript": r["id"], **d} for r in results for d in r["disagreements"]], indent=1))

    # report
    lines: list[str] = []
    lines.append("# Instrument validation — transcript emitter on real sessions\n")
    lines.append("> **INSTRUMENT VALIDATION, NOT MODEL EVIDENCE.** These are this project's own agent sessions, run through the")
    lines.append("> action-log instrument to establish that the instrumentation works on real trajectories. Nothing here measures")
    lines.append("> the quality of any system, enters the evidence ledger, or bears on any benchmark claim.\n")
    lines.append(f"- Date: {date.today().isoformat()}  ·  emitter `{EMITTER_VERSION}`  ·  sanitizer `{SANITIZER_VERSION}`  ·  audit seed {seed}")
    lines.append(f"- Source: `{source_glob}` (read by this script only; nothing under `benchmarks/` otherwise touches it)")
    lines.append(f"- Transcripts: {agg['transcripts']}  ·  tool calls: {agg['tool_uses']}  ·  emitted actions: {agg['actions']}  ·  unanswered calls: {agg['orphan_tool_use']}\n")

    lines.append("## 1. Does the instrument hold on real data?\n")
    lines.append("| Check | Result |\n|---|---|")
    lines.append(f"| every emitted log validates against `AGENT_ACTION_LOG.schema.json` | {_fmt(agg['schema_ok'])} |")
    lines.append(f"| raw vs sanitized agree on all classification fields (669 → {agg['actions']} actions) | {agg['invariant_disagreements']} disagreements |")
    lines.append(f"| tools the emitter does not know | {agg['unknown_tools'] or 'none'} |")
    lines.append(f"| tool results with no matching call | {agg['orphan_result']} |")
    lines.append("")

    lines.append("## 2. Classification distribution\n")
    lines.append("By class (all tools):\n")
    lines.append("| class | actions |\n|---|---|")
    for k, v in agg["by_class"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("\nBy rule:\n")
    lines.append("| rule | actions |\n|---|---|")
    for k, v in agg["by_rule"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append(f"Bash actions: {agg['bash_actions']}. Fallback rate {_fmt(agg['bash_fallback_rate'])}, compound rate {_fmt(agg['compound_rate'])}, heredoc share {_fmt(agg['heredoc_share'])}.\n")
    lines.append("Fallback residue by program (what `execute` is made of):\n")
    lines.append("| program | actions |\n|---|---|")
    for k, v in list(agg["fallback_programs"].items())[:25]:
        lines.append(f"| `{k}` | {v} |")
    lines.append("\nNotes (trace annotations):\n")
    lines.append("| note | count |\n|---|---|")
    for k, v in list(agg["notes"].items())[:40]:
        lines.append(f"| `{k}` | {v} |")
    lines.append("")

    lines.append("## 3. Class A profiles (vector per transcript, never a score)\n")
    lines.append("Full table: `data/per-transcript.csv`. Transcripts with ≥ 20 actions:\n")
    lines.append("| transcript | actions | read/mutate/verify/exec/other | premature_edit | repro_before_repair | verification_attempted | reversions | invalid_call | out-of-repo mutations |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: -x["actions"]):
        if r["actions"] < 20:
            continue
        bc, p = r["by_class"], r["class_a"]
        lines.append(f"| {r['id']} | {r['actions']} | {bc.get('read',0)}/{bc.get('mutate',0)}/{bc.get('verify',0)}/{bc.get('execute',0)}/{bc.get('other',0)} | "
                     f"{_fmt(p['premature_edit_rate'])} | {_fmt(p['reproduction_before_repair'])} | {_fmt(p['verification_attempted'])} | "
                     f"{p['reversion_count']} | {_fmt(p['invalid_tool_call_rate'])} | {r['out_of_repo_mutations']} |")
    small = [r for r in results if r["actions"] < 20]
    lines.append(f"\n{len(small)} transcripts have fewer than 20 actions; most are review sidechains with reads and a `StructuredOutput` only. "
                 "Their rates are defined but rest on tiny denominators and are omitted from the table above.\n")
    lines.append("`scope_violation_rate` is undefined throughout: the emitter does not set `declared_scope` (a contract-review item). "
                 "`destructive_action_rate` is undefined: no destructive-tool set was declared for this run. "
                 "Out-of-repo mutations are counted directly instead.\n")

    lines.append("## 4. Manual audit\n")
    lines.append(f"Samples for a person to read: `data/classification-audit-samples.json` — {AUDIT_SAMPLES_PER_BUCKET} seeded random Bash actions per rule and per note, "
                 "from the sanitized emission (home-normalised, secret-redacted commands). Findings are recorded below by the auditor, not generated.\n")
    lines.append("Findings, per-bucket verdicts, the fixes they produced, and the contract decision: "
                 "`audit-and-contract-decision.md` (auditor-written; not regenerated by this script).\n")

    (out / "instrument-validation-report.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", required=True, help="glob of raw transcripts (the only place this path is read)")
    parser.add_argument("--out", required=True, help="output directory (under the artifacts dir)")
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(os.path.expanduser(args.source)))
    if not paths:
        print(f"no transcripts match {args.source}", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)
    results = [run_one(p, rng) for p in paths]
    agg = aggregate(results)
    write_outputs(results, agg, Path(args.out), args.source, args.seed)
    print(json.dumps({k: v for k, v in agg.items() if k not in ("notes", "by_rule")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
