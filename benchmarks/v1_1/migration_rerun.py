"""The single authorized v1.0 / v1.1 side-by-side rerun over real sessions (D17).

MIGRATION / INSTRUMENT EVIDENCE — NOT MODEL EVIDENCE. This script exists to show
that every v1.0 → v1.1 delta on real trajectories is explained by the migration
categories, and to establish corpus-wide what the fixtures could only sample
(D18: the write-before-test ordering). It measures the instrument. Nothing it
produces is a benchmark result, enters the evidence ledger, or bears on the
agent that produced the transcripts.

It is the second component under `benchmarks/` permitted to read
`~/.claude/projects/`; the first is the frozen v1.0 validation script, whose
harness conventions (verify / read-only patterns) it imports unchanged so the
two contracts are compared under identical declarations.

Harness declarations (D19) — recorded verbatim in every emitted log
--------------------------------------------------------------------
declared_scope: **None**. No permission policy governed these historical
    sessions, so "permitted to mutate" is undeclared and scope_violation_rate
    stays undefined, exactly as under v1.0. Declaring "." retroactively would
    invent a policy the harness never had.
source_scope: the repository's code directories, listed in SOURCE_SCOPE below,
    derived from the owner's rule (artifacts/, reports, logs, checkpoints,
    caches and generated outputs are not source). `trajectory/` is included
    because it holds test scripts alongside evidence documents; that is flagged
    for the owner in the report rather than decided silently.

Known limitation recorded, not fixed: targets are relative to the recorded
session cwd. A command that `cd`s into a clone (`cd /tmp/lk && sed -i …
nanoscribe/x.py`) yields an in-repo-looking target; shell `cd` is not tracked
by either contract. Out of v1.1's scope.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.transcript_sanitizer import iter_records  # noqa: E402
from benchmarks.v1_1.emitter import EMITTER_VERSION  # noqa: E402
from benchmarks.v1_1.migration import Comparison, compare, render_markdown  # noqa: E402
from benchmarks.validation.instrument_validation_transcripts import (  # noqa: E402  (frozen v1.0 declarations)
    REPO_READ_ONLY_PATTERNS,
    REPO_VERIFY_PATTERNS,
    transcript_id,
)

DECLARED_SCOPE = None
SOURCE_SCOPE = ["benchmarks/", "nanoscribe/", "fabric/", "scripts/", "scribe/", "sft/", "pretrain/",
                "stage_m/", "scale/", "wedge_v1/", "trajectory/", ".github/"]

KEYS = ("action_count", "call_count", "invalid_tool_call_rate", "repeated_action_rate", "premature_edit_rate",
        "reproduction_before_repair", "verification_attempted", "verification_attempted_in_scope", "reversion_count",
        "files_inspected", "files_changed", "files_changed_in_scope", "mutations_without_target", "noop_mutations")


def write_outputs(comps: list[Comparison], out: Path, source_glob: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    unexplained = sum(len(c.unexplained) for c in comps)
    changed_keys: Counter = Counter()
    categories: Counter = Counter()
    flips = {"verification_attempted": Counter(), "reproduction_before_repair": Counter()}
    scoped_vs_unscoped: Counter = Counter()
    eff: Counter = Counter()
    for c in comps:
        for d in c.changed:
            if d.category == "new-key":
                continue
            changed_keys[d.key] += 1
            categories[d.category] += 1
        for k in flips:
            if c.p10[k] != c.p11[k]:
                flips[k][f"{c.p10[k]}→{c.p11[k]}"] += 1
        scoped_vs_unscoped[f"unscoped={c.p11['verification_attempted']} scoped={c.p11['verification_attempted_in_scope']}"] += 1
        for n, k in c.p11["effects_per_call"].items():
            eff[n] += k

    with open(out / "data" / "per-transcript-delta.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["transcript", "unexplained", *[f"v10:{k}" for k in KEYS], *[f"v11:{k}" for k in KEYS]])
        for c in comps:
            w.writerow([c.name, len(c.unexplained), *[c.p10.get(k) for k in KEYS], *[c.p11.get(k) for k in KEYS]])

    agg = {
        "date": date.today().isoformat(), "emitter_v11": EMITTER_VERSION, "source": source_glob,
        "transcripts": len(comps), "gate": "PASS" if unexplained == 0 else "FAIL", "unexplained": unexplained,
        "declared_scope": DECLARED_SCOPE, "source_scope": SOURCE_SCOPE,
        "calls": sum(c.p11["call_count"] for c in comps), "sub_actions": sum(c.p11["action_count"] for c in comps),
        "v10_actions": sum(c.p10["action_count"] for c in comps),
        "effects_per_call": dict(sorted(eff.items())),
        "changed_keys": dict(changed_keys.most_common()), "categories": dict(categories.most_common()),
        "verification_attempted_flips": dict(flips["verification_attempted"]),
        "reproduction_before_repair_flips": dict(flips["reproduction_before_repair"]),
        "scoped_vs_unscoped_verification": dict(scoped_vs_unscoped.most_common()),
        "new_mutate_targets_total": sum(len(c.new_mutate_targets) for c in comps),
        "new_mutate_target_origins": dict(sum((Counter(c.new_mutate_targets.values()) for c in comps), Counter())),
        "invariants_failed": {k: n for k, n in Counter(k for c in comps for k, (ok, _) in c.invariants.items() if not ok).items()},
    }
    (out / "data" / "aggregate.json").write_text(json.dumps(agg, indent=1))

    # full per-transcript sections only where something changed; the rest is one line each
    changed = [c for c in comps if c.changed and any(d.category != "new-key" for d in c.changed)]
    md = render_markdown(changed, "v1.1 migration rerun — side-by-side on 62 real sessions (changed transcripts)")
    head = ["# v1.1 migration rerun — summary\n",
            "> **MIGRATION / INSTRUMENT EVIDENCE, NOT MODEL EVIDENCE.** The one authorized side-by-side rerun (D17). It shows",
            "> whether every v1.0 → v1.1 delta on real trajectories is explained by the migration categories. It measures the",
            "> instrument, not the agent, and promotes nothing.\n",
            f"- Date {agg['date']} · source `{source_glob}` · transcripts {agg['transcripts']} · v1.0 actions {agg['v10_actions']} = calls {agg['calls']} → sub-actions {agg['sub_actions']}",
            f"- Harness declarations (D19): declared_scope = `{DECLARED_SCOPE}` (no permission policy existed; scope_violation_rate undefined as in v1.0); source_scope = `{SOURCE_SCOPE}`",
            f"\n**Gate: {agg['gate']}** — {unexplained} unexplained delta(s); invariants failed: {agg['invariants_failed'] or 'none'}\n",
            "| aggregate | value |\n|---|---|",
            f"| effects per call | {agg['effects_per_call']} |",
            f"| changed keys (transcripts affected) | {agg['changed_keys']} |",
            f"| explanation categories | {agg['categories']} |",
            f"| new mutate targets (hidden in v1.0) | {agg['new_mutate_targets_total']}, origins {agg['new_mutate_target_origins']} |",
            f"| verification_attempted flips v1.0→v1.1 | {agg['verification_attempted_flips'] or 'none'} |",
            f"| reproduction_before_repair flips | {agg['reproduction_before_repair_flips'] or 'none'} |",
            f"| unscoped vs scoped verification (v1.1) | {agg['scoped_vs_unscoped_verification']} |",
            "\nPer-transcript values: `data/per-transcript-delta.csv`; aggregate: `data/aggregate.json`.",
            f"\n{len(changed)} transcript(s) with at least one changed v1.0 key follow; the other {len(comps) - len(changed)} are identical on every v1.0 key.\n\n---\n"]
    (out / "migration-rerun-report.md").write_text("\n".join(head) + "\n" + md)
    return agg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    paths = sorted(glob.glob(os.path.expanduser(args.source)))
    if not paths:
        print(f"no transcripts match {args.source}", file=sys.stderr)
        return 1
    comps = [compare(transcript_id(p), iter_records(p), verify_patterns=REPO_VERIFY_PATTERNS, read_only_patterns=REPO_READ_ONLY_PATTERNS,
                     declared_scope=DECLARED_SCOPE, source_scope=SOURCE_SCOPE) for p in paths]
    agg = write_outputs(comps, Path(args.out), args.source)
    print(json.dumps(agg, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
