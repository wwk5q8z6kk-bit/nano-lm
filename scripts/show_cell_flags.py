#!/usr/bin/env python3
"""Read each ablation cell's leakage flags OUT OF GIT, never by importing.

Why this exists rather than being a thing to remember: importing
``nanoscribe.leakage`` to check a branch's flags gave two false readbacks in one
session and cost a rebuild cycle. CPython caches bytecode keyed on the source
file's mtime at one-second granularity, so a rapid checkout-write-import cycle
can load the PREVIOUS cell's compiled module while the source on disk is already
correct. The committed content was right both times; only the readback lied.

Git is the source of truth for what a run will actually execute — the runner
extracts the recorded commit into an isolated directory, so what matters is the
blob, not the working tree and certainly not a .pyc.

Usage:
  python3 scripts/show_cell_flags.py            # all orx/ branches
  python3 scripts/show_cell_flags.py --json
  python3 scripts/show_cell_flags.py --expect L000=orx/l000-...  # verify mapping
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

FLAGS = (
    ("C1", "PROMPT_ANSWER_TEMPLATE_GOLD_VALUE"),
    ("C2", "PARSER_RAW_VALUE_FALLBACK"),
    ("C3", "PROMPT_QUESTION_USES_GOLD_SURFACE"),
    ("Q_ID", "PROMPT_QUESTION_NAMES_CONCEPT"),
)
SOURCE = "nanoscribe/leakage.py"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def branches() -> list[str]:
    out = _git("branch", "--format=%(refname:short)")
    return [b for b in out.split() if b.startswith("orx/")]


def flags_for(branch: str) -> dict[str, bool] | None:
    try:
        blob = _git("show", f"{branch}:{SOURCE}")
    except subprocess.CalledProcessError:
        return None
    found: dict[str, bool] = {}
    for short, name in FLAGS:
        match = re.search(rf"^{name} = (True|False)$", blob, re.MULTILINE)
        if match is None:
            return None
        found[short] = match.group(1) == "True"
    return found


def cell_name(flags: dict[str, bool]) -> str:
    return "L" + "".join("1" if flags[k] else "0" for k in ("C1", "C2", "C3"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="CELL=branch",
        help="assert a cell maps to a branch; repeatable. Non-zero on mismatch.",
    )
    args = parser.parse_args()

    table = {}
    for branch in sorted(branches()):
        flags = flags_for(branch)
        if flags is not None:
            table[branch] = {**flags, "cell": cell_name(flags)}

    if args.json:
        print(json.dumps(table, indent=2, sort_keys=True))
    else:
        print(f"{'branch':<54} {'C1':<6} {'C2':<6} {'C3':<6} {'Q_ID':<6} cell")
        for branch, row in table.items():
            print(
                f"{branch:<54} {str(row['C1']):<6} {str(row['C2']):<6} "
                f"{str(row['C3']):<6} {str(row['Q_ID']):<6} {row['cell']}"
            )

    failures = []
    for spec in args.expect:
        cell, _, branch = spec.partition("=")
        row = table.get(branch) or table.get(f"orx/{branch}")
        if row is None:
            failures.append(f"{branch}: no such branch, or it lacks {SOURCE}")
        elif row["cell"] != cell:
            failures.append(f"{branch}: expected {cell}, git says {row['cell']}")
    for line in failures:
        print(f"MISMATCH {line}", file=sys.stderr)
    if failures:
        return 1

    q_id = {b: r["Q_ID"] for b, r in table.items() if not r["Q_ID"]}
    if q_id:
        print(f"WARNING Q_ID is pinned ON; off in: {sorted(q_id)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
