#!/usr/bin/env python3
"""Report where every path cited by the canonical docs actually resolves.

SPEC §0 keeps a table of this, and it goes stale — it read "21 paths, 4 local"
after the branch it described had been merged elsewhere. Regenerate the table
from this rather than maintaining it by eye.

Finds paths cited in backticks, then reports, per path: resolves on this branch,
resolves on some other local branch (named), or resolves nowhere. A path
resolving nowhere is the interesting case — it caught the E3 protocol cited as
`research/preregistrations/PREREG_E3_dual_clinician.md` when the file is
`trajectory/PREREG_E3_dual_clinician_arm.md`.

**REPORT ONLY — deliberately not a gate.** Its false-positive modes are not
characterized: illustrative paths, glob-ish citations and paths belonging to
branches that are not fetched all read as "nowhere" without being defects.
Promoting it to a gate needs that calibration first (SPEC §22).

    python3 scripts/audit_cited_paths.py            # human table
    python3 scripts/audit_cited_paths.py --json     # machine-readable
    python3 scripts/audit_cited_paths.py --unresolved-only
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("docs/NANO_VNEXT_MASTER_SPEC.md", "docs/RESEARCH_STATUS.md")

# A backticked token with a directory separator and a known source extension.
CITED = re.compile(r"`([A-Za-z0-9_./-]+/[A-Za-z0-9_./-]+\.(?:md|py|json))`")
# `de188a2:artifacts/...` — already pinned to a commit; resolve against that.
PINNED = re.compile(r"`([0-9a-f]{7,40}):([A-Za-z0-9_./-]+)`")


def _branches() -> list[str]:
    out = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout.split()
    # Prefer local branches; fall back to origin/* for ones already pruned.
    remote = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout.split()
    return out + [r for r in remote if r.split("/", 1)[-1] not in out]


def _exists_at(ref: str, path: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{path}"],
        capture_output=True, cwd=ROOT,
    ).returncode == 0


def collect() -> tuple[set[str], dict[str, str]]:
    cited: set[str] = set()
    pinned: dict[str, str] = {}
    for doc in DOCS:
        text = (ROOT / doc).read_text(encoding="utf-8")
        for sha, path in PINNED.findall(text):
            pinned[path] = sha
        cited.update(CITED.findall(text))
    return cited, pinned


def audit() -> list[dict]:
    cited, pinned = collect()
    branches = _branches()
    rows = []
    for path in sorted(cited):
        if (ROOT / path).exists():
            rows.append({"path": path, "where": "this branch", "kind": "local"})
            continue
        if (sha := pinned.get(path)) and _exists_at(sha, path):
            rows.append({"path": path, "where": sha, "kind": "pinned"})
            continue
        found = next((b for b in branches if _exists_at(b, path)), None)
        rows.append(
            {"path": path, "where": found or "NOWHERE",
             "kind": "cross-branch" if found else "unresolved"}
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--unresolved-only", action="store_true")
    args = ap.parse_args()

    rows = audit()
    if args.unresolved_only:
        rows = [r for r in rows if r["kind"] == "unresolved"]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("no unresolved cited paths")
        return 0
    width = max(len(r["path"]) for r in rows)
    for r in rows:
        print(f"{r['path']:<{width}}  {r['kind']:<12} {r['where']}")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
