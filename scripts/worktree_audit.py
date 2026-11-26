#!/usr/bin/env python3
"""Read-only audit of all git worktrees: drift, dirtiness, locks, conflicts, and shared-file overlaps.

Usage: python3 scripts/worktree_audit.py [--base origin/master] [--json] [--fail-on-overlap]
Never mutates any worktree or writes optional git locks.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time

SHARED = frozenset(
    {
        "NANOLM_CROSS_WORKTREE_LEDGER.json",
        "NANOLM_EXPERIMENT_REGISTRY.json",
        "docs/ACTIVE_NOW.json",
        "docs/ACTIVE_NOW.md",
        "AGENTS.md",
        "POST_ALPHA_EVIDENCE_FREEZE.json",
        "POST_ALPHA_EVIDENCE_FREEZE.md",
        "SHA256SUMS",
        "EVIDENCE_CURRENT.md",
    }
)


def git(path: str, *args: str) -> tuple[int, str, str]:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    r = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", path, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def worktrees(root: str) -> list[dict]:
    rc, out, _ = git(root, "worktree", "list", "--porcelain", "-z")
    if rc != 0:
        return []
    entries = []
    for record in out.split("\0\0"):
        if not record:
            continue
        cur: dict = {}
        for line in record.split("\0"):
            if not line:
                continue
            k, sep, v = line.partition(" ")
            cur[k] = v if sep else True
        if "worktree" in cur:
            entries.append(cur)
    return entries


def merge_conflicts_with_base(root: str, base: str, head_sha: str) -> list[str]:
    if not head_sha:
        return []
    rc, out, _ = git(root, "merge-tree", "--write-tree", "--name-only", base, head_sha)
    if rc == 0:
        return []
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    # Lines after the tree OID and blank line before messages contain CONFLICT (...)
    conflicts = [ln for ln in lines if ln.startswith("CONFLICT ")]
    return conflicts or ["CONFLICTS_WITH_BASE"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/master")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fail-on-overlap", action="store_true")
    a = ap.parse_args()

    rc, root, _ = git(".", "rev-parse", "--show-toplevel")
    if rc != 0 or not root:
        root = os.getcwd()

    rows, touched = [], {}
    for w in worktrees(root):
        p = w["worktree"]
        head_full = str(w.get("HEAD", ""))
        head = head_full[:7]
        br = str(w.get("branch", "(detached)")).replace("refs/heads/", "")
        rc_lr, lr_out, _ = git(p, "rev-list", "--left-right", "--count", f"{a.base}...HEAD")
        behind, ahead = (lr_out.split() if rc_lr == 0 and len(lr_out.split()) == 2 else ["0", "0"])
        _, st_out, _ = git(p, "status", "--porcelain")
        dirty = [ln[3:] for ln in st_out.splitlines() if ln]
        _, diff_out, _ = git(p, "diff", "--name-only", f"{a.base}...HEAD")
        changed = set(diff_out.splitlines()) | set(dirty)
        _, ts_out, _ = git(p, "log", "-1", "--format=%ct")
        age_h = (time.time() - int(ts_out or time.time())) / 3600.0
        touched[p] = changed

        locked_val = w.get("locked")
        locked = bool(locked_val)
        lock_reason = locked_val if isinstance(locked_val, str) else ""
        enrolled = "owner=" in lock_reason

        base_conflicts = merge_conflicts_with_base(root, a.base, head_full)

        flags = []
        if int(behind) > 50:
            flags.append("STALE_BASE")
        if locked and age_h > 24:
            flags.append("LOCKED_IDLE>24h" + ("(enrolled)" if enrolled else "(manual)"))
        if dirty and age_h > 24:
            flags.append("DIRTY_IDLE>24h")
        if w.get("detached"):
            flags.append("DETACHED")
        if any(s in changed for s in SHARED):
            flags.append("TOUCHES_SHARED")
        if base_conflicts:
            flags.append("CONFLICTS_WITH_BASE")

        rows.append(
            {
                "path": p,
                "branch": br,
                "head": head,
                "behind": int(behind),
                "ahead": int(ahead),
                "dirty": len(dirty),
                "locked": locked,
                "enrolled_lock": enrolled,
                "lock_reason": lock_reason,
                "age_h": round(age_h, 1),
                "base_conflicts": base_conflicts,
                "flags": flags,
            }
        )

    overlaps = []
    for (p1, c1), (p2, c2) in itertools.combinations(touched.items(), 2):
        both = c1 & c2
        hot = sorted(f for f in both if f in SHARED)
        if hot:
            overlaps.append({"a": p1, "b": p2, "files": hot})

    if a.json:
        print(json.dumps({"worktrees": rows, "shared_overlaps": overlaps}, indent=2))
    else:
        for r in rows:
            print(
                f"{r['branch'][:42]:42} behind={r['behind']:<4} ahead={r['ahead']:<3} "
                f"dirty={r['dirty']:<3} idle={r['age_h']:>6}h {'L' if r['locked'] else ' '} "
                f"{','.join(r['flags'])}"
            )
        print(f"\nShared-file overlaps: {len(overlaps)}")
        for o in overlaps:
            print(" ", os.path.basename(o["a"]), "<->", os.path.basename(o["b"]), o["files"])

    if a.fail_on_overlap and overlaps:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
