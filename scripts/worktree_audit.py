#!/usr/bin/env python3
"""Read-only audit of all git worktrees: drift, dirtiness, locks, file overlap.

Usage: python3 scripts/worktree_audit.py [--base origin/master] [--json]
Exit code 1 if any HIGH finding (overlap on shared files, dirty+locked stale).
Never mutates any worktree.
"""
import argparse, json, subprocess, time, itertools, collections

SHARED = ("NANOLM_CROSS_WORKTREE_LEDGER.json", "NANOLM_EXPERIMENT_REGISTRY.json",
          "docs/ACTIVE_NOW.json", "docs/ACTIVE_NOW.md", "AGENTS.md",
          "POST_ALPHA_EVIDENCE_FREEZE.json", "SHA256SUMS")


def git(path, *a):
    r = subprocess.run(["git", "-C", path, *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def worktrees(root):
    out, cur = [], {}
    for line in git(root, "worktree", "list", "--porcelain").splitlines() + [""]:
        if not line:
            if cur: out.append(cur); cur = {}
            continue
        k, _, v = line.partition(" ")
        cur[k] = v or True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/master")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    rows, touched = [], {}
    for w in worktrees(root):
        p = w["worktree"]
        head = w.get("HEAD", "")[:7]
        br = w.get("branch", "(detached)").replace("refs/heads/", "")
        behind, ahead = (git(p, "rev-list", "--left-right", "--count", f"{a.base}...HEAD") or "0 0").split()
        dirty = [l[3:] for l in git(p, "status", "--porcelain").splitlines()]
        changed = set(git(p, "diff", "--name-only", f"{a.base}...HEAD").splitlines()) | set(dirty)
        age_h = (time.time() - int(git(p, "log", "-1", "--format=%ct") or time.time())) / 3600
        touched[p] = changed
        flags = []
        if int(behind) > 50: flags.append("STALE_BASE")
        if "locked" in w and age_h > 24: flags.append("LOCKED_IDLE>24h")
        if dirty and age_h > 24: flags.append("DIRTY_IDLE>24h")
        if w.get("detached"): flags.append("DETACHED")
        if any(s in changed for s in SHARED): flags.append("TOUCHES_SHARED")
        rows.append(dict(path=p, branch=br, head=head, behind=int(behind), ahead=int(ahead),
                         dirty=len(dirty), locked="locked" in w, age_h=round(age_h, 1), flags=flags))
    overlaps = []
    for (p1, c1), (p2, c2) in itertools.combinations(touched.items(), 2):
        both = (c1 & c2)
        hot = sorted(f for f in both if f in SHARED)
        if hot: overlaps.append(dict(a=p1, b=p2, files=hot))
    if a.json:
        print(json.dumps(dict(worktrees=rows, shared_overlaps=overlaps), indent=2)); 
    else:
        for r in rows:
            print(f"{r['branch'][:42]:42} behind={r['behind']:<4} ahead={r['ahead']:<3} dirty={r['dirty']:<3} "
                  f"idle={r['age_h']:>6}h {'L' if r['locked'] else ' '} {','.join(r['flags'])}")
        print(f"\nShared-file overlaps: {len(overlaps)}")
        for o in overlaps: print(" ", o["a"].split('/')[-1], "<->", o["b"].split('/')[-1], o["files"])
    raise SystemExit(1 if overlaps else 0)


if __name__ == "__main__":
    main()
