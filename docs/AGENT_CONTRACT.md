# Agent Contract (multi-agent / multi-worktree)

Goal: every agent can work autonomously in any worktree without colliding. Rules are mechanical, enforced by `.githooks/` and CI, not by memory.

## Session start (every agent, every worktree)
```bash
bash scripts/agent_bootstrap.sh <agent-name> "<task>" [paths-you-will-edit...]
```
This sets `core.hooksPath=.githooks`, fetches, reports drift, rebases a clean branch onto `origin/master`, runs `agent_lease.py gc`, and claims a lease. Exit 2 = another live agent holds your paths: pick different work, do not force.

## Pre-authorized (no owner prompt)
- Create branches/worktrees, commit, push own branch, open PRs.
- Rebase/merge `origin/master` into own branch (only when working tree is clean).
- Run tests, benchmarks, Kaggle/free-tier jobs, routine RunPod within the active experiment budget.
- Take/heartbeat/release leases; reap *stale leases* (TTL 6h).
- Resolve conflicts on code in own branch; on conflict in shared files use the per-agent files below, never overwrite a peer's entries.

## Needs owner / experiment-scoped authorization
- Spend beyond the active experiment budget (\$180 envelope, \$200 cap, \$20 reserve).
- Freeze tags, evidence-protected paths (`AGENT_ALLOW_EVIDENCE=1` only in an authorized run), PHI/private data.
- Deleting another agent's worktree, unlocking a locked worktree, discarding its dirty files, force-push.

## Shared state rules
1. Never mutate sibling worktrees (read-only). Cross-agent info goes through git (PRs/master) or leases.
2. One writer per shared file. New registry/ledger entries go in `coordination/entries/<agent>-<utc>.json` (append-only, one file each); `scripts/worktree_audit.py` flags any branch that edits a shared single file.
3. `origin/master` is truth. Never push from stale local `master`; pre-push rejects branches >100 commits behind.
4. Heartbeat leases during long runs (`agent_lease.py heartbeat`); release at end. Stale leases are reaped, so a crashed agent never blocks others.
5. Compute: acquire a lease on path `compute/runpod` (or `compute/gpu0`) before launching pods; release on teardown. Conflicting claim = wait or use free backend.

## Cleanup policy
`python3 scripts/worktree_audit.py` is read-only. Candidates for reaping = locked/idle >24h or dirty/idle >24h; an agent may *report* them, only the owner removes dirty ones. Clean, merged, idle >72h worktrees may be removed by any agent via `git worktree remove` (never `--force`).
