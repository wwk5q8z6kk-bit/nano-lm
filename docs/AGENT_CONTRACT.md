# Agent Contract (multi-agent / multi-worktree)

Goal: every agent works autonomously across any worktree without colliding, overwriting peer edits, or drifting off `origin/master`. Rules are mechanical, enforced by `.githooks/`, `scripts/agent_coord.py`, and CI.

## Session start (every agent, every worktree)

```bash
bash scripts/agent_bootstrap.sh <agent-name> "<task>" [paths-you-will-edit...]
```

This installs shared hooks into `<git-common-dir>/hooks`, fetches `origin/master`, reaps expired leases, rebases a clean branch onto `origin/master` (stopping cleanly if conflicts exist), claims a path lease, announces the session, and prints the coordination board. Exit `2` means another live agent holds those paths: pick disjoint work, do not force.

## Pre-authorized (no owner prompt)

- Allocate a new worktree with the sanctioned allocator (`~/.claude/bin/worktree-new` — use `--long-lived` on `/Volumes/Express1M2/Worktrees/nano-lm` whenever internal `/` free space is `< 35 GB`), create feature branches, commit, push own branch, and open PRs.
- Keep build artifacts out of `/tmp` and internal disk by using `~/.claude/bin/build-scratch nano-lm`.
- Rebase or merge `origin/master` into own branch when the working tree is clean.
- Run tests, benchmarks, Kaggle/free-tier jobs, and routine RunPod within the active experiment budget.
- Claim/heartbeat/release own leases; run `agent_coord.py gc` to reap expired/dead-PID leases.
- Verify PR readiness with `~/.claude/bin/pr-ready <PR>` and retire *only* own proven-complete, clean, merged worktree via `~/.claude/bin/worktree-retire --check` then `--apply`.

## Needs owner / experiment-scoped authorization

- Spend beyond the active experiment budget (\$180 envelope, \$200 cap, \$20 RunPod owner reserve floor).
- Freeze tags (`paper-alpha*`, `*freeze*`), evidence-protected paths (`AGENT_ALLOW_EVIDENCE=1` only inside an authorized run), and PHI/private data.
- Unlocking or removing dirty, unmerged, or peer-locked worktrees, or force-pushing.

## Shared state & compute mutex rules

1. **Never mutate sibling worktrees.** Peer worktrees are strictly read-only. Cross-agent state goes through git PRs or [`scripts/agent_coord.py`](../scripts/agent_coord.py).
2. **Never edit monolithic shared ledgers.** Root `NANOLM_CROSS_WORKTREE_LEDGER.json` and `NANOLM_EXPERIMENT_REGISTRY.json` are blocked by `pre-commit` and CI. Record one file per entry instead:
   - Experiments: `python3 scripts/agent_coord.py record-experiment --id <ID> --status <STATUS> --verdict <VERDICT> --paid-usd 0` (writes `coordination/experiments/<ID>.json`)
   - Findings/status: `python3 scripts/agent_coord.py record-finding --id <ID> --focus "<focus>" --finding "<text>"` (writes `coordination/findings/<ID>.json`)
   - Live session status: `python3 scripts/agent_coord.py announce --agent <NAME> --status ACTIVE --scope "<scope>"`
   - Combined view: `python3 scripts/agent_coord.py board`
3. **Serialize shared hardware & enforce RunPod reserve.** Wrap MPS, CPU benchmarks, or pod launches in the kernel `flock` mutex:
   ```bash
   python3 scripts/agent_coord.py lock-run --resource mps --agent <NAME> --reason "<exp>" -- <cmd...>
   ```
   `--resource runpod` automatically refuses when `<git-common-dir>/agent-coord/budget.json` is at or below the \$20.00 owner reserve floor (override `AGENT_ALLOW_PAID_RUNPOD=1` only with owner authorization).
4. **Stay close to `origin/master`.** `pre-push` blocks direct pushes to `master`/`main`, blocks freeze tags, blocks branches `> 100` commits behind `origin/master`, and blocks branches that fail `git merge-tree --write-tree origin/master HEAD`.
5. **Audit safely.** Use [`scripts/worktree_audit.py`](../scripts/worktree_audit.py) (`GIT_OPTIONAL_LOCKS=0`, read-only) to inspect drift, stale locks, and merge-tree conflicts across all worktrees.
