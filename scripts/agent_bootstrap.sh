#!/usr/bin/env bash
# Usage: scripts/agent_bootstrap.sh <agent-name> "<task>" [paths...]
set -eu
agent="${1:?agent name}"; task="${2:-}"; shift 2 || true
root=$(git rev-parse --show-toplevel); cd "$root"
git config core.hooksPath .githooks
git fetch -q origin master
python3 scripts/agent_lease.py gc || true
behind=$(git rev-list --count HEAD..origin/master)
echo "[bootstrap] $(git branch --show-current) is $behind behind origin/master"
if [ "$behind" -gt 0 ] && [ -z "$(git status --porcelain)" ]; then
  git rebase origin/master >/dev/null 2>&1 && echo "[bootstrap] rebased clean" \
    || { git rebase --abort; echo "[bootstrap] rebase conflicts; staying put, resolve manually" >&2; }
elif [ "$behind" -gt 0 ]; then
  echo "[bootstrap] dirty tree; skipped rebase (commit/stash first)" >&2
fi
AGENT_NAME="$agent" python3 scripts/agent_lease.py claim --agent "$agent" --task "$task" --paths "$@"
python3 scripts/agent_lease.py status
