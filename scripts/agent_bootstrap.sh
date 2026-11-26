#!/usr/bin/env bash
# Session bootstrap for autonomous agents across all worktrees.
# Usage: bash scripts/agent_bootstrap.sh <agent-name> "<task>" [paths...]
set -eu

agent="${1:?agent name}"
task="${2:-}"
shift 2 || true

root=$(git rev-parse --show-toplevel)
cd "$root"

python3 scripts/install_coordination_hooks.py >/dev/null
git fetch -q origin master || true
python3 scripts/agent_coord.py gc || true

if git rev-parse --verify --quiet origin/master >/dev/null 2>&1; then
  behind=$(git rev-list --count HEAD..origin/master)
  echo "[bootstrap] $(git branch --show-current) is $behind behind origin/master"
  if [ "$behind" -gt 0 ] && [ -z "$(git status --porcelain)" ]; then
    if git rebase origin/master >/dev/null 2>&1; then
      echo "[bootstrap] rebased clean onto origin/master"
    else
      git rebase --abort 2>/dev/null || true
      echo "[bootstrap] rebase conflicts with origin/master; staying put, resolve manually" >&2
    fi
  elif [ "$behind" -gt 0 ]; then
    echo "[bootstrap] dirty tree; skipped auto-rebase (commit or stash first)" >&2
  fi
fi

AGENT_NAME="$agent" python3 scripts/agent_coord.py claim --agent "$agent" --task "$task" --paths "$@"
python3 scripts/agent_coord.py announce --agent "$agent" --status ACTIVE --scope "$task" >/dev/null
python3 scripts/agent_coord.py board
