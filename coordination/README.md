# coordination/

`entries/` holds append-only, one-file-per-agent JSON records (`<agent>-<UTC>.json`) that replace in-place edits to `NANOLM_CROSS_WORKTREE_LEDGER.json` / `NANOLM_EXPERIMENT_REGISTRY.json`. Because every entry is a new file, concurrent branches never conflict. See `docs/AGENT_CONTRACT.md`.
