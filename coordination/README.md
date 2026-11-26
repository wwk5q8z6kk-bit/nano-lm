# coordination/

Append-only, one-file-per-entry JSON shards that eliminate cross-worktree merge conflicts:

- `experiments/<ID>.json` — created via `python3 scripts/agent_coord.py record-experiment ...`
- `findings/<ID>.json` — created via `python3 scripts/agent_coord.py record-finding ...`

Validate and compile with `python3 scripts/agent_coord.py compile-registry --check`. See [`docs/AGENT_CONTRACT.md`](../docs/AGENT_CONTRACT.md).
