# Durable raw C3 JSONL — retrieval

## Location

Tracked git paths under `artifacts/durable_raw/c3/` (gitignore exception).

## Verify

```bash
cd /path/to/clone
shasum -a 256 -c artifacts/durable_raw/SHA256SUMS   # macOS
# sha256sum -c artifacts/durable_raw/SHA256SUMS     # Linux
```

## Semantics

| Role | Clean-clone source of truth |
|------|-----------------------------|
| C3 primary seeds 0–2 | `artifacts/durable_raw/c3/outputs_c3_seed*.jsonl` **or** `trajectory/outputs_c3_seed*.jsonl` |
| C3 replication seeds 0–2 | **`artifacts/durable_raw/c3/replication_*.jsonl` only** (trajectory replication path is gitignored) |

## Limitation

Checkpoint/tokenizer byte hashes may be absent from the C3 result JSON top-level; see `MANIFEST.json` → `checkpoint_tokenizer_note`.
