# artifacts/ — evidence freeze packaging

*Refreshed 2026-07-31T14:04:33.413228+00:00; HEAD `0e01d73205e9`.*

Schema version 1.0 manifests per closed experiment. See:

- `MANIFEST.json` — top-level index
- `ARTIFACT_INVENTORY.json` — full inventory with statuses
- `e1/MANIFEST.json`, `e3/MANIFEST.json` — post-α primary evidence
- `*/MANIFEST.json` — other closed experiments
- `local_raw_archive/` — SHA-verified copies of raw JSONL (C-1b/C-3; gitignored sources)
- `SHA256SUMS` — checksums for preserved freeze artifacts
- root `SHA256SUMS` — same checksums for freeze verification
- `POST_ALPHA_EVIDENCE_FREEZE.json` — freeze machine record

**Storage classes:** TRACKED · LOCAL_UNTRACKED · LOCAL_IGNORED · DURABLY_ARCHIVED · REFERENCED_BUT_MISSING · PARTIAL · NOT_REQUIRED

Do not invent hashes. Do not call artifacts immutable until committed, tagged, or content-addressed durably. Do not upload without owner authorization.
