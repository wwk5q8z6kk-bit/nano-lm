# Evidence artifacts

This directory contains compact manifests and durable copies of evidence that
cannot be reconstructed from a clean checkout alone.

- `MANIFEST.json` indexes the retained experiment manifests.
- `*/MANIFEST.json` records files, roles, hashes, and retrieval instructions for
  each closed experiment.
- `local_raw_archive/` contains checksum-verified copies of raw C-1b/C-3 JSONL
  whose source paths are intentionally gitignored.
- `nano_h2/` retains the compact H2 decision summary, RunPod metadata, and
  content hashes while the large checkpoints and row-level evaluation remain
  local.
- `nano_h3/` retains the compact H3 decision summary, RunPod closeout, and
  content hashes while the six checkpoints and row-level evaluation remain
  local and independently backed up.
- `nano_h4/` retains the frozen H4 data-only protocol, deterministic
  surface/value-family manifest, result summary, RunPod closeout, and content
  hashes while the six checkpoints and row-level development evaluation remain
  local and independently backed up.
- `durable_raw/` records durable-storage and retrieval details.
- `SHA256SUMS` covers selected preserved evidence files.

The narrative claim boundary is `papers/EVIDENCE_LEDGER.md`; the machine-readable
claim map is `papers/EVIDENCE_MANIFEST.json`. A manifest proves identity and
location, not scientific validity by itself.

Historical anchors:

- `paper-alpha-v1` → `0e01d73205e9c35ea32925fd4d6c7e5fceb61137`
- `post-alpha-evidence-freeze-2026-07-31` → `a9d12cb1c456f6c465284e1d469c6326cb14d329` (premature tag retained for history)
- `post-alpha-reconciled-evidence-freeze-2026-07-31` → `67bf87b1f968a38e68c0225b2b556f7bba5ea1cc`

Private corpus data, local review state, model checkpoints, and ignored runtime
traces remain local unless a compact content-addressed summary is explicitly
listed in this index.
