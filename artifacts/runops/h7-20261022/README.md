# H7 execution capture — 2026-10-22

`RUN_SPEC.json` is the immutable prelaunch specification, so its
`PREPARED_NOT_LAUNCHED` field records preparation rather than live status.
`RUN_STATE.json` records the latest observed execution state. Neither a launch
nor training success is a completed H7 screen result.

The scientific source is `d6ed212c32384c1d50089e50260918028af88bfd`.
The original training and evaluation archives remain separately authenticated
by their manifest and member hashes. `scripts/run_h7_kaggle.py` changes only
the dependency bootstrap and checksum command's working directory. The frozen
trainer, scorer, datasets, seeds, checkpoint-selection rules and stopping rules
are unchanged.

## Stages

- Training: private kernel `hassaneljesr/nano-h7-train-run-d6ed212`, version 1,
  with only private dataset `hassaneljesr/nano-h7-training-d6ed212`.
- Evaluation package: private dataset
  `hassaneljesr/nano-h7-evaluation-d6ed212`. Its presence is preparation, not
  permission to score incomplete training outputs.
- Evaluation must receive a separate, frozen treatment-results archive only
  after both successful training reports and selected checkpoint bytes have
  been verified. No H5 fit/calibration data are mounted in evaluation.

For the treatment dataset, retain the exact archive bytes under an opaque
`.bin` suffix so Kaggle does not automatically expand the archive during
upload. The frozen wrapper reads gzip content independently of its filename.
Verify the single uploaded file and its downloaded SHA-256 before reserving
evaluation; do not add checksum sidecars or other data files to that mount.

The canonical external reservation directory is recorded in the reservation
JSON. A fresh checkout, kernel name or output path does not authorize another
attempt. Do not release or recreate reservations to work around failures.
The runtime budget is cumulative: evaluation receives 14,400 seconds minus
the preserved successful training terminal's elapsed time.

The private training notebook is retained exactly as submitted, including its
embedded runner and reservation. Provider dataset metadata records privacy
checks. `RUN_EVENTS.json` is a readable snapshot of the local event log.

## Capture before interpretation

Preserve the provider output and full attempt-evidence archive. Verify its
SHA-256, `ATTEMPT_SHA256SUMS`, the treatment artifact inventory and both training
report hashes. Require the reserved attempt ID, successful terminal, both fixed
seeds completed, exact runtime versions, and verified selected checkpoints.
Any failed or unknown attempt remains visible; no automatic retry is allowed.
Evaluation results are imported into the comparison from preserved bytes, never
by rescoring. H6's original known-development evaluation remains terminal.

## Local-only material and cost

Raw account billing responses, old unsubmitted notebook drafts, downloaded
archives/checkpoints and transient provider logs stay local. They are excluded
from this public repository without deleting them. The budget reconciliation
records source hashes and aggregate observations; its account-window total is
not an attributed campaign balance. No paid compute was launched for H7.
The campaign's $180 autonomous envelope, $200 hard cap and $20 owner reserve
remain in force. Future backend choices account for wall-clock speed as well
as price; H7 retains its frozen $0 CPU execution specification.
