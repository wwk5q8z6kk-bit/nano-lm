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
- Evaluation: private kernel `hassaneljesr/nano-h7-eval-run-d6ed212`,
  version 2, with the private evaluation package and separate private treatment
  dataset `hassaneljesr/nano-h7-treatments-c6177067`. Both training reports,
  all six epoch checkpoints and the exact uploaded treatment archive were
  verified first. No H5 fit/calibration data are mounted in evaluation.

Evaluation notebook version 1 failed at its manifest-parent-name assertion,
before its only subprocess invocation. Kaggle had flattened the dataset root.
The [preserved failure review](evaluation-v1-preexecution-failure/FAILURE_REVIEW.json)
records the terminal provider error, exact downloaded runner/reservation bytes,
and independent review. Version 2 changes only manifest discovery to require
the sole manifest and its frozen checksum; it retains the identical runner,
canonical reservation and scientific invocation. This is a reviewed pre-execution
infrastructure repair; no scientific evaluation pass was repeated.

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

The private training notebook and both evaluation notebook versions are retained
exactly as submitted, including their embedded runner and reservation. Provider dataset metadata records privacy
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

## Terminal outcome

Measured training and evaluation stage runtimes total 4,017.465 seconds for $0,
excluding provisioning and the pre-execution version-1 failure. Exact result bytes
are preserved at `artifacts/nano_h7/kaggle/h7-boundary-margin-v1/` and imported by
hash into the P1 comparison. See [the terminal record](../../nano_h7/H7_TERMINAL_DECISION.json).

The frozen evaluator recorded REJECT, but its gold transport was invalid: all
5,000 fields were reconstructed as missing against a manifest containing only
250 missing fields. This is not a valid treatment rejection or accuracy result.
The [diagnosis](../../../docs/research/H7_EVALUATOR_DIAGNOSIS.md) is based on
stored diagnostics and frozen code, without rescoring or opening screen input.

`evaluation-evidence/ORIGINAL_CAPTURE_FAILURE.json` retains the checksum-temp
packaging failure. A separately preserved verifier accepts only the absent final
`results/SHA256SUMS.tmp` entry whose digest equals the exact preceding checksum
bytes; every actual archive member remains checked. The successful capture
validates stored arithmetic, not the correctness of reconstructed gold labels.
The screen remains consumed and no scientific retry is authorized.
