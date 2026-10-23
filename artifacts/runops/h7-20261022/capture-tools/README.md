# Preserved H7 capture tools

These files preserve the local capture implementation used for this operational
attempt. `nano_h7_capture_training.py`, `nano_h7_capture_evaluation.py`, and
`test_nano_h7_capture_evaluation.py` are exact copies of their `/private/tmp`
originals. `PRESERVATION.json` records their source paths, byte counts and SHA-256
digests. The originals were not changed.

The training helper authenticates a completed training attempt, its inventory,
reports and checkpoint bytes. The evaluation helper authenticates completed
attempt evidence and recomputes stored-result consistency. Capturing existing
results does not authorize another training or evaluation attempt. Neither
helper is a replacement for the canonical reservation or frozen runner.

## Local runtime and path limits

Both helpers hardcode `/Users/mac/Projects/nano-lm` as their repository root and
import its operational code. They are preserved operational scripts, not
portable command-line tools. The evaluation helper authenticates the relevant
local source files against the frozen package manifest. Both depend on the
local frozen package and reservation/evidence files. A clean clone without those
local inputs cannot reproduce a real capture.

The original fixture also imports the evaluation helper from `/private/tmp`.
`run_fixture_tests.py` redirects only that import to the preserved sibling file,
without editing either original. Use this launcher so the test target does not
depend on a later change to the temporary helper:

```sh
python3 artifacts/runops/h7-20261022/capture-tools/run_fixture_tests.py -v
```

The fixture uses invented output rows and archives. It checks successful capture,
rejection of malformed or inconsistent evidence, and preservation of failures.
It imports frozen Python modules and reads package manifest/shell/source members;
it does not open screen input members, run inference, load a checkpoint, or call
a provider. Torch and tokenizers must be installed because the frozen modules
import them, even though the fixture performs no model execution. This suite is
not part of the lightweight benchmark CI dependency set.

Run the helpers with ordinary Python, never `python -O`: the preserved training
helper uses assertions for validation and executes its CLI at import time. Do
not import it as a library. The evaluation helper exposes a guarded CLI and
uses explicit validation exceptions. The preserved fixture tests the evaluation
helper only; it is not an independent test suite for the training helper.

`TEST_RESULT.json` records the local fixture runtime, command and outcome. The
captured attempt's pinned Python/Torch runtime is separately checked by the
helpers; the local fixture interpreter is not evidence of provider execution.
Stored-row consistency checks cannot establish screen authenticity or model
execution independently of the preserved hashes, invocation markers and
canonical reservation. No clinical or capability claim follows from these
software checks.

## Reviewed checksum packaging defect

`nano_h7_capture_evaluation_checksum_review.py` is the separately named verifier
used for the successful capture of the completed evaluation. It and
`test_nano_h7_capture_evaluation_checksum_review.py` preserve their temporary
source files byte for byte; their hashes are recorded in `PRESERVATION.json`.
The original helper and fixture above remain unchanged.

The frozen evaluation shell writes its checksum list to `results/SHA256SUMS.tmp`
while enumerating every result file except `SHA256SUMS`, then renames that
temporary file. Consequently the stored list includes a final checksum for a
temporary pathname that no longer exists. The initial strict capture stopped
at this defect; its partial evidence and `CAPTURE_FAILURE.json` remain at
`../downloads/evaluation-capture/` so the initial failure is not hidden.

The reviewed verifier permits only that exact absent final temporary pathname,
after every actual result file has passed its checksum. Its recorded digest
must equal the exact preceding checksum bytes (199 bytes in this attempt).
The outer archive inventory and digests, the exact three-member result
inventory, and agreement between nested and separately preserved result files
remain mandatory. The successful recapture at
`../downloads/evaluation-capture-reviewed/CAPTURE_VALIDATION.json` records the
anomaly explicitly. It reads the same attempt archive, without rewriting
archives, checksums, result bytes, or scientific code and without rerunning the
evaluation. Capture success does not erase the packaging defect.

Use the separate launcher to test the repaired verifier and all original
regressions against preserved files:

```sh
python3 artifacts/runops/h7-20261022/capture-tools/run_checksum_review_fixture_tests.py -v
```

This launcher redirects the three known temporary imports to preserved sibling
files and rejects other temporary imports. `CHECKSUM_REVIEW_TEST_RESULT.json`
records this suite's local runtime and result; `TEST_RESULT.json` retains the
original suite result and links the reviewed suite. The runtime and source
limitations above apply to both suites. The additional cases check the exact
temporary-prefix proof and reject changed real results, a wrong temporary
digest, a real temporary file, missing checksums, an unknown pathname, and a
temporary entry in the wrong position.
