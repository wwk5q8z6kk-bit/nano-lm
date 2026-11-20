# Harness review before model inference

A separate review agent inspected only the new runner, toy harness tests and
product interfaces. It did not read the fresh sources, annotations, gold or
model outputs. Three issues were identified before any inference on the fresh
cases:

- Conflict exact-evidence scoring must require every annotated contradictory
  observation, rather than any two matches that might be compatible with one
  another. State recall remains separate. The adjudicated set has one C field
  with exactly two spans, so this hardening does not change its target or score.
- Manifest paths must be canonical relative paths, so input snapshots cannot
  accidentally target an absolute input path.
- Checkpoint initialization and hashing must be inside failure recording, so a
  setup failure cannot leave provenance marked as running.

These are harness corrections, tested with unrelated toy data before the model
runs. No product inference code or locked annotation input was changed for this
review. The annotation lock remains the original pre-inference lock. The review
found no gold leakage or final-note emission accounting mismatch in the inspected
path; that finding is code review, not proof against every measurement defect.
