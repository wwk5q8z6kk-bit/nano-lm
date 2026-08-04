# Benchmark integration

This directory contains a small, digest-bound integration sentinel for the
held-value task. It checks that benchmark manifests, task loading, scoring,
artifact hashes, and failure preservation behave reproducibly.

It is infrastructure, not a claim of benchmark superiority.

## Current contents

- `REGISTRY.yaml` and `REGISTRY.schema.json`: registered suites, task, and
  resource classes.
- `adapters/lm_eval/tasks/held_value_sentinel.yaml`: the operational sentinel.
- `adapters/lm_eval/fixtures/held_value_sentinel_n4.json`: four-record
  digest-bound fixture.
- `adapters/lm_eval/LM_EVAL_PIN.md`: pinned lm-eval version and commit.
- `adapters/lm_eval/tests/test_sentinel.py`: registry, digest, run-ID,
  artifact, and boundary tests.

The sentinel is explicitly ineligible for leaderboard or Evidence Ledger
promotion. A smoke pass means the integration path works; it says nothing about
model quality, Nano capability, or scientific leadership.

## Run

```bash
python3 -m benchmarks.adapters.lm_eval.cli validate
python3 -m benchmarks.adapters.lm_eval.cli smoke --mode deterministic
pytest -q benchmarks/adapters/lm_eval/tests/test_sentinel.py
```

Scientific status remains in
[`papers/EVIDENCE_LEDGER.md`](../papers/EVIDENCE_LEDGER.md). New benchmark
work should exist only when it answers a current AI-engineering or research question
under the project’s [decision gates](../papers/DECISION_GATES.md).
