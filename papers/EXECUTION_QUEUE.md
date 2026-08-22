# Execution Queue

**Superseded as living authority.**

| | |
|--|--|
| **Current status** | [`docs/ACTIVE_NOW.md`](../docs/ACTIVE_NOW.md) · [`docs/ACTIVE_NOW.json`](../docs/ACTIVE_NOW.json) |
| **Executable tasks** | [`docs/EXECUTION_PLAN.md`](../docs/EXECUTION_PLAN.md) |
| **Compute / RunPod** | [`docs/infrastructure/RUNPOD.md`](../docs/infrastructure/RUNPOD.md) |
| **Historical snapshot** | [`docs/archive/legacy/EXECUTION_QUEUE_20260731.md`](../docs/archive/legacy/EXECUTION_QUEUE_20260731.md) |

The 2026-07-31 queue text (`TRAINING: NOT_AUTHORIZED`) is **historical packaging** from the post-E1/E4 idle posture. It must **not** be read as current policy.

**Current compute posture:** RunPod is the active primary GPU training backend. Routine RunPod training is **ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET**. Materially costly runs are **EXPERIMENT_SCOPED**. Confirmatory evidential runs are **PREREG + EXPERIMENT_SCOPED**. No PHI or private owner material in current Nano experiments.
