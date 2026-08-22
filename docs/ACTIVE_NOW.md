# Active Now

**Updated:** 2026-08-22  
**Integration base:** `origin/master` @ `2ad06d2` (excludes evidence-reconciliation `9fe5b6b6`)

## Status (canonical — mirrored in `ACTIVE_NOW.json`)

| Field | Value |
|-------|-------|
| `program_execution_status` | `DOC_RESET_V2_AND_P1_SCRIBING` |
| `capability_frontier` | `P1_SCRIBING` |
| `current_gate` | `REPOSITORY_RECONCILIATION_AND_CANONICAL_DOC_AUTHORITY` |
| `evidence_core` | `FROZEN_UNTOUCHED_BY_DOC_PR` |
| `training_backend` | `RUNPOD` |
| `training_status` | `ACTIVE` |
| `local_zero_cost_exploratory_training` | `ALLOWED` |
| `routine_runpod_training` | `ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET` |
| `materially_costly_run` | `EXPERIMENT_SCOPED_AUTHORIZATION` |
| `confirmatory_evidential_run` | `PREREG_PLUS_EXPERIMENT_SCOPED_AUTHORIZATION` |
| `phi_on_cloud` | `NOT_AUTHORIZED` |
| `phi_or_private_data` | `NOT_AUTHORIZED` |
| `clinical_claims` | `FORBIDDEN_WITHOUT_EXTERNAL_HUMAN_VALIDATION` |

## COMPUTE_POSTURE

```text
training_backend = RUNPOD
training_status = ACTIVE

routine_runpod_training =
ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET

materially_costly_run =
EXPERIMENT_SCOPED_AUTHORIZATION

confirmatory_evidential_run =
PREREG_PLUS_EXPERIMENT_SCOPED_AUTHORIZATION

phi_or_private_data =
NOT_AUTHORIZED

Local Apple Silicon / CPU remains available for cheap
mechanism experiments, development, preprocessing,
evaluation, and other zero-cost exploratory work.
```

The distinction is **cost / risk / evidential significance**, not “free local versus paid cloud.” RunPod is the active primary GPU training backend. Routine training on that backend is in-workflow within the active experiment budget. A new materially costly run still needs experiment-scoped authorization. A confirmatory evidential run needs prereg plus experiment-scoped authorization.

## Current gate sequence

```text
REPOSITORY RECONCILIATION (docs v2, no Evidence Core diff)
        ↓
CANONICAL DOC AUTHORITY + CI integrity checks
        ↓
P1 — encounter representation + span/evidence bottleneck
        ↓
VERIFIED ENCOUNTER RECORD → NOTE REALIZATION
        ↓
EXTERNAL MEDICAL SCRIBE EVAL + HUMAN REVIEW
        ↓
P1 MASTERY DECISION
```

## Training / compute policy

| Class | Policy |
|-------|--------|
| Training backend | **RUNPOD** (active primary GPU training / adaptation / CUDA experiments) |
| Training status | **ACTIVE** |
| Local Apple Silicon / CPU | **ALLOWED** — development, smoke, analysis, preprocessing, evaluation, small/cheap experiments |
| Routine RunPod training | **ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET** |
| Materially costly run | **EXPERIMENT_SCOPED_AUTHORIZATION** |
| Confirmatory / evidential run | **PREREG + EXPERIMENT_SCOPED_AUTHORIZATION** |
| PHI / private owner material | **NOT_AUTHORIZED** |

Do not describe RunPod as a future or optional backend, and do not treat every paid GPU job as automatically owner-gated.

## Owner gates (standing)

Materially costly runs outside the active experiment budget, confirmatory evidential runs without required prereg, PHI/private clinical data, protected tag moves, clinical claims, merge of this doc reset to `master`.

## Links

- [PROJECT_CHARTER.md](PROJECT_CHARTER.md) · [EXECUTION_PLAN.md](EXECUTION_PLAN.md) · [infrastructure/RUNPOD.md](infrastructure/RUNPOD.md) · [domains/medical/SCRIBING.md](domains/medical/SCRIBING.md)
