# Active Now

**Updated:** 2026-08-22  
**Integration base:** `origin/master` @ `2ad06d2` (pre–evidence-reconciliation `9fe5b6b6`)

## Status (canonical — mirrored in `ACTIVE_NOW.json`)

| Field | Value |
|-------|-------|
| `program_execution_status` | `DOC_RESET_V2_AND_P1_SCRIBING` |
| `capability_frontier` | `P1_SCRIBING` |
| `current_gate` | `REPOSITORY_RECONCILIATION_AND_CANONICAL_DOC_AUTHORITY` |
| `evidence_core` | `FROZEN_UNTOUCHED_BY_DOC_PR` |
| `training_backend` | `RUNPOD` |
| `training_status` | `ACTIVE` |
| `paid_compute_policy` | `EXPERIMENT_SCOPED_AUTHORIZATION` |
| `frozen_confirmatory_execution` | `PREREG_PLUS_EXPERIMENT_SCOPED` |
| `phi_on_cloud` | `NOT_AUTHORIZED` |
| `phi_or_private_data` | `NOT_AUTHORIZED` |
| `clinical_claims` | `FORBIDDEN_WITHOUT_EXTERNAL_HUMAN_VALIDATION` |

## COMPUTE_POSTURE

```text
Primary GPU training backend:
RUNPOD

Local compute:
Apple Silicon / CPU for development, smoke tests,
small mechanism experiments, analysis, preprocessing,
evaluation, and zero-cost experiments where appropriate.

Training:
ACTIVE_ON_RUNPOD

Paid experiment policy:
RunPod is an established project infrastructure.
A new materially costly or confirmatory run still requires
the experiment-specific budget/authorization rules defined
by the active execution plan.

Data:
No PHI/private clinical data on RunPod under the current program.
Use public, synthetic, licensed, or appropriately deidentified data.
```

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
| Training backend | **RUNPOD** (primary GPU training / adaptation / CUDA experiments) |
| Training status | **ACTIVE** |
| Local Apple Silicon / CPU | Development, smoke tests, analysis, preprocessing, evaluation, small/cheap experiments |
| Paid / confirmatory experiments | **EXPERIMENT_SCOPED** — infrastructure is established; individual costly or confirmatory runs follow the active plan’s budget/auth rules |
| Frozen confirmatory execution | **PREREG + experiment-scoped authorization** |
| PHI / private clinical data on cloud | **NOT_AUTHORIZED** |

Distinguish:

```text
training infrastructure is authorized and active
        ≠
a particular costly/evidential run may require its own authorization
```

Do not describe RunPod as a future/optional backend, and do not treat Nano training itself as globally unauthorized.

## Owner gates (standing)

Experiment-scoped costly/confirmatory RunPod runs (per active plan), confirmatory runs without prereg when required, PHI/private clinical data on cloud, protected tag moves, clinical claims, merge doc reset to `master`.

## Links

- [PROJECT_CHARTER.md](PROJECT_CHARTER.md) · [EXECUTION_PLAN.md](EXECUTION_PLAN.md) · [infrastructure/RUNPOD.md](infrastructure/RUNPOD.md) · [domains/medical/SCRIBING.md](domains/medical/SCRIBING.md)
