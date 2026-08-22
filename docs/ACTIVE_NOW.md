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
| `local_zero_cost_exploratory_training` | `ALLOWED` |
| `paid_training` | `OWNER_GATED` |
| `frozen_confirmatory_execution` | `PREREG_PLUS_OWNER_GATED` |
| `phi_on_cloud` | `NOT_AUTHORIZED` |
| `phi_or_private_data` | `NOT_AUTHORIZED` |
| `clinical_claims` | `FORBIDDEN_WITHOUT_EXTERNAL_HUMAN_VALIDATION` |

## COMPUTE_POSTURE

```text
Primary GPU training backend (when owner-gated paid runs are authorized):
RUNPOD

Local compute (zero-cost exploratory training ALLOWED):
Apple Silicon / CPU — development, smoke tests, small mechanism
experiments, analysis, preprocessing, evaluation

Paid / confirmatory runs:
OWNER_GATED — prereg + explicit authorization per active execution plan

Data:
No PHI or private owner material in current Nano experiments.
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
| Local zero-cost exploratory training (MPS/CPU) | **ALLOWED** |
| Paid training (RunPod / GPU) | **OWNER_GATED** |
| Frozen confirmatory execution | **PREREG + OWNER_GATED** |
| PHI / private owner material | **NOT_AUTHORIZED** |

The hard gate is **paid compute and confirmatory evidence runs**, not “never train anything.”

## Owner gates (standing)

Paid or confirmatory RunPod runs, confirmatory runs without prereg when required, PHI/private clinical data, protected tag moves, clinical claims, merge doc reset to `master`.

## Links

- [PROJECT_CHARTER.md](PROJECT_CHARTER.md) · [EXECUTION_PLAN.md](EXECUTION_PLAN.md) · [infrastructure/RUNPOD.md](infrastructure/RUNPOD.md) · [domains/medical/SCRIBING.md](domains/medical/SCRIBING.md)
