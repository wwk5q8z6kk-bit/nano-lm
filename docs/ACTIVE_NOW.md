# Active Now

**Updated:** 2026-08-22  
**Integration base:** `origin/master` @ `c4822b9` (PR #41 — Qwen inference + three-track harness)

> **Evidence lives in [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md), not here.** This
> file is canonical for the *gate and compute posture* (`ACTIVE_NOW.json` §4 of
> [`PROJECT_AUTHORITY.md`](PROJECT_AUTHORITY.md)); it is **not** canonical for
> what has been measured. Where the two appear to disagree about a result,
> RESEARCH_STATUS wins and this file is stale.
>
> **Since 2026-08-22, not reflected in the status fields below** (the gate is
> unchanged, so no field moved): the eight-cell leakage ablation landed the
> honest span-port baseline at 2/192; E-DELIMIT completed with **arm B VOID**
> (invariance precondition failed — H5 untested, *not* refuted); and
> `docs/NANO_VNEXT_MASTER_SPEC.md` became the canonical architecture entry
> point. No experiment is currently authorized.

## Status (canonical — mirrored in `ACTIVE_NOW.json`)

| Field | Value |
|-------|-------|
| `program_execution_status` | `P1_SCRIBING_ACTIVE` |
| `capability_frontier` | `P1_SCRIBING` |
| `current_gate` | `P1_ENCOUNTER_REPRESENTATION_AND_EVIDENCE_TRANSPORT` |
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
P1 — encounter representation + span/evidence transport
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

Materially costly runs outside the active experiment budget, confirmatory evidential runs without required prereg, PHI/private clinical data, protected tag moves, clinical claims.

## Links

- [PROJECT_CHARTER.md](PROJECT_CHARTER.md) · [EXECUTION_PLAN.md](EXECUTION_PLAN.md) · [infrastructure/RUNPOD.md](infrastructure/RUNPOD.md) · [domains/medical/SCRIBING.md](domains/medical/SCRIBING.md)
