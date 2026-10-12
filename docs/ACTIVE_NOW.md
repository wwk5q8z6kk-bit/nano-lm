# Active Now

**Updated:** 2026-10-12
**Integration base:** `origin/master` @ `c778520` (PR #114 — H7 preregistration and frozen screen)

## Status (canonical — mirrored in `ACTIVE_NOW.json`)

| Field | Value |
|-------|-------|
| `program_execution_status` | `P1_SCRIBING_ACTIVE` |
| `capability_frontier` | `P1_SCRIBING` |
| `current_gate` | `P1_H7_EXECUTION_PACKAGE_PREFLIGHT` |
| `evidence_core` | `FROZEN_UNTOUCHED_BY_DOC_PR` |
| `training_backend` | `KAGGLE_AND_RUNPOD` |
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
training_backend = KAGGLE_AND_RUNPOD
training_status = ACTIVE

backend_policy =
KAGGLE_FREE_WHEN_RUNTIME_FITS;
RUNPOD_FOR_EXACT_GPU_RUNTIME_PERSISTENCE_OR_MANAGED_INFERENCE

routine_runpod_training =
ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET

materially_costly_run =
EXPERIMENT_SCOPED_AUTHORIZATION

confirmatory_evidential_run =
PREREG_PLUS_EXPERIMENT_SCOPED_AUTHORIZATION

phi_or_private_data =
NOT_AUTHORIZED

Local Apple Silicon / CPU remains available for cheap mechanism experiments,
development, preprocessing, evaluation, and other zero-cost exploratory work.
```

Choose the cheapest backend that satisfies the experiment. Kaggle is preferred
when its free runtime fits. RunPod is used for exact GPU/runtime requirements,
persistent volumes, or managed inference. A new materially costly run still
needs experiment-scoped authorization. A confirmatory evidential run needs
preregistration plus experiment-scoped authorization.

## Current empirical boundary

H6 is complete and **REJECTED**. The one-shot known-development evaluation was
consumed on Kaggle and is recorded in
[`artifacts/nano_h6/H6_TERMINAL_DECISION.json`](../artifacts/nano_h6/H6_TERMINAL_DECISION.json).
The selected checkpoint passed overall, held-value, missing-target, failure,
and state-balance gates, but failed absence, conflict, and uncertainty gates.
The exact H6 coupling must not be rerun or silently renamed.

Authenticated row-level diagnosis now accounts for all 5,000 field decisions:
3,901 were jointly correct; 241 were state-only errors; 771 were binding-only
errors; and 87 combined both errors. Absence (184/214 errors) and uncertainty
(53/60 errors) were dominated by state selection. Conflict had 57 binding-only
and 50 combined errors. Supported fields were dominated by exact-span binding
(714/718 errors). Threshold calibration, verifier effects, latency, and sealed
confirmation were not assessed because H6 stopped at uncalibrated admission.
Of 858 fields with a binding error, only 12 were purely disjoint; the largest
pure shapes were raw-span contains gold (467), gold contains raw (192), partial
overlap (48), and count mismatch (86). Mixed multi-span relations remain
separately reported rather than credited to a favorable category.
The diagnosis is reproducible with
`python3 -m nano_ai.training.diagnose_h6`; it authenticates the frozen result
against the terminal-decision hash and does not rerun the model.

H7-BM now has a frozen evaluation-only screen at
`artifacts/nano_h7/screen-v1/`: 1,000 records from 250 four-variant worlds,
dataset SHA-256 `9b18c5eef105d0fa2ff0a1e575e764c0014a4ff62e40e8b8205dc12674c436f2`.
The manifest proves zero exact-transcript, normalized line-component, and open
value overlap (excluding the fixed severity vocabulary) against legacy
train/dev and H4 fit/calibration, which are the source families replayed by
H5/H6. It contains no PHI or private material. The screen remains unopened by
either model arm.

The dedicated H7 trainer is now implemented and locally verified. It imports
the authenticated H6 recipe, preserves H6's model, data identities, optimizer,
schedule, exposure, seeds, calibration, and selection rules, and adds only the
preregistered boundary-margin treatment (margin 1.0, radius 2, weight 0.25).
The frozen H6 trainer remains untouched. The next step is a portable execution
package containing the authenticated H5 inputs plus independent control and
treatment scoring. The local checkout currently has the exact H5 fit and
calibration bytes but lacks their authenticated manifest; that manifest must be
restored byte-for-byte rather than regenerated before an end-to-end run.

## Current gate sequence

```text
P1 — package authenticated H5 inputs + H7 treatment/control scoring
        ↓
CHEAPEST VALID TWO-SEED MECHANISM SCREEN (Kaggle first)
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
| Training backends | **Kaggle + RunPod** selected by runtime fit and cost |
| Training status | **ACTIVE** |
| Local Apple Silicon / CPU | **ALLOWED** — development, smoke, analysis, preprocessing, evaluation, small/cheap experiments |
| Routine RunPod training | **ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET** |
| Materially costly run | **EXPERIMENT_SCOPED_AUTHORIZATION** |
| Confirmatory / evidential run | **PREREG + EXPERIMENT_SCOPED_AUTHORIZATION** |
| PHI / private owner material | **NOT_AUTHORIZED** |

Do not treat RunPod as the only backend, and do not treat every paid GPU job as automatically owner-gated.

## Owner gates (standing)

Materially costly runs outside the active experiment budget, confirmatory evidential runs without required prereg, PHI/private clinical data, protected tag moves, clinical claims.

## Links

- [PROJECT_CHARTER.md](PROJECT_CHARTER.md) · [EXECUTION_PLAN.md](EXECUTION_PLAN.md) · [infrastructure/RUNPOD.md](infrastructure/RUNPOD.md) · [domains/medical/SCRIBING.md](domains/medical/SCRIBING.md)
