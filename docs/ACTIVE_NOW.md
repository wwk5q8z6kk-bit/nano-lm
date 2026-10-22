# Active Now

**Updated:** 2026-10-22
**Integration base:** `origin/master` @ `d6ed212` (PR #116 — isolated H7 package)

## Status (canonical — mirrored in `ACTIVE_NOW.json`)

| Field | Value |
|-------|-------|
| `program_execution_status` | `P1_SCRIBING_ACTIVE` |
| `capability_frontier` | `P1_SCRIBING` |
| `current_gate` | `P1_H7_CONSUMED_INVALID_GOLD_INSTRUMENT` |
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

Choose the backend that maximizes useful evidence per dollar and wall-clock
hour. Kaggle is preferred when its free runtime fits and turnaround is
acceptable. RunPod is also appropriate when its compute speed justifies the
cost within the active experiment budget, as well as for exact GPU/runtime
requirements, persistent volumes, or managed inference. H7 retains its frozen
Kaggle CPU recipe and $0 ceiling. A new materially costly run still
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
H5/H6. It contains no PHI or private material. The screen is mounted only in the separately reserved evaluation stage.

The dedicated H7 trainer is implemented and locally verified. It imports
the authenticated H6 recipe, preserves H6's model, data identities, optimizer,
schedule, exposure, seeds, calibration, and selection rules, and adds only the
preregistered boundary-margin treatment (margin 1.0, radius 2, weight 0.25).
The frozen H6 trainer remains untouched. The exact historical H5 manifest has
been restored and authenticated alongside its unchanged fit and calibration
bytes.

The bounded H7 execution is complete and its screen is consumed. Both fixed
seeds selected epoch 2, with all six checkpoints and both reports authenticated.
The training stage took 3,694.707 seconds; the evaluation stage, containing one
control pass and two treatment passes, took 322.758 seconds. Their measured
runtime was 4,017.465 seconds total and **$0 new paid compute** (excluding
provider provisioning and the pre-execution version-1 failure).
The result bytes and recorded arithmetic are preserved in
[H7_TERMINAL_DECISION.json](../artifacts/nano_h7/H7_TERMINAL_DECISION.json).

**Scientific validity: INVALID_GOLD_ANNOTATIONS.** The evaluator recorded
`REJECT`, but reconstructed all 5,000 gold fields as missing; the authenticated
screen manifest specifies only 250 missing and 4,750 nonmissing fields. It used
a legacy fixture extractor instead of the screen's annotated targets. The
reported accuracy and boundary comparisons therefore cannot establish model
quality or a valid treatment rejection. The [stored-output/code diagnosis](research/H7_EVALUATOR_DIAGNOSIS.md)
does not rerun a model or open screen inputs. The next instrument must validate
annotated-target transport before any separately specified fresh experiment.
This consumed screen, its original result and H6's terminal verdict remain unchanged.

Private evaluation kernel `hassaneljesr/nano-h7-eval-run-d6ed212` version 2
completed under the original reservation. Version 1 stopped at a mount-path
assertion before the runner started; independent review verified the narrow
manifest-discovery repair. The frozen result checksum list also included a
renamed temporary file. A separate capture verifier proved that entry's exact
prefix digest while authenticating every actual member; both the first capture
failure and successful audited capture are retained. Neither repair repeated
the scientific evaluation. See the [execution record](../artifacts/runops/h7-20261022/RUN_STATE.json).

## Preserved comparison and compute availability

The [P1 comparison](../artifacts/p1_comparison/REPORT.md) now imports exact,
hash-pinned compact-model source results from their preserved branch alongside
six recovered OpenResearch prompt variants, the corrected larger-model C2
screen and terminal H6 evidence. The compact
Qwen2.5-1.5B route cleared its recorded two-field synthetic floors across three
seeds; its span agreement includes overlap and excludes unrelocated spans, so
it is not an exact-span or full-P1 score. Corrected C2 student and managed state
counts are 109/149 and 118/150; the gap is -5.51 percentage points, not the old
-32-point claim. H7 remains a separate comparison group, explicitly marked invalid for scientific interpretation.

The recovered OpenResearch study uses a fixed Qwen2.5-1.5B checkpoint and 24
fresh synthetic instances (384 slots), separate from its earlier 12 instances.
The shortest-span clause produced 17/240 grounded present slots versus 3/240
for the content-free clause. The verbatim clause improved location but lost six
correct abstentions. These are small, scoped prompt-mechanism findings; the
repeated slots and shared templates do not establish independent clinical
evidence or a trained-model gain. Exact exports, preregistrations, metric code,
and a sanitized audit matching all six retrieved run logs are preserved in the
comparison. Raw provider logs remain local.

Kaggle, RunPod, and OpenResearch (`orx`) authentication were verified live.
OpenResearch managed boxes bill the organization; none were launched. No RunPod
pods or serverless endpoints were active. Two retained network volumes remain.
The observed RunPod account-window billing is $71.282502, including storage;
this is not a fully attributed campaign balance. The historical campaign ledger
is preserved. H7 has a $0 ceiling and adds no paid compute. See the
[budget reconciliation](../artifacts/runops/h7-20261022/BUDGET_RECONCILIATION.json).

## Current gate sequence

The [bounded scribing path](../nanoscribe/SCRIBING_PATH.md) now connects
transcript-only solver input to the encounter schema, a five-field review draft,
and independent source/representation checking. The default is explicitly the
deterministic reference extractor. Forty-nine new adversarial boundary tests bring
the NanoScribe suite to 143 passing tests; this is software integration evidence,
not a learned-model result or clinical validation. Unsupported context and
corrections retain review evidence and can force abstention.

```text
P1 — H7 executed; frozen screen consumed; gold transport invalid
        ↓
VALIDATE ANNOTATED TARGET TRANSPORT FOR A NEW INSTRUMENT
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
