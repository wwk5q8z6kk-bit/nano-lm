# Preregistration — H7-BM boundary-margin screen

**Frozen before treatment training or evaluation.**
**Status:** PRE-FLIGHT; no result exists
**Experiment ID:** `nano-p1-h7-boundary-margin-screen-v1`

## Question

Does a local hard-negative boundary-ranking term improve Nano's exact evidence
transport on unseen transcripts while preserving its state decisions?

## Arms

| Arm | Model | Training difference |
|---|---|---|
| Control | frozen terminal H6 selected checkpoint, `seed-20260805-epoch-2`, SHA-256 `07547ec5c01b46fa87e8d53a40497a7c4b4a6bc92b596dddd67f4be4d18c89e0` | none; evaluated only on the new H7 screen |
| Treatment | H6 architecture trained from the same frozen Nano v0.1 anchor with the same seeds and recipe | add local start/end margin loss, margin `1.0`, radius `2`, weight `0.25` |

The H6 known-development result is not opened by the H7 evaluator and is not
rerun. Using the frozen H6 checkpoint as a control on **new H7 data** does not
change or relabel H6's terminal decision.

## Fixed data boundary

The screen generator must be committed before either arm is scored. It will
produce 1,000 synthetic records from 250 complete four-variant worlds, with:

- new world IDs, values, question forms, answer forms, denial forms, uncertain
  forms, and correction forms;
- zero exact transcript, component-template, or open-value overlap with H5/H6
  fit, calibration, and known development;
- exact manifest hashes and overlap-audit counts;
- no PHI, private owner material, benchmark records, or historical-fresh data;
- no training use and no treatment-specific filtering.

If these conditions are not mechanically proven, the screen is ineligible.

### Frozen screen identity

- Path: `artifacts/nano_h7/screen-v1/`
- Generator SHA-256: `1fe306df6cc0d553019f10d279efe2baa77125559ff74dbdd6e558afe43b8d03`
- Record bytes SHA-256: `9b18c5eef105d0fa2ff0a1e575e764c0014a4ff62e40e8b8205dc12674c436f2`
- Transcript multiset SHA-256: `df73f34e3b99ae8f52c91137c0625a5cd20e9f8931c0c44370ec77a664ce40c8`
- Isolation audit: zero exact transcript, normalized line-component, and open
  value overlap (excluding the fixed severity vocabulary) against legacy
  train/dev plus H4 fit/calibration; zero PHI/private records.

## Fixed training boundary

- Seeds: `20260805`, `20260806`.
- Same H6 architecture, anchor, tokenizer, H5 replay-mixture fit/calibration,
  batches, epochs, optimizer, schedule, selection order, state loss, pointer
  cross-entropy, and threshold policy.
- Treatment change: average the start and end local-margin terms, multiply by
  `0.25`, and add it to the unchanged H6 total loss.
- Local negatives: valid Patient tokens within two token positions of the gold
  boundary, excluding the gold token. No doctor, prompt, or padding token can be
  a negative.
- The selected treatment checkpoint is chosen using training-only calibration,
  before the H7 screen is opened.

Any other source, data, loss, inference, or verifier change voids the comparison.

## Metrics

All metrics are computed on all 5,000 field decisions unless the denominator is
named explicitly.

1. `joint_exact_accuracy` — state and complete span set both exact.
2. `state_accuracy` — state only.
3. `span_exact_on_shared_state_correct` — span exact on the fixed cohort where
   both control and treatment select the gold state.
4. `binding_error_fields` and the full relation taxonomy from
   `nano_ai.training.diagnose_h6`.
5. `disjoint_or_speaker_mismatch_fields` — safety diagnostic.
6. Decode failures and field-collapse counts.

The shared-state-correct cohort is defined symmetrically after both predictions
exist; neither arm chooses it. Overall joint and state metrics remain primary
guards against a favorable conditional denominator.

## Decision bars

This is a two-seed mechanism screen, not confirmation.

**PASS** only if both treatment seeds satisfy all of:

- at least 20% relative reduction in binding-error fields versus the frozen H6
  control on the H7 screen;
- at least +5 percentage points in `span_exact_on_shared_state_correct`;
- state accuracy no more than 1 percentage point below control;
- no increase in `disjoint_or_speaker_mismatch_fields`;
- zero decode failures and no field collapse.

**REJECT** if either seed violates state non-inferiority or safety, or if neither
seed reaches the boundary-improvement bars. Otherwise the result is **GRAY** and
does not license another run with moved bars.

## Termination and compute

Stop after local unit tests, deterministic data generation, one control scoring
pass, and the two fixed treatment seeds. No third seed, threshold sweep, prompt
change, H6 rescore, or sealed evaluation is allowed in this screen.

Use local CPU/Apple Silicon for implementation and data checks. Use Kaggle free
GPU if the frozen H6 runtime fits; use RunPod only if the exact runtime cannot be
obtained free. Before paid compute, freeze the git SHA, dataset manifest,
commands, artifact destination, maximum runtime, and dollar ceiling. No PHI.

## What a pass would not establish

No P1 mastery, no clinical claim, no human-validation result, no robustness to
real encounter distributions, no promotion to a production checkpoint, and no
permission to alter the H6 terminal record.
