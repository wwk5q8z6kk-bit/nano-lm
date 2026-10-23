# Preregistration — cross-field consistency filter on the augment adapters

**Frozen 2026-08-10, after the filter implementation was unit-tested and
its word lists validated model-free, before any scored filtered run.**
Floors will not move.

Authority: `papers/DECISION_SPAN_MECHANISM_TRADEOFF.md` option 3,
**owner-authorized 2026-08-10** ("take best decision forward and
proceed"). Implementation: `nano_ai/training/span_field_filter.py`
(5 unit tests; word-boundary regexes).

## Question

Does dropping proposed quotes that name the other field's topic — the
single mechanism behind every measured wrong-evidence cluster — make the
augment recipe the first configuration to pass all six floors on all
seeds?

## Model-free validation, recorded before the run

- False positives on gold evidence: **0/1000 medication and 0/1000
  allergy** gold spans name the other field's topic. The filter cannot
  reject correct evidence on this corpus.
- True positives on measured failures: **19/19** disjoint cases on
  augment seed 20260825; 5/6 on the P2 baseline (the 6th names no
  topic).
- Prediction: seed 20260825 disjoint 0.0429 → ≈ 0; missing_span
  0.0804 → ≈ 0.117 (still under G6's 0.15); agreement → ≈ 1.0. Seeds
  20260823/24 already pass and can only improve on G3/G5.

## Design

- Adapters: augment seeds 20260823/24/25, reused as-is. **No training,
  no decode change** — the filter runs on parsed quotes after
  generation, so per-arm state accuracies must equal the augment run's
  artifacts exactly (wiring diagnostic; any deviation voids the run).
- Both evaluators run with `--field-filter`; snap fallback unchanged;
  filtered quotes leave the emitted-span denominator and degrade to
  abstention (counted and reported as `field_filtered`).
- Floors: the identical six-gate set of `PREREG_REGISTER_AUGMENT.md`
  (G1 ≥ 0.20, G2 ≥ 0.75, G3 ≤ 0.03, G4 ≤ 0.10, G5 ≥ 0.95, G6 ≤ 0.15),
  per seed, all three seeds.

## Preregistered decision table

| outcome | condition | consequence |
|---|---|---|
| **accepted_all_floors** | all six gates, all seeds | the accepted route-(b) configuration is **augment recipe + snap + cross-field filter**; `span-fragile` is lifted; **P4 execution unblocks** |
| rejected_by_gates | any floor fails | back to `DECISION_SPAN_MECHANISM_TRADEOFF.md` options 1/2 with the filter's measured effect added; no further mechanism iterations without a fresh owner call |
| wiring mismatch | state ≠ augment artifacts or evaluator cross-check fails | fix and rerun; no decision |

## Status: CONFIRMATORY on existing adapters (deterministic; no new training randomness)

## What acceptance does not license

Nothing beyond the synthetic dev bundle and this two-field contract —
the filter's word lists are field-pair-specific and P4's real documents
must revalidate them; no promotion into Nano's frozen inference surface;
no reopening of prior rejections; the zero-FP validation is a property
of this corpus, not a general guarantee.
