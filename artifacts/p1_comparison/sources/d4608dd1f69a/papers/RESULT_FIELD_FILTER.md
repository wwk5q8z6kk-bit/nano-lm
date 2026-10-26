# Result — accepted: augment + snap + cross-field filter passes all six floors

**2026-08-11.** Reports `papers/PREREG_FIELD_FILTER.md` against its frozen
gates. Artifacts: `artifacts/nano_h6/field_filter/FIELD_FILTER_SUMMARY.json`,
per-seed probes/agreement, `field_filter_run.log`.

**Verdict: `accepted_all_floors`. The accepted route-(b) configuration is
the augment recipe + normalization snap + cross-field consistency filter.
`span-fragile` is lifted. P4 execution is unblocked.**

## 1. Against the frozen floors — all pass, all seeds

| metric (floor) | 20260823 | 20260824 | 20260825 |
|---|---|---|---|
| control worst-class recall (≥ 0.20) | 0.733 | 0.733 | 0.733 |
| min-arm state accuracy (≥ 0.75) | **1.000** | 0.833 | 0.767 |
| disjoint rate (≤ 0.03) | **0.000** | 0.0021 | **0.000** |
| post-snap no_match (≤ 0.10) | 0.0222 | 0.0284 | 0.0578 |
| gold agreement (≥ 0.95) | **1.000** | 0.9979 | **1.000** |
| missing_span (≤ 0.15) | 0.0275 | 0.0333 | 0.1176 |
| state identity vs augment artifacts | exact | exact | exact |
| spans filtered | 14 | 1 | 19 |

## 2. The prediction held

The prereg's model-free forecast for the failing seed (20260825):
disjoint 0.0429 → ≈ 0 (**observed 0.000**); missing_span ≈ 0.117
(**observed 0.1176**); agreement ≈ 1.0 (**observed 1.000**). The filter
converted every cross-field wrong-evidence assertion into abstention and
touched nothing else — state identity is exact on all seeds, and the
zero-FP gold validation held in vivo (agreement did not drop on any
seed; two seeds reached 1.000).

## 3. The accepted configuration, stated plainly

Training: balanced span-bearing LoRA on the **augment-injected** build
(natural denials retained beside register-injected twins; train sha
`e4cd6d3b…`). Inference: greedy decode with the baseline prompt →
**cross-field filter** (reject quotes naming the other field's topic) →
**normalization snap** (casing/unicode/whitespace canonicalization,
unique-match required) → unmodified binder (exact unique relocation or
abstention). Every stage after the model is deterministic, reversible,
and fails toward abstention; the binder remains the sole grounding
authority.

## 4. What acceptance licenses, and does not

Licensed: P4 execution planning on this configuration; closure of the
span-fragility arc opened by `RESULT_P3_MULTISEED.md`. Not licensed:
anything beyond the synthetic dev bundle and this two-field lexicon
(P4 exists precisely to test that); no promotion into Nano's frozen
inference surface; the filter's word lists must be revalidated on real
documents; three seeds remain three seeds — the RunPod Stage-B
distribution study (in flight) is the calibration follow-up.
