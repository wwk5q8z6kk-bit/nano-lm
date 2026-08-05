# A4 result — the risk-coverage curve the frozen policy never plotted

**Computed 2026-08-05** on the calibration partition (800 examples / 4,000
fields), H6 checkpoint `seed-20260805/epoch-2`, CPU, 120 operating points.
Artifact: `artifacts/nano_h6/analysis/threshold_curve_seed20260805_epoch2.json`.
Producer: `nano_ai/training/run_threshold_sweep.py`, which mirrors
`train_evidence_query._calibrate_model` exactly and then sweeps instead of
selecting one point.

## What the frozen policy does

`minimal_zero_wrong_presented_inclusive_v1` is one line
(`evidence_query_inference.py:492`):

```python
threshold = max(wrong_confidences, default=0.0)
```

It takes the confidence of the worst presented-and-wrong field, applies it
once, asserts risk reached zero, and returns. It evaluates **two** points and
reports risk without ever naming the coverage it spent.

## The curve

AURC **0.00510**. Best retained-correct at each risk tolerance:

| threshold | coverage | selective risk | **correct retained** | errors |
|---|---|---|---|---|
| 0.99988 | 13.5% | 0.0000 | **538** | 0 |
| 0.99987 | 14.4% | 0.0017 | 575 | 1 |
| **0.97468** | **71.1%** | **0.0049** | **2,831** | **14** |
| 0.93670 | 75.1% | 0.0093 | 2,977 | 28 |
| 0.75763 | 80.2% | 0.0184 | 3,147 | 59 |
| 0.35738 | 84.7% | 0.0407 | 3,251 | 138 |
| 0.00000 | 85.1% | 0.0420 | 3,260 | 143 |

**The knee is extremely sharp.** Moving from the zero-error corner to 0.49%
selective risk — accepting **14 errors in 4,000 fields** — recovers **2,293
correct answers**, a 5.3× increase in useful output. The frozen policy sits at
the far end of that cliff, and the shape was invisible from its two points.

Stated as the cost ratio: zero-risk gives up **19 correct answers per error
removed**. At the knee the exchange is roughly **1 error per 164 correct
answers gained**.

## What this does NOT license

**It does not re-litigate H5 or H6.** Both were rejected at the *uncalibrated*
admission gate, and the preregistered staged stop halted them before the
threshold stage ran — H6's own record shows `calibrated_raw: null` and
`threshold_applied: false`. The threshold policy was never reached, so it
cannot have caused either rejection. Those verdicts stand exactly as recorded.

**It is not a threshold change.** No policy is amended here. Per the owner
decision of 2026-08-05 ("plot first, decide after"), this publishes the
tradeoff; any replacement policy requires a fresh preregistration with frozen
criteria written before it is measured.

**It is one checkpoint on one partition.** Curve shape is a property of this
model's confidence ranking, not a universal constant. Anchoring on H5 as a
second reference point is the obvious next check.

## Why it matters going forward

Every future rung that *does* pass the uncalibrated gates will meet this policy
at the calibrated stage, and on this evidence that stage discards ~84% of
correct output to reach zero wrong-presented. A rung could be scientifically
successful and still die there — not because the model failed, but because the
operating point is chosen by a rule that optimizes one axis and never measures
the other.

That is the same defect this session found in `fabric/slice.py:248` (a gate
denominated in a system-controlled quantity) and in
`wedge_v1/runtime.py::_relevant_claim` (a conjunctive filter with no coverage
term). Three mechanisms, one shape: **risk is bounded, coverage is unpriced.**

## Reproduce

```bash
python3 -m nano_ai.training.run_threshold_sweep \
  --checkpoint artifacts/nano_h6/kaggle/results-20260805/results/seed-20260805/epoch-2.pt \
  --calibration artifacts/nano_h5/data \
  --tokenizer sft/tokenizer.json \
  --output artifacts/nano_h6/analysis/threshold_curve_seed20260805_epoch2.json
```
