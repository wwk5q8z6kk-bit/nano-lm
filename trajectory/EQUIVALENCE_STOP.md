# Stage T — pre-registered equivalence check FAILED at 410m (STOP-PER-PREREG)

Status: **PRELIMINARY** (1b rung + pulled JSONs pending; conclusion not expected
to change — see below). Recorded before any re-registration.

## What fired

`kaggle_arm1.py`'s equivalence gate (the PREREG contamination-control check:
same model scored on instance 0 vs instance T, per-metric |diff| < 5 pts) hard-
exited nonzero at 410m. Console measurements:

| rung | inst0 gap (95% CI) | instT gap (95% CI) | |diff| | verdict |
|---|---|---|---|---|
| 160m | 7.0 [3.0, 11.0] | 2.0 [0.0, 5.0] | 5.0 | pass (float boundary) |
| 410m | 8.0 [4.0, 12.0] | 2.0 [0.0, 5.0] | 6.0 | **FAIL** |

parse / recall / halluc / omission all passed equivalence at both rungs; only the
GAP metric failed. The gate did exactly what F3 hardened it to do: it stopped the
run instead of letting a non-equivalent instrument pass silently.

## Diagnosis (the direction rules out the actual threat)

The equivalence check exists for ONE purpose (PREREG "Instrument and contamination
control"): detect whether the public instance 0 is contaminated. Contamination
would make instance 0 EASIER on held values (model memorized the public held-value
dialogues → held recall inflated → gap SHRINKS). 

Observed: instance 0's gap is LARGER than instance T's, at both sizes. That is the
OPPOSITE of the contamination signature. **The threat the check was built to catch
is affirmatively absent** — instance 0 is if anything harder, not memorized.

What DID fail is benign: the two instances differ in held-value difficulty by ~5-6
pts because the gap is dominated by a handful of hard held tokens (only 3 held CC
values, 2 held meds, 1 held allergy, across 20 held dialogues × 5 fields = 100 held
field-measurements). Which specific hard values land in which instance is seed-
dependent, so single-instance gap has instance-level variance LARGER than the 5-pt
threshold. The threshold was under-powered for the estimator's noise — an
instrument-design miss, caught (as intended) before it corrupted a trajectory claim.

Cross-size consistency (inst0≈7-8, instT≈2 at BOTH 160m and 410m) confirms this is
fixed instance difficulty, not random per-run noise.

## What survives and what does not

SURVIVES (robust to the instance variance):
- The QUALITATIVE trajectory. Anchors: 22 pts (3M), 23 pts (10M). Pythia 160m/410m:
  gap in the 2-8 pt range on either instance. The drop (~15-21 pts) dwarfs the ~5-6
  pt instance variance. "The held-out copying gap collapses from 10M to 160M+" holds.
- Model-side quality: parse 100%, recall 96-99%, halluc 1-4% at both rungs — all
  beat every nano-stack result, on both instances.
- Contamination ruled out (direction argument above).

DOES NOT survive without re-instrumentation:
- Any PRECISE per-rung gap value or fine-grained gap-vs-scale shape. Single-instance
  gap at n=20 held dialogues is too noisy (±4-5 pt CIs; 5-6 pt inter-instance swing)
  to place rungs on a curve. The PERSISTS/THRESHOLD band call (which needs the gap
  CI, not just the point) is NOT yet supportable.

## Re-registration proposal (cheap — it is a SCORING fix, not a training fix)

The finetuned LoRA adapters are retained for every rung. Fixing the gap estimator
requires only RE-SCORING existing adapters on a better eval — no re-training.

Options (engineering, to be pre-registered as Stage T-v2 before re-scoring):
1. Larger held-eval: 20 → 100+ held dialogues/instance → gap SE ~halves.
2. Multi-instance gap: mean over K≥5 instances/model; CI across instances.
3. Equivalence threshold calibrated to measured instance variance (or a paired test).

Recommended default: (1)+(2) — regenerate a larger, multi-instance held-eval
(new seeds, recorded), re-score all retained adapters, re-register the gap
estimator + equivalence threshold with the measured instance-variance as the
power target. Raw 160m/410m/1b measurements from this run are retained as-is.

## Immediate actions (in flight)

- 410m and 1b headless jobs are running/finishing; their raw JSONs will be pulled
  and archived (410m headless reproduces the interactive result deterministically;
  1b adds the third equivalence datapoint — expected inst0>instT again).
- No trajectory/band claim is made until the gap estimator is re-registered and the
  adapters re-scored.
