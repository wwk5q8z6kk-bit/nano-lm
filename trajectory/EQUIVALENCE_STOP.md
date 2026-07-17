# Stage T — pre-registered equivalence check FAILED at 410m (STOP-PER-PREREG)

Status: **FINAL** (all three Pythia rungs measured; JSONs archived). Recorded
before any re-registration.

## What fired

`kaggle_arm1.py`'s equivalence gate (the PREREG contamination-control check:
same model scored on instance 0 vs instance T, per-metric |diff| < 5 pts) hard-
exited nonzero at 410m. All three rungs, archived JSONs:

| rung | params | inst0 gap (95% CI) | instT gap (95% CI) | |diff| | verdict |
|---|---|---|---|---|---|
| 160m | 160M | 7.0 [3.0, 11.0] | 2.0 [0.0, 5.0] | ~5.0 | pass (float boundary) |
| 410m | 410M | 8.0 [4.0, 12.0] | 2.0 [0.0, 5.0] | 6.0 | **FAIL** |
| 1b   | 1B   | 5.0 [1.0, 9.0]  | 1.0 [0.0, 3.0] | 4.0 | pass |

parse / recall / halluc / omission passed equivalence at all three rungs; only the
GAP metric flapped. The gate did exactly what F3 hardened it to do: it stopped the
run instead of letting a non-equivalent instrument pass silently.

The three gap-diffs — ~5.0, 6.0, 4.0 — cluster tightly around the 5-pt threshold.
That is the signature of a threshold set AT the signal magnitude: the true
inst0-harder-than-instT effect is ~5 pts, the threshold is 5 pts, so measurements
scatter to both sides and the pass/fail verdict is effectively a coin-flip. This is
under-powered instrumentation, confirmed now by three datapoints, not a one-off.

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

Cross-size consistency confirms this is fixed instance difficulty, not random per-
run noise: inst0 gap > instT gap at ALL THREE rungs (7>2, 8>2, 5>1). Instance 0 is
reliably harder on held values regardless of model size — so the contamination-
direction argument (inst0 harder, not memorized) holds at every rung.

## What survives and what does not

SURVIVES (given the instance variance):
- The QUALITATIVE direction. Anchors: 22 pts (3M), 23 pts (10M). Pythia 160M-1B: gap
  1-8 pts on either instance. The reduction (~15-22 pts) is far larger than the
  ~5-6 pt instance variance, so the DIRECTION is not in doubt: the severe sub-10M
  gap is SUBSTANTIALLY SMALLER in the tested Pythia models. The CAUSE is not
  established — the stack confound (below) is unresolved; this is not a claim that
  scaling per se removes the gap.
- Model-side quality: parse 100%, recall 96-100%, halluc 0.5-4% across all three
  rungs — all beat every nano-stack result, on both instances.
- Contamination ruled out (direction argument holds at all three rungs).

DOES NOT survive without re-instrumentation:
- Any PRECISE per-rung gap value or fine-grained gap-vs-scale shape WITHIN Pythia.
  Single-instance gap at n=20 held dialogues is too noisy (±4-5 pt CIs; 4-6 pt
  inter-instance swing) to place the 160M/410M/1B rungs on a curve relative to each
  other. The formal PERSISTS/THRESHOLD band call (needs the gap CI) is NOT yet
  supportable — though it leans THRESHOLD: the top-rung (1b) gap CI lower bounds are
  1.0 (inst0) and 0.0 (instT), nowhere near the ≥10 PERSISTS floor.

CONFOUND (pre-registered, now load-bearing): the collapse coincides with the
nano-stack → Pythia-stack change (different pretraining data, tokenizer, and
full-FT → LoRA), not scale alone. WITHIN Pythia the gap is already near-floor at
160M and shows no clean monotonic scale trend (inst0: 7, 8, 5). So the transition
happened at or below 160M / is stack-driven; this ladder captured the post-
transition floor, not the transition itself. Separating scale from stack needs
either a nano-stack model at ~160M or a Pythia model well below 160M.

RELATION TO STAGE S: Stage S found the gap unmoved 3M→10M ("capacity WEAKENED").
Stage T shows it largely GONE by 160M. Consistent: the 3M→10M step (3.2×) was too
small to cross it; the 10M→160M step (16×), or the stack change, does. The gap is a
property of the sub-10M / nano-stack regime, not a capacity-invariant tail failure.

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

## Reproducibility note

The headless-T4 410m run reproduced the interactive-T4 410m run byte-for-byte on
every metric (gap 8.0/2.0, recall 96/99, etc.) — a free confirmation that the
headless API path and the interactive path are the same instrument.

## Status of Stage T

- Arm 1 raw measurements: COMPLETE and archived (160m/410m/1b JSONs + retained
  adapters for 410m/1b; 160m adapters regenerable in ~7 min).
- Gap trajectory: NOT claimed pending re-instrumentation (Stage T-v2, below).
- Arm 2 (frontier prompted probe): still pending; its value is now reframed — given
  the collapse, it would confirm the gap is absent at frontier scale (supporting
  THRESHOLD), not adjudicate PERSISTS.
- No band call, no paper claim, until the gap estimator is re-registered and the
  retained adapters re-scored on the larger multi-instance eval.
