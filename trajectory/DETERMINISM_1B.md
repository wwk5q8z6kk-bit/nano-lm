# Stage T-v2 — the determinism cross-check FAILED at 1b (second methodological finding)

The T-v2 instrument assumed re-finetuning at the fixed seed regenerates the frozen
v1 adapter (so multi-instance scoring is a re-SCORE of the same model). This was
VERIFIED at 160m and 410m and REFUTED at 1b. The check earned its place.

## The evidence (same eval instance, two training runs)

| rung | v1 inst0 gap | v2 inst0 gap | v1 instT gap | v2 instT gap | determinism |
|---|---|---|---|---|---|
| 160m | 7.0 | 7.0 | 2.0 | 2.0 | HELD (exact) |
| 410m | 8.0 | 8.0 | 2.0 | 2.0 | HELD (exact) |
| 1b   | 5.0 | **0.0** | 1.0 | **0.0** | **FAILED** |

inst0 and instT are byte-identical files across v1 and v2 (same SHA-256, see
REPRODUCIBILITY.md). So the 1b difference cannot be eval-instance variance — it is
the SAME instance scored on two DIFFERENT trained models. The only thing that
differed between the v1 and v2 1b runs is the training run itself (identical seed,
hparams, data, code, library versions, GPU type). Therefore: **the 1b finetune is
non-deterministic, and the nondeterminism is large enough to move the held-out gap
from 5.0 to 0.0.**

## Why 1b and not 160m/410m

fp16 autocast + non-associative GPU reductions (atomicAdd in backward) make each
step's update depend on nondeterministic kernel scheduling; over 1125 steps × 1B
params the drift compounds. 160m/410m either reproduce exactly or are far enough
from the perfect-copy boundary that the drift does not flip any of the ~100 held
field-decisions. At 1b the model sits AT that boundary — a run either achieves
perfect held-out copying (gap 0) or leaves a few errors (gap 5). Small weight
differences tip it across.

## Consequence for the instrument

T-v2 fixed the EVAL-instance noise (down to SD ~0.7-0.9 at 160m/410m). But at 1b it
revealed that a LARGER, different noise source dominates: TRAINING-run variance.
Multi-instance eval is the wrong instrument for the 1b gap — the right one is
multi-training-SEED. The two 1b runs we have (gap 5.0 and gap 0.0) are two draws
from that distribution; the point estimate is not identified without more seeds.

## What holds regardless

- 160m: gap 3.5 ± 0.7 (eval SD) — determinism verified, trustworthy.
- 410m: gap 4.2 ± 0.9 — determinism verified, trustworthy.
- 1b: gap is SMALL and training-run-dependent — both runs agree it is in [0, 5],
  i.e. consistent with 160m/410m or lower. NOT precisely identified.
- The direction — 22/23 pts (3-10M nano) → single-digit (Pythia 160M-1B) — holds
  across BOTH noise sources characterized here (eval-instance AND training-run):
  the reduction is far larger than either. "Substantially reduced," not
  "eliminated," and not attributed to scale per se (stack confound stands).

## Bearing on the paper

Two methodological findings now sit alongside the empirical one, and they compound
the contribution:
1. Single-instance gap was under-powered (eval noise) — T-v2 fixed it.
2. At the top of the ladder, training-run nondeterminism dominates eval noise — the
   determinism cross-check caught it; the honest 1b claim is an interval, not a point.

The empirical headline is unchanged and honest: the severe sub-10M copying gap
largely resolves by Pythia scale, but the precise residual at the largest rung is
limited by training nondeterminism, not by the model — which is itself the point
about why careful measurement matters.

## Open decision (not resolved here)

Whether to run K training seeds at 1b (to point-identify the 1b gap and characterize
training variance) or to report 1b as an interval [0, 5] and proceed to write-up.
This is a precision-vs-scope call, teed up for the owner.
