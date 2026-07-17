# Stage T-v2 — re-instrumented gap estimator (pre-registered before re-scoring)

Motivation: Stage T Arm 1 (see EQUIVALENCE_STOP.md) established that single-
instance gap at n=20 held dialogues has instance-level variance (~5-6 pts) larger
than the pre-registered 5-pt equivalence threshold. The threshold flapped across
three rungs (diffs ~5.0/6.0/4.0). The qualitative collapse (22/23 pts at 3-10M →
1-8 pts at Pythia 160M-1B) is robust, but no PRECISE per-rung gap or formal band
call is supportable on the v1 instrument. This stage fixes the ESTIMATOR only.

## Key property: this is a re-SCORING, not a re-TRAINING (implemented via
## deterministic regeneration)

The scientific point is that the MODELS do not change — only the eval set and gap
estimator do. Implementation: rather than upload the locally-retained adapters back
to Kaggle, the v2 kernel re-finetunes at the fixed seed. This is equivalent because
the pipeline is deterministic on T4 — the headless-T4 410m reproduced the
interactive-T4 410m byte-for-byte on every metric. The re-run therefore regenerates
the frozen v1 adapter exactly, verified in-band: v2 re-scores inst0 and instT too,
and those gaps must match the v1 JSONs (determinism cross-check). No hyperparameter,
seed, or data changes; the trained model is byte-identical to v1.

## What changes (fixed before any re-scoring)

1. **Larger held-eval.** 20 → 100 held dialogues per instance (200 → 1000 held
   field-measurements). Gap standard error scales ~1/sqrt(n) → roughly halves.
   Seen stratum kept proportional (100 seen dialogues) so both strata tighten.
2. **Multi-instance gap.** Generate K=5 fresh instances (seeds recorded below).
   Per-model gap = mean over instances; uncertainty = SD across instances (the
   quantity v1 lacked). This measures instance variance directly instead of
   colliding with it.
3. **Equivalence/contamination recast.** The v1 check conflated two things. Split:
   - *Contamination* (the real threat): flag ONLY if the public instance 0's gap
     is LOWER than the fresh-instance mean by > 2 SD (memorization makes held
     EASIER). Direction matters; a higher inst0 gap is not contamination.
   - *Instance stability*: report the across-instance SD; no pass/fail gate on it
     (it is a measured property, not a tripwire).

## Fixed seeds (recorded before generation)

Held-eval instances: seeds 20260720, 20260721, 20260722, 20260723, 20260724
(K=5, each 100 held + 100 seen dialogues, v1 generator distribution — same
`build_scribe_data.py` eval loop, only N changed). Instance 0 (public, seed 7)
and instance T (seed 20260717) retained as-is for the contamination check.

## Procedure (one pass, no tuning)

For each rung (160m regenerated; 410m, 1b from retained adapters):
1. Score the final adapter on all 5 fresh instances + instance 0 (greedy, EOS-only
   stop, identical `gate_scribe` field rules).
2. gap_mean = mean over 5 fresh instances; gap_sd = SD; report both.
3. Contamination check per the direction rule above.
Emit results_arm1_v2_<tag>.json with per-instance gaps, mean, SD, and the check.

## Interpretation bands (unchanged from Stage T, now on gap_mean ± SD)

- PERSISTS: top-rung gap_mean − 2·gap_sd ≥ 10 pts.
- THRESHOLD: gap_mean declines across the ladder AND top-rung gap_mean + 2·gap_sd
  < 5 pts.
- DIVERGENT / inconclusive: as before, reported not forced.

## What is NOT re-opened

- No re-training; adapters are frozen v1 artifacts.
- The confound stays (nano-stack vs Pythia-stack); re-scoring does not separate
  scale from stack. That needs a new TRAINING rung (nano-stack ~160M or Pythia
  <160M) and is a separate stage, not this one.
- Raw v1 JSONs (160m/410m/1b) retained unchanged as the first-instrument record.

## Status

Drafted, not executed. Re-scoring runs headless on T4 (same venue), ~10-15 min
total across rungs (scoring only, no training). Awaiting go decision vs. writing
the v1 qualitative finding first (see EQUIVALENCE_STOP.md § Status of Stage T).
