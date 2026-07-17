# Stage T — Findings (calibrated write-up)

Authoritative, calibrated summary of Stage T / T-v2. Where this and the working
notes (EQUIVALENCE_STOP.md, DETERMINISM_1B.md) differ in tone, this file governs.
Claims are held to what was measured; the causal question is left open explicitly.

## Question

The held-out copying gap = seen-value recall − held-out-value recall on the scribe
faithfulness task. Stage S found it ~unchanged 3M→10M. Stage T asks how it behaves
across a wider capability range, using the Pythia open-weight ladder (160M/410M/1B)
finetuned on the same scribe recipe, scored by the same faithfulness scorer.

## Result

| Model | Params | Stack | Held-out gap | Instrument |
|---|---|---|---|---|
| nano | 3.15M | own | ~22 pts | single instance |
| scale | 10M | own | ~23 pts | single instance |
| pythia-160m | 160M | Pythia | 3.5 ± 0.7 pts | 5 instances × 100 held; determinism verified |
| pythia-410m | 410M | Pythia | 4.2 ± 0.9 pts | 5 instances × 100 held; determinism verified |
| pythia-1b | 1B | Pythia | [0, 5] pts, not point-identified | training nondeterminism dominates |

## Three findings (kept distinct)

### Empirical
The large held-out copying gap seen in the 3–10M nano models (~22–23 pts) is
**substantially smaller** in the tested Pythia models (single-digit; 3.5/4.2 pts at
160M/410M, [0,5] at 1B). This is a statement about the models measured, not a causal
claim about scale: the comparison changes parameter count AND base-model family,
tokenizer, pretraining corpus, and finetuning method (full-FT → LoRA) simultaneously
(the "stack confound"). Within Pythia alone the gap is already small at 160M with no
clean monotonic trend, so this ladder captured a low-gap regime, not a transition.
Isolating scale would require a nano-stack model at ~160M or a Pythia model well
below 160M — a separate experiment, not attempted here.

### Methodological (two, and they are part of the result)
1. **Single-instance evaluation was under-powered.** The pre-registered
   contamination/equivalence check flapped at the 5-pt threshold (gap diffs
   ~5/6/4 across rungs). Diagnosis: the gap rides on a handful of hard held tokens,
   so one 20-dialogue instance has ±5-6 pt variance. Averaging over 5 instances of
   100 held dialogues cut the across-instance SD to ~0.7-0.9 at 160M/410M. The
   direction of the equivalence discrepancy (public instance HARDER, not easier)
   ruled out the contamination the check was built to detect.
2. **Training nondeterminism dominates at the top of the ladder.** The determinism
   cross-check (re-scoring byte-identical instance 0 after re-finetuning at the
   fixed seed) matched v1 exactly at 160M/410M but diverged at 1B: gap 5.0 (run 1)
   vs 0.0 (run 2) on the SAME instance. With eval variance held constant, the
   difference is training-run variance (fp16 + non-associative GPU reductions over
   1125 steps × 1B params, at the perfect-copy boundary). At 1B the right instrument
   is multi-training-seed, not multi-eval-instance; we report an interval, not a
   point.

### Engineering (prior work, companion)
The two-axis verification architecture (Stage G/A) — grounding + absence verifiers
routing model errors to human review at measured precision/review-load — remains the
deployment-relevant contribution, motivated by the fact that faithfulness under
distribution shift is not settled by whether a model passes an average-case gate.

## Bounded, characterized uncertainty (the honest state)

- 160M/410M gap: point estimates with tight eval-instance SD; determinism verified.
- 1B gap: interval [0,5]; limited by training nondeterminism, which is itself a
  measured finding, not a gap in the analysis.
- Cause of the reduction (scale vs stack): open, explicitly not claimed.

## Decision taken

Report 1B as the interval [0,5] and proceed to write-up. Additional 1B training
seeds would refine the point estimate but every plausible outcome still supports
"substantially reduced from ~22 pts," so seeds are a precision follow-up, not a
prerequisite for the main claim (owner call, recorded).

## Reproducibility

All eval instances, recipe, scorer, and library versions content-addressed in
REPRODUCIBILITY.md; frozen at git tag stage-t-v2-results. Raw per-rung JSONs
(v1 and v2) retained unchanged.
