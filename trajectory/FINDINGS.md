# Stage T — Findings (calibrated write-up)

> **SRC baseline note (2026-07-30):** this file remains the calibrated Stage T
> write-up. Program-level interpretation lock and paper split live in
> `papers/RESEARCH_PROGRAM.md` / `papers/EMPIRICAL_FOUNDATION.md`. Subsequent
> within-stack / diversity / corner / Phase C results are summarized there and in
> the Paper-1 measurement companion (`papers/paper2_draft.md`, reclassified).


Authoritative, calibrated summary of Stage T / T-v2. Where this and the working
notes (EQUIVALENCE_STOP.md, DETERMINISM_1B.md) differ in tone, this file governs.
Claims are held to what was measured; the causal question is left open explicitly.

## Question

The held-out copying gap = seen-value recall − held-out-value recall on the scribe
faithfulness task. Stage S found it ~unchanged 3M→10M. Stage T asks how it behaves
across a wider capability range, using the Pythia open-weight ladder (160M/410M/1B)
finetuned on the same scribe recipe, scored by the same faithfulness scorer.

## Result

| Model | Params | Stack | Held-out gap (multi-instance) | inst0 | Instrument |
|---|---|---|---|---|---|
| nano | 3.15M | own | 18.3 ± 1.3 pts | 22.4 | 5 instances × 100 held; determinism verified (byte-exact vs gate_scribe_v2.log) |
| scale | 10M | own | 18.7 ± 1.5 pts | 23.0 | 5 instances × 100 held; determinism verified (exact vs Stage S, even CUDA→MPS) |
| pythia-160m | 160M | Pythia | 3.5 ± 0.7 pts | 7.0 | 5 instances × 100 held; determinism verified |
| pythia-410m | 410M | Pythia | 4.2 ± 0.9 pts | 8.0 | 5 instances × 100 held; determinism verified |
| pythia-1b | 1B | Pythia | [0, 5] pts, not point-identified | 5.0/0.0 | training nondeterminism dominates |

All five rungs are now on one instrument (5×(100 held + 100 seen), v1 distribution,
gap_mean ± across-instance SD). The anchors were re-scored from their frozen v0.1
checkpoints (`PREREG_anchors.md`; re-scoring only, native ChatML/greedy scorer). The
single public instance (inst0) is a **uniformly hard draw** — its gap exceeds the
multi-instance mean at every rung — so the earlier headline (single-instance anchor
~22–23 vs multi-instance Pythia 3.5) mixed instruments at the two ends and overstated
the contrast. On the consistent instrument the anchors read ~18 and the gap is still
an order of magnitude above the Pythia rungs.

## Three findings (kept distinct)

### Empirical
The large held-out copying gap seen in the 3–10M nano models (**18.3±1.3 / 18.7±1.5
pts** on the consistent multi-instance instrument; 22–23 single-instance) is
**substantially smaller** in the tested Pythia models (single-digit; 3.5/4.2 pts at
160M/410M, [0,5] at 1B). This is a statement about the models measured, not a causal
claim about scale: the comparison changes parameter count AND base-model family,
tokenizer, pretraining corpus, and finetuning method (full-FT → LoRA) simultaneously
(the "stack confound"). Within Pythia alone the gap is already small at 160M with no
clean monotonic trend, so this ladder captured a low-gap regime, not a transition.
Isolating scale would require a nano-stack model at ~160M or a Pythia model well
below 160M — a separate experiment, not attempted here.

Re-scoring the anchors on the powered instrument also **sharpened Stage S**: the
3M→10M step moves the gap by 0.4 pts (18.3±1.3 → 18.7±1.5), i.e. inside one SD — the
"scale did not move the gap" conclusion, originally read off two single-instance
points (22 vs 23), now holds with error bars.

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
"substantially reduced from the ~18-pt anchor," so seeds are a precision follow-up,
not a prerequisite for the main claim (owner call, recorded).

## Reproducibility

All eval instances, recipe, scorer, and library versions content-addressed in
REPRODUCIBILITY.md; frozen at git tag stage-t-v2-results. Raw per-rung JSONs
(v1 and v2) retained unchanged.


## E1 / E3 validation gates (2026-07-30)

### E1 — non-LM utility kill-gate: **KILL**

On frozen utility \(U = P - 0.5 M - 0.3\rho - 0.02 L - 0.05 C\) (verify-on; ρ =
review/flagged load), the best non-generative baseline **M1** (hand-template)
achieves \(U\approx0.999\) and strictly dominates official generative M0
(Pythia-160M LoRA \(U=0.925\); ownstack Chinchilla+LoRA \(U=0.898\); local 10M
scale \(U\approx0.854\)). **M2** (dict+span) \(U\approx0.886\) is below official
M0 but within the pre-registered \(\delta=0.05\) non-necessity margin of it.
Sensitivity grid does not flip the kill. This falsifies the generative-substrate
product thesis for that frozen regime; it does **not** establish that no
generative-value regime exists. Non-transformer methods also show open-slot
held gaps → ecology **general**. Primary artifact: `results_e1_utility.json`
(venue `runpod-cuda`).

### E3 — exact-match construct: **EXACT_SURVIVES** (scoped)

Frozen normalize-then-match rescues **0** of 486 M0 exact field failures (gap
shrink 0.0 pts; `results_e3_normalize_construct.json`).

A bounded **agent-applied rubric audit** of 100 sampled errors
(`results_e3_human.json`, rater `agent-rubric-pass-1`) found **zero** acceptable
semantic equivalents (faithful-rate 0.00). This does **not** substitute for
independent clinician annotation, inter-rater agreement, or validation of a
synonym ontology. Prereg: `PREREG_E3_faithfulness_construct.md`.

### E2 — LoRA universes: **GATED / STOP** (no RESULT)

Design frozen in `PREREG_E2_lora_universes.md`. Implementation present
(`trajectory/e2/run_u3_earlystop.py`). Partial RunPod residue exists under
`trajectory/runpod_partial/e2_*` (empty monitor log). **No**
`results_e2_*.json`. No complete external U3 artifact found in local search.
Active RunPod pods: none (checked 2026-07-31). Stopped after E1 KILL: mechanism
ID does not restore the killed product frame without written re-scope.
