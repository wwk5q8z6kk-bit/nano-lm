# PREREG — own-stack scale ladder to separate SCALE from STACK (planned follow-up)

**Pre-registered 2026-07-18. Not executed.** This is the documented follow-up that
resolves Paper 1's primary limitation (§7 scale-vs-stack confound). Owner chose
freeze-and-write; this is the highest-priority extension, to be run if reviewers press
on the confound or if drafting shows it is needed. Design fixed here *before* any run,
including the decision rule, so a later execution cannot be accused of post-hoc tuning.

## Question

The Stage T ladder shows own-stack 3.15M/10M at gap ~18.3/18.7 and Pythia-160M/410M at
~3.5/4.2, but the nano→Pythia step changes scale AND the whole stack (pretraining-data
quantity ~1500×, tokenizer 4098 vs ~50k, architecture, full-FT vs LoRA). This experiment
holds the **own stack fixed** and moves **only parameter count**, to ask: does the gap
shrink with scale *within one stack*, or does it persist (implicating the stack)?

## Design (fixed before any run)

**Models.** Own-stack GPT family (RoPE, GQA, SwiGLU, RMSNorm pre-norm, tied embeddings,
4098-vocab BPE, 512 ctx) at **~160M first**, then ~40M and ~80M only if 160M is
ambiguous (see decision rule). 160M is the decisive rung because it matches the Pythia
rung where the gap collapsed (3.5 ± 0.7); it dominates 40M/80M on information gain.

**Pinned 160M dims (recorded 2026-07-18, before training):** `d=1024, L=14, H=16, KV=4,
hd=64, ff=2752` → **159.3M** params (verified by the exact param formula that reproduces
nano 3.15M at d=192/L=6/H=6/KV=2/hd=32/ff=512 and scale 10.0M at d=320/L=8/H=8/KV=2/hd=40/
ff=864). This is a within-family scale-up: same RoPE/GQA/SwiGLU/RMSNorm/tied-emb recipe,
GQA 4:1 (matching scale), hd grown 40→64, ff/d ≈ 2.7 held. If 40M/80M are added, pin
their dims by the same formula before training (e.g. ~40M ≈ d=640/L=10, ~80M ≈ d=768/L=12).

**Finetuning-method control (2×2 at 160M).** Because full-FT vs LoRA is a first-class
confound (the consult panel's most-flagged under-controlled variable), run BOTH at 160M:
- **A: own-stack 160M, full FT** (matches the nano/scale recipe).
- **B: own-stack 160M, LoRA r=16 α=32** (matches the Pythia recipe).
Interpretation: A≈B≈18 ⇒ stack effect (architecture/pretraining), not method. A≈18,
B≈mid ⇒ LoRA regularization is part of the gap. Both drop ⇒ scale dominates at 160M.

**Held fixed.** Same ~200M-token FineWeb pretraining recipe, same scribe finetune data,
same 5-instance instrument (scribe_eval_m0..m4, 100 held + 100 seen), same native
ChatML/greedy scorer (`rescore_anchors.py` generalizes), same gap = seen − held recall,
mean ± across-instance SD.

**Undertraining control (mandatory reporting).** 200M pretraining tokens is ~Chinchilla
for 10M (Hoffmann et al., 2022) but far below compute-optimal for 160M. Therefore: (i)
report pretrain/val loss curves and token/param ratio per rung; (ii) pre-declare the
token budget — keep 200M for the "identical recipe" comparison AND, if quota allows, a
second 160M pretrained to ~Chinchilla (~3.2B tokens) to bound the undertraining
objection; (iii) frame any high-gap 160M result as "within this recipe, adding
parameters did not reproduce the low-gap regime," NOT "architecture is bad."

**Training-nondeterminism probe.** At 1B (Pythia) two fixed-seed retrains split 5/0. If
own-stack 160M lands low (<6), run a **duplicate finetune** to check whether the
precision floor has shifted from eval-instance variance to training-run variance in the
own stack too (a predicted "precision-floor transition").

## Decision rule (fixed before results)

On the 160M full-FT rung (Condition A), gap_mean over m0–m4:
- **≥ 14** → STACK-dominant: scale within the own stack does not reproduce the Pythia
  low-gap regime; the reduction is stack/pretraining/tokenizer/method. Paper's center of
  gravity becomes the stack effect. (Then train 40M/80M to show the curve is flat/high.)
- **≤ 6** → SCALE-plausible within the family: report the within-stack transition; run
  40M/80M to locate it, plus the duplicate finetune (low-gap → check nondeterminism).
- **6–14** (mixed) → run 40M/80M to resolve the curve shape; report as a graded story.

## Compute estimate

~8–12 T4-hours for the 160M pretrain + full-FT + eval; +4–5 for the LoRA arm; +8–12 if
40M/80M are added. Total ~15–25 T4-hours, within a Kaggle weekly quota (~30 h). Venue
notes inherit REPRODUCIBILITY.md (T4 pin, torchao uninstall, etc.).

## What this does and does NOT resolve

- Resolves: is the reduction attributable to parameter count *within the own stack*, and
  (via the 2×2) whether the finetuning method is implicated.
- Does NOT fully isolate: pretraining-corpus *content/diversity* and tokenizer *vocab*
  remain bundled into "own stack" unless separately varied (future stages). State this;
  do not over-read a high-gap 160M as "architecture."

## Status

Drafted, not executed. Frozen design; execution (if chosen) runs headless on Kaggle T4
and appends `results_ownstack_v2_160m_{fullft,lora}.json` in the results_anchors schema.
