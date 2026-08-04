# PREREG — type-controlled slot-diversity sweep (the program's flagship test)

**Pre-registered 2026-07-19, before any variant training data, eval instances, or runs
exist.** Direct test of the refined hypothesis recorded in
`papers/EMPIRICAL_FOUNDATION.md`: *abstraction
(copy competence) is induced per-slot when training-value diversity makes memorization
uncompetitive, given sufficient capacity.* Designed to survive the two adversarial
audits' objections: type-level n, string-identity confound, field position, tokenization.

## Question → Prediction → Measurement → Decision

**Q:** Holding model, base checkpoint, finetuning method, data size, and eval procedure
fixed, does raising ONE slot's training-value diversity causally induce copy competence
on held-out values of that slot?

**Predictions (falsifiable, stated before data):**
- **H-slot (the hypothesis):** held-type recall on the allergy slot rises with training
  diversity |ALG_TRAIN| ∈ {5 → 20 → 80}, at fixed everything-else.
- **H-string (the live alternative):** recall tracks held-type *identity* (tokenization/
  string difficulty), not arm diversity — the same types fail in every arm.
- **H-position (control):** the position-permuted arm matches its diversity-matched arm
  (field position is not the driver).
Per the type-flip finding (P2 §3.2), effects are expected to be **categorical per
(model, type)** — per-type reporting is therefore mandatory, and "recall" aggregates are
secondary to the per-type flip table.

## Design (fixed before any run)

**Arms (finetune-data variants; pretraining untouched):**
- **D5** — baseline: ALG_TRAIN = the original 5 values (recipe otherwise byte-identical
  to v2; this arm re-derives the known behavior under the new eval instrument).
- **D20** — ALG_TRAIN expanded to 20 values.
- **D80** — ALG_TRAIN expanded to 80 values.
- **D20-pos** — D20's data with the template's MED and ALG sections swapped (allergy no
  longer template-final) — the position control.
Expansion pool: common allergen strings (drugs, foods, environmental), fixed and
committed before runs; **the 6 held types (below) are excluded from every pool**. All
arms keep N=12,000 examples, same generator seed policy, same med/cc/dur/sev pools.

**Held allergy types (SAME 6 across all arms, chosen for tokenization spread, fixed
now):** `sulfa drugs` (the original), `ragweed pollen`, `bee stings`, `ibuprofen`*,
`wool`, `strawberries`. (*ibuprofen appears in MED_TRAIN as a medication — deliberately:
a token-sequence fully present in training outputs but never as an allergy value;
cleanly separates output-token-novelty from slot-binding. Its tokenization + train-output
token coverage will be reported for every held type.)

**Eval instrument:** K=5 fresh instances per the m-series procedure (200 dialogues, 100
held + 100 seen), regenerated so held-allergy items draw uniformly from the 6 held
types; seeds 20260730–20260734, committed before scoring. Both diluted and clean
metrics; **per-type recall is the primary output**.

**Model/base (cost-minimal, phenomenon-bearing):** scale-10M — reuse the frozen
`scale10m_pretrain.pt` (v0.1 release asset), full FT per arm (the anchors' native
method; ~5 min FT on T4-class). 10M is the cheapest rung exhibiting the categorical
pattern (melatonin flip at 10M; alg total failure). **Optional replication tier** (only
if the primary result is positive or ambiguous): 160M-Chinchilla base + LoRA (the
"good" corner; base preserved locally), arms D5/D80 only.

**Venue:** RunPod (runpodctl authenticated locally) or Kaggle T4 — 10M runs are venue-
light; record GPU per run. FT_SEED=0 primary; note single-seed limitation (Q2's seed
study bounds interpretation).

## Decision rule (fixed now)

Primary contrast: per-type flip table + mean held-type recall per arm at 10M/full-FT.
- **Diversity effect** = recall(D80) − recall(D5) on the same 6 held types:
  **≥ 30 pts → H-slot supported**; **≤ 10 pts → H-slot refuted** (H-string favored —
  report per-type identity analysis); 10–30 → graded, report per-type flips without a
  binary verdict.
- **Monotonicity check:** D20 between D5 and D80 (± 5 pts) strengthens; strong
  non-monotonicity flags training-draw noise → duplicate the offending arm before
  interpreting.
- **Position:** |D20-pos − D20| ≤ 5 pts → position innocent; larger → position is a
  live confound for ALL prior alg claims (report loudly).
- **ibuprofen probe:** if held-`ibuprofen` (all tokens train-output-covered) fails while
  novel-token types fail equally, token-novelty is *not* the mechanism; if ibuprofen
  alone succeeds, tokenization carries more than slot-binding.

## What this resolves / doesn't

Resolves: whether the 190/18/5 gradient is causal (diversity) or correlational
(string/position/tokenization) — the program's central claim. Doesn't: cross-stack
generality (Pythia arms would be a follow-up); seed variance (Q2's lane).

## Status

**EXECUTED** (primary 10M tier). Original locked design above is unchanged.
Generators/kernels were built after preregistration (as planned) and then run;
this RESULT does not amend the decision rule.

Artifacts: `trajectory/results_sweep_10m.json`, `trajectory/sweep_eval/`,
runner `trajectory/kaggle_sweep_10m.py`.

## RESULT (scoped summary)

| Arm | mean held-type recall % |
|-----|-------------------------|
| D5 | 0.00 |
| D20 | 24.53 |
| D80 | 66.67 |
| D20-pos | 27.65 |

- Diversity effect D80−D5 = **66.7 pts** → decision **H-slot SUPPORTED** (≥30 pts threshold).
- Position Δ (D20-pos − D20) = **3.11 pts** (≤5 → position innocent under prereg).
- Causal license: behavioral effect of slot training diversity on held-type recall under this instrument. **Not** a circuit/mechanism claim.
