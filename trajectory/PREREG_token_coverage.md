# PREREG — Phase C-1: token-coverage continuum (the residual's leading candidate)

**Pre-registered 2026-07-20, before any candidate pools, eval instances, or runs exist.**
Owner directive: the shared ~15–18-pt clean residual (alg-dominated, surviving strong
base + LoRA in both stacks) is the program's central object; token coverage is its
leading candidate explanation. This experiment turns coverage from "secondary but real"
into a measured effect size — or refutes it, elevating the deeper-binding account.

## Question → Prediction → Measurement → Decision

**Q:** Holding slot diversity fixed at a level where diversity is no longer binding
(D80), does the *token coverage* of a held value's subword sequence — the fraction of
its tokens that appear in training-output token distributions — causally determine
whether the type flips to copyable?

**Predictions:**
- **H-coverage:** flip probability increases monotonically with coverage band; the
  fitted relation *retrodicts* sulfa/ragweed's failure at D80 (their coverage is low).
- **H-binding (the deeper alternative):** flip rates are flat across coverage bands —
  novel-token emission is not the barrier; value→slot binding under low type frequency
  is, and the residual survives full coverage.
- Per the sweep, effects are expected to be categorical per type → per-type flip table
  primary; band-level rates secondary.

## Design (fixed before any run)

**Held-type bands (selection procedure frozen; realized pools committed before FT):**
Four coverage bands × ≥4 held types each, ALL sharing the same slot (allergy), same
diversity arm (D80 training pool), and **matched token-sequence lengths across bands**
(coverage correlates with length; lengths matched to break it):
- **B100** — every token of the value appears in D80-arm training outputs (ibuprofen-like).
- **B-hi** — ≥75% of tokens covered, ≥1 novel token.
- **B-lo** — 25–50% covered.
- **B0-ish** — ≤25% covered (sulfa/ragweed-like).
Selection: a committed candidate list of plausible allergen strings is tokenized against
the D80-arm's full training-output token set; band assignment is deterministic from the
computed coverage fraction; ties broken by a fixed seed. `sulfa drugs` and `ragweed
pollen` are retained as bridge types (retrodiction targets), excluded from band-balance
counts. All held types excluded from every training pool.

**Arms:** ONE training arm (D80 allergy pool, v2 recipe otherwise byte-identical,
scale-10M frozen base, full FT — the sweep's exact configuration) — the manipulation is
entirely on the *held-type side*, so a single trained model is scored against the full
band spectrum. A second seed of the same FT (FT_SEED=1) guards the categorical flips
against draw noise (the sweep's D20-pos wobble showed type-level draw sensitivity).

**Eval instances:** K=5 fresh instances, seeds 20260740–20260744, held dialogues cycling
uniformly over ALL band types (≥16 types → ~6 items/type/instance, ~30/type over K=5);
seen-alg from the D80 pool. Same generator procedure as `gen_slot_diversity_eval.py`.

**Measurement:** per-type recall (flip table) + per-band mean; coverage fraction and
token count recorded per type; logistic fit of flip vs coverage as the effect summary.

## Decision rule (fixed now)

- **B100 flip-rate − B0-ish flip-rate ≥ 50 pts** (types flipped, both seeds agreeing) →
  H-coverage CONFIRMED with large effect; report the fitted curve + sulfa/ragweed
  retrodiction.
- **≤ 15 pts** → H-coverage REFUTED as the residual's driver → the minimal binding
  probe (Phase C-3) is promoted to the next experiment.
- Between → graded; per-type table reported; no binary verdict forced.
- **Seed-disagreement gate:** any type whose flip state differs across the two FT seeds
  is reported as boundary-variance (per the flip-boundary account), excluded from the
  band rates, and counted separately.

## Cost

2 FTs (~5 min each at 10M) + 2 × 5-instance scoring (~30 min each unbatched; the
validated batched scorer may be used on CUDA after a one-time 10M equivalence check on
this venue) ≈ **~1.5 h GPU total**. Venue: RunPod (A6000-class) or Kaggle T4.

## Status

Pre-registered. Next artifacts, in order, each committed before the next step: candidate
list + band assignment (computed pools), eval instances, kernel. Nothing has been run.

## AMENDMENT 1 (2026-07-20, pre-run — design falsified at the dry-run stage)

Band construction against the real D80 output-token set, cross-checked against the
sweep's KNOWN flips, refutes both coverage formulations before any GPU was spent:
- wool flipped to 100% at D80 WITH a never-emitted token ('ool'); strawberries with two.
- bee stings: identical coverage (0.75) at D20 vs D80 gives 15% vs 100% (diversity, not
  coverage, moved it).
- sulfa (1 novel) = 0% vs bee stings (1 novel) = 100% at D80.
**H-coverage (fraction AND weakest-link forms) is refuted as the residual's driver by
existing data.** The discriminating pattern among D80 types is **lexical interference**:
the never-flippers collide with trained material — "ragweed pollen" CONTAINS the trained
value "pollen" (plus four trained "X pollen" siblings); "sulfa drugs" shares its head
with the question template's "drug allergies" — while all flippers are lexically
isolated. Redesign (C-1b): the primary axis becomes **interference class** at fixed
diversity (D80) and fixed length:
- **I-iso** — lexically isolated novel values (no content-token shared with any trained
  value or template vocabulary);
- **I-contain** — held value CONTAINS a trained value as head/substring ("X pollen"-form
  constructions with novel X);
- **I-template** — held value shares a content token with question-template vocabulary
  but no trained value;
- **I-cov** — retained coverage gradient WITHIN the isolated class (coverage as
  covariate, no longer the primary axis).
**Mandatory instrumentation change:** per-item model OUTPUTS logged for every held item
(the sweep saved only counts), enabling failure-mode classification per type —
substitution-by-contained-value is the predicted signature of I-contain failures.
Decision rule (fixed now): flip-rate(I-iso) − flip-rate(I-contain) ≥ 50 pts with
substitution-dominated failure modes in I-contain → interference account CONFIRMED;
≤ 15 pts → refuted, minimal binding probe (C-3) promoted. Bridges retained; predictions:
sulfa/ragweed pattern with I-template/I-contain respectively.
