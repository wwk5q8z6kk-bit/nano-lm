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

## AMENDMENT 2 (2026-07-20, pre-pool — classifier refined by two vocabulary discoveries)

Mechanical inspection of the D80 pool and the v2 template vocabulary, done BEFORE any
candidate pool was computed, falsifies AMENDMENT 1's binary isolated/interfering split:

- **"wasp stings" ∈ D80** — so the flipper "bee stings" SHARES its head word with a
  trained value, yet flipped to 100%. Word-sharing alone does not block copying.
  (Recheck: `python3 -c "import sys;sys.path.insert(0,'trajectory');from
  slot_diversity_pools import ALG_TRAIN_80;print('wasp stings' in ALG_TRAIN_80)"`)
- **"ibuprofen" ∈ MED_TRAIN** — the flipper "ibuprofen" IS a trained value of the med
  slot in the same dialogues, yet flipped to 100% as a held allergy. Cross-slot
  trained-value identity does not block copying.

The surviving discriminator among the six sweep types is **within-slot containment of a
FULL trained value**: "ragweed pollen" ⊃ "pollen" (a standalone D80 value) failed;
everything short of full-value containment flipped. The class taxonomy is accordingly
refined to five classes, each anchored by ≥1 existing bridge type:

| class | definition (deterministic, word-level) | bridge | predicted |
|---|---|---|---|
| I-contain | held value contains a complete trained alg value as contiguous word-subsequence | ragweed pollen | FAIL |
| I-template | no trained-word share; shares a content word with template vocabulary (strip-s) | sulfa drugs | FAIL |
| I-sib | shares ≥1 word with a trained alg value but contains no full value | bee stings | FLIP |
| I-xslot | is (or contains) a trained value of the med slot | ibuprofen | FLIP |
| I-iso | none of the above | wool, strawberries | FLIP |

Classifier precedence: I-contain > I-sib > I-xslot > I-template > I-iso. Trained set =
ALG_TRAIN_80; template vocabulary = all D_*/P_*/DISTRACT literal strings in
build_scribe_data_v2.py, content words only (function-word stoplist), singular/plural
normalized by strip-s on both sides. Coverage (vs the D80-arm output-token set) and
token count recorded for every type as covariates (I-cov = the gradient within I-iso).

**Decision rules (fixed now, before pools exist):**
- Primary (unchanged from AMENDMENT 1): flip(I-iso) − flip(I-contain) ≥ 50 pts with
  substitution-dominated failures in I-contain → containment-interference CONFIRMED;
  ≤ 15 pts → refuted, C-3 binding probe promoted.
- Secondary contrasts: I-sib vs I-contain separates containment from word-overlap;
  I-xslot vs I-iso tests slot-specificity of the trained-value effect; I-template vs
  I-iso tests the sulfa pattern's generality.
- Per the flip-matrix account, per-type flip states (both FT seeds agreeing) are
  primary; class rates are compositions. Seed-disagreeing types are reported as
  boundary variance and excluded from class rates.

Candidate hygiene rules (mechanical, enforced by the pool generator): no modifier word
reused across classes; no candidate sharing words with MED_TRAIN except in I-xslot; no
candidate in any training pool or existing held set; classifier must reproduce all six
sweep types' known classes and flip states or the pools are INVALID.
