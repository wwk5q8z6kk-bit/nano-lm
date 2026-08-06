# Result — surface harness, run 1: `absent` and `uncertain` fail differently

**2026-08-05/06.** First full run of the instrument built in
`nano_ai/surface.py`. Two checkpoints (seeds 20260805, 20260806) × two axes
(denial, hedge) × 30 arms = 60 inference passes over the 1,000 sealed
development documents. CPU, $0.

Artifact: `artifacts/nano_h6/analysis/surface_harness.json`.
EXPLORATORY — selects nothing, gates nothing, moves no threshold.

---

## 1. Validity check first

The `DEV` arm is an identity rewrite: it substitutes nothing and scores the
development documents unmodified. It must therefore reproduce the independently
computed `generalization_gap.json` figures exactly. It does:

| | independent probe (seed05) | harness `DEV` arm (seed05) |
|---|---|---|
| absent | 199/413 = 0.4818 | **199/413 = 0.4818** |
| uncertain | 190/250 = 0.7600 | **190/250 = 0.7600** |

Two separate code paths, same numbers to four decimals. The substitution
machinery is not silently corrupting the documents it rewrites.

## 2. The headline

| axis / state | scope | robust | mean | sensitivity | seed instability |
|---|---|---|---|---|---|
| denial / **absent** | in-distribution | 94.1% | **97.2%** | **5.5%** | 2.5% |
| denial / **absent** | held out | 24.3% | **60.0%** | **59.8%** | 31.4% |
| hedge / **uncertain** | in-distribution | 41.2% | **67.7%** | **39.8%** | 16.9% |
| hedge / **uncertain** | held out | 0.0% | **48.0%** | **89.2%** | 7.9% |

`robust` = min over arm means · `sensitivity` = spread of arm means ·
`seed instability` = mean within-arm spread across the two seeds.

**These two states are broken in different ways, and no previous measurement
could tell them apart.**

### `absent` — a clean surface-generalisation failure

In-distribution the model is tight: mean 97.2% across four training phrasings
with only **5.5 points** of spread and **2.5 points** of seed instability. It
reliably recognises the denials it was trained on. Held out, mean drops to 60.0%
and spread explodes to **59.8 points**. The competence is real and the transfer
is not. Sensitivity exceeds instability roughly 2:1, so the surface effect is
not merely seed noise — though see §4 for why the formal claim is still withheld.

### `uncertain` — not the same failure, and worse than it looked

In-distribution spread is **39.8 points** — seven times `absent`'s. The model
never mastered hedging *even on the phrasings it was trained on*. Its mean on
training-distribution hedges is 67.7%, below `absent`'s held-out mean.

This is a different diagnosis with a different remedy, and it contradicts the
natural reading of H6's gate table, where `uncertain` (76.0%) merely looked like
a milder version of `absent` (48.2%). It is not milder; it is a weaker grasp of
the concept itself.

### The number that should worry us most

The `DEV` arm — **identical, unmodified documents** — scores `uncertain` at
**76.0% on seed 20260805 and 43.6% on seed 20260806**. A 32.4-point swing on the
same data from a checkpoint differing only in seed.

H6's `uncertain_target` gate required 228/250. Seed 05 produced 190; seed 06
would have produced 109. The gate's verdict on that state is substantially a
function of which seed was selected, and nothing in the H-cycle reported it.

## 3. Cross-effects — the concepts are not independent

The harness scores every state under every rewrite, so interference is visible:

| rewrite | state observed | mean | sensitivity |
|---|---|---|---|
| denial | supported | 80.1% | **0.6%** |
| denial | missing | 99.4% | 9.8% |
| denial | uncertain | 70.0% | **13.4%** |
| denial | conflicting | 45.4% | 5.4% |
| hedge | absent | 48.1% | 4.1% |
| hedge | supported | 79.6% | **0.5%** |
| hedge | missing | 100.0% | 0.0% |
| hedge | conflicting | 45.8% | 0.0% |

`supported` is essentially immune (≤0.6 points) — rewriting a denial or hedge
elsewhere in the transcript does not disturb value copying. But changing the
**denial** phrase moves **`uncertain`** accuracy by up to 13.4 points. The
per-field decisions are entangled: a field's epistemic state depends on wording
that belongs to a different field.

That is worth knowing before anyone proposes per-field heads or independent
probes as an architecture — the independence such designs assume is measurably
absent here.

## 4. The instrument refused every arm-level claim, and that is correct

All four aggregates report `arm_comparison_supported: false`. Two conditions
must hold and only one does: `sensitivity > instability` is satisfied in three
of four cases, but `MIN_SEEDS_FOR_ARM_CLAIM = 3` is not — only two checkpoints
exist.

So the per-arm orderings below the aggregates are **not** reportable. The
2026-08-05 replication measured Kendall τ = 0.00 between seeds on external arm
rankings, and nothing here changes that. Read the means.

The guard was not weakened to produce a cleaner story. Adding a third seed
requires a free-tier training run (subtask 10); until then, arm-level claims
remain unsupported and that is the honest state.

## 5. Honest limits

- **Two seeds.** Every instability figure is a spread over n=2, which is a point
  estimate of a spread, not an estimate of variance.
- **The held-out hedge arms are author-constructed**, labelled
  `NOT independent` in `surface_arms.py` and enforced by a test. No open-licensed
  inventory of *patient-voice epistemic* hedges was found; medspacy's
  `POSSIBLE_EXISTENCE` is clinician-register diagnostic hedging and was
  deliberately rejected as a source. The 48.0% held-out hedge mean is therefore
  weaker evidence than the denial figures, which draw on negspacy and medspacy.
- **`robust` = 0.0% for held-out hedge** comes from one constructed arm
  (`Your guess is as good as mine.`) on one seed. With the arm-level guard
  unsatisfied, that number describes a worst case observed, not a worst case
  established.
- **All of it remains synthetic clinic dialogue.**

## 6. What this changes

`ENHANCED_PLAN_20260805.md` §2 ordered the states by "logical complexity" and
read `uncertain` as a mid-difficulty composite. The harness separates the
confound: `absent` is competent-but-non-transferring, `uncertain` is
not-yet-competent. They do not belong on the same axis, and a single fix aimed
at "composite states" would have been aimed at a distinction that does not exist.

Next: subtask 13 extends the harness to `conflicting`, the remaining state whose
30.3-point held-out drop occurred with **no** disjoint phrase pool and is
therefore the strongest candidate for a genuinely structural failure.
