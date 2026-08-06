# Result — per-state re-diagnosis: it is lexical unfamiliarity, in every state

**2026-08-06.** Subtask 14. Synthesis of four harness runs over the sealed
development documents, two frozen seeds, CPU, $0. Supersedes
`ENHANCED_PLAN_20260805.md` §2.

Artifacts: `surface_harness.json` (denial, hedge) ·
`surface_harness_conflicting.json` (conflicting_value, conflicting_structure) ·
`surface_harness_value_template.json` (value, template).
EXPLORATORY — selects nothing, gates nothing, moves no threshold.

---

## 1. What was believed, and what the instrument found

`ENHANCED_PLAN_20260805.md` §2 ordered the five states by *logical complexity* —
`supported`/`missing` as simple existentials, `uncertain` as existential + a
modality judgement, `conflicting` as a relation between two spans, `absent` as a
conjunction of a negative and a positive — and read the accuracy ordering as
evidence that composite judgements are what a single forward pass cannot do.

Six axes later, holding documents fixed and varying one thing at a time:

| state | axis varied | in-distribution | held out | Δ |
|---|---|---|---|---|
| **missing** | *(nothing to vary — no span)* | 100.0% | 100.0% | **0.0** |
| **supported** | open **value** | **96.3%** | 77.2% | −19.1 |
| **supported** | answer **template** | 79.7% | 77.2% | −2.5 |
| **absent** | denial phrase | **97.2%** | 60.0% | −37.2 |
| **uncertain** | hedge phrase | 67.7% | 48.0% | −19.7 |
| **conflicting** | two open **values** | **95.2%** | 42.0% | −53.2 |
| **conflicting** | **order / distance** | — | — | *no effect (below)* |

**Every state that carries a span recovers to 95–100% when its vocabulary is
familiar.** The one state with no lexical dependency at all — `missing`, decided
by the *absence* of a mention — has a zero gap. The ordering is not logical
complexity. It is how unfamiliar the words are.

## 2. The structural hypothesis was tested and failed

`conflicting` was the strongest candidate for a genuinely structural failure: it
requires comparing two spans, and it lost 30 points held out. Two structural
interventions on identical documents — swapping the **order** of the two
competing values, and varying the **distance** between the two mentions (1, 3,
and 6 turns apart):

```
conflicting_structure   sensitivity 10.2%   seed instability 17.3%
```

**Sensitivity is smaller than instability.** Reordering the values and moving
them six turns apart produces no effect distinguishable from changing the random
seed. Meanwhile substituting familiar values into those same documents lifts
`conflicting` from 42.0% to **95.2%**.

The two-span comparison is not the problem. The words are.

This corrects a claim made in `RESULT_DP1_AND_THE_VOCABULARY_CEILING.md` §2 —
that `conflicting` "drops with no disjoint phrase pool at all, so its cue is
structural." `conflicting` has no disjoint *denial/hedge* pool, but the
development manifest also declares `open_value_lexicons_disjoint: true`, and
`conflicting` is decided over two *open values*. The correction is recorded
inline in that paper.

## 3. The one state that is different

`uncertain` is not a transfer failure. Its **in-distribution** spread is
**39.8 points** — seven times `absent`'s 5.5 — and its in-distribution mean
(67.7%) is below `absent`'s *held-out* mean. The model never learned hedging
even on the six phrasings it was trained on.

The seed evidence is starker still: the **unmodified** development documents
score `uncertain` at **76.0% on seed 20260805 and 43.6% on seed 20260806**. H6's
`uncertain_target` gate required 228/250; seed 05 delivered 190 and seed 06 would
have delivered 109. That gate's verdict was substantially a seed draw.

So of five states: three are competent-but-non-transferring, one is solved, and
one was never learned.

## 4. Value versus frame — a clean factorial result

The `supported` axes decompose cleanly because they were varied independently
over the same documents:

| value | template | accuracy |
|---|---|---|
| development | development | 77.2% *(baseline)* |
| development | **calibration** | 79.7% *(+2.5)* |
| **calibration** | development | **96.3%** *(+19.1)* |

**The value carries the accuracy; the frame around it barely matters on
average.** But *which* frame matters more than that average suggests: spread
across the four calibration templates is **22.6 points** against only **3.3
points** of seed instability — a 6.8× ratio, the strongest sensitivity-to-noise
margin measured anywhere in this work. A phrasing effect that survives that
comparison is real even though the arm-level guard withholds the formal claim.

## 5. What this means for the next experiment

The failure is not architecture, not composition, not capacity, and not the
two-span relation. It is that the training vocabulary is tiny — **8 denial
phrasings, 6 hedges, 86 values** — and nothing in the recipe pressures the model
to generalise past it.

That points at **data diversity as the intervention**, and it retires three
proposals on evidence rather than taste:

- **H7 / a state head** — the state machinery reaches 95–100% whenever the words
  are familiar. A new head does not address vocabulary.
- **Deterministic composition of existential probes**
  (`ENHANCED_PLAN` §3) — it was motivated by composite states being hard.
  Composition is not what fails; the same paper's own rule generalises at ~3% to
  independent lexicons.
- **Rung-1 scale** — a bigger model trained on 8 denial phrasings learns 8
  denial phrasings. Scale is not the binding constraint.

## 6. Honest limits

- **Two seeds throughout.** Every arm-level ordering is withheld
  (`arm_comparison_supported: false` in all runs); only means are reported. A
  third seed is blocked by design — `RESULT_SURFACE_HARNESS_RUN1.md` §4b.
- **A denominator asymmetry in the `value` axis**: its DEV baseline scores 1,287
  fields while its TRAIN arms score 1,213 (−5.7%), because a document containing
  two pooled values is dropped from a substituting arm but not from the identity
  arm. Too small to explain a 19-point gap, but it is not zero and the `template`
  axis (1,287 throughout) is the cleaner comparison.
- **Held-out arms differ in independence.** Denial arms draw on negspacy and
  medspacy (MIT, vendored); hedge arms are author-constructed and labelled
  `NOT independent`; value and template arms use the development pools, which are
  generator output rather than an external inventory.
- **Synthetic clinic dialogue throughout.** Nothing here licenses a claim about
  real documents, and the open-licensed dogfood corpus remains the only route to
  one.
