# Plan — fix the instrument before anything else

**2026-08-05.** Supersedes `ENHANCED_PLAN_20260805.md` as the primary plan.
Derived from `RESULT_DP1_AND_THE_VOCABULARY_CEILING.md`, which measured
something none of the previous plans knew.

---

## 1. What we learned today, stated as decisions

Three results, in the order they arrived, each overturning the plan that
preceded it:

1. **DP-1 passed vacuously.** `absent` was already 95.8% on calibration; only 4
   fields were recoverable. The criterion was denominated in a quantity the data
   drove to ~0 — the same degeneracy documented at `fabric/slice.py:247` a day
   earlier, reproduced by me while holding the finding.
2. **The 48.2% development figure is a lexical artefact.** Development is a
   declared stress test (`denial_phrases_disjoint: true`) with **one novel denial
   phrase per field**. In-distribution the same checkpoint scores 95.8%.
3. **The controlled substitution refuted my own explanation.** Holding
   transcripts fixed and varying only the denial phrase: TRAIN 95.9%, EXTERNAL
   62.7%, DEV 48.2%, with a 71-point spread from best phrasing (99.3%) to worst
   (28.1%). The model *does* partially generalise. The hand-written rule
   generalises at **0%** on those same external phrasings.

### The one sentence that matters

**A 71-point accuracy swing is controlled by which of ten strings the benchmark
happens to use, and every gate in the H-cycle was denominated in a single
choice of those strings.**

H6 was rejected on `absent` 199/413 against a required 383. On three of the four
training-distribution phrasings the identical checkpoint scores 385, 409, and 410
— it clears the gate. H6's rejection stands as a preregistered result; its
*diagnosis* does not. The architecture was never the measured variable.

---

## 2. First principles — what an evaluation must do here

The system maps a **document** to a **record** plus **evidence**. Three
independent things vary in the world:

| axis | what varies | example |
|---|---|---|
| **W — world** | what is true | which medication, which duration |
| **S — surface** | how it is said | `No, nothing.` vs `Denies medications.` |
| **D — document** | structure and noise | turn count, interruptions, length |

The generator currently swaps **all three at once** between training and
development — the manifest declares seven simultaneous disjointness flags. So a
single held-out number confounds all three axes, and a regression on it cannot
be attributed. That is why "absent is broken" survived as a diagnosis for two
model generations: nothing in the harness could tell *broken at the concept*
from *broken at the wording*.

**The requirement follows directly: vary one axis at a time and report the
gradient.** `run_lexical_substitution_probe` is one axis, done once, and it
overturned a standing diagnosis in five minutes of compute. That is the shape of
the instrument the project is missing.

### The metric that replaces "held-out accuracy"

A single surface form is not a measurement, it is a sample of size one. Report:

```
surface_robust_accuracy = min over K surface arms      # what you can promise
surface_mean_accuracy   = mean over K surface arms     # typical case
surface_sensitivity     = max − min                    # how much the number is
                                                       # a property of the wording
```

`min` is the right aggregator because the product's claim is *trustworthiness*:
the cost of one bad phrasing in production is a wrong record, so the loss is
asymmetric and the worst case is what may be promised (`rules/math-toolkit.md`
§7, §11). Publishing `surface_sensitivity` beside every accuracy is the
non-negotiable part — it is the number that would have prevented today's
misdiagnosis.

**Gate-design rule, generalised from the C1 failure:** never denominate a
criterion in a quantity that the system or the data partition can drive to zero,
and never gate on a single surface realisation of the target concept. Pair every
accuracy bar with (a) a data-sufficiency clause and (b) a surface-arm count.

---

## 3. What this says about the three roads on the table

**Bigger model / broad datasets / multi-agent / general capability expansion —
rejected, and the evidence is now direct rather than inferred.** Not because
scale never helps, but because no gate here can currently detect whether it did:
the outcome is dominated by wording. Spending compute against an instrument with
±35 points of surface noise buys an unmeasurable result. This agrees with the
standing judgement and strengthens it.

**"Deterministic epistemic engine" for polarity — rejected on new evidence.**
`ENHANCED_PLAN_20260805.md` §3 proposed replacing composite model judgements
with deterministic rules, reasoning from E1 (solver 0.999 vs generative 0.925).
That comparison was in-distribution. Out of distribution the ordering **inverts**:
model 62.7%, rule 0%. Promoting `_is_field_denial` into the decision path would
hard-code a ten-item list into the product and fail silently on `Denies
medications.` — the single most common phrasing in a clinical note.

**Escalation router — survives, with its trigger redefined.** The rule's value is
not its recall (3%) but its precision: it fired 4/4 correctly on calibration and
flipped 0 of 3,833 correct `supported` in the earlier exploration. A
high-precision, low-recall detector is the wrong shape for a decision rule and
exactly the right shape for an **escalation signal**. Route on *disagreement*
between the model's state and the rule's verdict, not on the rule's output.

**H7 (state head) — deprioritised, with evidence.** The state machinery scores
95.8% in-distribution; the deficit is surface transfer, which a new head does not
address. This matches the prior expectation that "if absent crosses the gate, H7
is likely unnecessary" — it does cross, in distribution.

---

## 4. The work, in order

**P0 — Surface-variation harness ($0, days). The critical path.**
Generalise `run_lexical_substitution_probe` from a one-off into the standing
instrument: all five states, not just `absent`; denial, hedge, conflict, and
value phrasings each varied over K arms; `min`/`mean`/`spread` reported for
every state. External inventories vendored with provenance, evaluation-only:
negspacy `en_clinical` for negation (done — MIT, hashed), medspacy
`POSSIBLE_EXISTENCE` (18 rules, MIT) for hedging.

**Multi-seed is a requirement, not a refinement.** The seed replication (§5 of
the result) found mean |Δ| of 2.5 points in-distribution against 31.5 points
out-of-distribution, with **Kendall τ = 0.00** between seeds on which novel
phrasings the model handles. A single-seed arm-level number is not a
measurement. The harness reports `mean over (arms × seeds)` with the per-arm
figure marked unreliable below a stated seed count, and `min` is taken over arm
means, never over single observations.
*Exit:* every accuracy in the repo carries its surface-sensitivity and its seed
count.

**P1 — Re-diagnose all five states through P0 ($0).**
`uncertain` (76.0%) and `conflicting` (57.2%) have never been separated into
concept-versus-wording. `conflicting` is the interesting one: it degraded 30
points with **no** disjoint phrase pool, so its failure is genuinely structural
and its 0.572 span accuracy is a real retrieval problem that no lexical work
touches.
*Exit:* a defensible statement of what is actually broken, per state.

**P2 — Retire the vacuous gates.** C1's shape (`before + fraction × recoverable`)
exists elsewhere. Audit every threshold in the repo against the gate-design rule
in §2 and add the data-sufficiency clause where it is missing. `fabric` gate v2
already has the coverage floor; this extends the same discipline to the
`nano_ai` selection slices.

**P3 — Escalation on model/rule disagreement.** Cheap, measurable, and it
converts the brittle rule from a liability into a signal. Measure the
disagreement set's error rate against the agreement set's — if disagreement
predicts error, it is a router; if not, drop it.

**P4 — Structure validators (~20 lines).** Unchanged from the previous plan and
still cheap: markdown-table parse and Mermaid compile, serving as evaluator,
data filter, and RLVR reward.

**P5 — Real documents.** The whole result is on synthetic clinic dialogue. The
open-licensed dogfood corpus remains the only route to knowing whether any of
this holds, and P0 is what would make its results interpretable when it lands.

**Deferred, explicitly: rung-1 $150 scale, H7, any paid compute.** The standing
gate holds — no paid run until the instrument is trustworthy. Nothing in P0–P4
needs a provider, and today's five decisive measurements cost $0 on CPU.

---

## 5. What would change this plan

- **If P1 shows `uncertain` and `conflicting` are surface-insensitive**, then
  their failures are structural, the lexical story is `absent`-specific, and a
  representational hypothesis (H7-shaped) comes back on the table for those two
  states only.
- **If the model's surface robustness turns out to be seed noise** — the arms
  share one checkpoint and one seed — the 71-point spread collapses and P0 is
  measuring variance, not sensitivity. A second seed is the cheapest possible
  falsification and belongs in P0.
- **If real documents show the model transfers better than 62.7%**, the closed
  vocabulary was the whole problem and the priority shifts to corpus work.

## 6. Standing constraints, unchanged

Open-source-licensed data only, verified before use (negspacy: MIT, upstream
hash recorded, evaluation-only, never trained on). Instruments before training.
Preregistration with frozen thresholds. Negative results retained. Kaggle free
tier as default platform. No paid training pod without a complete, verified,
authorised dataset.
