# Enhanced plan — decompose the epistemic decision, don't retrain it

**2026-08-05.** Derived from a first-principles decomposition plus one
observation in H6's sealed evaluation that explains every failure row at once.
Supersedes the capability-breadth and router proposals as the *primary* plan;
routing survives as an implementation detail (§5).

---

## 1. Decomposition — what the system actually does

Given a source document and a field, the system must perform five separable
steps:

| # | Step | Question | Measured state |
|---|---|---|---|
| 1 | **Locate** | which text is relevant? | **works** — BM25 promotes the answering paragraph (margin 3.728 on the case that failed) |
| 2 | **Bind** | does a specific span literally support this? | **mostly works** — fixed for ≥3-token queries (W-ABSTAIN-2); 2-token conjunction open |
| 3 | **Decide state** | supported / absent / conflicting / uncertain / missing? | **broken** — absent 0.482, uncertain 0.760 |
| 4 | **Decide to present** | confident enough to say it? | **signal good, policy bad** — 14 of 143 errors sit in the top 71% of confidence |
| 5 | **Render** | is the output structurally valid? | **not built** |

Steps 3 and 4 are conflated in discussion but are different decisions: 3 is a
classification over epistemic states, 4 is a scalar gate. The evidence says the
system is good at 4 and bad at 3 — the opposite of the common "it doesn't know
what it knows" reading.

## 2. The observation that explains everything

Ordering H6's development results by the **logical form** of each state:

| state | logical form | state acc | span acc |
|---|---|---|---|
| supported | `∃ span asserting v` | **0.999** | 0.814 |
| missing | `∄ mention` | **1.000** | 1.000 |
| uncertain | `∃ span` ∧ *hedged* | 0.760 | 0.972 |
| conflicting | `∃ two spans` that disagree | 0.800 | 0.572 |
| absent | `∄ asserting span` ∧ `∃ denying span` | **0.482** | 0.927 |

**Accuracy tracks logical complexity, not reading difficulty.** A single
existential query is near-perfect. Adding a modality judgment costs ~24 points.
Adding a relation between two spans costs ~20. Requiring a *conjunction of a
negative and a positive* costs ~52.

And the span column is decisive: for `absent` and `uncertain` the model
**locates the right evidence** (92.7%, 97.2%) and fails only when combining.

**First-principles reading.** A single forward pass is naturally an existential
operator — attention finds *a* thing. Universal and relational claims ("no span
asserts", "two spans disagree") require quantifying over the whole document or
comparing candidates. That is a different computation, and the H2/H3/H6 record
is consistent: three attempts to make one pass emit a composite judgement, three
rejections.

## 3. The proposal — compose existential probes with deterministic rules

Stop asking the model for the composite label. Ask it only what it is good at,
and combine with rules.

```
ABSENT      := ¬(∃ span asserting v)  ∧  (∃ span denying field)
CONFLICTING := |{distinct v : ∃ span asserting v}| ≥ 2
UNCERTAIN   := (∃ span asserting v)   ∧  hedged(span)
SUPPORTED   := (∃ span asserting v)   ∧  ¬hedged  ∧  no conflict
MISSING     := ¬(∃ any span mentioning field)
```

Every left side is a composite the model scores 0.48–0.80 on. Every right side
is either an existential probe (0.927–0.999) or a deterministic operation —
set cardinality, string comparison, a hedge lexicon.

**This is E1's finding applied inside the model's decision rather than beside
it.** E1 measured a deterministic solver at **0.999** against the best
generative reference at **0.925**; the composition steps are exactly the
rule-shaped part. We keep the model for retrieval and binding, where it is
strong, and use rules for combination, where they are.

### Falsifiable prediction

If the account is right, a decomposed pipeline should beat the monolithic state
head **on the existing H6 checkpoint, with no retraining**, because it reuses
the same span-finding ability that already scores 92.7% on absent.

If a decomposed pipeline does *not* beat 0.482 on absent, the account is wrong
and the failure is in span *semantics* (the model finds a span but misreads what
it asserts) rather than in composition. That is a clean falsification and it is
worth knowing either way.

### Why this is the highest-leverage item available

- **It costs $0 and needs no training.** Inference on a checkpoint we hold.
- **It tests the program's central blocker** — the wall that stopped H5 and H6.
- **It is falsifiable in one run**, unlike another architecture hypothesis
  requiring a full train-and-evaluate cycle.
- **If it works, H7 becomes unnecessary**, and the next model rung changes
  entirely — the model would need to be good only at existential retrieval.

## 4. Plan, in priority order

**P1 — Decomposition probe ($0, days).** Implement the five rules above over the
H6 checkpoint's existential outputs; evaluate on the **calibration** partition
first (development is spent). Preregister thresholds before measuring.
*Gate:* absent joint ≥ 0.70 (vs 0.482) with no regression on supported.

**P2 — Structure validators (~20 lines, hours).** Markdown-table parse and
Mermaid compile checks. Serve simultaneously as evaluator, synthetic-data
filter, and RLVR reward. This is where "charts and diagrams" belongs — as
validation, not as a training corpus.

**P3 — The model seam, one backend (days).** Make `lm/probe.py:LMBackend` real
with exactly one implementation. Every backend needs its own span-binding
contract, so N backends is N adapters — prove one end to end before fan-out.

**P4 — Confidence routing (after P3).** Justified by measurement, not
architecture fashion: 14 of 143 errors in the top 71% of confidence means a
three-zone router is supported by the data. Escalate the low-confidence tail —
which is also, usefully, where the absent-state errors live.

**P5 — H7, only if P1 fails.** If composition is not the problem, then a
state head is back on the table and the H7 preregistration applies.

**P6 — Rung-1 scale.** Still deferred. Nothing above needs it, and the residual
copying gap traces to a five-value training pool, not to capacity.

## 5. What survives from the other proposals

- **Routing:** yes, as P4, and now *evidence-backed* rather than assumed. It is
  the implementation of the model seam, not a competing architecture.
- **Distillation:** only to manufacture structured-transformation pairs, filtered
  by the P2 validators — the validator, not the teacher, is the source of trust.
- **Everything else** (broad capability training, 40-dataset mixtures, agents,
  multimodal, speech): unchanged verdict — unreachable at this scale, and
  available through the seam.

## 6. Honest uncertainties

1. **Does the model's span actually assert what we think?** P1 assumes a located
   span carries the semantics we ascribe. If it doesn't, P1 fails and the
   diagnosis moves to span semantics.
2. **Is the hedge lexicon domain-general?** `uncertain` detection via lexicon
   works on synthetic clinic text; real documents hedge in more ways.
3. **`conflicting` has a genuine span problem** (0.572), not just composition —
   it needs two distinct spans. Rules help with the comparison, not the retrieval.
4. **All of this is still closed-world**, on synthetic clinic dialogue. The
   open-licensed dogfood corpus remains the only route to knowing otherwise.
