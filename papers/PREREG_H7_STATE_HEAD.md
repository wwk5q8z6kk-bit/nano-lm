# Preregistration — H7: the failure is state classification, not span location

**Status: DRAFT, frozen before any training.** No compute is authorized by this
document. It exists so the next experiment is chosen by decomposition rather
than by the aggregate label "abstention calibration."

## 1. The decomposition that changes the diagnosis

From H6's own sealed development evaluation
(`artifacts/nano_h6/kaggle/eval-20260805/results/development_evaluation.json`,
`uncalibrated_raw.by_gold_state`):

| gold state | n | **state accuracy** | **span accuracy** | joint | presented |
|---|---|---|---|---|---|
| **absent** | 413 | **0.482** | 0.927 | 0.482 | 376 |
| **uncertain** | 250 | **0.760** | 0.972 | 0.760 | 53 |
| **conflicting** | 250 | 0.800 | **0.572** | 0.572 | 50 |
| supported | 3,837 | 0.999 | 0.814 | 0.813 | 3,833 |
| missing | 250 | 1.000 | 1.000 | 1.000 | 0 |

Two distinct mechanisms, which the aggregate gate scores had merged:

**(a) State misclassification with correct spans — `absent` and `uncertain`.**
The model locates the right evidence (92.7% and 97.2% span accuracy) and then
assigns the wrong epistemic state. Joint accuracy tracks state accuracy exactly
in both rows (0.482 = 0.482; 0.760 = 0.760), so **spans are not the limiter —
the state decision is.** For `absent`, 376 of 413 fields were *presented*: the
model is not withholding, it is confidently asserting the wrong state.

**(b) Span failure with correct state — `conflicting`.** State is 80% right
while spans are 57.2%. `CONFLICTING` requires **two** distinct spans
(`nano_ai/contract.py` rejects duplicates in both offsets and normalized text),
which is a strictly harder retrieval problem than the single span every other
state needs.

**Consequence:** "fix abstention calibration" is the wrong instruction. Absent
and uncertain need a better *state decision*; conflicting needs *multi-span
retrieval*. An intervention aimed at one will not move the other, and H6 is the
evidence — its residual acted on boundary queries (spans) and moved uncertainty
+11.2 while dropping absence −19.6.

## 2. Hypothesis under test

**H7:** absent/uncertain failures are a state-decision problem that a
dedicated, better-supervised state head can fix without touching the span
machinery — and the fix must not cost `supported` accuracy, which is 99.9% and
carries 3,837 of 5,000 fields.

Deliberately **out of scope**: `conflicting`. Its span failure is a different
mechanism and gets its own preregistration. Bundling them is exactly the error
that made H1–H6 hard to interpret.

## 3. Candidate interventions (one to be selected before freezing)

All are data- or objective-side; none adds a new head geometry, since H2 (direct
pointer supervision) and H3 (evidence-query head) were both rejected.

1. **Class-weighted state loss.** `absent` is 413/5,000 (8.3%) and `uncertain`
   250/5,000 (5%); `supported` is 76.7%. Existing `STATE_CLASS_WEIGHTS` in
   `train_pointer.py` are already a lever and were never swept.
2. **State-decision curriculum.** Fit-set rebalancing toward the three minority
   states, holding architecture, optimizer, steps, and evaluator fixed — the
   H4/H5 pattern, which is data-only and therefore cheap.
3. **Denial-evidence supervision for `absent`.** The contract requires a
   *positive denial span* for ABSENT; whether the training target actually
   teaches that distinction is unverified and must be checked before this option
   is selectable.

## 4. Preregistered gates (frozen before any run)

Primary, on the sealed development partition, one shot, staged stop as always:

| gate | H6 baseline | H7 requirement |
|---|---|---|
| **absent joint** | 199/413 (48.2%) | **≥ 289/413 (70%)** |
| **uncertain joint** | 190/250 (76.0%) | **≥ 200/250 (80%)** |
| **supported joint** | 3,120/3,837 (81.3%) | **≥ 3,120 — no regression** |
| **overall joint** | 3,901/5,000 (78.0%) | **≥ 3,901 — no regression** |
| conflicting joint | 143/250 | reported, **not gated** (out of scope) |
| decode failures | 0 | ≤ 10/1,000 |

**ACCEPT** iff all four gated rows hold. Any regression in `supported` or
overall rejects outright: 3,837 fields cannot be traded for 663.

The absent threshold (70%) is set at roughly the midpoint between H6's 48.2%
and its own span accuracy of 92.7% — i.e. capturing about half of the headroom
the spans already demonstrate is reachable. It is a judgement, fixed here
before measurement, and it is not to be moved afterward.

## 5. Preconditions before this may be authorized

- [ ] Option selected from §3 and this document re-frozen with it named
- [ ] Option 3's premise verified (does the target teach denial evidence?)
- [ ] H5 anchored through the same decomposition, so the pattern has two points
- [ ] Platform: Kaggle free tier (it ran the whole H-cycle for $0)
- [ ] Owner authorization

## 6. Why not the alternative

**D-b (slot-diversity × the winning corner)** targets the residual copying gap
in the allergy slot. That gap is real, but held-value copying **passes its gate**
(2,277/2,987) and was never what rejected H5 or H6. D-b is the better *paper*
experiment; D-a is the better *program* experiment, because it attacks the wall
that has now stopped two consecutive hypotheses.
