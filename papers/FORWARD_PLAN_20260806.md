# Forward plan — first-principles, post-control

**2026-08-06.** Written after the LoRA control overturned the previous plan.
Supersedes `DECISION_MEMO_20260806.md` §2 and the recommendation in
`PLAN_20260805_SURFACE_ROBUSTNESS.md` §4.

---

## 1. Executive outcome

**Concluded.** Nano's epistemic-state failures are lexical-transfer failures, and
transfer is supplied by **pretraining**, not by finetune vocabulary breadth. A
3B pretrained base, LoRA'd on Nano's *exact* training partition, generalises to
unseen denial phrasings at 90.3% where Nano — same data, same task — reaches
60.0%.

**Recommended direction.** Stop training Nano from scratch. Initialise from the
smallest openly-licensed pretrained base that carries English lexical priors,
and port the span-pointing contract onto it.

**Confidence.** *Strongly supported* that pretraining supplies the transfer.
*Unknown* whether a base small enough to preserve the local-first thesis
supplies enough of it — that is the next experiment, and it is cheap.

**Most important unresolved risk.** The control measured **state only**. Nano's
product is state **plus grounded span**. No evidence yet shows a pretrained base
can produce spans under Nano's contract, and the tokenizer change makes that a
non-trivial port, not a swap.

---

## 2. The assumption that broke

First-principles restatement of the product: *turn a document into a structured
record where every asserted value carries the span that grounds it, abstain when
grounding fails, and run locally so private documents never leave the machine.*

Ask what that actually requires:

| requirement | is it fundamental? |
|---|---|
| runs locally | **yes** — it is the privacy claim |
| small enough for the target device | **yes**, but "small" is set by the device, never measured |
| emits evidence spans | **yes** — it is the trust claim |
| abstains under a calibrated rule | **yes** |
| **trained from scratch** | **no. Nothing requires this.** |

**"Small" and "from scratch" were conflated, and only one of them is a product
constraint.** Training from scratch was a research choice inherited from the
project's origin as a nanoGPT-scale study. Every result since — H1 through H6,
and the entire surface-robustness cycle — has been optimising within an
assumption that was never load-bearing.

This is the highest-value thing the cycle produced. It is also a warning: the
constraint survived six experiments without being questioned because it was
never written down as a constraint.

---

## 3. Hurdles, with root causes

**H-1. No lexical priors.** Root cause: from-scratch training. *Verified* by the
LoRA control. Fix: pretrained initialisation. **Not yet fixed.**

**H-2. The span contract does not survive a base swap.** Nano is a *pointer*
model over its own tokenizer (`sft/tokenizer.json`, sha `bae49648…`); its
evidence-query head predicts start/end token indices in that encoding. A
pretrained base brings its own tokenizer, so spans must be re-derived — either
by porting the pointer head onto the new encoder, or by generating spans as text
and re-locating them by string match (which `state_span.py` already does via
`_locate_unique_patient_span`). **This is the real engineering work, and the
control did not touch it.** *Unknown* which route is better.

**H-3. Evaluation vocabulary is ten strings.** Partly addressed: negspacy and
medspacy vendored (MIT, hashed, evaluation-only), surface harness built. Still
open: the *training* vocabulary is equally narrow, and any expansion must keep
train/eval lexicons disjoint and hashed **before** use or the instrument dies
silently.

**H-4. Two seeds; arm-level claims unsupported.** Blocked by design for the H6
family (`train_evidence_query.py` pins its own SHA and enforces the seed tuple).
Fix belongs in the next preregistration: ≥3 seeds declared up front, justified by
the measured 31.5-point OOD instability at Kendall τ = 0.00.

**H-5. `uncertain` is weak *in-distribution*** (39.8-point spread on trained
phrasings; 76.0% vs 43.6% across seeds on identical documents). Neither
pretraining nor vocabulary breadth obviously addresses a concept never learned.
Root cause *unknown*. Cheapest discriminating test: does the tuned 3B handle
hedges? The hedge arms exist; the probe does not yet run them.

**H-6. Everything is synthetic.** No claim transfers to real documents. The
product side is worse than unproven — wedge over-abstains (3/10 useful on real
documents). The open-licensed dogfood corpus is the only route.

**H-7. Systematic weakness: denominator degeneracy.** Four instances this cycle
(`fabric/slice.py:247`, DP-1's C1, the crossmodel probe v1, and its own v2
guard); three were mine. Mitigation now in place:
`nano_ai/tests/test_gate_degeneracy_safety.py` pins the pattern, and every new
probe carries a balanced control block. **This is a process fix, not a code fix,
and it will recur without the checklist.**

---

## 4. The decision-relevant unknown

Exactly one measurement separates the current plan from a committed one:

> **What is the smallest pretrained base that recovers most of the transfer?**

Everything between 13M (Nano, 60.0% held-out) and 3B (90.3%) is unmeasured. The
curve could saturate at 135M — in which case the local-first thesis survives
intact and Nano simply changes its initialisation — or it could climb to 3B, in
which case the product's device target has to be renegotiated against the
privacy claim.

No amount of reasoning substitutes for the curve, and the curve is cheap: same
LoRA recipe, same arms, minutes per point, locally, $0.

---

## 5. Plan

**P0 — the transfer curve ($0, hours). Critical path.**
LoRA the same `fit.jsonl` onto openly-licensed bases spanning the gap
(≈135M / ≈360M / ≈500M–1B / 1.7B), run the identical denial arms, plot held-out
accuracy against base size. *Gate:* the balanced control block must pass for each
point or that point is uninterpretable. *Exit:* the smallest base clearing a
stated held-out bar — and the bar is preregistered before the curve is plotted.

**P1 — hedge arms through the tuned model ($0, minutes).**
Answers H-5's discriminating question at near-zero cost: if the tuned 3B handles
hedges well, `uncertain` is also a transfer problem and P0 fixes it too; if not,
`uncertain` is a genuinely different failure needing its own hypothesis.

**P2 — span port design (days).**
Resolve H-2. Two candidate architectures, prototyped and compared, not argued:
(a) pointer head on the pretrained encoder; (b) generate span text, re-locate by
unique string match using the existing `_locate_unique_patient_span`. Route (b)
reuses machinery that already exists and degrades to abstention when a span is
not uniquely locatable — which is the correct failure mode for this product.
*This is the largest genuine unknown and it should be prototyped before P3.*

**P3 — preregister and run the initialise-from-pretrained experiment.**
Only after P0 names a base and P2 names a span route. Must fix: ≥3 seeds,
`surface_robust_accuracy` (min over arm means) as the primary gate, absolute
floors over fixed gold denominators, and the train/eval lexicon split hashed
before use.

**P4 — dogfood on real open-licensed documents.**
The only test of H-6, and the one that determines whether any of this matters.

**Deferred with reasons:** rung-1 from-scratch scale (§2 removes its rationale);
H7 state head (`RESULT_PER_STATE_DIAGNOSIS` — the state machinery reaches
95–100% on familiar wording); deterministic composition as a *decision* rule
(3% external recall), retained only as the escalation router, which is
evidence-backed at 4.3–5.0× error lift.

---

## 6. Adversarial review of this plan

**"The control is state-only; the whole conclusion may not survive spans."**
Correct, and it is H-2. The inference that joint ≈ state holds for `absent`
(99.4% of Nano's mislabelled fields had the right span) but not for
`conflicting` (span accuracy 0.572). **P2 is sequenced before P3 precisely
because this objection could invalidate the direction.**

**"A pretrained base may have seen clinic-like text; the transfer could be
contamination."** Partly. The documents are synthetic and unpublished, so there
is no train/test leak, but the *phrasings* are ordinary English the base has seen
— which is the mechanism being claimed, not a confound. It does mean the result
will not transfer to a domain whose vocabulary is genuinely absent from
pretraining.

**"Nano may no longer be justified if a 3B does the job."** Honest. A 3B-4bit is
1.7 GB and runs on a laptop. If the device target admits that, from-scratch Nano
is a research artifact. **This is a product-boundary question the evidence now
forces, and it belongs to the owner.** The engineering answer is P0: measure the
curve first, because if 135M suffices the question dissolves.

**"Four denominator bugs in one cycle suggests the process, not the people."**
Agreed — H-7. Every probe now ships a balanced control that runs first and gates
interpretation. That is a real structural fix; the checklist discipline is not.

---

## 7. Confidence ledger

| claim | status |
|---|---|
| Nano's state failures are lexical-transfer failures | **verified** (six axes, two seeds, factorial) |
| `conflicting` is not a structural failure | **verified** (sensitivity 10.2% < instability 17.3%) |
| Pretraining supplies transfer that finetune breadth does not | **strongly supported** (one control, one base, state-only) |
| A small pretrained base suffices for local-first | **unknown** — P0 |
| A pretrained base can carry the span contract | **unknown** — P2 |
| `uncertain` is a different failure | **plausible** — P1 |
| Any of this transfers to real documents | **unknown** — P4 |
