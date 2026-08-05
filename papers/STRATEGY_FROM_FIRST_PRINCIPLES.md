# Strategy from first principles — rebuilt 2026-08-05

Written in response to a proposed 10-phase, ~40-dataset plan to train a
multi-capability assistant. I am rejecting most of it. This document explains
why on evidence, and rebuilds from the ground up.

---

## 1. Executive summary

**The proposed plan optimizes the one axis where this project cannot win, and
ignores the axis where it already leads.**

It proposes teaching a 160M-parameter model to write code, do math, read
charts, generate Mermaid, draft clinical notes, use tools, and act as an agent.
Every frontier lab does all of that better, with 100–1000× the parameters and
budgets this project will never have. Competing there is a guaranteed loss.

Meanwhile the thing nobody ships well — and that this repository has three
weeks of frozen evidence about — is **provable grounding**: output where every
asserted value carries a source span, absence is never inferred from silence,
and the system withholds rather than invents. `fabric` drove presented error to
**0.0% at 81.5–91% coverage with zero correct answers lost** across 24 cells.
That is a real, measured, unusual result.

The strategic inversion: **stop trying to make the model capable; make any
model's output verifiable.** The verification layer wraps a frontier API when
privacy allows, and a small local model when it does not. The 160M model stops
being the product and becomes one interchangeable backend for the private path.

Under that framing, ~90% of the proposed dataset work is unnecessary, and the
two things that actually matter are both already half-built and cost nothing.

---

## 2. Critical issues with the proposed plan

**(a) It contradicts this project's strongest measured finding.** E1 recorded a
pre-registered utility where the *deterministic, non-generative* solver M1
scored **0.999** and the best generative reference scored **0.925**. Verdict:
KILL. For structured extraction from constrained dialogue, the generative
substrate lost to rules. The plan's entire premise is "train a generative model
to do everything" — an experiment this project already ran and lost.

**(b) 160M cannot do what the plan describes, and the plan half-admits it.**
Its body proposes ten capability phases; its final paragraph concedes the model
"won't match the breadth of much larger frontier models." Those two statements
cannot both be acted on. The current anchor is d=192, L=6, V=4098, ctx=512.
Even at 160M, agentic tool-use and multimodal reasoning are not reachable
capabilities — they are aspirations attached to the wrong artifact.

**(c) It recommends a dataset this project has formally prohibited.** MIMIC-IV
appears twice. `papers/TRAINING_DATA_REGISTRY.md` records it as a **standing
prohibition**: its DUA forbids sending the data to third-party services, so the
popular "prompt an LLM API to reverse-engineer dialogues from MIMIC notes"
recipe — which the plan implicitly endorses — violates it. This is a legal
exposure, not a preference.

**(d) Licensing is treated casually elsewhere too.** LAION-5B (withdrawn over
CSAM findings), SAMSum (non-commercial), Common Crawl derivatives (ToU
obligations). This project has a standing **open-source-only rule** with pinned
revisions, per-file hashes, and recorded licenses. A plan that lists forty
datasets without a license column is unusable here.

**(e) It addresses none of the project's measured blockers.** The two things
actually stopping progress are: the model's **state classification** on absent
(0.482 accuracy while span accuracy is 0.927) and uncertain (0.760 vs 0.972),
and the product's **over-abstention** (3/10 useful on the only real corpus).
None of the ten phases touches either. Adding chart data does not fix a model
that picks the wrong epistemic state while looking at the right evidence.

**(f) Phase ordering inverts risk.** Memory arrives at phase 8, multimodal at 9,
after nine phases of investment. The cheapest disconfirming experiments belong
first, not last.

**(g) It is a generic plan.** Nothing in it is specific to this project. That is
the deepest problem: it could have been written without reading a single file
in this repository, and it was.

---

## 3. First-principles analysis

Strip the assumptions and ask what is actually true.

**What is the real goal?** Not "a model." An assistant whose output can be
*trusted* on documents that matter — clinical, legal, financial — where a
confident wrong answer is worse than no answer.

**What must be true for that to happen?**
1. Every asserted value is bound to evidence a human can check.
2. The system withholds when it cannot ground — and withholds *no more than
   necessary*, or it is useless.
3. It runs where the data is allowed to be.
4. It is cheap enough to run on every document, not just sampled ones.

**Which of those requires a trained model?** Only (3), and only when data
cannot leave the building. Requirements 1, 2 and 4 are *architecture*, not
parameters — which is exactly why E1's rules-based solver won and why fabric's
verifier drove error to zero regardless of which model generated the text.

**What is genuinely scarce?** Not capability — frontier APIs sell that cheaply.
What is scarce is *warranted trust*: a machine-checkable claim that an output
is grounded. Nobody ships that well. Hallucination, not capability, is what
stalls adoption in regulated documentation.

**Therefore the correct architecture is an inversion of the proposed one:**

```
          ┌─────────────────────────────────────────┐
          │        VERIFICATION LAYER (the product)  │
          │  spans · abstention · contradiction ·    │
          │  structure validation · audit trail      │
          └───────────────┬─────────────────────────┘
                          │  model-agnostic seam
        ┌─────────────────┼──────────────────┐
        │                 │                  │
   deterministic     small local model   frontier API
   extractor         (private path)      (when allowed)
   (E1 winner)       (Nano, 160M)        (best capability)
```

The model is a **swappable backend**. The layer is the asset. This also means
capability improvements from frontier labs become *our* improvements for free,
instead of a treadmill we lose.

**What does the 160M model become?** The privacy path only, doing the one thing
a model that size can do well: constrained structured extraction with grounding.
Not code, not math, not charts, not agents.

---

## 4. Optimized master plan

### Principle: two workstreams, not ten phases

**Workstream A — make the verification layer usable (the product).**
Everything here is free, local, and already half-built.

| # | Work | Why it is the highest leverage available |
|---|---|---|
| A1 | **Fix over-abstention** | 3/10 useful is the measured blocker. One mechanism already fixed (W-ABSTAIN-2); the 2-token conjunction remains. Nothing else matters until this moves. |
| A2 | **Ship the model-agnostic seam** | `lm/probe.py:LMBackend` is a Protocol with one deterministic stub. Making it real is what turns a research artifact into a product — and it is the *only* way frontier capability enters this system. |
| A3 | **Generalize contradiction detection** | Currently three hardcoded fields (`ttl_seconds`, `metformin_dose_mg`, `sample_n`). The marquee differentiator is inert on anyone else's documents. |
| A4 | **Structure validation** (tables, Mermaid) | The one item worth keeping from the proposal — but as *validators*, not training data. A Mermaid compile check is 20 lines and works with any backend. This is where "charts and diagrams" actually belongs. |
| A5 | **Real-corpus dogfood** on open-licensed documents | The only honest measure of whether any of this works. |

### Workstream B — the model, narrowed hard

| # | Work | Why |
|---|---|---|
| B1 | **H7: fix state classification** | absent 0.482 vs span 0.927 — the model finds evidence and picks the wrong state. This is the measured wall that stopped H5 and H6. Free on Kaggle. |
| B2 | **Then, and only then, consider scale** | Rung-1 remains deferred: the residual copying gap traces to a five-value training pool, a data-composition defect, not a scale deficit. |

**Everything else in the proposal is deleted.** See §8.

---

## 5. Roadmap

**Phase 1 — Make it usable (weeks, $0).** A1 over-abstention, A2 model seam,
A4 structure validators. Milestone: *a person who is not the author points it
at a folder and gets verified output they'd act on.*
Dependency: none. This is entirely unblocked today.

**Phase 2 — Prove it on documents we didn't write ($0).** A3 generalized
contradiction detection, A5 open-licensed dogfood corpus. Milestone: *U measured
on a corpus this project did not author, with coverage reported beside risk.*
Dependency: Phase 1.

**Phase 3 — The private path (free on Kaggle).** B1/H7 state classification.
Milestone: *the local model clears the epistemic-state gates that stopped H5 and
H6.* Dependency: none technically — it can run in parallel with Phase 1.

**Phase 4 — Scale, only if earned (≤$150).** Rung-1, and only if Phase 3's
diagnosis says capability rather than data composition is the constraint.

**Phase 5 — Publish.** Paper α is camera-ready and unsubmitted since July 31.
The selective-grounding instrument is a second, narrower contribution.
Dependency: an owner decision, not engineering.

---

## 6. Quick wins (do these first)

1. **Submit Paper α.** Camera-ready, council-approved, sitting unsubmitted for
   five days. Highest value-per-minute action available and it is not
   engineering.
2. **Fix the 2-token conjunction** in `_relevant_claim`. Hours. Directly moves
   the measured blocker.
3. **Wire one real LM backend** behind the existing Protocol. A day. Converts
   the whole system from "research artifact" to "usable with any model."
4. **Add a Mermaid/table compile validator.** ~20 lines, reusable as evaluator,
   data filter, and RLVR reward — captures the proposal's one good idea at 1%
   of its cost.
5. **Send the RunPod ticket.** Already drafted; unblocks paid compute if ever
   needed.

---

## 7. Risks and trade-offs

- **The inversion cedes capability to frontier labs.** Accepted deliberately:
  we cannot win there, and wrapping them means their improvements accrue to us.
- **The verification layer may not generalize** beyond synthetic clinic
  dialogue. Every headline number here is closed-world. Phase 2 exists to find
  out, cheaply, before more is invested.
- **Over-abstention may be structural, not a bug.** If grounded output is
  inherently low-coverage on real documents, the product thesis weakens. That is
  the single most important open risk, and Phase 1–2 are designed to expose it.
- **A frontier-API backend breaks the local-first privacy claim** for that path.
  Mitigation: it is a *choice per deployment*, and the local path remains.
- **Not training broadly means the model stays narrow.** Correct, and intended.

---

## 8. What to eliminate

Delete from the plan, with reasons:

- **Code, math, agents, tool-use, multimodal, speech training** — unreachable at
  this scale and available from any frontier backend through the seam.
- **MIMIC-IV / MIMIC-CXR** — prohibited by DUA; legal exposure.
- **LAION-5B** — withdrawn over CSAM findings.
- **SAMSum** — non-commercial license; incompatible with the approved posture.
- **The 10-phase structure** — sequences investment before disconfirmation.
- **"Generate 100,000+ Mermaid examples"** — a validator is 20 lines and works
  immediately; the dataset is months and only helps a model too small to use it.
- **Commercial labeling vendors** — nothing here is blocked on annotation.
- **Rung-1 pretraining, for now** — deferred on this project's own evidence.

---

## 9. Open questions

1. **Does grounded extraction stay useful at acceptable coverage on documents
   we did not write?** The whole thesis rests on this and it is unmeasured
   outside fixtures.
2. **Is the 2-token conjunction the last over-abstention mechanism, or the
   second of many?** n=10 cannot tell us.
3. **Which backend does the seam target first** — a local 3–7B model (keeps the
   privacy claim intact) or a frontier API (best capability, weakest privacy)?
4. **Does the training target teach denial evidence for ABSENT at all?** H7's
   design depends on the answer and it is unverified.
5. **Who is the first non-author user?** Without one, "usable" stays a guess.

---

## The 20% that produces 80%

**Fix over-abstention, and ship the model seam.** Those two turn an inspectable
research artifact into something a person can actually use, cost nothing, and
are both already half-built. Everything else in the proposed plan is either
unreachable at this scale, already refuted by this project's own evidence, or
legally prohibited.

---

## Appendix — "Dataset-first vs distillation vs hybrid?"

A follow-up proposal framed this as the key design decision and recommended a
hybrid mix (50–60% general text, 15–25% instruction, 15–20% domain, 10–20%
synthetic). **Under the inverted architecture the question mostly dissolves,
and the recommended mix is wrong for this project.**

### Why the question dissolves

Distillation exists to move a capability you cannot afford *into* a model you
can run. But the model-agnostic seam means the capability does not need to move
at all — when policy allows, you **call the frontier model and verify its
output**. Distilling frontier behavior into 160M to avoid an API call is paying
a large price to get a much worse version of something already available.

Distillation is therefore only relevant to the **privacy path**, where data
cannot leave the building. And on that path the requirement is not
conversational style, formatting, or tool-calling patterns — it is *narrow
structured extraction with grounding*. Almost everything distillation is good
at (per the proposal's own list: response style, formatting, organization) is
irrelevant to it.

### Why the recommended mix is wrong here

**E1 already ran this experiment.** On a pre-registered utility, the
deterministic solver scored **0.999** against the best generative reference at
**0.925**. Distilling generative behavior into a small model, to perform a task
where rules already beat generation, inverts the evidence.

**Capacity allocation.** The proposal's own strongest argument — a small model's
capacity should go to the skills you care about — argues *against* its own
recommendation. Spending 50–60% of a 160M model's capacity on general web text,
so it can be mediocre at everything, is the opposite of concentrating capacity
on the one task it must do locally.

**Teacher-output licensing is a real constraint, not a footnote.** Major
providers' terms restrict using outputs to train competing models. This project
has a standing open-source-only rule with pinned revisions and recorded
licenses; "check the teacher's terms" is not a note in a table here, it is a
gate. Open-weight teachers are the only clean option.

### Where distillation genuinely fits — one narrow, valuable place

**Generating training data for structured transformations, filtered by a
validator.** Text → Mermaid, table → chart spec, transcript → structured note.

The reason this is safe is not that the teacher is trustworthy. It is that the
**validator is the filter**: a generated Mermaid block either compiles or it
does not; a markdown table either parses or it does not; a claimed span either
matches the source text or it does not. Rejection-sample the teacher, keep only
rows that pass, and the teacher's errors never enter training.

That is the same machinery as `nano_ai/selective.py` and the fabric gate, reused
as a data filter — and it is the honest version of "use synthetic data for
structured outputs." Recorded already in `papers/NANO_V2_AMBITION.md` as the
rejection-sampling delta.

### The answer, stated plainly

- **Not dataset-first.** Broad corpora buy breadth this model cannot hold.
- **Not distillation-for-capability.** The seam makes it unnecessary, and the
  licensing is a genuine constraint.
- **Yes to distillation-under-validation**, narrowly, to manufacture verified
  structured-transformation pairs — where the validator, not the teacher, is
  the source of trust.

Build the validator first. It is 20 lines, it works with any backend today, and
it is the precondition for the only kind of synthetic data worth generating.
