# Decision record — Question Before Architecture

**Date:** 2026-08-26
**Decider:** owner
**Status:** in force
**Governs:** `docs/NANO_VNEXT_MASTER_SPEC.md` §20 (neural architecture research
program) · §22 (evaluation strategy) · §25 (approved next experiments)
**Spec read at:** `work/e-delimit-result` @ `d26fd33`
**Relation to prior record:** extends
[`2026-08-26-ratify-consolidation-as-canonical-state.md`](2026-08-26-ratify-consolidation-as-canonical-state.md)
— same day, same owner, **not a contradiction**. See § *Continuity* below.

---

## The governing rule

> Before proposing, implementing, or authorizing any architecture change or
> experiment, first establish that the experiment is **capable of answering the
> question being asked**.

The causal chain is explicit and directional:

```
observed failure → hypothesis → targeted mechanism → controlled manipulation
→ preserved invariants → measurable prediction → falsifiable result
```

and never:

```
interesting architecture → implementation → benchmark score → post-hoc explanation
```

No architecture is added because it is theoretically attractive, biologically
inspired, physics-inspired, fashionable, or intuitively plausible. **A mechanism
earns its place by resolving a measured failure mode under a controlled
experiment.**

Three consequences that are ordering rules, not preferences:

1. **Do not optimize the architecture before establishing that the instrument
   can discriminate the architectural hypothesis.** If the instrument cannot
   distinguish competing explanations, improve the instrument before expanding
   the architecture.
2. **If a manipulation changes the thing being measured rather than the
   mechanism intended to affect it, the result is not evidence for or against
   the hypothesis.** Declare VOID, diagnose the failed invariance condition,
   redesign.
3. **Retrieval and delimitation cannot be tested by the same manipulation.** If
   a model fails because it cannot retrieve the relevant evidence, test
   retrieval mechanisms. If it retrieves correctly but fails to delimit, test
   delimitation mechanisms. A manipulation that changes retrieval cannot
   simultaneously serve as a clean test of delimitation. This generalises the
   arm-B VOID (`4de84c18`) past the span-port line.

---

## The eighteen-field standard

Every experiment states all eighteen **before it runs**. This supersedes the
twelve-field chain ratified earlier the same day in §20; nothing in that chain
is dropped.

| # | Field | What it must answer |
|---|---|---|
| 1 | **Product question** | What real Nano capability or product requirement is ultimately being improved? |
| 2 | **Scientific question** | What specific uncertainty about the system is being resolved? |
| 3 | **Instrument** | Which benchmark, task, workload, or capability line actually measures that question? |
| 4 | **Measured bottleneck** | What failure mode has *already been observed* on that instrument? |
| 5 | **Hypothesis** | What mechanism addresses that bottleneck, and what causal prediction does it make? |
| 6 | **Baseline / control** | What is the comparison, and what must remain unchanged? |
| 7 | **Manipulation** | What exactly changes between conditions? |
| 8 | **Invariance requirements** | What must remain statistically or operationally equivalent for the experiment to answer its question? |
| 9 | **Confound analysis** | What alternative explanations could produce the observed result, and how does the design distinguish them? |
| 10 | **Outcome measures** | What quantities are measured — capability, quality, compute, latency, memory, cost? |
| 11 | **Decision rule** | What constitutes support, refutation, null/inconclusive, or methodological failure? |
| 12 | **Kill condition** | What observation makes the experiment *incapable of answering its question*, requiring VOID rather than interpretation? |
| 13 | **Falsifier** | What result would genuinely count against the proposed mechanism? |
| 14 | **Authorization** | Exploratory or confirmatory, and does it hold the required preregistration and experiment-scoped authorization? |
| 15 | **Provenance** | Can another researcher reconstruct exactly which code, model, data, configuration, seed, analyzer, and evaluation rules produced the result? |
| 16 | **Resource accounting** | What compute, latency, memory, energy, and monetary cost are required, and is the experiment proportionate to the uncertainty it resolves? |
| 17 | **Reproducibility** | What must be repeated across seeds, instances, machines, or analyzers before the result is reliable? |
| 18 | **Interpretation boundary** | What does this establish, what does it not establish, and what claims remain untested? |

### Continuity — how the eighteen relate to the twelve

Twelve carried forward, six genuinely new. **No field was removed.**

| Twelve-field chain (§20, ratified 2026-08-26) | Becomes | Change |
|---|---|---|
| hypothesis | Hypothesis (5) | unchanged |
| instrument | Instrument (3) | unchanged |
| measured bottleneck | Measured bottleneck (4) | unchanged |
| manipulation | Manipulation (7) | unchanged |
| invariance condition | Invariance requirements (8) | pluralised — more than one capacity may need holding |
| baseline | Baseline / control (6) | widened — control is named separately from baseline |
| expected benefit | Outcome measures (10) | widened — capability *and* compute, latency, memory, cost |
| cost | Resource accounting (16) | widened — adds energy and an explicit **proportionality** test |
| falsifier | Falsifier (13) | unchanged |
| preregistered decision rule | Decision rule (11) | widened — must name all four outcome classes, not only pass/fail |
| authorization | Authorization (14) | widened — exploratory vs confirmatory declared up front |
| provenance | Provenance (15) | unchanged; **reproducibility split out** as its own field |

| New field | Why it is new, and what it prevents |
|---|---|
| **Product question** (1) | The spec's §2 optimization target was never a per-experiment field. Prevents scientifically tidy experiments that improve nothing the product needs. |
| **Scientific question** (2) | Distinct from hypothesis. A hypothesis is a proposed answer; the scientific question is the uncertainty. Naming both prevents a mechanism in search of a question. |
| **Confound analysis** (9) | **Continuity, not novelty** — this is the §25 readiness gate's question 4 (*what competing explanations does the experiment distinguish?*) promoted from gate to preregistered field. It ranks candidates. |
| **Kill condition** (12) | Previously implicit in "breach ⇒ VOID". Now a named field, and **distinct from the falsifier**: a falsifier is a result that counts against the hypothesis; a kill condition is a result that means the instrument failed and the hypothesis was never tested. |
| **Reproducibility** (17) | Split from provenance. Provenance is *can it be reconstructed*; reproducibility is *what must repeat before it is believed*. The native30 seed-spread null is why. |
| **Interpretation boundary** (18) | What the experiment does **not** establish. Prevents scope creep between a result and the claim made from it. |

---

## Experiment verdicts

Every experiment closes in exactly one of six states. **Never convert a VOID
into a negative result because the observed number is unfavourable.**

| Verdict | Meaning |
|---|---|
| **SUPPORTED** | The preregistered evidence supports the hypothesis. |
| **REFUTED** | The experiment validly tested the hypothesis and its prediction failed. |
| **NULL / INCONCLUSIVE** | A valid experiment that did not distinguish the alternatives. |
| **VOID** | The experiment's assumptions or invariance requirements failed; the result cannot answer the question. |
| **PENDING** | Required validation or replication has not yet occurred. |
| **NOT AUTHORIZED** | The experiment has not been permitted to run. |

**A failed experiment is not necessarily a failed hypothesis.**

### Reconciliation with the ledger's claim buckets

These are **experiment verdicts**, not claim buckets. They do not replace the
`docs/RESEARCH_STATUS.md` vocabulary that `docs/PROJECT_AUTHORITY.md` §1 makes
canonical. The two compose: a verdict is what one run concluded; a bucket is the
standing of a claim after all runs bearing on it.

| Experiment verdict | Moves the claim toward bucket |
|---|---|
| SUPPORTED | `SUPPORTED BUT NOT CONFIRMED` → `ESTABLISHED` only after replication (field 17) |
| REFUTED | out of `HYPOTHESES`; recorded as refuted, not deleted |
| NULL / INCONCLUSIVE | stays in `HYPOTHESES` — the claim is untested, not weakened |
| VOID | `VOID RESULTS`; the hypothesis returns to `HYPOTHESES` **untouched** |
| PENDING | `PENDING REVALIDATION` |
| NOT AUTHORIZED | `NOT AUTHORIZED` |

`RESEARCH_STATUS.md` remains canonical for which bucket a claim is in, and is
never canonical for a number.

---

## Three substantive corrections

These change existing text rather than adding to it.

**1. Cheapness is not evidential validity.** The program should prefer the
smallest experiment that discriminates between competing explanations. But cost
is not the criterion for evidential validity:

> A \$0 local run can still be a confirmatory experiment requiring
> authorization. An expensive run can still be scientifically worthless if its
> invariants are violated.

This does not relax `docs/ACTIVE_NOW.md`'s compute posture — it clarifies that
**free ≠ exploratory**. §20 already noted authorization is "not implied by the
run being free"; that is elevated from a parenthetical to a rule, and it
qualifies §26's observation that the four E-DELIMIT runs cost \$0.

**2. Foundational questions are separated from the architecture ladder.** If the
capability floor may itself be contaminated by representation, tokenizer,
context-length, evaluation, or instrumentation constraints, those foundational
questions are resolved **separately** rather than allowed to contaminate
architectural claims.

This reclassifies the current top-ranked candidate. §24 hypothesis 1 — *is the
native30 capability floor a tokenizer-context artifact?* — is a **foundational
instrument question, not a rung on the architecture ladder**. It is prerequisite
to architectural claims at that scale, and it does not itself test any
mechanism.

**3. Retrieval and delimitation are separate instruments.** Stated in full under
*The governing rule* above. The span-port and scribe lines have different
measured bottlenecks (§23); neither line's manipulation may be used to answer
the other's question.

---

## What this record does not change

- **The consolidation ratification stands.** Every settled item in
  [`2026-08-26-ratify-consolidation-as-canonical-state.md`](2026-08-26-ratify-consolidation-as-canonical-state.md)
  remains in force.
- **No result is reclassified by this record.** E-DELIMIT arm B remains VOID on
  the evidence already recorded in
  `research/negative_results/RESULT_2026-08-25_E_DELIMIT.md`, for exactly the
  reason this standard now generalises. H5 remains untested.
- **History is not rewritten.** `research/preregistrations/PREREG_E_DELIMIT.md`
  predates this standard and is **not** retrofitted to it. Its missing
  LOCATED-invariance check is the documented cause of the arm-B VOID and is
  evidence, not a defect to be edited away. (§3 principle 8.)
- **The A0→A7 ladder remains not approved.** The 5–10M mechanism tier remains
  not automatic.
- **Authorization status is unchanged: no experiment is authorized.**

---

## Appendix — the vocabulary applied to two open items

Worked examples, to fix the meaning of the verdicts. **This appendix is not
canonical for claim buckets.** `docs/RESEARCH_STATUS.md` is
(`docs/PROJECT_AUTHORITY.md` §1), and the ledger rows below are for its owning
session to write.

**1. Cross-regime selection probe** — run `d222465e`, commit `3205a64`. Currently
**absent from `RESEARCH_STATUS.md`, therefore unclassified**, which is the gap
this example exists to flag.

The run is valid: the instrument worked, task and statistic were held fixed, the
model axis was crossed, and a SEEN control was present. Measured HELD first-token
top-1 **11/28 = 39.3%**, against a SEEN control of **102/127 = 80.3%** — a
held-vs-seen gap of **+41.0 points**. The two pre-registered anchors were *held
near 92%* (selection is a small-trunk property) and *held near 21%* (selection is
task-intrinsic).

Two different verdicts apply to two different propositions, and collapsing them
would lose the result:

- The claim that Qwen *"already exhibits the retrieval competence the nano trunk
  was measured to lack"* is **REFUTED**. The 95% Wilson interval excludes 92%.
  This was the author's own premise, and it was tested and failed — a valid
  refutation, not a VOID.
- The **two-regime dichotomy itself is NULL / INCONCLUSIVE**. 39.3% is near
  neither anchor; the interval excludes 21% as well as 92%, so the experiment did
  not distinguish "small-trunk property" from "task-intrinsic". The registered
  third branch (*report as-is beside the control*) is what fired.

Interpretation boundary (field 18): this establishes that content-addressed
selection does not generalise OOD for Qwen either, and that **both** regimes
carry a selection deficit. It does **not** establish where the deficit comes
from. The delimitation measurement is untouched by it.

**2. native30 capability floor.** Already correctly split in §23 and the prior
ratification; restated here in verdict terms because the split is exactly what
the vocabulary is for. The capability floor is **SUPPORTED in its permissible
form only** — *30M at 1800 steps with a tokenizer that cannot fit 83% of eval
prompts* is below the floor. The arms are **NULL / INCONCLUSIVE**
(`NOT_SEPARATED`; effect 0.0133 below seed spread 0.04). Wave 1 is **VOID** (a
non-causal decoder — a false null, and it must not be banked). The claim that
the wave ran cleanly under the integrity gate is **PENDING** the
`reval30_*_fixed_*` re-run.

Per correction 2 above, resolving the tokenizer confound is a **foundational**
question, not an architecture rung.

## Objective

The objective is not to discover a particular architecture. It is to discover
the **smallest set of empirically justified mechanisms** that produces the
required Nano capabilities at the best verified capability per unit of compute,
latency, memory, energy, and cost.

> Earn every component. Preserve every invariant. Predefine every decision.
> Record every limitation. Kill experiments that cannot answer their own
> question.

Every experiment must make Nano more **scientifically legible**, not merely more
architecturally elaborate.
