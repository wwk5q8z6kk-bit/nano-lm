# Research Program — from a copying gap to a theory of abstraction

*Written 2026-07-18. The organizing frame for the nano-lm work: every experiment is a
consequence of this program, not the other way around. Paper 1 is the first test of a
broader idea, not the destination.*

## Mission

**Understand when and why language models transition from memorizing surface forms to
learning reusable representations.** The held-out *value-copying* failure we measure is
one observable of that deeper phenomenon — a window into representation learning, not an
isolated benchmark.

## Long-term objective

A **predictive theory** of how representational capacity, optimization, and data interact
to produce *abstraction* instead of *memorization* — stated well enough to predict, for a
new (size, stack, objective), whether held-out copying will be near-total or near-solved.

## The scientific question

Not "does the copying gap shrink?" but **"what governs the transition from copying to
abstraction?"** The first framing yields one number; the second generates hypotheses.

## Core hypothesis (refined 2026-07-19 by the Pythia fieldwise data)

A **capacity/representation threshold** exists. Below it: memorization and literal copying
dominate, representations are local, and held-out values fail to copy (we measure a
near-total ~80–87-pt value-level gap at 3–10M). Above it: abstraction becomes cheaper,
reusable copy/retrieval features emerge, and the held-out gap collapses (single digits by
Pythia-160M). Paper 1 is *evidence consistent with* this hypothesis — not proof, because
the nano→Pythia step also changes the training stack.

**Refinement — the threshold is per-slot, set by training diversity.** The clean ladder
(2026-07-19) shows the transition is not one capability switching on: Pythia solves the
~190-value complaint slot and the 18-value medication slot outright (clean gap 0.0) yet
fails the 5-value allergy slot *totally* (100% at 410M — identical to the 3M anchor).
Copy-vs-classify is settled slot-by-slot: a slot whose training-value diversity is low
enough to memorize as a closed set never induces a copy mechanism, at any tested scale.
This upgrades the hypothesis from "capacity buys abstraction" to **"abstraction is
induced per-slot when diversity makes memorization uncompetitive, given sufficient
capacity"** — and it makes P4 concrete: vary slot diversity at fixed scale and find the
diversity threshold curve; predict the allergy failure persists in much larger models on
this recipe (a cheap frontier-model external-validation test).

## Null hypotheses (stated up front)

- **H0a** — the gap is independent of model size.
- **H0b** — the gap is entirely a training-stack effect (data/tokenizer/architecture/FT).
- **H0c** — the gap is an evaluation artifact.
- **H0d** — the gap disappears under proper measurement.

Status: **H0c and H0d are substantially rejected** — the multi-instance instrument, the
determinism cross-checks, and the dilution correction each made the phenomenon *sharper*,
not smaller (single-instance was a biased hard draw; the clean value-level gap is ~80–87,
larger than the diluted 18). **H0b is the live one** — the OWNSTACK_160M experiment (below)
is designed to test it. **H0a** is only partially addressed (3M→10M flat within-stack).

## Theory tree (separate theory from evidence; each arrow needs its own evidence)

```
capacity ──▶ representation ──▶ retrieval/copy circuits ──▶ held-out copying ──▶ the gap
   (P2)          (P3)                  (P3)                       (P1 ✓)          (P1 ✓)
```

## Paper roadmap

| Paper | Claim | Question | State |
|---|---|---|---|
| **P1** | Observation — the gap collapses; near-total on real held-out values | *What is the phenomenon?* | **submission-ready** (Reviewer-#2 clean; workshop/Findings) |
| **P2** | Causality — scale vs. training stack | *Does scale alone explain it?* | kernel built (`trajectory/kaggle_ownstack_160m.py`), pre-registered, awaiting a T4 run |
| **P3** | Mechanism — *why?* | *Which circuits replace copying?* | design drafted (`trajectory/PREREG_stageM.md`): the non-gated within-stack *failure* mechanism Q(M) is specced + locally runnable; the gated scale-collapse contrast is a post-P2 slot |
| **P4** | Generality | *Does it hold across data/arch/tokenizer/objective?* | future (Stage F/O/R matrix, paper §8) |
| **P5** | Theory | Unified capacity→abstraction account with predictive power | the end goal |

## The experimental ladder (every experiment answers exactly one question)

> **Question → Prediction → Measurement → Decision.**

Worked example (P2): *Does scale alone explain the collapse?* → Prediction: an own-stack
160M, same recipe, also collapses (gap → single digits). → Measurement: diluted + clean
gap on m0–m4. → Decision (pre-registered): gap ≥14 ⇒ STACK-dominant; ≤6 ⇒ scale plausible
within the family; 6–14 ⇒ add 40M/80M. No experiment runs without this four-line spec.

## Measurement principles (discovered the hard way; now non-negotiable)

1. **Consistency** — one instrument across all rungs (the single→multi-instance fix).
2. **Reproducibility** — content-addressed inputs, frozen tags, byte-exact re-score checks.
3. **Power** — mean ± across-instance SD; enough held items that a few hard tokens don't
   swing the estimate.
4. **Calibration** — the metric measures what it names (the diluted→clean value-level fix;
   the template-vs-value control).
5. **Independent validation** — adversarial audit + a fresh-eyes Reviewer pass before any
   claim ships (AAEA audit and the T1.5 review both caught real errors here).

## Decision framework (gate every proposed experiment)

Run it only if it does at least one of: **improve measurement · test causality · test
mechanism · test generality.** If it does none, it does not run — this is what keeps the
program from drifting into infrastructure or one-more-model churn.

## Success ladder (do not define success as "publish")

- **L1 — interesting observation.** ✅ reached (the gap; the field-localization).
- **L2 — robust empirical law.** ~here (consistent instrument, near-total clean gap,
  measurement lessons) — completed by P1 submission + P2's within-stack curve.
- **L3 — causal explanation.** P2 (scale vs. stack).
- **L4 — mechanistic explanation.** P3 (Stage M).
- **L5 — predictive theory.** P5 — the destination.

## Resource allocation (guard against infrastructure-heaviness)

≈ **40% writing · 30% experiments · 20% theory · 10% infrastructure.** Theory is
under-weighted in most projects and is the highest-leverage under-investment here.

## Pointers

- Tactical roadmap + dates: `~/.claude/plans/calm-frolicking-newell.md` (O1–O5, M1–M5).
- P1 manuscript: `papers/paper1_draft.md`; running audit: `papers/writing_audit.md`.
- P2 design + kernel: `trajectory/PREREG_ownstack_160m.md`, `trajectory/kaggle_ownstack_160m.py`.
- P4 axes matrix: paper §8 (Stage F/O/R/M).

## The one-sentence annual target

> *We identify a reproducible transition in held-out copying behavior between nano-scale
> and larger language models, show it is robust to measurement artifacts, disentangle
> scaling from training-stack effects, characterize its mechanistic basis, and develop a
> predictive framework linking capacity to abstraction* — a statement that progresses
> observation → causality → mechanism → theory, which is the whole program in one line.
