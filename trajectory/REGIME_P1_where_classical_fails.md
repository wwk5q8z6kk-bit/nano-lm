# P1 annex — Regime R★ (hardened)

*Product-path boundary. Decision lock: `trajectory/DECISION_P1_program_lock.md`.
Sequential plan: `papers/SEQUENTIAL_PIPELINE.md`. Does **not** authorize E4 runs
until Stage 3 (P2) freezes \(U_{R★}\).*

*Not part of Paper α. Gate 2 PASS recorded 2026-07-31 in `PIPELINE_GATE_LOG.md`.*

## Status

| Field | Value |
|-------|--------|
| Artifact class | Hardened regime definition (Stage 2) |
| Gate 2 | **PASS** — R★ non-empty and testable |
| Next | Stage 3 / **P2** — write \(U_{R★}\) + baselines + KILL/SURVIVE/GRADED rule |
| Forbidden until Gate 3 | Any E4 run, E2, fabric, old-task runs under `OLD_TASK_U` |
| Paper α | FROZEN; Stage 1 **agent-applied rubric audit** executed (Gate 1 PASS); dual-clinician IAA + synonym ontology open |
| E4 measurement | **Not present** — protocol aspirational until Stage 4 authorized |

## Why this note exists

E1 **KILL**: on the closed scribe task, M1 templates dominate generative refs under
`OLD_TASK_U`. Product work on that world is empty.

R★ is the **different** input regime where classical methods are *expected* to
break hard enough that a generative proposer *could* matter. Expectation ≠ proof.
Win/loss is **E4** after **P2**.

## Non-regime (explicitly excluded) — OLD TASK

| Property | Why classical wins |
|----------|-------------------|
| Seeded dialogues isomorphic to hand template families | M1 mirrors the generator |
| Values mostly contiguous verbatim spans under stable cues | Span/dict heuristics suffice when cues fire |
| Fixed `CC\|DUR\|SEV\|MED\|ALG` with closed DUR/SEV | Control fields rule-trivial |
| Held strings present in dialogue under familiar Q/A | M1 still solves; M2 weaker is irrelevant while M1 wins |

**Ban:** no E1-world re-bakeoff under `OLD_TASK_U`.

## Classical = freeze-set for R★ evaluation

When E4 runs (later), classical means at least:

| ID | Class | Freeze rule |
|----|-------|-------------|
| C-M1 | Template/regex | Rule budget frozen **before** eval; no post-hoc patterns after seeing eval |
| C-M2 | Train-dict + span | Train lexicon only; leakage check; no eval lexicon |
| C-M3+ | Optional span/CRF/constrained | Trained only on train split; same schema |

Generative reference(s) named in P2 only. No mid-flight method adds (Stage 4 rule).

---

## Measurable “classical break”

Classical **breaks** on an eval slice \(S\) when **any** of the following hold
under the frozen C-M1 + C-M2 budgets (P2 will pick primary metric via \(U_{R★}\);
these are the **measurable** predicates Gate 2 requires):

| ID | Predicate (measurable) | Intent |
|----|------------------------|--------|
| B1 | **Cue-hit rate** of frozen C-M1 on \(S\) \(< \tau_{\mathrm{cue}}\) | Templates do not fire (axis A/E) |
| B2 | **Verbatim-recoverable rate** among gold open-slot values \(< \tau_{\mathrm{span}}\) | Gold not recoverable by cue∪dict∪contiguous copy (axis B) |
| B3 | **Binding error rate** (wrong candidate among ≥2 present) \(\ge \tau_{\mathrm{bind}}\) | Multiplicity/reference (axis D) |
| B4 | **Train-dict coverage** of gold open values \(< \tau_{\mathrm{dict}}\) **and** cue-hit low | Ontology lag without easy copy (axis C) |

**Default floors for declaring a slice “in R★”** (P2 may re-weight into \(U\), but
may **not** silently delete these predicates):

```text
τ_cue  = 0.60    # fewer than 60% of docs get a successful C-M1 cue path for the target open slot
τ_span = 0.50    # fewer than 50% of gold open values are contiguous copy-recoverable under frozen heuristics
τ_bind = 0.20    # ≥20% of multi-candidate docs bind wrong under C-M1/C-M2
τ_dict = 0.50    # train dict covers <50% of gold open values on the slice
```

**Slice ∈ R★** iff it is built to the inclusion recipe below **and** at least
**two** of {B1,B2,B3,B4} are true on a locked classical probe pass (probe = score
C-M1/C-M2 only — **not** a substrate bakeoff; no generative run at Gate 2).

*Note:* The probe pass is Stage 4-adjacent instrumentation; Gate 2 passes on
**testability of the recipe**, not on having run the probe yet. E4 must include
the classical probe as a precondition check that the eval slice is still in R★.

---

## Regime axes (input conditions)

A product-relevant slice must instantiate **≥2 strong axes**:

| Axis | Inputs | Classical failure mode |
|------|--------|------------------------|
| **A** Surface-form explosion | Paraphrase, disfluency, ASR/OCR noise, style shift — not from the frozen rule family’s templates | C-M1 cue-hit collapses (B1) |
| **B** Non-verbatim / non-contiguous | Relative dates, “same as last time,” split mentions, implied negation | Not recoverable by contiguous copy (B2) |
| **C** Open ontology growth | Long-tail meds/allergies/complaints absent from train dict **and** weak cues | Dict miss + bad span (B4) |
| **D** Multiplicity / reference | Lists, corrections, two CCs, med changes | Wrong bind (B3) |
| **E** Weak cues | Free-text notes without stable Q/A anchors | C-M1 never fires (B1) |

---

## R★ inclusion recipe (testable instantiation)

**Schema (v1 wedge):** keep `CC | DUR | SEV | MED | ALG` for comparability, but
**change the delivery process** so the old M1 isomorphism breaks.

### Must include (generator / corpus constraints)

1. **Held template families** for surface forms (axis A): eval phrasings drawn from
   a pool **disjoint** from any C-M1 patterns frozen at train-rule-lock time.  
2. **≥30% of open-slot gold values** require normalization or multi-span assembly
   (axis B) — e.g. duration from relative date; allergy from correction turn.  
3. **Train-dict incompleteness** (axis C): ≥40% of eval open gold strings absent
   from train lexicon (leakage check as E1).  
4. **≥20% of dialogues** contain ≥2 competing values for at least one open slot
   (axis D), with a single gold after discourse.  
5. **≥30% of docs** lack the canonical cue strings C-M1 relies on (axis E) —
   e.g. prose notes rather than “any allergies?” adjacency.

### Must exclude (out of scope for R★ v1)

- Full EHR / billing / coding product claims  
- Open-ended clinical advice  
- Multilingual / multimodal (defer)  
- Changing `OLD_TASK_U` to re-litigate E1  
- Unbounded schema (arbitrary nested JSON) in v1 — schema stays fixed; **delivery** changes  

### In-distribution vs out-of-scope (summary)

| In R★ v1 | Out of scope |
|----------|--------------|
| English clinical-ish dialogue or short note → five fields | Arbitrary documents, no schema |
| Stress on open slots CC/MED/ALG under A–E | Claiming DUR/SEV difficulty as the product thesis |
| Frozen classical budgets | Post-hoc rule writing after eval reveal |
| Later: verifier relation \(R\) in Stage 5 | Open-world “zero hallucination” |

---

## Gate 2 decision

| Criterion | Assessment |
|-----------|------------|
| Is R★ empty? | **No** — inclusion recipe forces ≥2 axes + measurable B1–B4 |
| Is R★ testable? | **Yes** — generator/corpus constraints + classical probe predicates + fixed schema |
| Does classical still cover “everything we care about”? | **No** for product intent — we care about arrivals that violate cue/span/dict isomorphism (the non-regime is exactly what we **don’t** build product on) |

**Gate 2: PASS.** Proceed to Stage 3 (P2).

**Falsifier for Gate 2 (retroactive):** if a good-faith R★ v1 slice is built and
classical probe shows &lt;2 of {B1..B4}, the slice is **not** R★ — rebuild slice or
**STOP product** (Gate 2 fails on that instantiation). Max **one** R★ revision
after a failed E4 KILL per sequential plan.

---

## Evidence already in hand (do not over-read)

| Fact | Allowed | Forbidden |
|------|---------|-----------|
| E1 M1 ≫ M0 on old task | Classical wins closed world | Classical always wins |
| M2 weaker than M1 on old task | Dict fragility | LM necessary |
| α open-slot LM gaps | LMs can fail copy | Product need for LM on old task |

## Anti-goals

- No fabric / NanoScribe revival on old task  
- No E2 prose as product unlock  
- No \(U_{R★}\) invented here (Stage 3)  
- No E4 until Gate 3 PASS  
- No old-task runs under `OLD_TASK_U`

## Exit → Stage 3 (P2)

P2 must freeze:

1. \(U_{R★}\) (weights + δ; may differ from `OLD_TASK_U`)  
2. Baseline list (C-M1, C-M2, ≥1 generative ref; optional C-M3+)  
3. Eval slice builder satisfying inclusion recipe  
4. KILL / SURVIVE / GRADED rule  
5. Precondition: classical probe confirms slice ∈ R★  

Executing that protocol = **E4**.

## One-sentence freeze

**Product evolution serves R★ only; the E1 world remains a dead generative-substrate thesis.**
