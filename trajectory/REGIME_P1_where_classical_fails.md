# P1 annex — Regime R★ (design-hardened)

*Product-path boundary. Decision lock: `trajectory/DECISION_P1_program_lock.md`.
Sequential plan: `papers/SEQUENTIAL_PIPELINE.md`. Ambition frame: `papers/AMBITION.md`.*

*Not part of Paper α. Gate 2 PASS recorded 2026-07-31 in `PIPELINE_GATE_LOG.md`.
Under `AUTHORIZE_E4_DESIGN_ONLY` (2026-07-31): design hardening only — **no** E4
execution, **no** R★ world freeze, **no** dataset generation that constitutes eval lock.*

## Status

| Field | Value |
|-------|--------|
| Artifact class | Design-hardened regime definition |
| Gate 2 | **PASS** — R★ non-empty and testable (protocol) |
| E4 | `DESIGN_IN_PROGRESS` / `EXECUTION_BLOCKED` |
| Next authorized | Harden P2 design (`PREREG_E4_Rstar_killgate.md`); **not** Stage 4 runs |
| Forbidden | E4 execution, E2, fabric/NanoScribe expansion, old-task runs under `OLD_TASK_U`, paid compute for E4 |
| Paper α | FROZEN; Stage 1 **agent-applied rubric audit** (Gate 1 PASS); dual-clinician IAA open |
| E4 measurement | **Absent** — no `results_e4_*` |

## Why this note exists

E1 **KILL**: on the closed scribe task, M1 templates dominate generative refs under
`OLD_TASK_U`. Product work *on that world* is empty.

Ambition continues elsewhere: define a **different** input regime **R★** where
classical methods are *expected* (by construction of delivery process) to be
stressed enough that a generative proposer *could* matter. Expectation ≠ proof.
Win/loss is **E4** only after separate execution authorization against frozen P2.

## Anti-circularity (load-bearing)

**Forbidden circular selection:** pick or filter eval documents *because* classical
methods already scored poorly on them (or because generative scored well).

| Layer | Role | May depend on classical scores? |
|-------|------|----------------------------------|
| **Inclusion predicates I\*** | How documents are *generated / curated* (process properties) | **No** — fixed before any method scores |
| **Exclusion predicates X\*** | What must not enter R★ v1 | **No** |
| **Probe predicates B\*** | Post-build sanity that the locked slice still stresses frozen classical budgets | **Yes, but only as VOID/rebuild** — before any generative scoring; never as post-hoc cherry-pick |

**Rule:** Lock the eval instance set from I\*/X\* alone. Run classical probe once.
If fewer than 2 of {B1..B4} fire → slice ∉ R★ → rebuild or STOP. If probe passes → freeze
the set; **do not** drop instances after seeing G-ref outcomes.

---

## Non-regime (explicitly excluded) — OLD TASK

| Property | Why classical wins |
|----------|-------------------|
| Seeded dialogues isomorphic to hand template families | M1 mirrors the generator |
| Values mostly contiguous verbatim spans under stable cues | Span/dict heuristics suffice when cues fire |
| Fixed `CC\|DUR\|SEV\|MED\|ALG` with closed DUR/SEV | Control fields rule-trivial |
| Held strings present in dialogue under familiar Q/A | M1 still solves; M2 weaker is irrelevant while M1 wins |

**Ban:** no E1-world re-bakeoff under `OLD_TASK_U`. E1 KILL stays scoped to that
frozen U and world — not a universal “LMs never extract” claim
(`WITHDRAWAL_SPEC` W-KILL-UNIVERSAL).

## Classical = freeze-set for later E4 evaluation

When E4 *execution* is authorized (later), classical means at least:

| ID | Class | Freeze rule |
|----|-------|-------------|
| C-M1 | Template/regex | Rule budget frozen **before** eval reveal; no post-hoc patterns |
| C-M2 | Train-dict + span | Train lexicon only; leakage check; no eval lexicon |
| C-M3+ / C-M4 | Optional span/CRF/constrained | Trained only on train split; same schema |

Generative reference(s) named in P2 only. Information-parity matrix lives in
`PREREG_E4_Rstar_killgate.md` §2.5.

---

## Inclusion predicates (I*) — independent of classical scores

These are **process / gold-construction** constraints. Satisfaction is checked from
generator metadata + gold annotations **without** running C-M1/C-M2 accuracy.

| ID | Predicate (measurable from builder artifacts) | Axis |
|----|-----------------------------------------------|------|
| **I1** | Eval surface-form template family ID set is **disjoint** from the C-M1 rule family’s template IDs (committed at rule-lock time) | A |
| **I2** | ≥30% of open-slot gold values tagged `needs_norm_or_multispan=true` in gold | B |
| **I3** | ≥40% of eval open gold strings absent from the **train** lexicon file (hash-locked); leakage report attached | C |
| **I4** | ≥20% of dialogues annotated with ≥2 competing candidate values for ≥1 open slot and a single discourse-resolved gold | D |
| **I5** | ≥30% of docs tagged `cue_family=none\|weak` (canonical C-M1 cue strings absent by construction) | E |

**Slice satisfies inclusion** iff **all** of I1–I5 hold on the locked eval set
**and** ≥2 of axes A–E are marked “strong” in the builder manifest (recommend all five
for v1 wedge).

### Must exclude (X*)

| ID | Exclusion |
|----|-----------|
| **X1** | m0–m4 / v1–v2 isomorphic dialogues under old M1 families |
| **X2** | Full EHR / billing / coding product claims; open clinical advice |
| **X3** | Multilingual / multimodal as primary (defer) |
| **X4** | Changing `OLD_TASK_U` to re-litigate E1 |
| **X5** | Unbounded schema (arbitrary nested JSON) in v1 — schema stays `CC\|DUR\|SEV\|MED\|ALG` |
| **X6** | Selecting/filtering instances using classical or generative scores after peek |

---

## Probe predicates (B*) — post-build, pre-generative only

Classical **probe** (not the bakeoff) on locked slice \(S\) with frozen C-M1/C-M2:

| ID | Predicate | Intent |
|----|-----------|--------|
| B1 | Cue-hit rate of frozen C-M1 on \(S\) \(< \tau_{\mathrm{cue}}\) | Templates rarely fire |
| B2 | Verbatim-recoverable rate among gold open-slot values \(< \tau_{\mathrm{span}}\) | Gold not contiguous-copy recoverable |
| B3 | Binding error rate (wrong candidate among ≥2 present) \(\ge \tau_{\mathrm{bind}}\) | Multiplicity stress |
| B4 | Train-dict coverage of gold open values \(< \tau_{\mathrm{dict}}\) **and** cue-hit low | Ontology lag |

**Default τ (design draft — may be amended only before execution auth):**

```text
τ_cue  = 0.60
τ_span = 0.50
τ_bind = 0.20
τ_dict = 0.50
```

**Probe pass:** ≥2 of {B1..B4} true → `in_Rstar: true`. Else rebuild or STOP.
Probe failure is **not** permission to hand-pick easy generative wins.

---

## Regime axes (input conditions)

| Axis | Inputs | Expected classical stress (descriptive) |
|------|--------|------------------------------------------|
| **A** | Paraphrase, disfluency, ASR/OCR noise, style shift — not from frozen rule family’s templates | C-M1 cue-hit collapses (B1) |
| **B** | Relative dates, “same as last time,” split mentions, implied negation | Not contiguous-copy recoverable (B2) |
| **C** | Long-tail meds/allergies/complaints absent from train dict **and** weak cues | Dict miss + bad span (B4) |
| **D** | Lists, corrections, two CCs, med changes | Wrong bind (B3) |
| **E** | Free-text notes without stable Q/A anchors | C-M1 never fires (B1) |

---

## Schema / scope (v1 wedge)

Keep `CC | DUR | SEV | MED | ALG` for comparability; **change delivery** so old M1
isomorphism breaks. Open slots stressed: **CC, MED, ALG**. DUR/SEV = controls.

| In R★ v1 | Out of scope |
|----------|--------------|
| English clinical-ish dialogue or short note → five fields | Arbitrary documents, no schema |
| Stress on open slots under I1–I5 | Claiming DUR/SEV difficulty as the product thesis |
| Frozen classical budgets + information parity | Post-hoc rule writing after eval reveal |
| Later: verifier relation \(R\) only if execution authorized | Open-world “zero hallucination” |

---

## Gate 2 decision (unchanged)

| Criterion | Assessment |
|-----------|------------|
| Is R★ empty? | **No** — I\* + axes force a non-empty design space |
| Is R★ testable? | **Yes** — builder constraints + probe B\* + fixed schema |
| Circular? | **Mitigated** — inclusion ≠ observed classical failure |

**Gate 2: PASS** (protocol). Instantiation still requires a builder under later
execution auth.

**Falsifier:** good-faith slice built to I\*/X\* but probe fires fewer than 2 of B\* → not R★;
max **one** R★ revision after a failed E4 KILL per sequential plan.

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
- No E4 execution under design-only auth
- No old-task runs under `OLD_TASK_U`
- No claiming R★ generative value without RESULT

## Exit → P2 design / later execution

P2 design package (`PREREG_E4_Rstar_killgate.md`) freezes draft \(U_{R★}\),
baselines, fairness matrix, consequences, builder checklist.

**Executing** that protocol = Stage 4 / E4 — requires separate owner authorization.

## One-sentence freeze

**Product evolution may serve R★ only after a non-circular, precommitted kill gate;
the E1 world remains a dead generative-substrate thesis under `OLD_TASK_U`.**
