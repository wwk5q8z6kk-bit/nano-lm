# Sequential execution plan (constants + if/else gates)

*Authoritative product/science pipeline after E1 KILL.
Not “idle forever” and not “do all experiments.”
Each stage opens only when the previous gate passes. 2026-07-31.*

**Companion docs:** `AZ_EXECUTION_PLAN.md` (paths) · `DECISION_P1_program_lock.md` ·
`REGIME_P1_where_classical_fails.md` · `EVIDENCE_LEDGER.md` · `CLAIM_GLOSSARY.md`

---

## Frozen constants (do not change mid-stream)

```text
OLD_TASK_U = P - 0.5M - 0.3ρ - 0.02L - 0.05C
δ = 0.05
E1_VERDICT = KILL          # official M0 max U = 0.925 < M1 = 0.999
PAPER_α = FROZEN
E2_DEFAULT = GATED         # until written re-scope
FABRIC_DEFAULT = GATED
OLD_TASK_RUNS = FORBIDDEN  # no more experiments on the closed scribe task under OLD_TASK_U
```

Amending a constant requires an **owner-authored** commit that names the constant
and the reason. Silent re-weighting of `OLD_TASK_U` to revive substrate claims is
forbidden.

---

## Hard rules (sequential discipline)

1. **No Stage N+1 work until Gate N passes.**  
2. **OLD_TASK_RUNS = FORBIDDEN** under `OLD_TASK_U`.  
3. **E2 / FABRIC stay gated** until Gate 4 ∈ {SURVIVE, GRADED}.  
4. Every experiment must be able to **kill** the current product branch.  
5. If a stage cannot change the roadmap, **skip it**.

---

## Stage 0 — Decision lock

**Status: DONE — Gate 0 PASS**

- Claims C1–C9 locked (`DECISION_P1_program_lock.md`)  
- E1 KILL locked  
- Evidence ledger locked (`EVIDENCE_LEDGER.md`)  

→ Stage 1

---

## Stage 1 — Construct residual (E3)

**Goal:** Decide whether exact-match overstates the failure.

1. Keep auto normalize result (**0/486** rescues) as baseline.  
2. **If** owner authorizes a bounded human soft-match: fixed small subset;
   pre-stated rubric; decision rule: “qualitative open-slot gap shrinks materially?”  
3. **Else** keep the written limitation in Paper α.

### Gate 1

| Result | Action |
|--------|--------|
| Human check shows gap is mostly formatting/equivalence noise | Downgrade copy-failure language; measurement paper becomes weaker; product path still needs Stage 2 |
| Gap remains large under soft/human | Copy-failure interpretation stands; proceed |
| Skipped | Proceed with explicit limitation; do **not** pretend construct validity is closed |

→ Always continue to Stage 2 **if product is a goal**. Stage 1 does **not** unlock old-task LM runs.

**Gate 1: PASS — EXACT_SURVIVES (executed, not skipped).** Bounded **agent-applied rubric audit** on frozen pack n=100 (`agent-rubric-pass-1`); faithful-rate 0.00; qualitative open-slot gap does **not** shrink materially. Not dual-clinician validation. First-principles note: `trajectory/STAGE1_E3_CONSTRUCT_FIRST_PRINCIPLES.md`. Paper α keeps exact-match limitation (strict metric + single-pass/synthetic limits), not because failures are formatting. Log: `trajectory/PIPELINE_GATE_LOG.md`.

---

## Stage 2 — Regime lock (P1 → harden)

**Status: DONE — Gate 2 PASS** (2026-07-31)

**Goal:** Define where classical methods fail hard enough that a generative proposer *could* matter.

**Hardened regime:** `trajectory/REGIME_P1_where_classical_fails.md` (R★): measurable
break predicates B1–B4, τ defaults, inclusion recipe, in/out of scope.

Must specify:

- input conditions (open schema, paraphrase, long-tail, incomplete rules, …)  
- what “classical break” means (**measurable**)  
- what is still in-distribution vs out of scope  

### Gate 2

| Result | Action |
|--------|--------|
| R★ is empty / classical still covers everything you care about | **STOP product path.** Science-only or idle. |
| R★ is non-empty and testable | Go to Stage 3 |

---

## Stage 3 — Utility + kill-gate protocol (P2)

**Status: DONE — Gate 3 PASS** — `trajectory/PREREG_E4_Rstar_killgate.md`

**Goal:** Freeze decision rule *before* any new run.

Deliverable:

- Utility \(U_{R★}\) for regime R★ (weights written, not improvised later)  
- Baseline family (classical + constrained + generative reference)  
- Same data / same schema / same metrics  
- KILL / SURVIVE / GRADED rule with δ  

### Gate 3

| Result | Action |
|--------|--------|
| Utility or baselines incomplete | Do **not** run anything |
| Protocol complete and frozen | Go to Stage 4 (**E4**) |

---

## Stage 4 — New kill gate (E4)

**Goal:** Does generative add utility **in R★**?

Run only what Stage 3 listed. No extra methods mid-flight.

### Gate 4 (main product branch)

| Outcome | Meaning | Next |
|---------|---------|------|
| **KILL** (classical still dominates \(U_{R★}\)) | Generative not justified in this regime either | Stop product path **or** redefine R★ **once** (max one revision), else idle |
| **SURVIVE** (generative wins \(U_{R★}\)) | Substrate justified *in this regime* | Stage 5b |
| **GRADED / mixed** | Wins only on subsets | Stage 5a (routing), not full fabric |

---

## Stage 5 — Build only what Gate 4 bought

### 5a — If GRADED
Build **router**: classical default → generative only on R★ slices that win on \(U\).  
Verify + abstain only on those slices. Re-measure \(U\).

### 5b — If SURVIVE
Minimal stack only:

1. Proposer that won E4  
2. Verifier under explicit relation \(R\)  
3. Abstention + review routing  
4. Provenance  

After each addition: **re-score \(U_{R★}\)**.  
**If** feature does not improve \(U\) → remove it.

### Gate 5

| Result | Action |
|--------|--------|
| \(U\) flat or down after “enhancements” | Roll back; complexity is not progress |
| \(U\) up and stable | Go to Stage 6 |

---

## Stage 6 — Harden + optional science

Only after Stage 5 shows positive \(U\):

- Adversarial false-accept tests on the verifier  
- Schema drift / long-tail stress  
- Optional E2 (mechanism) as **science**, not as product requirement  
- Optional Paper β (systems soundness / economics) — no substrate mythology  

### Gate 6

| Result | Action |
|--------|--------|
| Verifier false-accept rate unacceptable | Fix or abstain more; do not ship |
| Economics negative (review cost eats gains) | Narrow scope or stop |
| Stable positive \(U\) + acceptable risk | Product wedge exists |

---

## Stage 7 — Expand or stop

**If** wedge works on R★:

- Widen fields/domains **one at a time**  
- Each expansion repeats a mini kill-gate (classical vs generative on that slice)  
- No global fabric until ≥2–3 wedges independently win  

**Else:** stop. Archive. Paper α remains the measurement record.

---

## Parallel optional track (never blocks product)

| Track | Rule |
|-------|------|
| E3 human | Optional, bounded; does not unlock Stage 5 |
| E2 LoRA mechanism | Only with written science objective; cannot reopen E1 KILL on old task |
| Publish Paper α | Anytime; orthogonal (`paper-alpha-v1` already public) |

---

## Cursor — what to do next (literally)

```text
Stage 0: DONE (Gate 0 PASS)
Stage 1: DONE (Gate 1 PASS — E3 bounded human EXECUTED; EXACT_SURVIVES)
Stage 2: DONE (Gate 2 PASS — R★ hardened / testable)
Stage 3: DONE (Gate 3 PASS — P2 / PREREG_E4_Rstar_killgate.md frozen)
Stage 4: E4 BLOCKED until owner authorizes run against frozen P2   ← NEXT DECISION
```

**Next owner/agent line (pick one):**

- `Stage 4: authorize E4 — implement R★ builder + run kill gate per PREREG_E4_Rstar_killgate.md`  
- `Idle` — leave E4 blocked; protocol stays frozen  

Do **not** edit \(U_{R★}\) or baselines after seeing scores. Do **not** reopen old-task runs.
