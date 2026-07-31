# nano-lm A→Z — corrected for E1 KILL

*Directionally strong as a decision pipeline; **out of date** if read as “run Days 1–7.”
E1 already fired. This file is the corrected roadmap. 2026-07-31.*

**Authoritative sequential plan (gates + constants):** `papers/SEQUENTIAL_PIPELINE.md`.

**One rule:** never run an experiment whose result cannot change the roadmap.  
On the **old** closed task, more LM runs **cannot** change the roadmap — E1 already did.  
On a **new regime** where classical methods fail, a kill gate **can** — that is the only
product-facing experiment worth funding next (after written regime + utility).

| Doc | Role |
|-----|------|
| `papers/EVIDENCE_LEDGER.md` | Claim strength |
| `papers/CLAIM_GLOSSARY.md` | Forbidden / approved language |
| `trajectory/DECISION_P1_program_lock.md` | E1/E3/E2 lock (science program) |
| `trajectory/REGIME_P1_where_classical_fails.md` | **P1 regime note (DONE)** — R★ |
| `papers/EMPIRICAL_FOUNDATION.md` | Owner lockfile |
| Paper α / `paper-alpha-v1` | Public measurement freeze |

---

## What is already done

| Plan step | Status |
|-----------|--------|
| Phase 0 freeze / evidence boundary | **DONE** (ledger, glossary, foundation, Paper α) |
| **Phase 1 E1 kill gate** | **DONE — KILL (H-substrate)** |
| Official M0 vs M1/M2 under written \(U\) | **DONE** |
| Paper α measurement track | **FROZEN** on GitHub |
| Forbidden claims / scope lock | **DONE** |

**Outcome 1 already happened:**

> Non-LM wins → program changes from “LM reliability architecture” to  
> **“When do generative models add value over structured extraction?”**

Do **not** rebuild baselines or re-run E1 on m0–m4 / v1–v2.

---

## What the original plan still gets right (keep)

- Utility **before** experiments  
- Same benchmark, same rules  
- Construct validation (E3) as a real *optional* residual  
- Mechanism work only after premise is settled (and **not** as product work on the old task)  
- Paper split (α measurement / β systems)  
- Product decision only after evidence  
- **Never run an experiment that cannot change the roadmap**

---

## Corrected phases (post-KILL)

### Phase 0 — DONE
Evidence locked. Paper α frozen. **No reopen** of the old generative-substrate claim on this task.

### Phase 1 — DONE
E1 = KILL. Generative LM is **not** the preferred substrate *for this closed task under this \(U\)*.

### Phase 2 — Optional residual science (only if you care)
- E3 construct validity (exact vs soft/human) — limitation already in α; full human study **optional**  
- Do **not** fund E2 / mechanism / residual / scaling on the old task as product work  
- Do **not** fund “more supporting evidence” for α gaps that cannot change the product roadmap  

### Phase 3 — Product path (only path that matches product intent)

**Replaces** “run more LM experiments on the old task.”

| Step | Action | Status | Output |
|------|--------|--------|--------|
| **P1** | Write the regime where templates/dictionaries fail | **DONE** | `trajectory/REGIME_P1_where_classical_fails.md` (regime **R★**) |
| **P2** | Write utility + kill-gate protocol **for that regime** | **DONE** | `trajectory/PREREG_E4_Rstar_killgate.md` |
| **P3 / E4** | Kill-gate classical vs generative *in that regime* | **BLOCKED** | Awaits owner authorize Stage 4 against frozen P2 |
| **P4** | Branch | After P3 | Generative wins → build/verify; classical still wins → classical product, selective generative help, or stop |

Rules for Phase 3 (same discipline as E1):

1. Utility first (no bakeoff without \(U_{R★}\)).  
2. Non-LM baselines first / alongside — not LM-only.  
3. Same inputs, schema, metrics for all methods.  
4. Decision margin and falsifiers written before scoring.  
5. If classical still wins under \(U_{R★}\) → product is **not** “small LM.”

### Phase 4 — Systems (optional, private)
Verifier / provenance / abstention notes — **no** claim that the proposer must be a generative LM.  
**No** fabric expansion as product revival on the E1 world.

### Phase 5 — Publish
Only if/when you want. **Not** required for product work. α already public.

---

## Immediate next (replaces historical “Day 1–14”)

| Step | Action | Output |
|------|--------|--------|
| 1 | Leave Paper α frozen | No α edits needed |
| 2 | ~~P1: Regime note~~ | **DONE** — R★ |
| 3 | **P2: Utility + kill-gate protocol** for R★ | Written \(U\) + baselines + decision rule |
| 4 | Run that **new** kill gate | Only if P2 accepted — sole product-roadmap experiment |
| 5 | Branch | Generative adds value → build; else → classical / stop |

If you do **not** want to write P2 yet: **Idle**. Correct and allowed.

---

## Explicitly obsolete

| Historical item | Disposition |
|-----------------|-------------|
| Days 1–7 build E1 baselines / run E1 | **Do not do** — already KILL |
| Day 8 “decision meeting” on E1 | **Already decided** — Outcome 1 |
| E2 as next funded science for product | **No** |
| More LM scaling / residual sweeps on old task | **No** — cannot change roadmap |
| Fabric / NanoScribe enhancement on E1 world | **No** — contradicts KILL |

---

## Residual experiment IDs (order and justification)

| ID | What it is | When justified |
|----|------------|----------------|
| **E1** | Old-task substrate kill gate | **Done — KILL** |
| **E3** | Construct validity (exact vs soft/human) | Optional; improves Paper α interpretation; **does not unlock product** |
| **E2** | LoRA / mechanism discrimination | Only if you still care *scientifically* why LoRA helped on the old stack — **not** for product on the closed task |
| **E4** | New-regime kill gate (classical vs generative **where templates break**) | **The real next product experiment** — only after regime (**P1**, done) + utility/protocol (**P2**) are written |

“After E2 E3 E4” is meaningless without choosing a branch. E2/E3/E4 only make sense on specific branches below.

**Naming:** E4 = Stage 2 of the product path = run of the P2 protocol on regime **R★**. Docs call the protocol P2; the executed experiment is E4.

---

## Path A — Idle / science-only

1. Paper α stays frozen  
2. Optional **E3** (soft/human match) → stronger limitation or revised interpretation  
3. Optional **E2** (mechanism) → pure science; **no product claim**  
4. **End:** benchmark / measurement program. **No product.**

Allowed: Idle, E3 human, E2 (science only).  
Forbidden: fabric revival on E1 world; treating E2 as product unlock.

---

## Path B — Product path

### Stage 1 — docs only
| Step | Status |
|------|--------|
| Regime note (where templates/dicts fail) | **P1 DONE** → R★ |
| Utility for that regime | **P2 NEXT** |
| Kill-gate protocol (baselines + decision rule) | **P2 NEXT** |

### Stage 2 — E4 (new kill gate)
Run classical vs generative **in regime R★** under \(U_{R★}\).

| E4 outcome | What comes next |
|------------|-----------------|
| **Classical still wins** | Product is classical IE (+ maybe light assist). **Stop** generative substrate work. Optional: publish “when LMs don’t help.” |
| **Mixed by regime** | Product becomes **routing**: classical default, generative only in failure regimes. Build router + verify **only where it changes \(U\)**. |
| **Generative wins on utility** | Enhancement is justified → Stage 3 |

### Stage 3 — only if generative wins on E4
1. **Minimal proposer** that wins the regime (**not** full fabric)  
2. **Verification / abstention** scoped to the new \(R\) (legitimate systems track)  
3. **Provenance + review routing** (HITL economics)  
4. **Hardening:** adversarial false-accept, schema drift, long-tail eval  
5. **Product loop:** measure \(U\) in production-like conditions → cut features that don’t move \(U\)  

### Stage 4 — after a working wedge
- Narrow Paper β (soundness, abstention economics) — optional  
- Scale the wedge (fields/domains) **only where \(U\) stays positive**  
- Do **not** rebuild NanoScribe-as-general-architecture until one wedge wins repeatedly  

---

## Path C — Pure research sequence (“after E2 E3 …” as science)

Not a product unlock:

1. **E3** — Is exact-match overstating failure?  
2. **E2** — Why did LoRA help (geometry vs optimization vs early-stop vs hparams)?  
3. Diversity / coverage generalization — other slots, other domains  
4. Cross-domain replication — code, legal, tables, …  
5. Interpretability — only after behavioral mechanisms are pinned  
6. **End:** scientific benchmark + mechanism papers; **still no product claim**

---

## Branch summary

```
 E1 KILL (done)
        │
        ├── Path A: Idle / α freeze ── optional E3 ── optional E2 ── end (science)
        │
        ├── Path B: P2 docs ── E4 on R★ ─┬─ classical wins → classical product / stop
        │                                 ├─ mixed → route by regime + verify where U moves
        │                                 └─ generative wins → minimal proposer → verify →
        │                                      abstain → review → harden → optional β
        │
        └── Path C: E3 → E2 → diversity/domain → interpretability → science papers
```

**Short answer**

- **E1** is over.  
- **E3** is optional construct cleanup.  
- **E2** is optional old-task mechanism science.  
- **E4** (new-regime kill gate) is the only experiment that can unlock a product.  

Nothing after that should be “more experiments on the old closed task.”  
Everything after is **idle**, **science on open residual questions**, or **product in a regime where classical methods actually fail**.
