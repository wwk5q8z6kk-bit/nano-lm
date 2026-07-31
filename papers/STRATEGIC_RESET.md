# nano-lm — Strategic Reset

**Date:** 2026-07-31  
**Status:** Owner decision document (not Layer-1 evidence; not a new constitution)  
**Context:** Program 0 done. E1/E4 KILL stand. Stop infrastructure expansion. Re-center on useful capability, not "tiny LM scores better."

---

## 1. Opening question (locked)

Wrong start:

> How do we make a tiny language model score better?

Right start:

> **What useful capability can a small system deliver better than larger, more expensive, less controllable systems?**

---

## 2. What is nano-lm?

> **A first-principles research and engineering project for building small, efficient, verification-gated AI systems that outperform larger model-only approaches on useful real-world tasks.**

It is **one research + product vehicle**, not a laboratory franchise, not a benchmark company, and not NanoScribe.

```text
Mission (Level 0)
  → Product thesis / research questions (Level 1)
  → nano-lm as vehicle (Level 2)
  → Current experiments / infra / components (Level 3)
```

Paper α, E1, Fabric, Program 0, and any future census sit at **Level 3**. They serve the mission; they are not the mission.

---

## 3. Core product thesis

> **A small, local, efficient AI system that combines compact models, deterministic tools, retrieval, memory, and verification to complete useful tasks with high reliability and low cost.**

The key word is **system**, not model.

**First product thesis (operating rule):**

> **Use the smallest sufficient solver, verify every consequential output, and escalate only when necessary.**

E1 already taught the routing rule:

- when rules solve it better → use rules;
- when retrieval solves it better → use retrieval;
- when a small model adds flexibility → use the model;
- when the answer cannot be verified → abstain or escalate to review.

```text
User task
   ↓
Task classifier
   ↓
Choose cheapest sufficient solver
   ├── deterministic parser
   ├── search / retrieval
   ├── symbolic tool
   ├── small specialized model
   └── larger external model only when required
   ↓
Verification
   ↓
Useful output, abstention, or escalation
```

That is more valuable than "a small chatbot."

---

## 4. Definitions (product language)

### Small

Not only parameter count. Means: local where possible; low memory / latency / energy / cost; easy to deploy; private by default; understandable components; bounded context; minimal external dependencies; replaceable models; reliable offline behavior.

A 500M model plus exact tools may be a smaller and better system than a 3B model doing everything itself.

### Powerful

Completing real workflows; handling unfamiliar inputs; using tools correctly; remembering validated information; providing evidence; knowing when uncertain; recovering from failure; improving via modular upgrades; using larger models selectively.

Not merely generating impressive prose.

### Useful

Saves measurable time, money, cognitive effort, mistakes, review burden, privacy risk, or infrastructure cost.

For every proposed feature:

$$
\\mathrm{Value} \\propto \\frac{\\mathrm{time\ saved} + \\mathrm{errors\ prevented} + \\mathrm{new\ capability}}{\\mathrm{review\ burden} + \\mathrm{latency} + \\mathrm{cost}}
$$

---

## 5. Nano Runtime (Directions 1 + 2)

**Chosen product shape:**

> **A local-first, verification-gated task and knowledge engine that uses the smallest sufficient method for every operation.**

Not Direction 3 (full agent runtime / DevX) as the start.

### Initial capabilities (target, not built)

1. Ingest a document or structured dataset.
2. Identify a requested task.
3. Choose a deterministic or model-based solver.
4. Produce structured claims.
5. Attach evidence spans.
6. Verify claims.
7. Abstain when unsupported.
8. Store validated results.
9. Report latency, cost, confidence, and provenance.

### Minimum viable system (build order after wedge lock)

| Component | Role |
|-----------|------|
| Task router | Classify: extract / search / calculate / summarize / compare / synthesize / act. Deterministic first. |
| Solver registry | Each solver declares capabilities, I/O, cost, latency, confidence domain, verification method. |
| Evidence ledger | claim, source, span, solver, timestamp, confidence, verification state, contradictions. |
| Verifier | Match claim type: exact, arithmetic, code, evidence-backed fact, contradiction, abstain. |
| Validated memory | Only verified items persist; states: confirmed / probable / disputed / superseded / expired. |
| Evaluation harness | Program 0 / existing eval infra measures whether each component improves utility. |

### Success metrics (first workflow)

$$
U = Q - \\lambda_e E - \\lambda_r R - \\lambda_l L - \\lambda_c C
$$

Also track: evidence coverage; unsupported-claim rate; abstention rate; time saved; local completion rate; escalation rate; memory error rate; user correction rate.

LM stays only when hybrid utility beats classical utility.

---

## 6. Current scientific assets (earned — unchanged)

| Asset | Meaning (scoped) |
|-------|------------------|
| **Paper α / held-value instrument** | Reproducible measurement of held-out copying failure in small LMs; field localization; diversity / stack effects. |
| **E1 KILL** | Under frozen U, classical template extraction beats generative M0 on the old closed scribe task. Generation is not the default for that regime. |
| **E3 (partial)** | Exact-match not rescued by normalize; agent-rubric exact survives; dual-clinician IAA still open. |
| **E2** | Prereg frozen; GATED / STOP (no RESULT). |
| **E4 / R★** | KILL on frozen R★ v1; execution-blocked until owner re-authorizes a revision (budget 1). |
| **Claim discipline** | Evidence Ledger + freeze tags are Layer-1; ambition stays here / portfolio / roadmap. |

---

## 7. Current technical assets (tools, not identity)

| Asset | Role |
|-------|------|
| **`trajectory/`** | Empirical instruments, preregs, scored results. |
| **`fabric/`** | Scoped verification slice (propose→verify→abstain)—not a cognitive OS; not NanoScribe. |
| **Evaluation infrastructure (Program 0)** | One digest-bound sentinel path, manifests, Gate 0. Done. Serves Nano Runtime component evaluation. |
| **Governance docs** | How work is authorized in-repo. Supporting only. |

---

## 8. Falsified / stopped paths (do not revive casually)

- Generative substrate as preferred solution for the old closed extraction task under frozen OLD_TASK_U / official E1 U.
- Treating Fabric or NanoScribe packaging as proven architecture.
- LoRA "geometry preservation" as established mechanism.
- Letting infrastructure / governance displace useful capability work.
- Auto-starting Program 1 because Program 0 exists.
- Beginning with a general assistant or full agent IDE.

---

## 9. Development roadmap (product)

| Phase | Work | Gate |
|-------|------|------|
| **1** | Define one concrete wedge + 20–50 representative tasks | Owner lock on workflow |
| **2** | Classical baseline: parse, retrieve, rules, deterministic verify | Measurable U_classical |
| **3** | Add small model only where baseline fails | Keep only if Delta U_LM > 0 |
| **4** | Verified memory | Improves repeat workflows without stale/false recall |
| **5** | Size / efficiency research (distill, quantize, adapters, …) | After wedge value is real |
| **6** | Expand workflows | Only after first wedge pays |

**No broad architecture before Phase 1 lock.**

---

## 10. Exact next authorized work

```text
PROGRAM0 = DONE
EVAL_INFRA = AVAILABLE
PROGRAM1 = NOT AUTHORIZED
TRAINING = NOT AUTHORIZED
E4_EXECUTE = BLOCKED
NANOSCRIBE = STOP
INFRA_EXPANSION = STOP
LAB_STRUCTURE_EXPANSION = STOP
```

**Authorized / locked:**

1. ~~Select the first concrete workflow (Phase 1)~~ **DONE** — `papers/WEDGE_V1.md`  
   (`local_research_document_intelligence`, task pack n=40).
2. ~~Phase 2 classical baseline~~ **DONE** — `AUTHORIZE_WEDGE_V1_CLASSICAL_BASELINE`; artifact `wedge_v1/results_wedge_v1_classical.json` (draft U≈0.891 on clean synthetic track).
3. ~~Phase 3 E-class cheapest-sufficient probes~~ **DONE** — `AUTHORIZE_WEDGE_V1_PHASE3_ECLASS_PROBE`; verdict `ECLASS_CLOSED_WITHOUT_LM` (acc=1.0, ΔU≈+0.009 < δ; LM not invoked).
4. Standing: protect Layer-1 freeze / tags; do not casually commit remaining dirty freeze/audit files.

**Next (optional auth):** `AUTHORIZE_WEDGE_V1_U_FREEZE` | `AUTHORIZE_WEDGE_V1_OWNER_CORPUS` | `AUTHORIZE_WEDGE_V1_NOISY_TRACK` | idle/hygiene. Owner `continue` = `CONTINUE_SESSION` (M0 only), not execute. `AUTHORIZE_WEDGE_V1_PHASE3_LM_PROBE` is **not indicated**.

**Not authorized:** Program 1 census; new governance layers; training runs; E4 execute; NanoScribe / fabric-v2 / agents / memory product builds; expanding "Benchmark Supremacy Lab" identity; Nano Runtime component builds without Phase 3/expand auth.

---

## 11. Decision rule

Authorize a unit only if it:

1. advances useful capability under the product thesis (§3);
2. can change the roadmap if it fails;
3. uses classical baselines and existing eval infra before adding models;
4. stays scoped to nano-lm as one vehicle.

```text
Infrastructure must serve useful capability, not replace it.
Next work = first concrete workflow — not more laboratory structure.
```
