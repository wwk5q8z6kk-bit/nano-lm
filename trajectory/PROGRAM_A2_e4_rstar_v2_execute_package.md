# Program A2 — E4′ / R★ v2 execute package (prep only)

**Program:** A — Regime value of generation  
**Unit:** A2 — execute-readiness package for the **sole remaining** R★ revision  
**Design source:** `trajectory/PROGRAM_A1_rstar_revision_design.md` (`OWNER_ACCEPTED`)  
**Mitigations:** `trajectory/PROGRAM_A_FIRST_PRINCIPLES_MITIGATIONS.md`  
**Prep auth:** `AUTHORIZE_PROGRAM_A2_EXECUTE_PACKAGE_PREP`  
**Execute auth:** `AUTHORIZE_E4_RSTAR_V2_EXECUTE` — **NOT GRANTED**

```text
A2_STATUS = PREP_COMPLETE_AWAITING_EXECUTE_AUTH
WORLD_REBUILD = FORBIDDEN_UNTIL_EXECUTE_AUTH
GREF_TRAIN = FORBIDDEN_UNTIL_EXECUTE_AUTH
E4_SCORE = FORBIDDEN_UNTIL_EXECUTE_AUTH
PAID_COMPUTE = FORBIDDEN_UNTIL_EXECUTE_AUTH
```

---

## 1. Purpose

Turn accepted A1 design into a single **execute-ready checklist** so that when (if) the owner grants `AUTHORIZE_E4_RSTAR_V2_EXECUTE`, builders do not invent mid-stream policy.

This file does **not** run the experiment.

---

## 2. Frozen design pointers (from A1)

| Topic | Pointer |
|-------|---------|
| Postmortem / H* | A1 §5 — primary **H1**, secondary **H4** |
| I*′ / X*′ / B*′ | A1 §6 — **B2 mandatory**, τ_span=0.35; I6–I8 |
| Classical set | A1 §7 — C-M1, C-M2, C-M4 (+ optional C-Ret) |
| U candidate | A1 §8 + mitigations §3 — \(U_{dep}\) official; \(U_{cap}\) diagnostic only |
| Anti-span construction | Mitigations §2 recipes + span oracle B2 |
| Decision table | A1 §9 — SURVIVE / KILL / VOID |
| Compute ceiling draft | A1 §10 |

v1 regime doc remains historical: `trajectory/REGIME_P1_where_classical_fails.md`.  
v1 prereg remains historical: `trajectory/PREREG_E4_Rstar_killgate.md`.  
**Do not mutate v1 result artifacts.**

---

## 3. Builder checklist (unlocked only by execute auth)

Copy structure of v1 `trajectory/e4/BUILDER_CHECKLIST_STATUS.json`, new schema id `nano-lm.e4.builder_checklist.v2`.

| Step | Action | Gate |
|------|--------|------|
| B0a–c | Constructor recipes + I8 self-check + span-oracle B2 (see mitigations §8) | execute auth |
| B1 | Implement / extend generator for I6–I8 + I2′/I4′ (new code under `trajectory/e4/` or `trajectory/e4_v2/`) | execute auth |
| B2 | Emit train/dev/eval splits + world manifest; **no score peek** | execute auth |
| B3 | Leakage report (train lexicon vs eval gold) | execute auth |
| B4 | Freeze C-M1 rules **before** eval reveal | execute auth |
| B5 | Classical probe with **B2 mandatory**; VOID if fail | execute auth |
| B6 | G-ref train under pre-registered recipe + matched budget | execute auth |
| B7 | Verify-on / verify-off scoring | execute auth |
| B8 | Utility rows + decision.json | execute auth |
| B9 | SHA256SUMS + recipe freeze + auth record | execute auth |

**Anti-circularity:** no dropping instances after classical or gen scores.

---

## 4. Proposed auth record fields (for owner execute grant)

When issuing `AUTHORIZE_E4_RSTAR_V2_EXECUTE`, record at minimum:

```text
authorize_e4_rstar_v2_execute: true
design_ref: trajectory/PROGRAM_A1_rstar_revision_design.md
package_ref: trajectory/PROGRAM_A2_e4_rstar_v2_execute_package.md
revision_budget: 1   # final; KILL ends product-path gen for this line
venue: <local-mps | runpod-cuda | ...>
max_usd: <number or 0 for local-only>
max_wall_hours: <number>
C_schedule_ack: true   # G-ref C not silently zeroed
U_default_only_verdict: true
b2_mandatory: true
tau_span: 0.35
old_task_u: forbidden
program1: not_authorized
```

Suggested path for the auth file (create only at grant time):  
`trajectory/e4/AUTH_RECORD_RSTAR_V2.md`

---

## 5. Explicit non-actions under A2 prep

- Do not call `build_rstar.py` to replace v1 worlds  
- Do not train or overwrite `gref_nano_rstar_sft_v1.pt`  
- Do not write `results_e4_*` replacements  
- Do not spend paid compute  
- Do not start Program 1  

---

## 6. After KILL or SURVIVE (reminder)

| Verdict | Product-path consequence |
|---------|--------------------------|
| SURVIVE | Scoped claim only under frozen U/R★′; still ≠ NanoScribe |
| KILL | Generative product track for this line **STOP** (budget spent) |
| VOID | No claim; new auth required to rebuild |

---

## 7. Ready state

```text
A1 = OWNER_ACCEPTED
A2 = PREP_COMPLETE_AWAITING_EXECUTE_AUTH
NEXT_OWNER_STRING = AUTHORIZE_E4_RSTAR_V2_EXECUTE
```
