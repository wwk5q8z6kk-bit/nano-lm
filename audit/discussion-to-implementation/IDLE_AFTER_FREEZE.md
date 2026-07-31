# IDLE After Freeze — Governance Note

**Decision gate:** `IDLE_AFTER_FREEZE` (evidence packaging complete)  
**Design carve-out:** `AUTHORIZE_E4_DESIGN_ONLY` (2026-07-31)  
**Date:** 2026-07-31  
**Tag:** `post-alpha-evidence-freeze-2026-07-31`  
**E4:** `DESIGN_IN_PROGRESS` / `EXECUTION_BLOCKED` — this note is **not**
authorization to **run** E4.

## IDLE ≠ project halted

Evidence freeze completed. Ambition continues under design discipline — see
`papers/AMBITION.md`. The research question remains: whether ∃ R★ with
\(U_{\mathrm{gen+verify}}(R★) > U_{\mathrm{classical}}(R★)\) under matched costs.

## Default posture (still in force)

After the evidence + claim-sync commits and annotated tag:

1. No new **experiments** / Stage 4 scoring.
2. No paid compute / GPU for E4 or E2.
3. No fabric / NanoScribe expansion.
4. No E2 runs.
5. No E4 builder/data/result work that freezes an R★ world — until
   `AUTHORIZE_E4_EXECUTE` (or equivalent).

**Allowed under design carve-out:** harden R★ / P2 design docs, fairness matrix,
draft \(U_{R★}\), consequences, builder checklists, ambition/status language.

Canonical statuses: `CANONICAL_STATUS_TABLE.md`.

## E4 precommitted consequence table (governance)

If E4 is ever **executed** later under a frozen \(U_{R★}\) and locked classical/generative
set, interpret outcomes as:

| Outcome | Meaning | Program consequence |
|---------|---------|---------------------|
| **KILL** | Best classical under \(U_{R★}\) beats or ties (within \(\delta\)) best generative | Generative still not preferred in R★; product thesis fails again; do not expand generative substrate |
| **GRADED** | Generative wins on some locked slices / axes but not the primary \(U_{R★}\) decision | Partial value only; expand only along winning slices; no blanket architecture claim |
| **SURVIVE** | Best generative strictly beats classical by \(>\delta\) on primary \(U_{R★}\) | Generative value in R★ supported for that regime only; still not NanoScribe product authorization |
| **VOID** | Protocol/data/builder violation, leakage, or undecidable \(U\) | No scientific update; fix protocol; do not interpret as SURVIVE |

Full table + \(U\) draft: `trajectory/PREREG_E4_Rstar_killgate.md`.  
This table does **not** authorize Stage 4 execution.
