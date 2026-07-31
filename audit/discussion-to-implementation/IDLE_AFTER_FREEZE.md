# IDLE After Freeze — Governance Note

**PROGRAM_EXECUTION_STATUS:** `IDLE_AFTER_FREEZE`  
**AUTHORIZED_NONEXECUTION_WORK:** `E4_DESIGN_ONLY`  
**E4_PROTOCOL_STATUS:** `DESIGN_DRAFT`  
**E4_EXECUTION_STATUS:** `BLOCKED`  
**Evidence freeze tag (immutable):** `post-alpha-evidence-freeze-2026-07-31` → `a9d12cb1c456f6c465284e1d469c6326cb14d329`  
**Date:** 2026-07-31  
**Companions:** `CANONICAL_STATUS_TABLE.md`, `papers/AMBITION.md`, `DIRTY_TREE_CLOSURE.md`

## Execution is idle; design docs are allowed

`IDLE_AFTER_FREEZE` means **no experiment execution**. It does **not** mean the research ambition is abandoned.

Allowed under `E4_DESIGN_ONLY`: harden R★ / P2 design docs, fairness matrix, draft U_R★, consequences, builder checklists, ambition/status language.

**Not allowed:** E4 world freeze, data generation, Stage 4 scoring, paid GPU, E2 runs, Fabric/NanoScribe product expansion, old-task re-runs under `OLD_TASK_U`.

Canonical statuses: `CANONICAL_STATUS_TABLE.md`.

## E4 precommitted consequence table (governance)

If E4 is ever **executed** later under a frozen U_R★ and locked classical/generative set:

| Outcome | Meaning | Program consequence |
|---------|---------|---------------------|
| **KILL** | Best classical under U_R★ beats or ties (within δ) best generative | **Stop** generative-substrate product track for tested R★. At most **one** preregistered R★ revision then re-gate; else remain idle. No automatic redesign loop. No NanoScribe/fabric expansion. |
| **GRADED** | Generative wins only on locked slices / axes | Preserve **only** the exact winning slice(s). No general platform or NanoScribe justification. |
| **SURVIVE** | Best generative strictly beats classical by >δ on primary U_R★ | Evidence for **this frozen R★ only**. Does **not** authorize NanoScribe or full product construction. Separate product feasibility gate required. |
| **VOID** | Protocol/data/builder violation or undecidable U | Correct the instrument. **No** scientific update for/against generative value. |

Full table + U draft: `trajectory/PREREG_E4_Rstar_killgate.md`.  
This table does **not** authorize Stage 4 execution. Committed design ≠ frozen protocol.
