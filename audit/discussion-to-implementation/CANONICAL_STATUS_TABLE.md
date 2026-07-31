# Canonical Status Table

**Authority:** this file + `CANONICAL_STATUS_TABLE.json`  
**Updated:** 2026-07-31 (E4 EXECUTE → KILL)  
**Rule:** every status-bearing doc must derive from this table.

## Program posture (exact separation)

```text
PROGRAM_EXECUTION_STATUS: IDLE_PARKED
AUTHORIZED_NONEXECUTION_WORK: NONE
SCIENCE_TRACK_STATUS: IDLE_AFTER_E4_KILL
WEDGE_PROGRESS: PHASES_1_3 + NOISY + DOGFOOD_8_8
E4_PROTOCOL_STATUS: EXECUTED
E4_EXECUTION_STATUS: COMPLETE
E4_WORLD_STATUS: FROZEN
E4_DATA_STATUS: LOCKED
E4_RESULT_STATUS: KILL
FABRIC_STATUS: SCOPED_VERIFICATION_SLICE
NANOSCRIBE_STATUS: ARCHITECTURAL_RESEARCH_PROGRAM
OLD_TASK_GENERATIVE_SUBSTRATE: FALSIFIED_UNDER_FROZEN_U
NANOSCRIBE_PRODUCT_EXPANSION: STOP
RSTAR_PRODUCT_TRACK: STOP_FOR_TESTED_RSTAR
RSTAR_REVISION_BUDGET_REMAINING: 1
PRODUCT_WEDGE_V1: RUNTIME_SLICE_LIVE  # classical+E-class CLI; LM not indicated
# Operating detail: papers/LABORATORY_CONSTITUTION.md / EXECUTION_QUEUE.md
# Optional next needs typed AUTHORIZE_WEDGE_V1_* (U_FREEZE | OWNER_CORPUS | …)
```

**Meaning:** E4 executed under `AUTHORIZE_E4_BUILDER_AND_EXECUTE`. Verdict **KILL**.
Generative+verify does not beat classical under frozen \(U_{R★}\) on locked R★.
Wedge v1 Phases 1–3 + noisy + papers dogfood **done** (8/8); LM probe **not indicated**; runtime CLI live.
No NanoScribe/fabric expansion. At most one preregistered R★ revision then re-gate.
Next product step requires typed `AUTHORIZE_WEDGE_V1_*` (not bare `proceed`/`continue`).

## Object table

| Object | Canonical status | Notes |
|--------|------------------|-------|
| Evidence freeze tag | `IMMUTABLE_HISTORICAL` | `post-alpha-evidence-freeze-2026-07-31` → `a9d12cb1c456f6c465284e1d469c6326cb14d329` |
| Paper α core | `PUBLIC_FROZEN_CORRECTED` | 32.8M nano; schedule-aware scale; M1-specific E1; E3 agent-rubric |
| E1 | `PUBLIC_EVIDENCE_ARCHIVED` | KILL; M1 > official M0; M2 within δ, does not dominate |
| E2 | `GATED_STOP` / `NO_RESULT` | Not in flight |
| E3 normalize | `PUBLIC_EVIDENCE_ARCHIVED` | 0/486 |
| E3 agent audit | `AGENT_SINGLE_PASS` / `NO_IAA` | agent-rubric-pass-1; not clinician |
| E3 human arm | `NOT_RUN` | Dual-clinician/IAA open |
| Fabric | `SCOPED_VERIFICATION_SLICE` / `NOT_PRODUCT` | ≠ NanoScribe |
| R★ | `WORLD_FROZEN` | `trajectory/e4/data/rstar_world_manifest.json` |
| E4 | `EXECUTED` / `KILL` | `results_e4_utility.json`; U_class≈0.638 (C-M2) vs U_gen≈−1.623 |
| Program execution | `IDLE_AFTER_DOGFOOD` | science track still `IDLE_AFTER_E4_KILL` |
| Authorized nonexecution work | `NONE` | |

## Transition log

| When | Change |
|------|--------|
| Freeze tag | Evidence boundary at `a9d12cb1c456f6c465284e1d469c6326cb14d329` |
| AUTHORIZE_E4_DESIGN_ONLY | Design carve-out; execution blocked |
| POST-FREEZE DESIGN-CLOSURE | Split PROGRAM_EXECUTION vs AUTHORIZED_NONEXECUTION_WORK; E4=DESIGN_DRAFT |
| AUTHORIZE_E4_BUILDER_AND_EXECUTE | Builder+Stage 4 unlocked 2026-07-31 |
| E4 KILL | Gate 4 KILL; R★ product track STOP; revision budget 1 remaining |
