# Execution Queue

**Operating doc — what we are authorized to do *now*.**  
**Adopted:** 2026-07-31  
**Constitution:** `LABORATORY_CONSTITUTION.md`  
**Strategic center:** `papers/STRATEGIC_RESET.md`  
**Active wedge:** `papers/WEDGE_V1.md`

```text
PROGRAM_EXECUTION_STATUS: IDLE_AFTER_WEDGE_V1_PHASE3_ECLASS
AUTHORIZED_NONEXECUTION_WORK: NONE
PHASE2_RESULT: wedge_v1/results_wedge_v1_classical.json
PHASE3_ECLASS_RESULT: wedge_v1/results_wedge_v1_phase3_eclass.json
PHASE3_ECLASS_VERDICT: ECLASS_CLOSED_WITHOUT_LM
LM_PROBE: NOT_INDICATED
TRAINING: NOT_AUTHORIZED
A_D_LM_REOPEN: FORBIDDEN
E4_EXECUTION_STATUS: BLOCKED
NANOSCRIBE_PRODUCT_EXPANSION: STOP
OLD_TASK_RUNS_UNDER_OLD_TASK_U: FORBIDDEN
```

## Queue

| Priority | Item | Auth | Type | Notes |
|----------|------|------|------|-------|
| 0 | Freeze integrity | Standing | Ops | Do not move premature freeze tag |
| 1 | Wedge Phase 1–2 classical | Done | Measure | Clean-track baseline |
| 2 | Phase 3 E-class non-LM | Done | Measure | `ECLASS_CLOSED_WITHOUT_LM`; ΔU≈0.0090 < δ; LM not invoked |
| — | Phase 3 LM probe | — | — | **Empty** — `lm_still_needed=false` |
| — | General LM productization | — | — | Forbidden |

## Explicitly not queued

- `AUTHORIZE_WEDGE_V1_PHASE3_LM_PROBE` (not indicated)  
- Reopening A–D with LM · training · E4′ · NanoScribe · agents/memory  

## How items enter this queue

Only via `DECISION_GATES.md` + explicit owner authorization string.
