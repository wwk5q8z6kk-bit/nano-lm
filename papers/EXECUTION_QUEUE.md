# Execution Queue

**Operating doc — what we are authorized to do *now*.**  
**Adopted:** 2026-07-31  
**Constitution:** `LABORATORY_CONSTITUTION.md`  
**Strategic center:** `papers/STRATEGIC_RESET.md`

```text
PROGRAM_EXECUTION_STATUS: PROGRAM_A_A1_ACCEPTED
AUTHORIZED_NONEXECUTION_WORK: OWNER_ACCEPT_A1_DESIGN
BENCHMARK_PROGRAM0_STATUS: DONE
EVAL_INFRA_STATUS: AVAILABLE
PROGRAM_CHOICE: A
UNIT: A1_RSTAR_REVISION_DESIGN
A1_STATUS: OWNER_ACCEPTED
PROGRAM1: NOT_AUTHORIZED
TRAINING: NOT_AUTHORIZED
E4_RSTAR_V2_EXECUTE: NOT_AUTHORIZED
E4_EXECUTION_STATUS: BLOCKED
NANOSCRIBE_PRODUCT_EXPANSION: STOP
OLD_TASK_RUNS_UNDER_OLD_TASK_U: FORBIDDEN
```

## Queue (intentionally short)

| Priority | Item | Auth | Type | Notes |
|----------|------|------|------|-------|
| 0 | Maintain evidence freeze integrity | Standing | Ops | Do not move freeze / alpha tags; leave unrelated dirty freeze/audit files alone |
| 1 | A1 design (closed) | `OWNER_ACCEPT_A1_DESIGN` | Docs | Accepted; see decision record |
| — | E4′ / R★ v2 execute | — | — | **Empty** until `AUTHORIZE_E4_RSTAR_V2_EXECUTE` |

## Explicitly not queued

- `AUTHORIZE_E4_RSTAR_V2_EXECUTE` (world rebuild, G-ref train, E4′ score)  
- Program 1 world census  
- E2 · Fabric V2 · NanoScribe  
- Training / paid compute  
- Old-task generative revival under `OLD_TASK_U`  

## How items enter this queue

Only via `DECISION_GATES.md` + explicit owner authorization string.

## Standing hygiene (not experiments)

See `FIRST_PRINCIPLES_RISK_MITIGATION.md` §5 M0 checklist:

- Auth lint / allowlist (B1)
- Freeze tag preserve (B2)
- Dirty-tree allowlists (B11)
- Anomaly log / NONCLAIM banners (B12)

These do **not** replace `AUTHORIZE_WEDGE_V1_CLASSICAL_BASELINE`.

