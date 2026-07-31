# Execution Queue

**Operating doc — what we are authorized to do *now*.**  
**Adopted:** 2026-07-31  
**Constitution:** `LABORATORY_CONSTITUTION.md`  
**Strategic center:** `papers/STRATEGIC_RESET.md`  
**Active wedge:** `papers/WEDGE_V1.md`

```text
PROGRAM_EXECUTION_STATUS: WEDGE_V1_CLASSICAL_BASELINE_COMPLETE
AUTHORIZED_WORK: AUTHORIZE_WEDGE_V1_CLASSICAL_BASELINE
RESULT: wedge_v1/results_wedge_v1_classical.json
OWNER_TRIGGER: proceed (2026-07-31T18:12:37.114478+00:00)
BENCHMARK_PROGRAM0_STATUS: DONE
TRAINING: NOT_AUTHORIZED
LM_SOLVERS: NOT_AUTHORIZED
E4_EXECUTION_STATUS: BLOCKED
E4_RSTAR_V2_EXECUTE: NOT_AUTHORIZED
NANOSCRIBE_PRODUCT_EXPANSION: STOP
OLD_TASK_RUNS_UNDER_OLD_TASK_U: FORBIDDEN
```

## Queue

| Priority | Item | Auth | Type | Notes |
|----------|------|------|------|-------|
| 0 | Freeze tag integrity | Standing | Ops | Do not move premature freeze tag |
| 1 | Wedge v1 Phase 1 lock | `WEDGE_V1` | Docs | Done |
| 2 | Classical baseline (clean track) | `AUTHORIZE_WEDGE_V1_CLASSICAL_BASELINE` | Measure | **DONE** — U≈0.891; Q=1.0 on 50 checks |
| — | Phase 3 LM solvers | — | — | Needs separate auth + ΔU expectation |

## Explicitly not queued

- Program A1 / E4-prime / R★ v2 execute  
- Training / paid compute / NanoScribe / Fabric V2 / agents / memory  
- Old-task generative revival under `OLD_TASK_U`  

## Standing hygiene

`FIRST_PRINCIPLES_RISK_MITIGATION.md` §5; `scripts/lint_claim_auth.py`.
