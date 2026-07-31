# Execution Queue

**Operating doc — what we are authorized to do *now*.**  
**Adopted:** 2026-07-31  
**Constitution:** `LABORATORY_CONSTITUTION.md` (in-repo governance; Program 0 = authorized benchmark infra)

```text
PROGRAM_EXECUTION_STATUS: IDLE_AFTER_FREEZE
AUTHORIZED_NONEXECUTION_WORK: E4_DESIGN_ONLY | BENCHMARK_SUPREMACY_LAB_PROGRAM0_INFRA
BENCHMARK_LAB_STATUS: INFRASTRUCTURE_ONLY
E4_EXECUTION_STATUS: BLOCKED
NANOSCRIBE_PRODUCT_EXPANSION: STOP
OLD_TASK_RUNS_UNDER_OLD_TASK_U: FORBIDDEN
```

## Queue (intentionally short)

| Priority | Item | Auth | Type | Notes |
|----------|------|------|------|-------|
| 0 | Maintain evidence freeze integrity | Standing | Ops | Do not move `post-alpha-evidence-freeze-2026-07-31` |
| 1 | E4/R★ **design docs only** | `AUTHORIZE_E4_DESIGN_ONLY` | Docs | No world, data, GPU, result |
| 2 | Benchmark Program 0 | `BENCHMARK_SUPREMACY_LAB_PROGRAM0_INFRA` | Infra | One reproducible benchmark path inside nano-lm; Program 1 NOT included |
| — | *(empty for measurement experiments)* | — | — | Idle is valid |

## Explicitly not queued

- E4 Stage 4 execution  
- E2 runs  
- Fabric V2 / NanoScribe control plane / memory/routing/UI  
- Old-task generative revival under `OLD_TASK_U`  
- “Implement the Technology Roadmap”

## How items enter this queue

Only via `DECISION_GATES.md` + explicit owner authorization string.  
Portfolio and Roadmap items do **not** auto-enter.

## Idle ≠ stop dreaming

An empty experiment queue is the correct state when no gate is passed.  
See `RESEARCH_PORTFOLIO.md` and `TECHNOLOGY_ROADMAP.md` for ambition.


## Explicitly not queued (benchmark lab)

- Program 1 world benchmark census
- MMLU / HELM / MLPerf / full Tier-B suites
- Leaderboard promotion or Evidence Ledger rows from Program 0
- Training, paid compute, E4 execute


```text
PROGRAM1 = DEFERRED_PENDING_GATE0
```
