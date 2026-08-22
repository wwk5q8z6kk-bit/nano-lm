> **ARCHIVED** 2026-08-22 — preserved from `papers/EXECUTION_QUEUE.md` at `origin/master` (2ad06d2).

# Execution Queue

**Operating doc — what we are authorized to do *now*.**  
**Adopted:** 2026-07-31  
**Constitution:** `LABORATORY_CONSTITUTION.md`  
**Strategic center:** `papers/STRATEGIC_RESET.md`  
**Active wedge:** `papers/WEDGE_V1.md`

```text
PROGRAM_EXECUTION_STATUS: IDLE_AFTER_DOGFOOD
AUTHORIZED_NONEXECUTION_WORK: NONE
RUNTIME_SLICE: LIVE
DOGFOOD_RESULT: wedge_v1/results_wedge_v1_dogfood.json
DOGFOOD_ACCURACY: 1.0 (8/8)
LM_PROBE: NOT_INDICATED
TRAINING: NOT_AUTHORIZED
NANOSCRIBE_PRODUCT_EXPANSION: STOP
OLD_TASK_RUNS_UNDER_OLD_TASK_U: FORBIDDEN
```

## Queue

| Priority | Item | Auth | Type | Notes |
|----------|------|------|------|-------|
| 0 | Freeze integrity | Standing | Ops | Do not move premature freeze tag |
| 1 | Classical + E-class | Done | Measure | LM not indicated |
| 2 | Runtime CLI | Done | Build | `ask|find|scan|smoke|dogfood` |
| 3 | Papers dogfood | Done | Measure | 8/8 classical; fail-closed OOS |
| — | LM / memory / NanoScribe | — | — | Forbidden |

## Explicitly not queued

- LM probe · training · E4′ · agents · memory · Fabric v2  

## How items enter this queue

Only via `DECISION_GATES.md` + explicit owner authorization string.

## Speech-act note

Owner `continue` ⇒ `CONTINUE_SESSION` (`papers/OWNER_SPEECH_ACTS.md`): ungated M0 only.  
Does **not** authorize `U_FREEZE`, `OWNER_CORPUS`, commit, tag, push, or LM.
