# Wedge v1 Phase 2 — Authorization Record

```yaml
doc_type: auth_record
valid_only_if_queued: true
queue_path: papers/EXECUTION_QUEUE.md
auth_ids: [AUTHORIZE_WEDGE_V1_CLASSICAL_BASELINE]
governance_status: PENDING_QUEUE_ROW  # queue currently says NEXT_REQUIRES_AUTH for this string
may_authorize_execution: false  # becomes true only when EXECUTION_QUEUE lists this auth_id as live
```


**Date:** 2026-07-31
**Auth string:** `AUTHORIZE_WEDGE_V1_CLASSICAL_BASELINE`
**Trigger:** Owner "continue" after Wedge v1 Phase 1 + first-principles hardening
**Lock:** `papers/WEDGE_V1.md`

## Authorized

| Item | Status |
|------|--------|
| Synthetic mini-corpus under frozen I*/X* | Authorized |
| Classical solvers + verifier for T01–T40 | Authorized |
| Score frozen draft U on clean track | Authorized |
| Write `results_wedge_v1_classical.json` | Authorized |

## Not authorized

```text
TRAINING = NOT_AUTHORIZED
LM_SOLVERS = NOT_AUTHORIZED
E4_EXECUTE = BLOCKED
NANOSCRIBE = STOP
MEMORY_AGENTS_UI = STOP
PROGRAM1 = NOT_AUTHORIZED
OLD_TASK_U = FORBIDDEN
```

## Consequence

Phase 2 RESULT informs whether classical+verify already delivers useful U on this
wedge. Phase 3 (LM) requires a separate auth only if Delta U_LM is expected positive
on E-class failures.
