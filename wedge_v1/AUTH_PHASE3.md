
## Capability scope (B14)

```text
scope_bits: [execute_eval, commit]
valid_only_if_queued: true
```

# Wedge v1 Phase 3 — Authorization Record

**Date:** 2026-07-31
**Auth string:** `AUTHORIZE_WEDGE_V1_PHASE3_ECLASS_PROBE`
**Trigger:** Owner "proceed" after Phase 2 classical baseline
**Scope:** Close classical abstentions on E-class tasks (T35/T36/T39) via
**cheapest sufficient** probes. LM inference only if non-LM probes fail.

```yaml
governance_status: OWNER_PROCEED_2026-07-31
may_authorize_execution: true
owner_trigger: "proceed"
activated_at: "2026-07-31T18:14:28.342057+00:00"
result: wedge_v1/results_wedge_v1_phase3_eclass.json
verdict: ECLASS_CLOSED_WITHOUT_LM
```

## Authorized

| Item | Status |
|------|--------|
| Query expansion / symbolic E-class probes | **DONE** |
| Re-score hybrid U vs classical baseline | **DONE** |
| Report Delta U on E-class | **DONE** |
| Frozen-checkpoint LM inference | **Not invoked** — non-LM probes closed T35/T36/T39 |

## Not authorized

```text
TRAINING = NOT_AUTHORIZED
NANOSCRIBE = STOP
MEMORY_AGENTS_UI = STOP
E4_EXECUTE = BLOCKED
OLD_TASK_U = FORBIDDEN
GENERAL_LM_PRODUCTIZATION = STOP
```


## RESULT (2026-07-31)

```text
VERDICT = ECLASS_CLOSED_WITHOUT_LM
ECLASS_ACCURACY = 1.0
DELTA_U_HYBRID = 0.008989   # vs δ=0.05 → no material hybrid gain
LM_STILL_NEEDED = false
LM_INVOKED = false
```

**Product implication:** Admit non-LM E-class solvers (query expand / symbolic compare / coref-lite) to the wedge registry. Do **not** authorize LM probe or Nano Runtime expansion on this RESULT alone.
