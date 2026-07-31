# Canonical Status Table

**Authority:** this file + `CANONICAL_STATUS_TABLE.json`  
**Updated:** 2026-07-31 (POST-FREEZE DESIGN-CLOSURE)  
**Rule:** every status-bearing doc must derive from this table.

## Program posture (exact separation)

```text
PROGRAM_EXECUTION_STATUS: IDLE_AFTER_FREEZE
AUTHORIZED_NONEXECUTION_WORK: E4_DESIGN_ONLY
E4_PROTOCOL_STATUS: DESIGN_DRAFT
E4_EXECUTION_STATUS: BLOCKED
E4_WORLD_STATUS: NOT_FROZEN
E4_DATA_STATUS: NONE
E4_RESULT_STATUS: NONE
FABRIC_STATUS: SCOPED_VERIFICATION_SLICE
NANOSCRIBE_STATUS: ARCHITECTURAL_RESEARCH_PROGRAM
OLD_TASK_GENERATIVE_SUBSTRATE: FALSIFIED_UNDER_FROZEN_U
NANOSCRIBE_PRODUCT_EXPANSION: STOP
```

**Meaning:** program **execution** is idle. Documentation and protocol design are allowed under `E4_DESIGN_ONLY`. No experiment is authorized.

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
| R★ | `DESIGN_HARDENED` (protocol text) | World NOT_FROZEN |
| E4 | `DESIGN_DRAFT` / `EXECUTION_BLOCKED` / `WORLD_NOT_FROZEN` / `NO_DATA` / `NO_RESULT` | Committed design ≠ freeze |
| Program execution | `IDLE_AFTER_FREEZE` | |
| Authorized nonexecution work | `E4_DESIGN_ONLY` | |

## Transition log

| When | Change |
|------|--------|
| Freeze tag | Evidence boundary at `a9d12cb1c456f6c465284e1d469c6326cb14d329` |
| AUTHORIZE_E4_DESIGN_ONLY | Design carve-out; execution blocked |
| POST-FREEZE DESIGN-CLOSURE | Split PROGRAM_EXECUTION vs AUTHORIZED_NONEXECUTION_WORK; E4=DESIGN_DRAFT |
