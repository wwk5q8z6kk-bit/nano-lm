# Execution Queue

**Operating doc — what we are authorized to do *now*.**  
**Adopted:** 2026-07-31  
**Park:** `papers/PARK_NOTE_2026-07-31.md`

```text
PROGRAM_EXECUTION_STATUS: IDLE_PARKED
AUTHORIZED_WORK: NONE
LAST_PUSH: origin/master @2ad06d24c4f7
RECONCILED_FREEZE_TAG: post-alpha-reconciled-evidence-freeze-2026-07-31 @67bf87b1f968
PREMATURE_FREEZE_TAG: post-alpha-evidence-freeze-2026-07-31 UNMOVED
LM_PROBE: NOT_INDICATED
TRAINING: NOT_AUTHORIZED
E4_RESULT: KILL  # EXECUTED; not blocked/untested
NANOSCRIBE: STOP
```

## Queue

| Priority | Item | Status |
|----------|------|--------|
| 0 | Freeze integrity | Standing — protected tags unmoved |
| — | All execute recipes | Empty until typed `AUTHORIZE_*` |

Idle is the default. Owner speech acts: `papers/OWNER_SPEECH_ACTS.md`.
