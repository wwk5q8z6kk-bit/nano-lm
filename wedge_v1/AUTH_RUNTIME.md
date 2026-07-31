# Wedge v1 — Runtime slice authorization

```yaml
doc_type: auth_record
queue_path: papers/EXECUTION_QUEUE.md
auth_ids: [AUTHORIZE_WEDGE_V1_RUNTIME_SLICE]
scope_bits: [execute_eval, build]
valid_only_if_queued: true
owner_trigger_historical: continue
speech_act_note: >
  Historical activation used owner locution "continue" (pre-B23).
  Under papers/OWNER_SPEECH_ACTS.md, CONTINUE_SESSION grants no bits.
  This receipt remains valid only because EXECUTION_QUEUE lists the auth_id;
  future minting requires AUTHORIZE_* in queue + typed force, not "continue".
activated_at: 2026-07-31T18:16:46.329928+00:00
hardened_at: 2026-07-31T18:17Z
```

**Auth:** `AUTHORIZE_WEDGE_V1_RUNTIME_SLICE`  
**Trigger (historical):** Owner "continue" after `ECLASS_CLOSED_WITHOUT_LM` — **recorded, not a precedent**.  
**Scope bits:** `execute_eval`, `build` — **No LM. No memory. No NanoScribe. No freeze tag. No push.**

## Authorized

| Item | Status |
|------|--------|
| `wedge_v1/runtime.py` ask/scan API | Authorized |
| `python -m wedge_v1 ask|scan` | Authorized |
| Fail-closed ABSTAIN / NO_CORPUS | Required |

## Not authorized

```text
TRAINING = NOT_AUTHORIZED
LM_SOLVERS = NOT_AUTHORIZED
MEMORY_AGENTS_UI = STOP
NANOSCRIBE = STOP
E4_EXECUTE = BLOCKED
FREEZE_TAG = NOT_AUTHORIZED
COMMIT_PUSH = NOT_IN_THIS_RECEIPT
```

## Smoke

```bash
python -m wedge_v1 ask "How long before cached entries expire?"
python -m wedge_v1 ask "What is the clinical accuracy of NanoScribe in hospitals?"  # ABSTAIN
python -m wedge_v1 scan
```
