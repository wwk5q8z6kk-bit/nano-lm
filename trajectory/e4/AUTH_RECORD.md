# E4 authorization record

```yaml
doc_type: auth_record
valid_only_if_queued: true
queue_path: papers/EXECUTION_QUEUE.md
auth_ids: [AUTHORIZE_E4_BUILDER_AND_EXECUTE]
governance_status: AMBIGUOUS_PENDING_OWNER  # chat "authorized" was agent-expanded; see SWARM_QUEEN_SYNTHESIS
may_authorize_execution: false  # false until EXECUTION_QUEUE lists auth_ids and owner RATIFY/VOID/PARK
```


**Owner call (verbatim context):** `authorized` (typo for authorized)  
**Interpreted as:** `AUTHORIZE_E4_BUILDER_AND_EXECUTE`  
**Date:** 2026-07-31  
**Repo:** `/Users/mac/Projects/nano-lm`

## Scope unlocked

| Item | Status |
|------|--------|
| Builder / data checklist (PREREG §5 B1–B14) | **Authorized** |
| R★ world freeze + content-addressed splits | **Authorized** |
| Classical probe + Stage 4 scoring under frozen \(U_{R★}\) | **Authorized** |
| Paid GPU / RunPod if needed for G-ref | **Authorized** (prefer cheapest fit; document venue) |

## Still forbidden

- Reopen E1 / old-task substrate under `OLD_TASK_U`
- E2 mechanism claims
- Equating Fabric with NanoScribe / claiming NanoScribe implemented
- Mislabeling E3 agent rubrics as human/clinician evaluation
- Altering E1 frozen U weights
- Silent mid-stream \(U_{R★}\) weight edits after scores (VOID)

## Design freeze at auth

Weights / thresholds / consequences frozen from
`trajectory/PREREG_E4_Rstar_killgate.md` at the SHA recorded in
`trajectory/e4/recipe_freeze.json`. Amendments after unlock require explicit
owner note before scoring; post-score weight edits VOID the decision.

## Budget (B13)

| Field | Value |
|-------|-------|
| Preferred venue | Local Apple MPS (no paid GPU unless needed) |
| Fallback | Cheapest RunPod/CUDA fit if MPS insufficient |
| Owner acceptance | Implied by `AUTHORIZE_E4_BUILDER_AND_EXECUTE` + “Paid GPU/RunPod OK” |
