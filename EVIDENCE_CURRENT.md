# EVIDENCE_CURRENT

**Status:** RECONCILED TAG **DEFERRED** (not proposed-for-immediate-tag on current tip)  
**Updated (UTC):** 2026-07-31T18:18Z
**`origin/master`:** `b9c3b604453b55c59a4cdb49107183e356447407`  
**Local HEAD:** `b9c3b604453b55c59a4cdb49107183e356447407` (== origin)
**Proposed tag name:** `post-alpha-reconciled-evidence-freeze-2026-07-31` — **ABSENT**  
**Tag decision:** **DEFER** — current tip ancestry includes E4 execute commit `6af178d` (`ancestor=YES`). Do **not** label this tip as a freeze.

This file is the **single root pointer** for “what is the current post-Paper-α evidence tip?”  
Stratigraphy: [`audit/discussion-to-implementation/STRATIGRAPHY.md`](audit/discussion-to-implementation/STRATIGRAPHY.md)  
Closeout: [`audit/discussion-to-implementation/COUNCIL_HYBRID_CLOSEOUT.md`](audit/discussion-to-implementation/COUNCIL_HYBRID_CLOSEOUT.md)

## Immutable public layers

| Layer | Object | SHA | Role |
|------:|--------|-----|------|
| 1 | `paper-alpha-v1` | `0e01d73205e9c35ea32925fd4d6c7e5fceb61137` | Paper α science freeze — **DO NOT MOVE** |
| 2 | `post-alpha-evidence-freeze-2026-07-31` | `a9d12cb1c456f6c465284e1d469c6326cb14d329` | PREMATURE public evidence tag — **PRESERVE / DO NOT MOVE** |
| 3 | reconciled freeze tag | — | **DEFERRED** until clean lineage or explicit non-freeze snapshot under `OWNER_TAG_OK` |

## What is on origin (post-freeze chronology; not a freeze brand)

Includes (among other commits): claim corrections `1fc8eea`, durable C3 `ea001d4`, later docs, **and** E4 R★ kill-gate commit `6af178d`.  
E4 ancestry ≠ E4 authorization for further runs. Do not fold E4 into a freeze brand.

## Residuals (honest; do not overclaim)

```text
E2 = GATED_STOP
E4 = BLOCKED                 # further curiosity runs; ancestry on tip ≠ authorization
FABRIC = GATED               # ≠ NanoScribe; no expansion implied
E3 human/clinician           = UNRESOLVED (agent-rubric faithful only)
tokenizer_hash (C3 base)     = ABSENT_FROM_RESULTS_JSON
PUBLIC_EVIDENCE_FREEZE       = HISTORICAL_TAG_ONLY  # premature present; reconciled ABSENT/INCOMPLETE
PROGRAM_STATE                = IDLE_AFTER_HYBRID_COMMIT
```

Unrelated local dirty/untracked work (wedge_v1, PROGRAM_A*, etc.) is **not** authorized freeze evidence.

## Non-authorizations

No E2 GPU · no E4 execution reopen · no fabric/v2 · no NanoScribe build · no Stage M / OLD_TASK_U · no clinical readiness · no force-moved tags · no inventing measurements · no freeze-brand tag on E4-containing tip without explicit OWNER_TAG_OK naming a chosen tip.

## Owner options (from closeout)

1. Remain deferred (default)
2. Clean lineage: branch from premature freeze tag, cherry-pick non-E4 freeze docs, tag under `OWNER_TAG_OK` (`authorize_tag: true`)
3. Non-freeze snapshot tag on HEAD — name must **not** claim freeze; still needs `OWNER_TAG_OK`
