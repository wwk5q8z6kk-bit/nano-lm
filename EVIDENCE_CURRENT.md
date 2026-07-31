# EVIDENCE_CURRENT

**Status:** Reconciled freeze tag **RATIFIED** under owner `authorize tag clean-lineage`  
**Updated (UTC):** 2026-07-31T18:32Z  
**`origin/master`:** `2ad06d24c4f72b292f73ef098fdcc0ce2a008659` (working tip; E4 ancestry=YES — **not** a freeze brand)  
**Clean-lineage tip:** `67bf87b1f968a38e68c0225b2b556f7bba5ea1cc` (branch `freeze/clean-lineage-2026-07-31`)  
**Tag:** `post-alpha-reconciled-evidence-freeze-2026-07-31` → peeled `67bf87b1f968a38e68c0225b2b556f7bba5ea1cc` (tag object `1bdb6586d8ae69de6731eccb11af71d5741bdfaa`)  
**Owner receipt:** `.autonomous/post-alpha-freeze-hybrid/OWNER_TAG_OK` (tip_policy=clean-lineage; authorize_tag_push=false)

Stratigraphy: [`audit/discussion-to-implementation/STRATIGRAPHY.md`](audit/discussion-to-implementation/STRATIGRAPHY.md)  
Verify log: `.autonomous/post-alpha-freeze-hybrid/TAG_CLEAN_LINEAGE_VERIFY.txt`

## Immutable public layers

| Layer | Object | SHA | Role |
|------:|--------|-----|------|
| 1 | `paper-alpha-v1` | `0e01d73205e9c35ea32925fd4d6c7e5fceb61137` | Paper α science freeze — **DO NOT MOVE** |
| 2 | `post-alpha-evidence-freeze-2026-07-31` | `a9d12cb1c456f6c465284e1d469c6326cb14d329` | PREMATURE public evidence tag — **PRESERVE / DO NOT MOVE** |
| 3 | `post-alpha-reconciled-evidence-freeze-2026-07-31` | peeled `67bf87b1f968a38e68c0225b2b556f7bba5ea1cc` | Clean-lineage reconciled freeze (E4 `6af178d` **not** ancestor) — **ratified; do not retarget** |

## Residuals (honest; do not overclaim)

```text
E2 = GATED_STOP
E4 = BLOCKED                 # further curiosity runs; kill on master ≠ freeze fold-in
FABRIC = GATED
E3 human/clinician = UNRESOLVED
tokenizer_hash (C3 base) = ABSENT_FROM_RESULTS_JSON
PUBLIC_EVIDENCE_FREEZE = CLEAN_LINEAGE_TAG_PRESENT
PROGRAM_STATE = IDLE_AFTER_CLEAN_LINEAGE_TAG_RATIFY
```

## Non-freeze snapshot (not a stratigraphy layer)

Owner `authorize tag non-freeze-snapshot` (2026-07-31T18:33Z): local tag `snapshot/master-2ad06d2-2026-07-31` @ `2ad06d24c4f72b292f73ef098fdcc0ce2a008659` **ratified**.
**Not** a freeze brand. **Not** pushed. Does not alter layers 1–3 above.
Verify: `.autonomous/post-alpha-freeze-hybrid/TAG_NONFREEZE_SNAPSHOT_VERIFY.txt`

## Verdict annotation (not a stratigraphy layer)

Owner `authorize tag verdict-annotation` (2026-07-31T18:33Z): local additive `verdict/reconciled-freeze-clean-lineage@67bf87b` @ `67bf87b1f968a38e68c0225b2b556f7bba5ea1cc`.
Discloses clean-lineage freeze tip vs E4-containing `origin/master`. **Not** a freeze product. **Not** pushed.
Verify: `.autonomous/post-alpha-freeze-hybrid/TAG_VERDICT_ANNOTATION_VERIFY.txt`

## Non-authorizations

No E2/E4 reopen · no fabric/NanoScribe · no Stage M · no clinical readiness · no force-moved tags · no inventing measurements · no retarget of this reconciled tag under this receipt.
