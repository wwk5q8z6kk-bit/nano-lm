# FREEZE_VERIFICATION_REPORT

*Generated 2026-07-31T14:05:23.760984+00:00*
*HEAD `0e01d73205e9c35ea32925fd4d6c7e5fceb61137`*
*Proposed tag (NOT created): `post-alpha-evidence-freeze-2026-07-31`*

## Checks

| Check | Result |
|---|---|
| pytest (configured) | PASS (15) |
| fabric/test_fabric.py | PASS (8) |
| trajectory/test_recompute_c3.py | PASS (7) |
| E1 U recompute from rows | PASS |
| E1 decision M1>M0 | PASS (0.998999 > 0.925217) |
| E3 auto 0/486 | PASS |
| E3 agent-rubric rater | PASS (`agent-rubric-pass-1`) |
| Artifact manifests present | PASS (`artifacts/e1`, `artifacts/e3`, root) |
| SHA256SUMS present | PASS (root + artifacts/) |
| Active pods | ID	NAME	GPU	IMAGE NAME	STATUS |
| Tag created | NO (owner-only) |

## Stale-term scan (public/lock files)

Hits remaining after remediation: **1** (see below; audit/history files excluded from this scan).


| File | Line | Pattern | Text |
|---|---:|---|---|
| `trajectory/DECISION_P1_program_lock.md` | 131 | `human arm` | ### 3.2 Human arm requirements (optional; only if information is worth the rater cost) |

## Corrected / weakened / preserved

See `POST_ALPHA_EVIDENCE_FREEZE.md`.

## Verdict

- Paper α: **READY_AFTER_OWNER_APPROVAL**
- Freeze: **FREEZE_CANDIDATE_READY**
- Next decision: **OWNER_APPROVAL_REQUIRED** (tag + idle); default after owner freeze: **IDLE_AFTER_FREEZE**
- E4: **not recommended** as momentum continuation
