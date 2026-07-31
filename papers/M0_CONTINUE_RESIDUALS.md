# M0 residuals after CONTINUE_SESSION (2026-07-31T18:18Z)

**Force used:** `continue` → CONTINUE_SESSION (no scope bits).  
**Lint:** `scripts/lint_claim_auth.py` PASS.  
**Not done:** commit / tag / push / execute / E4 / owner-corpus.

## Verified this session (M0)

| Check | Result |
|-------|--------|
| Speech-act classifier on `continue` | CONTINUE_SESSION |
| Auth lint | PASS |
| α LaTeX nano token budget | **32.8M** present in `papers/latex/paper1.tex` methods |
| α draft nano token budget | **32.8M** present in `papers/paper1_draft.md` |
| Public one-pager | Present |
| Mitigation scorecard | Present + refreshed |
| Runtime CLI smoke (prior) | TTL ask SUPPORTED — not re-executed this continue |

## Still needs a typed force (not CONTINUE_SESSION)

| Residual | Say |
|----------|-----|
| Commit dirty mitigation/docs tree | `authorize commit` |
| Real-corpus classical contact (B19/B20) | Queue + `AUTHORIZE_WEDGE_V1_OWNER_CORPUS` |
| Freeze wedge U | `AUTHORIZE_WEDGE_V1_U_FREEZE` |
| E4 disposition | `PARK_AS_EXPLORATORY` \| `VOID_E4_AUTH` \| `RATIFY_E4_EXECUTE` |
| Optional L/C replay | `AUTHORIZE_E1_LC_REPLAY` (if queued) |
| Tag policy | `authorize tag` + tip policy |

## Default if no typed force

Remain on runtime-slice + measurement record. Product contact stays open as B19/B20 until owner-corpus auth or explicit idle/park.

Re-run: `python3 scripts/m0_consistency_audit.py` → `papers/M0_CONSISTENCY_AUDIT.md`.
