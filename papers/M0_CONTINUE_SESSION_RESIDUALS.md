# M0 CONTINUE_SESSION residuals

**NONCLAIM / ops hygiene.** Updated: 2026-07-31T18:20:42.210045+00:00  
**Force:** `CONTINUE_SESSION` only (`OWNER_SPEECH_ACTS.md`). Never execute/commit/tag/push.

## Cleared under M0 (do not re-do)

| Item | Evidence |
|------|----------|
| Auth lint | `scripts/lint_claim_auth.py` PASS |
| Speech-act classifier | `scripts/classify_owner_speech_act.py` |
| Wedge Phases 1–3 + noisy status sync | queue / WEDGE_V1 / AGENTS / constitution |
| B15 gateway posture | AGENTS.md + FIRST_PRINCIPLES B15 |
| B16 `context_of_use` schema note | `EVIDENCE_LEDGER.json` (E1 atoms) |
| E1 decision vs cost split design | `trajectory/E1_DECISION_VS_COST_SPLIT.md` |
| NONCLAIM banners | portfolio / roadmap / ambition |

## Still owner-gated (continue will refuse)

| Need | Typed force |
|------|-------------|
| Freeze draft wedge \(U\) | `AUTHORIZE_WEDGE_V1_U_FREEZE` |
| Private owner corpus | `AUTHORIZE_WEDGE_V1_OWNER_CORPUS` |
| E4 surface disposition | `RATIFY_E4_EXECUTE` \| `VOID_E4_AUTH` \| `PARK_AS_EXPLORATORY` |
| Commit listed paths | `authorize commit` |
| Tag policy | `authorize tag` + tip policy |
| LM probe | **not indicated** (`lm_still_needed=false`) |

## If owner keeps saying `continue`

Refresh refuse log only. Do **not** invent new labs, experiments, or auth strings.
