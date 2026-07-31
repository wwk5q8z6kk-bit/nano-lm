# Mitigation status scorecard (live)

**Updated:** 2026-07-31T18:18Z · Method: `FIRST_PRINCIPLES_RISK_MITIGATION.md`  
**Rule:** CLOSED = exit criterion met with enforcing artifact; PARTIAL = design/docs exist, gate incomplete; OPEN = residual risk; PARKED = science/curiosity only.  
**Namespace:** `LAB.B*` = this scorecard / first-principles map; `WED.B*` = `papers/WEDGE_V1.md` risk register.

| ID | Blocker | Status | Enforcing artifact / residual |
|----|---------|--------|-------------------------------|
| B1 | Auth forgery | **PARTIAL** | `lint_claim_auth.py` PASS; runners must still fail-closed on missing queue |
| B2 | Premature freeze recreate | **CLOSED** | Lint forbids recreate; tags preserved |
| B3 | E1 L/C PUBLIC_PARTIAL | **PARTIAL** | Protocol exists; replay needs auth |
| B4 | E2 STOP | **CLOSED** (parked) | No RESULT; product path does not need E2 |
| B5 | E3 clinician IAA | **OPEN** (parked) | Agent-rubric labeled; human arm NOT_RUN |
| B6 | E4 / substrate revival | **CLOSED→watch** | E4 EXECUTED/KILL on tested R★; product track STOP; revision budget 1; do not reopen without typed auth |
| B7 | Scale/50× overclaim | **PARTIAL** | Correction note + ledger split; watch public α PDF |
| B8 | C3 L / morph | **PARKED** | Scopes honest |
| B9 | Fabric≠OS | **PARTIAL** | README + boundary; watch ambition docs |
| B10 | Clinical / zero-halluc | **PARTIAL** | Lint phrases; ledger FORBIDDEN rows |
| B11 | Dirty-tree contamination | **PARTIAL** | Allowlists; tree still dirty at times |
| B12 | Ambition↔evidence | **PARTIAL** | Constitution + `PUBLIC_ONE_PAGER.md` |
| B13 | Wedge before classical | **CLOSED** | Phase 2+3 E-class `ECLASS_CLOSED_WITHOUT_LM`; LM not indicated; watch A–D reopen |
| B14 | Auth scope overgrant | **PARTIAL→CLOSED watch** | `auth_gate.py` enforces bits; eval AUTH narrowed to `[execute_eval]` (commit unbundled) |
| B15 | Gateway-only consult | **CLOSED** | Documented as intended |
| B16 | Context-of-use drift | **PARTIAL** | E1 GATE rows have `context_of_use` in ledger JSON; full validation incomplete |
| B17 | Freeze-tag honesty | **PARTIAL** | Deferred; `CLEAN_LINEAGE_FREEZE_RECIPE.md` design landed; still needs OWNER_TAG_OK |
| B18 | Decision vs cost split | **PARTIAL** | Design note + offline decision tests; ledger claim IDs pending owner |
| B19 | Synthetic perfect-U overclaim | **PARTIAL** | SYNTHETIC_MINI banners + one-pager; dogfood≠owner corpus; CONTACT_CLOCK design |
| B20 | Governance cosplay | **PARTIAL** | One-pager refreshed; CONTACT_CLOCK; still need OWNER_CORPUS or PRODUCT_STOP |
| B21 | Token methods residual in α PDF | **PARTIAL→mostly closed** | `paper1.tex` + draft cite 32.8M for nano; confirm PDF rebuild if shipping camera-ready |
| B22 | Multi-agent/API process failure | **PARTIAL** | Council/CLI credit failures; prefer gateway+Cursor dual-path |
| B23 | Session-continue ≠ authority | **CLOSED** | Speech-act table + classifier + unit tests; AUTH_RUNTIME hardened (historical continue ≠ precedent) |

| B24 | Draft-U Goodhart | **OPEN** | Need U_FREEZE or mandatory DRAFT label on all citations |
| B25 | Gold-leaking classical heuristics | **CLOSED→watch** | T35 classical ABSTAIN; T29 extracts from docs; eclass probes restored; re-scored |
| B26 | Task-check vs claim-level U | **PARTIAL** | `claim_level` in classical RESULT; U still DRAFT until U_FREEZE |
| B27 | Product-contact vacuum | **OPEN** | `papers/CONTACT_CLOCK.md` design ready; contact event still missing |
| B28 | Dual estimand unwired | **PARTIAL** | Design exists; schema not in RESULT yet |
| B29 | Recipe/solver hash drift | **OPEN** | Pin hashes in recipe_freeze; runner check |
| B30 | LAB.B* vs WED.B* namespace | **PARTIAL** | Rule in §9; migrate citations gradually |

## Priority residual (do these)

1. **B25** — Prediction/scoring firewall (no gold imports in classical proposers)  
2. **B24/B26** — DRAFT-U labeling + claim-level reporting (U_FREEZE when owner types it)  
3. **B19/B20/B27** — Owner real-corpus classical contact (or explicit product STOP)  
4. **B14** — Enforce `scope_bits` in runners (lint alone is theater)  
5. **B16/B18** — Owner commit ledger `context_of_use` + decision/cost claim IDs  
6. **B6** — Owner one-liner: PARK / VOID / RATIFY E4 surface  
7. **B23** — Keep speech-act classifier on gated turns; `continue` ≠ execute  
8. **B3** — Optional `AUTHORIZE_E1_LC_REPLAY`  

## Forbidden “mitigations”

- Reopening old-task generative under `OLD_TASK_U`  
- Authorizing E4 to “fix packaging”  
- Treating agent-rubric as clinician IAA  
- Retargeting protected freeze tags

Companion this session: `papers/M0_CONTINUE_RESIDUALS.md`.
