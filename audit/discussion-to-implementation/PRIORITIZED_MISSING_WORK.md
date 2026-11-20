# Prioritized Missing Work

Priority = (Scientific value × Risk reduction × Dependency centrality × Evidence readiness) / (Cost × Complexity × Collision risk), each factor 1–5.

| ID | Item | Class | Sci | Risk | Dep | EvReady | Cost | Cx | Coll | Priority | Notes |
|----|------|-------|-----|------|-----|---------|------|----|------|----------|-------|
| M1 | Commit or formally declare working-tree post-α locks/results | MUST_FIX_BEFORE_CLAIM | 5 | 5 | 5 | 5 | 2 | 2 | 2 | **7.8** | Archival truth gap |
| M2 | Converge E2 status to GATED/STOP; no RESULT | MUST_FIX_BEFORE_CLAIM | 4 | 5 | 4 | 5 | 1 | 1 | 2 | **20.0** | Cheap doc sync |
| M3 | Sync E3 human status; clarify agent-rubric ≠ clinician | MUST_FIX_BEFORE_CLAIM | 5 | 5 | 4 | 5 | 1 | 1 | 2 | **25.0** | Highest cheap win |
| M4 | Fix ρ definition in DECISION_P1 | MUST_FIX_BEFORE_CLAIM | 3 | 4 | 3 | 5 | 1 | 1 | 1 | **18.0** | Accuracy |
| M5 | Soften fabric Intent/append-only claims | MUST_FIX_BEFORE_CLAIM | 3 | 4 | 3 | 5 | 1 | 1 | 1 | **18.0** | Anti-overclaim |
| M6 | Add RESULT sections to stale preregs (sweep/ownstack/Tv2) | MUST_FIX_BEFORE_SUBMISSION | 4 | 3 | 3 | 5 | 2 | 2 | 1 | **7.5** | |
| M7 | Paper α limitation wording vs Stage 1 | MUST_FIX_BEFORE_SUBMISSION | 4 | 4 | 3 | 4 | 2 | 2 | 2 | **6.0** | Owner |
| M8 | Unit tests for E1 U scorer / ρ | MUST_FIX_BEFORE_NEXT_EXPERIMENT | 3 | 4 | 4 | 4 | 2 | 2 | 1 | **6.0** | |
| M9 | Adversarial fabric tests (dup/stale/malformed) | MUST_FIX_BEFORE_FABRIC_V2 | 3 | 4 | 3 | 4 | 2 | 3 | 2 | **2.7** | |
| M10 | JSONL release bundle or documented local-only policy | MUST_FIX_BEFORE_SUBMISSION | 3 | 3 | 4 | 5 | 2 | 2 | 1 | **7.5** | |
| M11 | Dual-clinician IAA on E3 pack | VALIDATED_LATER | 4 | 3 | 2 | 3 | 4 | 3 | 2 | **1.0** | |
| M12 | R★ builder + classical probe (E4 precondition) | HIGH_VALUE_NEXT | 5 | 5 | 5 | 2 | 4 | 4 | 3 | **1.0** | Needs owner Stage 4 auth |
| M13 | E2 U3 science run | SPECULATIVE_DEFER | 2 | 1 | 1 | 2 | 3 | 3 | 4 | **0.1** | Negative EV post-KILL for product |
| M14 | NanoScribe kernel/memory/UI | REMOVE_FROM_PLAN | 1 | 1 | 1 | 1 | 5 | 5 | 5 | **0.01** | Until E4 SURVIVE |
| M15 | OSS Outlines/LangExtract integrate | SPECULATIVE_DEFER | 2 | 1 | 1 | 1 | 3 | 3 | 3 | **0.07** | |
| M16 | Morphology causal prereg | VALIDATED_LATER | 3 | 2 | 2 | 2 | 3 | 3 | 2 | **0.7** | |
| M17 | Stage M induction measure | SPECULATIVE_DEFER | 2 | 1 | 1 | 2 | 3 | 3 | 3 | **0.1** | |

## Ranking note

Highest priorities are **documentation/archival truth repairs** (M2–M5, M1), not new experiments. The only experiment-shaped HIGH_VALUE_NEXT is **E4 preconditioning**, and it remains unauthorized.
