# Dirty-tree closure record

*Generated 2026-07-31T15:20:33.520884+00:00*
*Mode: POST-FREEZE DESIGN-CLOSURE / CONTINUE_E4_DESIGN_ONLY*
*Pre-closure HEAD:* `6f3a82362027d5e1bef22417f49a12ad8db90c30`

## Three repository states (do not conflate)

| State | Identifier | Meaning |
|-------|------------|---------|
| **1. Evidence freeze** | tag `post-alpha-evidence-freeze-2026-07-31` → `a9d12cb1c456f6c465284e1d469c6326cb14d329` | **Immutable** public evidence boundary. Do not move/delete/recreate/retarget. |
| **2. Post-freeze hygiene** | commits after freeze through `6f3a82362027` | Tests, packaging, AAEA, Paper α polish, E4 design-only ambition track. |
| **3. Design/status closure (this unit)** | commits produced in this unit | Reconciles dirty claim/status/design files; **not** a new evidence freeze; E4 remains DESIGN_DRAFT. |

## Classification table

| Path | Why changed | Scientific meaning changed? | Keep | Intended commit |
|------|-------------|----------------------------|------|-----------------|
| `audit/discussion-to-implementation/DIRTY_TREE_CLOSURE.md` | closure inventory | no | keep | `docs: canonicalize post-freeze program status` |
| `audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.md` | status separation | no | keep | `docs: canonicalize post-freeze program status` |
| `audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.json` | status machine | no | keep | `docs: canonicalize post-freeze program status` |
| `audit/discussion-to-implementation/IDLE_AFTER_FREEZE.md` | execution vs design carve-out | no | keep | `docs: canonicalize post-freeze program status` |
| `audit/discussion-to-implementation/TAG_AUDIT_POST_ALPHA.md` | immutable tag audit | no | keep | `docs: canonicalize post-freeze program status` |
| `audit/discussion-to-implementation/OWNER_APPROVAL_REQUIRED_DIFFS.md` | banner sync | no | keep | `docs: canonicalize post-freeze program status` |
| `AGENTS.md` | agent posture sync | no | keep | `docs: canonicalize post-freeze program status` |
| `trajectory/DECISION_P1_program_lock.md` | C7 agent-rubric wording | wording | keep | `docs: canonicalize post-freeze program status` |
| `trajectory/PREREG_E3_faithfulness_construct.md` | human arm -> agent audit | wording | keep | `docs: canonicalize post-freeze program status` |
| `trajectory/STAGE1_E3_CONSTRUCT_FIRST_PRINCIPLES.md` | agent-rubric clarity | wording | keep | `docs: canonicalize post-freeze program status` |
| `papers/AMBITION.md` | ambition under design-only | no | keep | `docs: harden E4 design-only governance` |
| `trajectory/PREREG_E4_Rstar_killgate.md` | E4 DESIGN_DRAFT + consequences | protocol design | keep | `docs: harden E4 design-only governance` |
| `trajectory/REGIME_P1_where_classical_fails.md` | admissibility rejects | protocol design | keep | `docs: harden E4 design-only governance` |
| `papers/MASTER_PLAN.md` | architecture memory banner; Phases 3-4 STOP | no | keep | `docs: harden E4 design-only governance` |
| `trajectory/test_e1_utility_recompute.py` | offline invariants | no | keep | `test: strengthen offline E1 and E3 invariants` |
| `trajectory/test_e3_normalize.py` | offline invariants | no | keep | `test: strengthen offline E1 and E3 invariants` |
| `papers/paper1_draft.md` | M2 delta + agent-rubric wording | clarity | keep | `paper: synchronize post-freeze factual corrections` |
| `papers/latex/paper1.tex` | match draft | clarity | keep | `paper: synchronize post-freeze factual corrections` |
| `papers/latex/paper1.pdf` | rebuild from tex | render | keep | `paper: synchronize post-freeze factual corrections` |
| `papers/paper2_draft.md` | align wording | clarity | keep | `paper: synchronize post-freeze factual corrections` |
| `papers/PAPER_ALPHA_CORRECTION_NOTE.md` | correction log | no | keep | `paper: synchronize post-freeze factual corrections` |
| `audit/discussion-to-implementation/MEGAPLAN_FULL_REPORT.md` | megaplan reconciliation | governance | keep | `docs: add megaplan reconciliation report` |
| `audit/discussion-to-implementation/MEGAPLAN_FULL_REPORT.json` | machine twin | no | keep | `docs: add megaplan reconciliation report` |
| `audit/discussion-to-implementation/EVIDENCE_LEDGER_PROPOSED.md` | proposal-only DIFF E | no | keep | `docs: add megaplan reconciliation report` |
| `audit/discussion-to-implementation/EVIDENCE_LEDGER_PROPOSED.json` | proposal twin | no | keep | `docs: add megaplan reconciliation report` |

## Explicitly not in this unit

- E4 world builder / datasets / frozen instances / scorers on E4 data
- GPU/RunPod kernels / adapters
- Fabric V2 / NanoScribe control plane / memory/routing/tools/UI
- Stage M / E2 runs
- Moving or recreating `post-alpha-evidence-freeze-2026-07-31`

## Post-closure program status (canonical)

```text
PROGRAM_EXECUTION_STATUS: IDLE_AFTER_FREEZE
AUTHORIZED_NONEXECUTION_WORK: E4_DESIGN_ONLY
E4_PROTOCOL_STATUS: DESIGN_DRAFT
E4_EXECUTION_STATUS: BLOCKED
E4_WORLD_STATUS: NOT_FROZEN
E4_DATA_STATUS: NONE
E4_RESULT_STATUS: NONE
FABRIC_STATUS: SCOPED_VERIFICATION_SLICE
NANOSCRIBE_STATUS: ARCHITECTURAL_RESEARCH_PROGRAM
OLD_TASK_GENERATIVE_SUBSTRATE: FALSIFIED_UNDER_FROZEN_U
NANOSCRIBE_PRODUCT_EXPANSION: STOP
```

Optional distinct (non-evidence) tag after push: `e4-design-draft-2026-07-31` — **not** created automatically.


## Closure verification

| Check | Value |
|-------|-------|
| Final HEAD (pre-push) | `c2475c73169b983ee3c9c17888f0a9c7dada3c62` |
| Evidence freeze tag target | `a9d12cb1c456f6c465284e1d469c6326cb14d329` (unchanged) |
| Tag moved? | NO |
| E4 world/data/result added? | NO |
| Tests | see push verification |
