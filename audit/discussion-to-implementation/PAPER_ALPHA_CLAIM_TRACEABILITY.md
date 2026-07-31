# Paper α Claim Traceability

*Generated 2026-07-31T14:04:33.413228+00:00; HEAD `0e01d73205e9c35ea32925fd4d6c7e5fceb61137`*

Audit target: `papers/paper1_draft.md` + `papers/latex/paper1.tex` (post DIFF H/I/J).

## Abstract / conclusion substantive sentences

| Sentence (paraphrase of current text) | Claim ID | Evidence artifact | Recompute path | Scope valid? | Required edit |
|---|---|---|---|---|---|
| Across evaluated own-stack configurations, gap did not decrease monotonically with parameter count; not a parameter-only scale law (token budgets unequal). | C_SCALE_DESCRIPTIVE | TOKEN_BUDGET_RECONCILIATION; ownstack JSONs; paper1 methods | re-read audits + results JSON | yes after DIFF H | applied |
| M1 U≈0.999 exceeds official M0 U≈0.925 under frozen U (KILL). | C_E1_M1_KILL | results_e1_utility.json | trajectory/e1/common.py reconstruct U | yes | applied DIFF I |
| Field-localized open-vocab gap; closed fields zero. | C_FIELD_LOCAL | anchors/fieldwise + Paper α ladder | scorer recompute | yes | none |
| Slot diversity +66.7. | C_DIVERSITY | results_sweep_10m.json / PREREG_slot_diversity | sweep eval scripts | yes | none |
| Exact-match dual-clinician unvalidated; agent-rubric 0/100. | C_E3_EXACT | results_e3_*; PREREG_E3 | normalize + rubric JSON | yes scoped | applied DIFF J |

## Methods token budgets

| Config | Tokens | Source |
|---|---:|---|
| 3.15M | 32.8M | `pretrain/AUDIT.md` |
| 10M | ~200M | `scale/AUDIT.md` |
| 160M weak | 200M | `results_ownstack_v2_160m_fullft.json` |
| 160M Chinchilla | 3.2B | `results_ownstack_v2_160m_chinchilla.json` |

## Remaining readiness gaps

- E1/E3 primary JSON currently LOCAL_UNTRACKED until freeze commits.
- C-1b/C-3 raw JSONL: local archive present; gitignored; RESULT_ACCEPTED_WITH_REPRODUCIBILITY_LIMITATION if not durably published.
- Camera-ready PDF not rebuilt in this remediation pass.
- Owner must authorize tag `post-alpha-evidence-freeze-2026-07-31`.
