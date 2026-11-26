# P1 reply-retention development repair — 2026-10-26

The development path now shares bounded complete-clause observations between the binder, constrained decoder and final note checker. A recognized valid clause can survive a separate field's uncertainty or a neutral trailing sentence. This fixes the new implementation regressions below; it does not establish general model quality. Fresh independently annotated validation is recorded [separately](../p1_retention_validation_20261026/REPORT.md).

## Behavior changes

- Bounded question grammar accepts additional ordinary request operators, polite wrappers and literal discourse prefixes. Allergy reaction requests target ALG. Compound, quoted, historical, hypothetical and third-party questions still do not authorize positive field binding.
- Reply segmentation preserves original character offsets, including CRLF. The first clause may inherit a known question; later concrete clauses require an explicit patient assertion. Unknown clauses retain local uncertainty and full multi-clause replies remain reviewable evidence.
- Concrete duration, severity, medication and allergy clauses share the same observations across source binding, all five decoder fields and final checking. Arbitrary source fragments no longer become supported complaint choices.
- Medication administration adjuncts such as “today” are excluded from the medicine name. Relation words prevent “an allergy test” from becoming a medication and “the medication” from becoming a named allergen.
- Corrections, historical reassignments and experiencer changes keep their local target across consecutive patient turns. A question answered with a different field retains that field but leaves the requested field uncertain.
- A narrowly recognized medication-efficacy reply does not retract actual medication use. Its text remains uncovered for review. Unrecognized follow-ups still require review.

Policy: `context-source-scope-v4`; decoder: `typed-context-source-trie-v8`; diagnostics remain `nanoscribe.context-state-diagnostics.v3`. Frozen `nano_ai` parsing, denial vocabulary and unique-text transport are unchanged. New natural-language denial forms and duplicate textual evidence remain known limitations.

## Development and independent review

New implementation tests are [test_question_coverage_dev.py](../../nanoscribe/test_question_coverage_dev.py) (46 cases) and [test_reply_retention_dev.py](../../nanoscribe/test_reply_retention_dev.py) (34 cases). They use injected outputs and synthetic cases separate from the fresh validation cohort. They do not measure model quality.

An independent AI reviewer supplied adversarial new reproductions and verified repairs for temporal names, non-medication phrases, consecutive-turn retractions, unresolved requested fields, efficacy follow-ups and unnamed allergens. [Review evidence](independent_review.md) records the exact outcomes. This review is not clinical review.

The implementation coordinator did not read fresh validation sources or annotations until [173 source hashes were locked](implementation_lock.json). There were no implementation changes after that lock. Two independently initialized AI annotators and a third blind adjudicator produced the new cohort's labels before inference. All scored outputs use that locked code.

## Verification

[Final verification record](verification_final.json) preserves commands, exit codes and logs. The initial full-suite record and logs remain present. NanoScribe was rerun after the last two review regression tests; other suite groups were unaffected by that final local guard change.

| Group | Passed | Skipped |
| --- | ---: | ---: |
| NanoScribe | 472 | 0 |
| Nano Core | 316 | 0 |
| Learned Nano AI | 499 | 1 |
| Benchmarks | 500 | 1 |
| Fabric, trajectory and Wedge | 32 | 0 |
| Evaluation/run scripts | 121 | 0 |
| Total pytest | **1,940** | **2** |

Direct fabric and trajectory checks also pass; their tests overlap with pytest and are not added again. The NanoScribe software simulation reports four exact spans, zero critical errors and zero unnecessary abstentions. It loads no model weights and is not evidence of clinical transfer. [Documentation checks](documentation_checks.json) and [final integrity verification](integrity.json) both pass.

## Preservation

[protected_before.json](protected_before.json) covers 1,713 previously existing protected files, including frozen evidence and all prior independent development results and source snapshots. All hashes match after this work. No frozen run was repeated, no evidence relabelled, no freeze tag changed, and no checkpoint replaced. Existing work in this checkout was retained. The new model diagnostic uses the cached base checkpoint offline with zero incremental paid compute.
