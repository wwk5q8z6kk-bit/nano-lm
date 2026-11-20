# Fresh P1 retention validation — 2026-10-26

The bounded question/reply repair passes its software regressions, but **fresh development transfer still fails**. On 24 independently authored synthetic encounters, free generation ends with 24 abstentions; constrained generation ends with 23 abstentions and one review draft containing one normalized-correct complaint. Neither mode emits an assertion matching the adjudicated exact evidence. The checker discards 13 exact-correct assertions recoverable from malformed free responses. This is an unresolved retention problem, not a model-quality pass.

## Design and independence

A fresh-context AI author produced 24 dialogues across six equally sized strata: ordinary current facts, field-local uncertainty, medication/allergy scope, corrections/conflicts, irrelevant history/other speakers, and missing/compound requests. These cases were separate from implementation regressions. Two independently initialized AI agents annotated the sources using the existing rubric, without code, predictions or each other's labels. A third blinded AI adjudicator resolved evidence-boundary disagreements before inference. Original A/B state agreement was 120/120; exact evidence agreement was 108/120, with 12 field disagreements across 11 cases. Gold contains 61 supported, 19 absent, 20 missing, 16 uncertain and four conflicting fields; 80 fields are asserted S/A. Every case has at least one usable gold assertion.

The coordinator locked [173 implementation files](implementation_lock.json) before inspecting this cohort. The [protocol](protocol.md), sources, labels and process records were then locked in [annotation_lock.json](annotation_lock.json) before inference. No implementation or annotation was tuned after exposure, and no prediction was rerun for selection. Procedural blinding among AI agents is not human or clinician dual-rater validation, nor independence across model families.

The model is cached base Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, weights SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`. Both runs used local CPU, offline greedy inference and the existing 256-token limit (per slot in constrained mode). Free took 123.225 seconds; constrained took 207.703 seconds. Incremental paid compute was $0. The constrained raw proposal already uses source/grammar restrictions and is not an unaided-model baseline. Full provenance, prompts and input/source snapshots accompany each run.

## Primary comparison

All 24 cases and all 120 fields remain in primary denominators, including malformed responses and annotations rejected by legacy transport. Raw and checked measurements use the same prediction for each case. Checked extraction precedes final-note withholding.

| Measure | Free raw | Free checked | Constrained raw | Constrained checked |
| --- | ---: | ---: | ---: | ---: |
| Correct state /120 | 0 | 12 | 24 | 22 |
| Correct normalized content /120 | 0 | 12 | 24 | 22 |
| Grounded exact field /120 | 0 | 12 | 22 | 21 |
| Complete evidence exact field /120 | 0 | 6 | 19 | 14 |
| Emitted asserted fields | 0 | 0 | 2 | 1 |
| Exact-correct assertions /80 gold assertions | 0 | 0 | 0 | 0 |
| Asserted state/content mismatches | 0 | 0 | 0 | 0 |
| Asserted exact-evidence mismatches | 0 | 0 | 2 | 1 |
| Conflict state recalled /4 | 0 | 0 | 0 | 0 |

Grounded exact scoring gives U/M credit for correct state alone; complete-evidence scoring additionally requires identical evidence sets. Thus the 12 free checked credits are all U states. The 21 constrained checked credits comprise 16 U and five M states. These are not retained factual assertions. The normalized-content metric checks a normalized value for S, but only state for A/U/M/C; it does not independently validate denial meaning. Zero state/content mismatches is uninformative about broad safety when almost no content survives.

All 24 free responses fail strict whole-response parsing. Seventeen permit independent-slot recovery; seven remain inference failures. The final free path emits no assertion and abstains on 24/24 usable sources. Constrained parsing succeeds on all 24; final verification yields one review draft and 23 abstentions. That draft is `retain-07`: complaint `dry cough`, source quote `A dry cough`. The adjudicated quote is `dry cough`, so the value is normalized-correct but the exact span differs. This mismatch is not an adjudicated medical error. The draft still omits the available severity information. Full [free](free/comparison.json) and [constrained](constrained/comparison.json) comparisons include per-field/stratum results, confusion matrices, failure counts and paired changes.

Checking salvages 12 and loses 13 grounded-exact fields in the constrained run, a net loss of one; no exact-correct assertion is present in its strict raw denominator to retain. It removes one normalized-correct assertion while retaining the other. Conflict recall remains zero.

## Locally recoverable raw information and policy coverage

The prespecified [local-slot diagnostic](posthoc_diagnostics.json) examines saved parser-admitted slots without changing the strict primary scores. Free responses contain 34 independently valid raw slots: 13 exact-correct supported assertions and 16 assertions credited by the normalized-content metric. **All 13 exact-correct supported assertions are lost during checking.** All 16 normalized-credited slots are also rejected, but that metric gives A credit from state alone: `retain-11` medication incorrectly quotes historical ibuprofen use as a denial and is not a valid lost assertion. The other two normalized-only slots are `retain-05` complaint with an extra article and `retain-09` allergy with shortened denial evidence. This distinguishes whole-response formatting failures from excessive field rejection. The checked free path is not rescuing the valid source-grounded content available inside malformed outputs.

No fresh source activates global whole-note uncertainty. The remaining losses arise from local interpretation/rejection: 34 free recovered slots receive `independent_binder_state_mismatch`; constrained diagnostics contain 88 such mismatches. Removing a global fallback alone therefore does not establish retention. Only 24 of 102 clinician turns are recognized as singleton requests; this is implementation coverage, not annotated question-attribution accuracy, since some turns are compound requests or statements.

The [menu coverage control](coverage_controls.json) finds concrete S/A/C candidates in **only four of 120 slots**: two complaints, one duration and one medication; zero severity or allergy slots. Twenty-seven slots have only M/U[] and the rest can still offer uncertainty text without a factual choice. Coverage assumes M for previous slots, so actual selections can further restrict later menus. The independent audit finds that none of these four concrete choices matches adjudicated exact evidence: `Four days` differs from `Four days ago`; `A dry cough`, `A headache` and `A saline spray` include articles outside the annotated spans. Three choices match the normalized value. Because prior selections only remove available candidates, exact assertion recall is structurally zero for this locked candidate set. The measured constrained 0/80 is therefore not a pure finding about model extraction ability. The deterministic binder yields 97 U, 21 M and two S fields, with zero exact-correct assertions. These diagnostics implicate the bounded source interpretation and candidate restrictions as major bottlenecks; they are not a causal estimate of an alternative decoder's performance.

## Controls, transport and sensitivity

The annotation-native oracle scores 120/120 and the all-M control 20/120. The all-U control scores 16/120 grounded-exact but 0/120 complete-evidence, illustrating how uncertainty-state credit can conceal zero utility. These are scorer checks, not model results.

The unchanged legacy contract rejects **19 annotated absent fields** because its denial vocabulary does not recognize their natural-language spans. Only nine cases fully transport through that contract. The primary annotation-owned comparison retains all cases and all original states; there is no silent exclusion, all-missing replacement or relabeling to fit the implementation. The secondary transportable subset is a diagnostic only. This limits claims about full end-to-end target transport even though the primary scorer handles the annotations.

[Original-label sensitivity](annotation_sensitivity.json) rescores the same saved outputs against both annotators. Free checked stays 12/120 grounded-exact; constrained checked stays 21/120, with zero exact-correct checked assertions for all three references. Constrained raw is 23/120 with one exact assertion against A, versus 22/120 with zero against B or adjudicated gold. Boundary choice affects that raw count, but does not change the severe retention failure. No labels were altered after inference. An [independent result audit](independent_result_audit.md) checks the calculations and interpretations separately.

## Verification, preservation and remaining work

The [implementation report](../p1_retention_repair_dev_20261026/REPORT.md) records the bounded changes, independent adversarial review and full software suite: **1,940 passed, two environment skips**. All 1,713 prior protected files, including frozen evidence and previous development cohorts, retain their hashes. The implementation lock and both run source/input snapshots also match; see [final integrity](../p1_retention_repair_dev_20261026/integrity.json). No H6/H7 run was repeated, verdict relabeled, freeze tag changed or checkpoint replaced.

This cohort is now exposed development evidence. Further work should use separate development cases to improve semantic question/reply interpretation, denial transport, evidence boundaries and concrete candidate coverage, then test another fresh independently annotated cohort. Adding tested phrase forms has not established broad retention. Earlier cohorts use different cases and runtime settings, so their scores do not establish a paired improvement or regression here. No P1 mastery, clinical completeness or deployment claim follows from these results.
