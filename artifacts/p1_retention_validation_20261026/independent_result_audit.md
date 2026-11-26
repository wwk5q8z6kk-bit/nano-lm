# Independent result audit — fresh retention development validation

Audited on 2026-10-26 from locked sources/annotations, saved free and constrained predictions, comparison files, coverage controls, supplemental diagnostics, annotation sensitivity, protocol, process notes and the locked harness. The primary arithmetic is reproducible and the result supports an unresolved retention/coverage failure. It does not establish clinical quality, broad safety, a trained-model gain, or a paired improvement over an earlier cohort.

The only persistent additions from this audit are this report and `independent_audit_recompute.py`. No model inference, source/label correction, product change or frozen-artifact change was made. The reviewer is a separate AI agent, not a clinician or human rater. This review has access to results and therefore is not a blinded annotation pass.

## Verification performed

The read-only recomputation script independently implements the published state, normalized-content, evidence-alignment and complete-evidence definitions rather than calling the harness scorer. It reconstructs annotation offsets from source text, checks that every evidence span lies inside a Patient turn, reparses the saved model text with the existing parser, reconstructs supplemental local fields from saved inference traces, and recomputes the principal counters against adjudicated gold and both original annotators. It also independently counts the negative/oracle controls, checks legacy contract transport, and scores every concrete menu choice against gold. Run it with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 artifacts/p1_retention_validation_20261026/independent_audit_recompute.py
```

This command passed. All 173 locked live implementation files match `implementation_lock.json`. Their copies in the free, constrained and preflight source snapshots also match. All 11 files covered by `annotation_lock.json`, and that lock's own hash in each saved input snapshot, match. These checks establish present byte identity and consistency of the saved snapshots; procedural blindness and the historical absence of hidden reruns remain supported by the protocol/process records, not independently witnessed by this reviewer. Prior protected-artifact preservation and the software test-suite claims are outside this result audit's independently rerun checks.

## Recomputed primary numbers

There are 24 cases, 120 fields, and 80 gold asserted fields: S=61, A=19, M=20, U=16, C=4. All 24 sources contain at least one gold S/A field. A/B state agreement is 120/120 and ordered exact evidence agreement is 108/120.

| Counter | Free strict raw | Free checked extraction | Constrained raw | Constrained checked extraction |
| --- | ---: | ---: | ---: | ---: |
| Correct state /120 | 0 | 12 | 24 | 22 |
| Normalized content /120 | 0 | 12 | 24 | 22 |
| Grounded exact /120 | 0 | 12 | 22 | 21 |
| Complete exact evidence /120 | 0 | 6 | 19 | 14 |
| Asserted fields emitted | 0 | 0 | 2 | 1 |
| Exact-correct assertions /80 | 0 | 0 | 0 | 0 |
| Asserted state/content mismatches | 0 | 0 | 0 | 0 |
| Asserted exact-field mismatches | 0 | 0 | 2 | 1 |
| Conflict state or exact recall /4 | 0 | 0 | 0 | 0 |

Strict free parsing fails on 24/24 cases. The failures comprise 14 non-exact/non-patient evidence failures, six records without five ordered slots, two malformed C payloads, one empty/malformed A payload, and one ambiguous patient evidence span. Seventeen cases retain a checked extraction via local recovery; seven do not. None reaches the saved token-limit flag. Free final disposition is ABSTAIN in all 24 cases and no final asserted claim is emitted.

Constrained parsing succeeds on all 24 cases. Final disposition is ABSTAIN in 23 cases and REVIEW in one. There is one actual final asserted claim, with normalized-correct content and a boundary mismatch. Checked-extraction counts precede final-note handling; actual note claims were separately inspected. The existence of a saved note object with nonasserted review fields is not counted as a useful factual assertion.

Free checking's 12 grounded-exact credits are entirely gold U states. Constrained checking salvages 12 gold U fields and loses 13 gold M fields, producing a net loss of one grounded-exact field. Its 21 credits are 16 U plus five M, not factual assertions retained. Under this metric, U/M need only the right state. Complete-evidence scores of 6 and 14 expose additional missing/different uncertainty evidence. “Grounded exact” must not be read as evidence completeness or useful note completeness.

## The surviving assertion and real losses

The surviving constrained assertion is `retain-07` CC. Source: `Patient: A dry cough.` Both raw/checked extraction and the final note represent `dry cough`, supported by `[44,55)` = `A dry cough`. Gold uses `[46,55)` = `dry cough`. The extra article explains the exact mismatch; this is not evidence of a wrong complaint or a medical error. The final note nevertheless abstains on supported severity (`three out of ten`) along with the genuinely uncertain fields.

The other constrained raw assertion is `retain-13` CC, `A headache`, normalized to `headache`. Gold's first span is `headache`, with an additional location span. Checking turns the complaint into U and includes the conflicting duration statements as evidence. The duration conflict does not make the identity of the headache ambiguous under the rubric. Thus one of two normalized-correct constrained assertions is retained and one is lost, even though neither meets adjudicated exact evidence at raw admission. Against annotator A, `A headache` matches an exact gold span, so that raw assertion earns exact credit; it still lacks A's full additional location evidence and is still lost on checking.

The free supplemental analysis recovers 34 locally parse-admitted fields from malformed records. It gives 19 state credits, 16 normalized-content credits and 13 exact/complete credits, with the original 120-field denominator retained. All 13 exact-correct fields are supported assertions and all are rejected with `independent_binder_state_mismatch`:

| Case | Exact supported values lost |
| --- | --- |
| retain-01 | CC `sore throat`; DUR `three days` |
| retain-02 | SEV `mild` |
| retain-06 | ALG `latex` |
| retain-08 | DUR `the evening before last` |
| retain-09 | DUR `since I woke up this morning` |
| retain-10 | SEV `moderate` |
| retain-14 | CC `itchy patch on my left calf`; DUR `one week`; SEV `mild` |
| retain-15 | SEV `moderate` |
| retain-21 | CC `persistent cough` |
| retain-22 | CC `left ear is ringing` |

The three extra normalized credits require separate interpretation. `retain-05` CC differs only by a leading article. `retain-09` ALG uses the shortened denial `no allergies to medicines` rather than the full gold clause. Crucially, `retain-11` MED predicts A with evidence `ibuprofen for a shoulder injury last winter`; the source's actual current-treatment denial is a different clause. This is wrong-scope, non-denial evidence. It receives normalized credit because the metric checks only state for A. The number 16 must therefore not be described as 16 semantically verified correct assertions lost. Thirteen exact supported losses is the clear supported claim. None of these supplemental credits changes the strict free baseline of zero: all 24 whole responses are malformed.

## Candidate coverage is a major measurement limitation

Only four of 120 all-M-prefix menus offer any S/A/C choice, all four being S. The choices are:

| Case/field | Concrete choice | Gold value/evidence | Normalized value matches | Exact assertion matches |
| --- | --- | --- | --- | --- |
| retain-06 DUR | `Four days` | `Four days ago` | No | No |
| retain-07 CC | `A dry cough` | `dry cough` | Yes | No |
| retain-13 CC | `A headache` | `headache` plus location | Yes | No |
| retain-16 MED | `A saline spray` | `saline spray` | Yes | No |

No severity or allergy menu has a concrete choice; no menu has an A or C choice. Twenty-seven menus offer only M/U[], while 89 more offer uncertainty text without a concrete assertion. The remaining four are listed above. Inspection of `compact.py` confirms earlier choices can consume/remove spans from these all-M-prefix menus; they do not create new concrete spans. Consequently, in this saved cohort the locked constrained candidate policy has **zero adjudicated exact-correct assertion choices**, and only three slots with a normalized-correct assertion choice. Its measured zero exact assertion recall is partly predetermined by the grammar and evidence boundaries. It cannot be attributed solely to model selection. This is a diagnostic finding, not a counterfactual model score or a reason to revise gold.

The saved binder emits 97 U, 21 M and two S fields with zero exact-correct assertions. No source activates the global-unsafe flag. Singleton-request recognition is 24/102 clinician turns, which is implementation coverage, not independently scored attribution accuracy: that denominator also includes compound requests and statements. These observations localize remaining issues to bounded interpretation, candidate generation and checking, without proving how much a replacement would improve.

## Transport, controls and scope

All **19 gold absent fields fail the unchanged legacy `FieldOutput` denial contract**, across 15 cases. Only nine complete cases transport through it. Their natural-language gold states remain in the 120-field primary denominator, which is appropriate; silently dropping them or converting them to M would invalidate the comparison. The annotation-native oracle's 120/120 verifies the scorer/annotation route, not complete product transport. The secondary nine-case oracle does not close that gap.

The all-M control gives 20/120; all-U gives 16/120 grounded-exact but zero complete-evidence credit. These controls make blanket abstention's limited accuracy visible. With zero free assertions and one constrained assertion, zero state/content mismatches cannot establish safety. A/U/M/C “normalized content” credit is only state agreement, not evidence or semantic verification.

Sensitivity rescoring matches the stored results. All three references give free checked 12/120 and constrained checked 21/120 grounded-exact, with zero checked exact-correct assertions. Constrained raw is 23/120 and one exact-correct assertion against A, versus 22/120 and zero against B/adjudicated gold. The conclusion of severe retention failure survives this boundary sensitivity.

The current `REPORT.md` was reviewed and its primary numbers, strict/supplemental separation, surviving assertion, 13 exact supported losses, controls and annotation limitations agree with this audit. It includes both the state-only A metric caveat and the zero-exact-choice ceiling. Its software-suite and prior-artifact-preservation claims require their separately cited evidence.

This small, deliberately stratified synthetic cohort was authored and annotated by AI agents under a shared rubric. Procedural separation is valuable but does not create independent model families, human dual-rater reliability, clinical realism, population representativeness or external validation. The cohort is now exposed development evidence. Future product work requires separate cases; this audit provides no authorization to tune on these sources, rerun for favorable outputs, or alter any consumed/frozen experiment.
