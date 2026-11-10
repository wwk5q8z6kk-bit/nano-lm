# Independent review of fresh scope-repair validation

Reviewed 2026-10-23, after both saved runs completed. This is an independent AI result review of the current synthetic development cohort, not clinician review or confirmatory evidence. Review inputs were the current protocol, annotations/adjudication, controls, locked run artifacts and source snapshots, scoring runner, sensitivity and post-hoc diagnostics, and current repair report. No model inference, implementation changes, annotation changes, tuning, or historical frozen-cohort inspection was performed. A separate read-only reviewer independently recomputed sensitivity arithmetic without importing the product scorer.

The evidence supports **failure to transfer the repair into usable scribing on this cohort**. Both checked modes emit zero assertions and abstain on all 24 encounters despite usable annotated content in every encounter. The improvement in the reported grounded score is entirely uncertainty-state credit. Syntax recovery works in many free-generation records, but checking removes every correct assertion it encounters. The constrained path additionally excludes many correct answers before model selection.

## Verified results and denominator discipline

All 24 cases and 120 fields remain in the primary and sensitivity denominators. The primary gold contains 80 supported (S), 12 absent (A), 27 uncertain (U), one conflicting (C), and zero missing (M) fields. “Assertion” below means S/A. Saved input/source hashes, annotation lock entries, and run provenance checked during review agree; both runs record completion and post-inference lock verification. This establishes local consistency, not independently authenticated execution history.

| Primary measure | Free raw | Free checked | Constrained raw | Constrained checked |
|---|---:|---:|---:|---:|
| State correct / 120 | 7 | 26 | 16 | 26 |
| Normalized content correct / 120 | 3 | 26 | 9 | 26 |
| Grounded exact correct / 120 | 2 | 26 | 9 | 26 |
| Complete evidence exact correct / 120 | 2 | 8 | 7 | 12 |
| Assertions emitted in extraction | 10 | 0 | 11 | 0 |
| Exact-correct assertions | 2 | 0 | 3 | 0 |
| Whole-record parse/inference failures / 24 | 22 | 1 | 0 | 0 |

Final notes contain zero assertions and abstain in 24/24 cases in both modes. Free checked output has 115 U fields and five failed fields; constrained checked output has 106 U and 14 M fields. Neither recovers the one gold C state. Every one of the 26 checked grounded credits is U.

The primary paired counts are arithmetically correct: free salvages 26 exact fields and loses two; constrained salvages 20 and loses three. None of the salvaged fields is an assertion. Both remove eight exact-mismatching assertions from the strict raw baseline, while also removing all exact-correct assertions. A zero emitted-error count with zero assertions does not establish useful selective accuracy or clinical safety.

## Correct content lost by checking

The strict whole-record baseline deliberately scores every field in a malformed record as a failure. Its free-generation loss count therefore understates how much locally valid content checking rejects. The separately labelled post-hoc diagnostic retains 120 scoring rows, including the five fields without diagnostics from the unrecoverable record; it does not reduce the denominator to parsed slots.

Free generation has 47 locally parse-admitted slots, including 16 exact-correct and 19 normalized-correct assertions. **All 16 exact-correct and all 19 normalized-correct assertions are discarded**, each with `independent_binder_state_mismatch`. Of the 47 slots, 37 come from otherwise malformed records. Recovery of 21 whole records and 37 slots is a parsing result, not recovery of usable content. Constrained local scoring reproduces its primary results and all three exact assertion losses.

Concrete primary exact losses are:

| Mode / case / field | Correct raw evidence | Checked result |
|---|---|---|
| Free / scope-08 / duration | `two days` | U; binder state mismatch |
| Free / scope-15 / severity | `moderate` | U; binder state mismatch |
| Constrained / scope-07 / complaint | `right shoulder hurts when I reach up to a shelf` | U; binder state mismatch |
| Constrained / scope-09 / severity | `Mild` | U; binder state mismatch |
| Constrained / scope-16 / complaint | `sore right ankle` | U; binder state mismatch |

Additional exact losses exposed by the free local-slot diagnostic include scope-01 `sore throat` and `mild`, scope-03 `dry cough`, scope-11 `itchy patch on my left forearm` and `a week`, and scope-18's corrected `moderate`. These are affirmative examples of checker over-abstention, beyond a general complaint that outputs are sparse. The strict free baseline also loses a third normalized-correct assertion in scope-08 medication, whose evidence boundary does not satisfy exact scoring.

## Remaining mechanisms and source fidelity

**Constrained candidate coverage is a substantial bottleneck.** The saved generation trace records only two choices for 76/120 slots. The locked decoder constructs these two as `:M` and `:U[]`; concrete S/A/C options are additional choices. Those 76 slots have gold states S=56, A=6, C=1, U=13. Thus the correct concrete state is unavailable for at least 63 fields, including 62 assertions, before the model chooses anything. This is a lower bound; having more than two choices does not prove the correct value is present. The constrained raw output is M for 100/120 fields, including nine entirely M records. It is not reasonable to attribute this result solely to the base model.

The saved source-policy replay reports zero globally unsafe cases and singleton-request recognition on 44/142 clinician turns. This is implementation coverage, not an accuracy/recall estimate: some clinician turns are not singleton requests. Nonetheless, readily interpretable questions are unrecognized, including scope-01's `How far back does that go?` and `How much is it bothering you?`, and scope-03's `Is there anything on your allergy list?`. Combined with the candidate and rejection traces, the evidence points to limited question/reply attribution and field-local evidence policy, rather than a deliberate global-unsafe veto explaining universal abstention.

The model still makes substantive source errors when allowed an answer. Constrained scope-18 chooses the earlier `severe` after an explicit correction to moderate; scope-19 proposes Penicillin after the history is reassigned to the sister. Free scope-24 reduces a combined blocked-nose/sore-throat complaint to one symptom and selects mild despite the second symptom's moderate severity. Scope-17 free output has six slots and remains unrecoverable. Case and paraphrase changes also cause exact-source parse failures, such as `three days` versus source `Three days` and `6 days` versus `Six days`. Neither mode reaches its generation token limit, so truncation is not the observed explanation.

Checked evidence itself can remain poorly attributed even when quarantined. Scope-01 medication evidence includes the penicillin allergy reply; scope-10 duration includes a medication/coworker reply; scope-11 duration/severity include an explicitly unattributed notebook; scope-12 medication includes an article's other-person tramadol report. These are not final asserted facts, but they compromise field evidence fidelity and help explain excessive quarantine. Scope-17 checked complaint evidence retains the earlier right-ear report without the later left-ear correction. Scope-20 checked medication retains the incompatible daily/no-use observations but becomes U and omits part of the gold clarification evidence; the full source remains available, so this is not deletion of the underlying transcript. The one conflict is not successfully represented.

## Metric and annotation sensitivity

The implemented `grounded_exact` definition requires only state agreement for U/M; no U evidence agreement is required. S/A need an intersecting exact annotated span, while C needs all gold spans. Consequently, 26/120 (21.7%) checked “grounded” accuracy should not be described as 26 faithful evidence extractions. Complete evidence equality is only 8/120 free and 12/120 constrained. A constant-U output would receive 27/120 under primary gold by this definition—an arithmetic counterfactual, not another model run. The all-M negative control cannot reveal this degenerate uncertainty-credit mechanism.

The prespecified sensitivities are implemented correctly, using the same saved responses and all 120 fields. The eight-label variant changes exactly the specified ALG U entries back to original A labels/evidence; original A and B are each also scored in full. No denominator or arithmetic discrepancy was found.

| Gold policy | Free raw grounded | Constrained raw grounded | Checked grounded, either mode | Checked complete evidence, free / constrained |
|---|---:|---:|---:|---:|
| Adjudicated primary | 2 | 9 | 26 | 8 / 12 |
| Original annotator A | 2 | 5 | 17 | 4 / 8 |
| Original annotator B | 2 | 6 | 18 | 4 / 7 |
| Restore only eight ALG overrides | 2 | 6 | 18 | 4 / 7 |

Every table entry has denominator 120. Every variant still has zero checked assertions and 24/24 abstaining notes. Restoring the eight conventional no-known-allergies labels removes eight checked credits; the primary interpretation therefore materially raises the aggregate score, without changing utility. The interpretation was documented before inference and the sensitivity is appropriate; this is not evidence of output-driven relabelling. Scope-13's pollen interpretation accounts for the remaining original-A state difference. Original-A complaint boundaries also change constrained exact assertion losses from three to two. Annotation matches and mismatches must not be translated directly into clinical error counts.

Exact boundary sensitivity remains real: adjudication strips possessives/articles in several disputed complaint spans while retaining similar wrappers in some shared spans. The original annotators agree on 119/120 states but only 100/120 exact evidence sets (111/120 state plus normalized value). These figures measure two AI contexts under one rubric, not clinician agreement or independent model-family replication.

## Preflight and scope of the conclusion

The dictionary oracle's 120/120 and all-M control's 0/120 validate the custom scoring path. They do not show that the current product contract can transport all gold. Preflight records 12 `absence_without_denial` failures across 11 cases, covering every primary A field, and only 13 wholly transportable cases for the secondary contract oracle. Keeping those natural-language denials and all 120 fields is correct; silently restricting evaluation to the transportable subset would conceal a material representational limitation. The absence of gold M and presence of only one C also prevent meaningful estimates of missing-state specificity and conflict generality.

The cohort is small, synthetic, deliberately stratified, and now exposed development evidence. Procedural blinding and pre-inference locks reduce direct implementation/label adaptation, but do not confer human validity or population representativeness. Free and constrained modes use different prompting as well as decoding policy; their difference is a pipeline comparison, not an isolated causal effect of constraining the same model response. Successful software tests in the repair report are compatible with the failures observed here.

The concrete research result is syntax recovery without usable assertion retention, severe constrained candidate undercoverage, and unresolved source/correction handling. The evidence does not support useful faithful scribing or a safety claim from the higher aggregate checked score. It does support directing subsequent development toward candidate availability, field attribution and preservation of correct assertions, while preserving these results as the completed failed-transfer diagnostic.
