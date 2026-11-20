# Independent adjudication of the synthetic P1 retention development cohort

## Scope, blindness, and limitations

The adjudicator read the applicable repository instructions and only the supplied new `sources.json`, `annotations_a.json`, `annotations_b.json`, their two process notes, and the existing independent semantic annotation rubric. The adjudicator did not inspect implementation, tests, earlier source/gold cohorts, or any extractor/model outputs. No production extractor was invoked. Each source and both annotations were reviewed directly. All source and annotation files were preserved; only `gold.json` and this report were created.

The scope is the five patient-reported fields CC, DUR, SEV, MED, and ALG. Source statements and ordinary local conversational context govern the judgments. No clinical knowledge, diagnosis, medication identity, or unreported timeline was inferred. This is a synthetic development cohort, not a held-out confirmatory evaluation. Annotator A, annotator B, and the adjudicator are AI agents. This is an agent-applied rubric exercise, not human or clinician review and not human dual-annotator agreement.

## Agreement before adjudication

Agreement counts compare states and exact ordered evidence spans independently of explanatory rationale text. Explicit offsets are included in exact evidence comparison. Thus different wording in rationales is not counted as an annotation disagreement.

| Field | State agreement | Exact evidence agreement | Evidence disagreements |
| --- | ---: | ---: | ---: |
| CC | 24/24 | 17/24 | 7 |
| DUR | 24/24 | 21/24 | 3 |
| SEV | 24/24 | 23/24 | 1 |
| MED | 24/24 | 24/24 | 0 |
| ALG | 24/24 | 23/24 | 1 |
| Total | 120/120 (100%) | 108/120 (90%) | 12 |

All 24 cases have complete five-field state agreement. Thirteen of 24 cases also have exact agreement on every evidence list. The 12 evidence disagreements occur in 11 cases. The first represented span agrees exactly in 50/61 S fields; this measures literal span identity, not semantic equivalence. Annotator A supplied 107 spans; annotator B supplied 106. Both assigned S=61, A=19, M=20, U=16, C=4.

No state changes were needed after independent source review. The evidence differences concern phrase boundaries, retention of qualifiers, or one supporting location span. The following decisions were made from the rubric and source, without reference to implementation behavior.

## Evidence disagreements and decisions

| Case and field | Difference | Adjudicated decision |
| --- | --- | --- |
| retain-02 CC | A includes `My`; B omits it. | Use `left ankle is sore`. The patient attribution is established; laterality, body part, and symptom remain. |
| retain-02 DUR | A retains `after I got home`; B keeps only `Yesterday afternoon`. | Use `Yesterday afternoon, after I got home`. The home-arrival clause refines the reported onset within that afternoon. |
| retain-03 DUR | A retains `so far`; B omits it. | Use `Two days so far`, preserving the explicit accrued-to-present qualifier. |
| retain-04 ALG | A retains the complete second denial clause; B retains both clauses. | Use `I don't have medication allergies.` This is a complete explicit denial clause, which the rubric permits; `There aren't any` adds no separate information. |
| retain-06 SEV | A retains `but annoying`; B uses `Mild`. | Use `Mild, but annoying`. This is one qualified reported severity, not unresolved alternatives; preserve the short qualification of how it feels. |
| retain-09 CC | A includes `My`; B omits it. | Use `neck has felt stiff`; the possessive is redundant in the patient context. |
| retain-10 CC | A includes `My`; B omits it. | Use `right elbow aches`, retaining laterality and the full complaint. |
| retain-11 CC | A includes `My`; B omits it. | Use `shoulder started aching`, retaining the reported symptom without the redundant possessive. |
| retain-13 CC | A uses `A headache` plus a later location span; B uses an offset-qualified `headache` only. | Use `{"text":"headache","start":42}` as the first value and `Mostly at the back of my head` as additional support. This combines the shortest value with the explicitly reported location without fabricating a composite quote. The repeated word requires the offset. |
| retain-17 CC | A includes `My`; B omits it. | Use `knee aches`, with patient attribution supplied by the turn. |
| retain-20 DUR | A includes `since`; B omits it. | Use `since this morning`, preserving the explicit onset-to-present relation. |
| retain-22 CC | A includes `My`; B omits it. | Use `left ear is ringing`, retaining laterality, site, and symptom. |

Six disputed evidence lists follow B, five follow A, and one combines supported choices from both. These selections reflect the adjudicator's source judgment, not a majority vote. Explicit contextual qualifiers in the duration and severity examples are retained, while patient-ownership wrappers are unnecessary for already-attributed complaints. The first S span is always an exact source phrase; additional support never creates an invented quote.

## State and scalar semantics independently checked

- Retain-08 MED and ALG are U: the two treatments and two allergies can coexist, so a scalar field cannot retain all values. Their concrete individual names remain as evidence. These are not contradictions.
- Retain-12 MED is U: actual use of a tablet is affirmed, but its identity is explicitly unknown. The full statement preserves both certainty about use and uncertainty about identity. No drug name is inferred.
- Retain-05 through retain-08 and retain-20 retain field-local uncertainty. A clear value in another field remains supported. Retain-21's `persistent` qualifies the cough but does not supply an onset or duration value; the unanswered duration and severity questions remain M.
- Retain-09's routine thyroid medication, retain-11's prior-winter treatment, retain-18's husband's bottle and sister's allergy, retain-19's unrelated prior dizziness, and retain-20's other speaker do not establish current patient field values. Retain-10 explicitly distinguishes a historical side effect from allergy. Retain-02 names penicillin directly in the medication-allergy context; no additional clinical interpretation is made.
- Retain-15 explicitly retracts `Last night` as another matter, resolving DUR to `this morning`. Retain-16 explicitly retracts the aspirin intake entry, resolving ALG to `Penicillin`. Retractions do not create unresolved conflict states.
- Clinician-only assertions, unanswered questions, and unrelated address/travel/hearing exchanges do not fill missing values or make unrelated supported fields uncertain. Retain-17's short denial is explicit in its immediately preceding allergy-question context.

All four C fields preserve at least two distinct incompatible patient observations, without replacing them with generic uncertainty:

| Case and field | Preserved competing observations | Why unresolved |
| --- | --- | --- |
| retain-13 DUR | `This headache began two days ago.` / `This headache began five days ago.` | Both concern the same headache; neither is withdrawn. |
| retain-14 MED | `I haven't taken or applied anything for this itch at any point this week.` / `I used hydrocortisone on this itch every night this week.` | The blanket denial and positive report cover the same itch and week. |
| retain-15 CC | `My right wrist hurts right now; my left wrist doesn't hurt.` / `My left wrist hurts right now; my right wrist doesn't hurt.` | Opposing current laterality reports include opposing explicit denials; neither is corrected. |
| retain-16 SEV | `It has been mild all day today.` / `It has been severe all day today.` | Both assert severity throughout the same day with no retraction. |

## Final counts and mechanical validation

The gold array contains 24 unique cases and 120 field decisions, with S=61, A=19, M=20, U=16, C=4. It contains 107 evidence spans, including two explicit offsets. All spans were mechanically verified as exact source substrings located wholly inside a Patient turn. Every string span has exactly one transcript occurrence. Explicit offsets select `toothache` at character 46 in retain-12 and `headache` at character 42 in retain-13; these words recur elsewhere in their transcripts. Offsets are zero-based transcript character positions.

Validation also checked exact case and field coverage; permitted states; nonempty S/A evidence; empty M evidence; MED/ALG-only A states; and at least two distinct evidence observations for each C field. The same mechanical checks passed for both source annotation files. The original six annotation inputs were hashed before writing and checked unchanged afterward. These checks validate transcription and schema integrity, not clinical truth or external model performance.
