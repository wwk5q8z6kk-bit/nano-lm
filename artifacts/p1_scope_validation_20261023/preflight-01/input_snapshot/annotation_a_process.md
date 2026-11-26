# Annotation A process

This is an independent AI semantic annotation, not a human, clinician, or clinical adjudication exercise. I assigned labels from ordinary conversational meaning under the supplied rubric; no clinical expertise, medical inference, model inference run, or empirical system evaluation is claimed.

## Inputs and blindness

During annotation I read only `annotation_rubric.md` and `sources.json` in this cohort directory. Repository operating instructions were already supplied in the task context. I did not open implementation, tests, historical cases or results, other annotations, or model outputs. I did not coordinate label choices with another annotator or inspect their work. The cohort is synthetic and consists of 24 cases; this pass covers all five requested fields for every case.

Labels, evidence, and rationales were authored independently. A temporary Python command serialized those authored decisions and located quoted strings in the supplied transcript; it did not generate semantic labels, run a model, inspect implementation, or change code. Repeated text was assigned a zero-based transcript character offset to its first matching patient occurrence, and each affected rationale explains the selection.

## Annotation choices and limitations

I treated explicit corrections as resolving the earlier statement and preserved uncertainty locally to the affected field. Different symptom- or activity-specific values were treated as compatible but unresolved when the scalar schema could not keep them all. I excluded claims expressly attributed to relatives, coworkers, an unrelated article, or an unattributed notebook. Clinician assertions were not used as evidence of patient facts. Quoted evidence is restricted to patient text, although question context and clear pronoun antecedents were used to interpret it.

Some rubric boundaries remain judgment calls. In particular, I classified conventional direct statements of no known allergies as absence of reported known allergy, but classified personal-awareness-qualified answers as uncertain under the rubric's qualified-lack-of-knowledge rule. This distinction is debatable and is disclosed in the affected rationales. Other disclosed judgment calls concern current-complaint attribution of a regular medication, whether an environmental trigger is an explicit material allergy report, and unresolved incompatible medication histories. These choices are annotations to be evaluated independently, not established ground truth.

The five-field scalar schema cannot preserve every clinically or conversationally relevant detail. This synthetic cohort and AI-only pass cannot establish clinician agreement, human inter-annotator agreement, medical validity, or generalization to real encounters. Familiarity with the shared rubric and task instructions is unavoidable; blindness here means absence of exposure to implementation, other annotations, previous outcomes, or model answers during this pass.

## Validation

Validated JSON parsing, exactly 24 unique source case identifiers, all five fields per case (120 annotations), required keys and allowed states, state-specific minimum evidence counts, absence labels restricted to MED/ALG, nonempty rationales, exact transcript substring matching, patient-only span boundaries, unique unoffset quotations, and correctness of every explicit repeated-text offset. No repository tests or implementation checks were run.

Validation checked 129 evidence spans, including 4 spans with explicit offsets. State counts: S=81, A=20, M=0, U=18, C=1.
