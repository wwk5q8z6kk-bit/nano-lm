# Independent annotation A process

Read only the supplied `sources.json` and the independent semantic annotation rubric at `artifacts/p1_scope_validation_20261023/annotation_rubric.md`. Did not read the implementation, tests, previous source or gold datasets, other annotations, or model outputs. Did not invoke a production extractor or coordinate judgments with another annotator.

Annotated all 24 cases and all five fields (120 field decisions) directly from patient utterances and ordinary conversational context. Used the rubric's S/A/M/U/C states and exact patient-text spans. No clinical inference was used. The source file was not changed.

Mechanically checked the case set, field set, state and evidence requirements, exact substring matches, and location of every span wholly inside a Patient turn. Quotes that repeat elsewhere in a transcript use explicit zero-based character offsets selecting their unique patient occurrence; corresponding field rationales explain this. Additional supporting location evidence is included for retain-13's chief complaint.

Judgment limitations are recorded in field rationales where useful. In particular, retain-12 medication use is certain but its identity is expressly unknown, so it is U; the two coexisting medications and two allergies in retain-08 are U under the scalar schema; persistent cough in retain-21 supplies no duration value; reports about other speakers or unrelated episodes do not supply current patient field values. Explicit corrections resolve the duration in retain-15 and allergy in retain-16. Unretracted incompatible reports remain C.

This is an independent agent-applied semantic annotation, not a clinician evaluation or human dual-annotator agreement study.
