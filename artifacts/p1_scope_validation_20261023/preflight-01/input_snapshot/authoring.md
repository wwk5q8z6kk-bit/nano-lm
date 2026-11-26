# Synthetic source authoring record

Created on 2026-10-23 for development validation of source fidelity in five scalar fields: complaint, duration, severity, medication use, and allergy.

The corpus contains 24 newly written fictional encounters, with four encounters in each assigned stratum:

- ordinary_question_variation
- field_local_uncertainty
- irrelevant_and_unattributed_context
- medication_allergy_scope
- correction_and_conflict
- mixed_or_ambiguous_questions

The dialogues were composed directly in natural English rather than generated from a template grammar. They use Doctor:/Patient: turns, varied question wording, short and extended replies, and familiar everyday complaints. Their content includes current-patient reports, uncertainty limited to particular facts, absent information, denials, changes in intensity, concurrent complaints, historical medication use, corrections, unresolved inconsistencies, family information, hypothetical discussion, unattributed context, compound questions, and clinician statements that receive no clear confirmation. Every encounter contains some useful current-patient information. The dialogues do not provide medical recommendations.

All people, circumstances, and utterances are invented. No actual patient or private owner material was used. The author received only the assignment specifying the six strata, format, size, and source-fidelity purpose, plus operating instructions already supplied to the session. The author did not inspect repository implementation, tests, prior cases, historical evidence, annotations, model outputs, or parent-chat content while performing this task, and did not communicate with implementation agents or annotators. The existing-file check was limited to the two requested destination paths. Any validation was limited to these newly authored files and their structural requirements.

The JSON contains only case identifiers, stratum names, and transcripts. There are no gold labels, expected extracted values, medical judgments, or predicted system outcomes. Stratum names describe the requested authoring coverage rather than answer annotations. This is a deliberately balanced development corpus; its size and sampling method do not support claims about clinical performance or population prevalence.
