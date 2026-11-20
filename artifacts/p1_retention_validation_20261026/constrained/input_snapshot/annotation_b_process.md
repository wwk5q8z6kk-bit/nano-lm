# Independent annotation B process

I read only `artifacts/p1_retention_validation_20261026/sources.json` and
`artifacts/p1_scope_validation_20261023/annotation_rubric.md` for this annotation.
I did not inspect implementation, tests, earlier examples or gold labels, other
annotations, or model/extractor outputs. I did not coordinate with another
annotator or invoke a production extractor.

I manually assigned every case all five fields, using patient statements and
local conversational context. These are textual semantic judgments, not
clinical judgments or a human/clinician reliability evaluation. No clinical
inference was used.

For supported values I copied a short complete phrase, retaining relevant
qualifiers. For denials I copied a complete sentence or denial clause. Explicit
uncertainty remained field-local. Coexisting medication/allergy values were
marked U because of the scalar schema. Explicit corrections resolved earlier
claims; unresolved incompatible reports retained C. Clinician assertions,
other people's histories, unrelated timelines, and unanswered questions did
not establish a patient fact.

Interpretation limitations are recorded as rationales in the annotations. In
particular, retain-12 MED is U: taking a tablet is certain, but its identity is
explicitly unknown. Retain-21 DUR is M because “persistent” supplies no duration
and the timing question is unanswered. Retain-02 ALG uses the directly named
medicine in the allergy-question context without inferring anything beyond
that report.

A mechanical check used only the source transcripts and these manually
entered annotations in memory. It verified coverage, state constraints, exact
substring matches, patient-speaker attribution, and zero-based offsets. It
did not derive or revise semantic labels. Wherever a quote recurs anywhere
in the transcript, an explicit offset selects the intended patient occurrence;
the first patient occurrence was the intended one for each repeated quote
listed below.

- retain-12 CC: "toothache", start 46.
- retain-13 CC: "headache", start 42.

Validation: 24 unique cases, 120 field labels, 106 exact patient spans.
State counts: S=61, A=19, M=20, U=16, C=4.
