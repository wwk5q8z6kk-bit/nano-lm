# Independent semantic annotation rubric

Read only this rubric and sources.json. Do not read implementation, other
annotations, previous examples, or model outputs. Independently annotate every
case and all five fields: CC (chief complaint), DUR (duration of complaint),
SEV (reported severity), MED (medication use or treatment tried for the current
complaint), ALG (reported medication allergy; named material allergies also
count if volunteered in this context). These are textual reports, not medical
judgments. Do not infer symptoms, drugs, allergy status, or timelines from
medical knowledge.

Assign one state to each field:

- S: One unambiguous reported value can be faithfully represented. Copy the
  shortest complete value phrase from the patient, excluding wrappers such as
  "I have". Preserve qualifiers needed to avoid changing meaning. Supporting
  antecedents may be additional spans if needed. Do not combine nonadjacent
  words into an invented quote.
- A: The patient explicitly denies the field in the relevant question context.
  For this task A applies to MED and ALG. Copy the complete denial sentence or
  complete denial clause. Natural wording is valid; no fixed vocabulary is
  required. Qualified lack of knowledge is U, not A.
- M: The patient provides no information about the field. Evidence list empty.
  A clinician assertion alone does not establish a patient fact. A doctor
  question without an answer remains M.
- U: The patient explicitly expresses uncertainty, the report cannot be
  resolved to one scalar value, or attribution/context is genuinely ambiguous.
  Copy the uncertainty clause(s) or competing compatible values. Two different
  medications or allergies can coexist, so label U, not C, when the scalar
  schema cannot retain both. Do not mark a clear field U merely because another
  field is uncertain or an unrelated exchange exists.
- C: Two or more incompatible patient reports address the same field and
  relevant time/context without a clear resolution. Copy at least two distinct
  observations. A positive medication report and a blanket denial for the same
  period can conflict. An explicit retraction/correction or clearly different
  times resolves apparent contradiction; annotate the resolved value/state
  rather than C. Differences in drug/allergy names alone do not imply C.

Interpret conversational context and ordinary English, including clear local
pronoun antecedents. Unrelated small talk has no effect on supported fields.
Do not use a whole-note uncertainty rule. If multiple symptoms form one clear
chief-complaint phrase, that complete phrase can be the scalar value. If an
important annotation is debatable, record why, but still give your best label.

Output a JSON list with every source case_id exactly once:

```json
[
  {
    "case_id": "fresh-01",
    "fields": {
      "CC": {"state": "M", "spans": [], "rationale": "No patient report."},
      "DUR": {"state": "M", "spans": [], "rationale": "No patient report."},
      "SEV": {"state": "M", "spans": [], "rationale": "No patient report."},
      "MED": {"state": "M", "spans": [], "rationale": "No patient report."},
      "ALG": {"state": "M", "spans": [], "rationale": "No patient report."}
    }
  }
]
```

Every quoted span must be copied exactly from patient text. Prefer a unique
occurrence. If a necessary quote repeats, represent that span as
`{"text": "exact quote", "start": 123}` with a zero-based transcript character
offset instead of an ambiguous string, and explain it. S requires at least
one evidence span, and its first span supplies the represented value. A requires
at least one denial span, C requires at least two incompatible observations,
M has no spans, and U may have one or more evidence spans (or none only if
context prevents attribution). Record your blindness and any limitations in a
separate annotation-process note. Do not coordinate with the other annotator.
