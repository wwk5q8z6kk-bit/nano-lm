# Bounded transcript-to-draft path

This executable increment connects the existing `nano_ai` transcript-only
solver contract to the NanoScribe encounter schema, a deterministic note view,
and separate source/representation checking. It does not establish learned
model success, clinical adequacy, or a P1 exit result.

From the repository root:

```sh
python3 -m nanoscribe.scribe --transcript nanoscribe/examples/synthetic_encounter.txt --format text
python3 -m nanoscribe.scribe --transcript nanoscribe/examples/synthetic_encounter.txt > /tmp/nanoscribe-draft.json
python3 -m pytest -q nanoscribe/test_scribe.py
```

The example is invented, contains no private patient data, and is a software
smoke fixture. Its successful output is not an extraction benchmark. The CLI
runs locally without model downloads, network access, paid inference, training,
or frozen evaluation data. A successful execution exits zero even when the
disposition is `abstain`; invalid input or failed inference/checking exits 2.

The text example produces:

```text
Status: draft; disposition: review
Patient-reported encounter draft — human review required
Chief complaint: patient reports "headache" [e:chief_complaint:0]
Duration: patient reports "3 days" [e:duration:0]
Severity: patient reports "mild" [e:severity:0]
Medication: patient reports "acetaminophen" [e:medication:0]
Allergy: patient denial "No allergies." [e:allergy:0]
Review scope: Five patient-reported fields only; source checks are bounded and human review is required. No diagnosis, plan, or full encounter completeness is established.
```

The input is UTF-8, one labeled `Doctor:` or `Patient:` turn per line. Source
bytes are decoded without newline conversion. Spans use Unicode character
offsets, not byte offsets. The full original source, speakers, containing turns,
exact evidence, candidate extraction, checked encounter, draft, source hash,
solver descriptor, and checking result are retained in JSON for review.

## Execution and boundaries

`scribe(transcript, encounter_id=..., solver=...)` sends only `NanoInput`
(`item_id`, `transcript`, `schema_version`) to the existing `run_inference`
boundary. It never supplies target values, gold spans, `AtomSpec`, or scoring
data. The default is the existing `DeterministicV0Solver`: an explicitly labeled
**reference solver for the synthetic five-field grammar**, not Nano's trained
model. Python callers can supply another existing `NanoSolver`; the checker
retains the same narrow acceptance boundary. This increment does not load H6,
rerun its evaluation, or promote any checkpoint.

`check_note.assess_fields` independently inspects source context. It requires
an allowlisted preceding question, a complete supported reply template, and a
plain value in its bounded grammar. It deliberately recognizes fewer phrasings
than the reference extractor. Explicit medication/allergy denial is retained
as a denial quote; silence is missing. Qualified denial such as “Not that I
know of” is quarantined even if the reference solver calls it absent.
Uncertain/conflicting fields retain their evidence but have no asserted value.
Any unhandled source turn quarantines all otherwise presented fields, because
an ignored correction or instruction about another person could affect any of
them. This includes clinician turns other than the exact allowlisted questions.
So does an
evidence-bearing reply outside the checked observation grammar, even when the
solver assigned it to uncertainty in a different field. This conservative rule
prevents a consumed correction from leaving an earlier denial in the draft.
Unhandled clinician content is preserved as a review item; clinician quotations
cannot serve as patient evidence. Contextual words for non-use, contemplated use,
or qualification are also excluded from scalar values: “Just avoiding aspirin”
and “potentially penicillin” remain uncertain despite a positive reply wrapper.
Chief-complaint scalars also exclude explicit allergy/medication relation words
(such as “allergy”, “medicine”, “effects”, or “reaction”). Thus “penicillin
allergy” or “aspirin side effects” cannot count as a handled complaint while a
separate denial remains asserted. The checker quarantines the values; it does
not infer or resolve a clinical relationship between fields.
These exclusions are conservative lexical guards, not a complete language model.

The checked projection uses existing `EncounterRecord` atoms. Chief complaint,
duration, and severity are symptom-related scalar atoms identified by their
field IDs; this is not a new comprehensive clinical ontology or a relationship
inference. Medication/allergy have their existing atom types. Temporality is
unknown, every atom requires review, and uncertain/conflicting observations do
not assert a patient experiencer. Literal scalar values do not capture every
nuance (for example, approximate duration); the original full reply remains
the authoritative context for a reviewer.

`render_note` produces a fixed five-field view. `check_note` never calls the
extractor or renderer. It checks against the original input: exact source and
speaker binding, every field's state/value/evidence, atom metadata, conflicts,
unresolved reasons/references, and all visible note text. Unexpected, omitted,
or changed claims fail the check. Inference errors stay explicit failures with
no fabricated missing output. Failed final verification withholds the note.
Encounter-schema failures (including an encounter ID colliding with an
internal ID) are explicit projection failures, also with no draft.

The final disposition uses `nano.runtime.Disposition`: useful checked content
is `review`; no safely presented content is `abstain`. This path never returns
`present`. A passing check establishes only the bounded source/representation
properties above. A phrase allowlist plus lexical exclusions is not a general
semantic or clinical verifier, and it can abstain excessively or miss meaning
outside the tested cases. It does not prove completeness, medical correctness,
clinical usefulness, clinician review, or readiness to put a note in a chart.

## Boundary verification

Tests exercise values derived from the transcript, original Unicode/CRLF
offsets, absence versus silence, qualified denials, conflicts, negative/hedged
and other-person language, injected instructions, unconsumed corrections,
clinician context/corrections, medication non-use and contemplated use,
cross-field context inside complaint scalars,
wrong-field and wrong-speaker source spans, solver exceptions, and CLI error
behavior. Mutation tests change draft prose/value/state/references, remove or
add claims, alter atom metadata, remove conflicts, change unresolved review
metadata, and show that the pipeline withholds a corrupted draft. These are
software boundary checks, not a held-out medical evaluation.
