"""Personal subjects, literal duration bounds and typographic English in grounded evidence."""

import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import _observation_kind, transcript_source
from nanoscribe.grounded import grounded_candidate_choices
from nanoscribe.test_grounded_followup import check

FIELDS = {"MED": FieldName.MEDICATION, "ALG": FieldName.ALLERGY, "DUR": FieldName.DURATION, "SEV": FieldName.SEVERITY}


@pytest.mark.parametrize("slot,report,quote,supported", [
    ("MED", "He took aspirin.", "aspirin", False),
    ("ALG", "She is allergic to latex.", "latex", False),
    ("MED", "They took aspirin.", "aspirin", False),
    ("DUR", "They have had a cough for six days.", "six days", False),
    ("SEV", "She said the pain was severe.", "severe", False),
    ("MED", "I ate lunch and he took aspirin.", "aspirin", False),
    ("ALG", "She is allergic to nickel but I am allergic to latex.", "latex", True),
    ("ALG", "She is allergic to nickel but I am allergic to latex.", "nickel", False),
    ("MED", "They watched while I took aspirin.", "aspirin", True),
    ("MED", "I took aspirin because she recommended it.", "aspirin", True),
    ("MED", "I heard that he took aspirin.", "aspirin", False),
    ("DUR", "My headaches did not stop. They lasted four days.", "four days", True),
    ("DUR", "My headaches began last week. They’ve lasted four days.", "four days", True),
    ("DUR", "My headaches began last week. They have lasted four days.", "four days", True),
    ("ALG", "They’re allergic to latex.", "latex", False),
    ("MED", "I took aspirin. She is not sure whether I took it.", "aspirin", True),
    ("MED", "Do you think he took aspirin and I took ibuprofen?", "ibuprofen", False),
    ("MED", "I am unsure whether he took aspirin and I took ibuprofen.", "ibuprofen", False),
    ("MED", "The chart says he took aspirin and I took ibuprofen.", "ibuprofen", False),
    ("DUR", "The cough did not last no more than four days.", "no more than four days", False),
    ("MED", "I haven’t taken aspirin.", "aspirin", False),
    ("MED", "I didn’t take aspirin.", "aspirin", False),
    ("ALG", "Latex isn’t something I am allergic to.", "Latex", False),
    ("ALG", "I’m allergic to latex.", "latex", True),
    ("MED", "I’ve been taking aspirin.", "aspirin", True),
    ("MED", "I didn’t take aspirin. I took aspirin.", "aspirin", True),
    ("ALG", "I’m allergic to St John’s wort.", "St John’s wort", True),
    ("DUR", "I missed at least two classes. The cough lasted four days.", "four days", True),
])
def test_evidence_keeps_its_subject_polarity_and_source_spelling(slot, report, quote, supported):
    source = "Patient: " + report
    check(source, slot, FIELDS[slot], quote, source.rindex(quote), supported)


@pytest.mark.parametrize("qualifier", [
    "less than four days", "no more than four days", "more than four days", "at least four days",
    "at most four days", "under four days", "over four days", "up to four days",
    "four days or less", "four days or more", "four days at most", "four days at least",
])
@pytest.mark.parametrize("clip", [False, True])
def test_duration_cannot_become_exact_by_dropping_its_bound(qualifier, clip):
    source = f"Patient: The cough lasted {qualifier}."
    quote = "four days" if clip else qualifier
    check(source, "DUR", FieldName.DURATION, quote, source.index(quote), not clip)


@pytest.mark.parametrize("report,field", [("He took no medications.", FieldName.MEDICATION),
                                         ("She has no allergies.", FieldName.ALLERGY)])
def test_a_third_party_denial_does_not_become_the_patients_denial(report, field):
    source = "Patient: " + report
    doc = transcript_source(NanoInput("subjects", source))
    span = EvidenceSpan(source.index(report), len(source), report)
    assert _observation_kind(doc, field, span, context_policy="grounded") is not FieldState.ABSENT
    assert f":A[{report}]" not in grounded_candidate_choices(source, field, set())
