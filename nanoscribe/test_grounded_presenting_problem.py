"""Check the literal symptom value in patient-owned presenting problem statements."""
import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import _observation_kind, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import (
    build_grounded_english_menu, grounded_candidate_choices, serialize_grounded_slot,
    verify_grounded_span,
)
from nanoscribe.scribe import scribe


@pytest.mark.parametrize("text,quote,valid", [
    ("Doctor: What brings you in?\nPatient: The main problem is a throbbing headache.", "throbbing headache", True),
    ("Doctor: What brings you in?\nPatient: The main problem is a throbbing headache.", "a throbbing headache", False),
    ("Doctor: Tell me your main symptom.\nPatient: The primary concern is joint pain.", "joint pain", True),
    ("Doctor: Which symptom are you here for?\nPatient: The main problem is a rash on my arm.", "rash on my arm", True),
    ("Doctor: What brings you in?\nPatient: The main problem is a cough.\nPatient: I misspoke about my symptom. The main problem is joint pain.", "cough", False),
    ("Doctor: What brings you in?\nPatient: The main problem is a cough.\nPatient: I misspoke about my symptom. The main problem is joint pain.", "joint pain", True),
    ("Doctor: What brings you in?\nPatient: The main problem is a cough.\nPatient: I misspoke about my medication.", "cough", True),
    ("Doctor: What brings you in?\nPatient: The main problem is a cough.\nOther: I misspoke about my symptom.", "cough", True),
    ("Patient: I misspoke about my parking pass. The main problem is a cough.", "cough", False),
    ("Doctor: What brings you in?\nPatient: The main problem is pain that went away months ago.", "pain", False),
    ("Doctor: What brings you in?\nPatient: The main problem is pain that could start tomorrow.", "pain", False),
    ("Doctor: What brings you in?\nPatient: The main problem is pain that is expected next year.", "pain", False),
    ("Doctor: What brings you in?\nPatient: The main problem is pain that will develop after surgery.", "pain", False),
    ("Doctor: What brings you in?\nPatient: The main problem is a rash that will appear after the procedure.", "rash", False),
    ("Doctor: What brings you in?\nPatient: The main problem is nausea that is going to start after the infusion.", "nausea", False),
    ("Doctor: What brings you in?\nPatient: The main problem is pain I will experience after surgery.", "pain", False),
    ("Doctor: What brings you in?\nPatient: The main problem is pain that starts when I bend my knee.", "pain that starts when I bend my knee", True),
    ("Doctor: What brings you in?\nPatient: The main problem is pain from a form I was reading.", "pain", False),
    ("Doctor: What brings you in?\nPatient: The main problem is a rash in an example in this worksheet.", "rash", False),
    ("Doctor: What brings you in?\nPatient: The main problem is a cough on this chart.", "cough", False),
    ("Doctor: What brings you in?\nPatient: The main problem is pain from a twisted ankle.", "pain from a twisted ankle", True),
    ("Doctor: What brings you in?\nPatient: The main problem is pain from the form I was reading.", "pain", False),
    ("Doctor: What brings you in?\nPatient: The main problem is pain that went away but then returned.", "pain that went away but then returned", True),
    ("Doctor: What brings you in?\nPatient: The main problem is pain in my own ankle.", "pain in my own ankle", True),
    ("Patient: My presenting problem is nausea.", "nausea", True),
    ("Patient: My main problem is a cough. I also need a receipt.", "cough", True),
    ("Patient: My main problem is a cough. I also need a receipt.", "receipt", False),
    ("Doctor: Why did the printer stop?\nPatient: The main problem is a headache.", "headache", False),
    ("Doctor: What medication are you taking?\nPatient: The main problem is a cough.", "cough", False),
    ("Doctor: What brings you in?\nPatient: The main problem is a headache diary.", "headache", False),
    ("Doctor: What brings you in?\nPatient: The main problem is not a headache.", "headache", False),
    ("Doctor: What brings you in?\nPatient: The main problem is my son's earache.", "earache", False),
    ("Doctor: What brings you in?\nPatient: The main problem is a rash that I might develop tomorrow.", "rash", False),
    ("Doctor: What brings you in?\nPatient: The main problem is a headache that disappeared last year.", "headache", False),
    ("Doctor: What brings you in?\nPatient: The main problem is parking. I have sharp joint pain.", "sharp joint pain", True),
    ("Doctor: What brings you in?\nPatient: The main problem is a headache.\nPatient: I was mistaken about my symptom.", "headache", False),
])
def test_presenting_problem_owns_the_current_symptom(text, quote, valid):
    field = FieldName.CHIEF_COMPLAINT
    source = transcript_source(NanoInput("presenting", text))
    start = text.index(quote)
    span = EvidenceSpan(start, start + len(quote), quote)
    assert verify_grounded_span(source, field, FieldState.SUPPORTED, span)[0] == valid
    assert (_observation_kind(source, field, span, context_policy="grounded") is FieldState.SUPPORTED) == valid
    slot = serialize_grounded_slot(source, field, "S", (span,))
    assert (slot[2:] in grounded_candidate_choices(text, field, set())) == valid
    assert (("S", (span,)) in build_grounded_english_menu(source, field, set()).values()) == valid
    solver = compact_solver(lambda _: slot + "|DUR:M|SEV:M|MED:M|ALG:M", binder="grounded")
    reply = scribe(text, solver=solver, context_policy="grounded")
    assert reply["verification"]["passed"]
    claim = next(row for row in reply["note"]["claims"] if row["field"] == field.value)
    assert (claim["state"] == "supported") == valid
