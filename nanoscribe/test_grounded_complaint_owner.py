"""A complaint citation identifies the clinical report made by the patient."""
from dataclasses import replace

import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import assess_fields, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import build_grounded_english_menu, format_grounded_span_token


@pytest.mark.parametrize("transcript,quote,expected", [
    ("Patient: I took aspirin.", "aspirin", False),
    ("Patient: I took aspirin for a headache.", "headache", False),
    ("Doctor: What did you take?\nPatient: Aspirin.", "Aspirin", False),
    ("Patient: I am allergic to latex.", "latex", False),
    ("Doctor: Where do you work?\nPatient: I work at a bank.", "bank", False),
    ("Doctor: What is bothering you?\nPatient: I work at the pain clinic.", "pain", False),
    ("Patient: I have a copy of the report about wrist pain.", "wrist pain", False),
    ('Patient: Please print the report that says I have chills and my head hurts.', "head hurts", False),
    ("Doctor: What brings you in?\nPatient: I carry a coin in my hand.", "coin in my hand", False),
    ("Patient: I have a headache after visiting the bank.", "bank", False),
    ("Patient: I work at a bank, but I have knee pain.", "bank, but I have knee pain", False),
    ("Patient: I work at a bank, but I have knee pain.", "knee pain", True),
    ("Patient: My ankle aches and I need my invoice.", "ankle aches", True),
    ("Patient: My ankle aches and I need my invoice.", "invoice", False),
    ("Patient: I have chest pressure when I check my bills.", "chest pressure when I check my bills", True),
    ("Doctor: Which medication did you take?\nPatient: Aspirin. I have aching shoulders.", "aching shoulders", True),
    ("Doctor: Where do you work?\nPatient: I work at the clinic. I have blurred vision.", "blurred vision", True),
    ("Doctor: What brings you in today?\nPatient: Jagged flashes across my vision.", "Jagged flashes across my vision", True),
    ("Doctor: Which medication did you take?\nPatient: Cough syrup.", "Cough", False),
    ("Patient: I have cough drops.", "cough", False),
    ("Patient: I have a headache diary.", "headache", False),
    ("Patient: I have chest pain and an invoice to pay.", "chest pain and an invoice to pay", False),
    ("Patient: I have chest pain and an invoice to pay.", "chest pain", True),
    ("Patient: I have shortness of breath.", "shortness of breath", True),
    ("Doctor: What brings you in?\nPatient: Loss of taste.", "Loss of taste", True),
    ("Patient: My hands shake.", "hands shake", True),
    ("Patient: Please update my telephone number. I also have an itchy rash.", "itchy rash", True),
    ("Patient: I still have shoulder pain, and I also want a copy of my bill.", "shoulder pain", True),
    ("Patient: I have joint pain.", "joint pain", True),
    ("Patient: I have twitching.", "twitching", True),
    ("Patient: I have acute pain.", "acute pain", True),
    ("Doctor: What brings you in?\nPatient: I am on my period with cramps.", "cramps", True),
    ("Patient: I'm menstruating with sharp abdominal pain.", "sharp abdominal pain", True),
    ("Patient: I am on my period tracker.", "period tracker", False),
    ("Patient: I am on aspirin for cramps.", "cramps", False),
    ("Doctor: What brings you in?\nPatient: I am on edge.", "on edge", True),
    ("Patient: I've been feeling on edge.", "on edge", True),
    ("Patient: I am on edge of the platform.", "on edge", False),
    ("Patient: I am on the edge of my seat.", "edge", False),
    ("Patient: I need a refill and my wrist hurts.", "my wrist hurts", True),
    ("Patient: I need a refill and my wrist hurts.", "refill", False),
    ("Patient: I took a fall and my left ankle aches.", "my left ankle aches", True),
])
def test_grounded_complaint_and_injected_supported_record(transcript, quote, expected):
    item = NanoInput("clinical-complaint", transcript)
    source = transcript_source(item)
    start = transcript.index(quote, transcript.index("Patient:"))
    span = EvidenceSpan(start, start + len(quote), quote)
    raw = f"CC:S[{format_grounded_span_token(source, span)}]|DUR:M|SEV:M|MED:M|ALG:M"
    result, _ = compact_solver(lambda _: raw, binder="grounded").infer_grounded(item)
    field = FieldName.CHIEF_COMPLAINT
    complaint = next(entry for entry in result.fields if entry.field is field)
    assert (complaint.state is FieldState.SUPPORTED) == expected

    supplied = replace(complaint, state=FieldState.SUPPORTED, value=quote, evidence=(span,))
    extraction = replace(result, fields=tuple(supplied if entry.field is field else entry for entry in result.fields))
    checked = next(entry for entry in assess_fields(item, extraction, context_policy="grounded") if entry.field is field)
    assert (checked.state is FieldState.SUPPORTED) == expected
    choices = build_grounded_english_menu(source, field, set())
    assert any(code == "S" and span in evidence for code, evidence in choices.values()) == expected
