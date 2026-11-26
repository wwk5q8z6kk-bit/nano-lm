"""Complaint evidence is verified against complete words in the cited source turn."""
from dataclasses import replace

import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import assess_fields, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import build_grounded_english_menu, format_grounded_span_token, grounded_candidate_choices


@pytest.mark.parametrize("report,value,last,supported", [
    ("I have painless swelling.", "pain", False, False),
    ("I have painless swelling.", "less swelling", False, False),
    ("I have painless swelling.", "painless swell", False, False),
    ("I have earache.", "ache", False, False),
    ("I have a headache-free morning.", "headache", False, False),
    ("I have non‑tender swelling.", "tender swelling", False, False),
    ("I have swollen fingers.", "swollen finger", False, False),
    ("I have cafe\u0301 spots.", "cafe", False, False),
    ("I have swimmer’s ear.", "s ear", False, False),
    ("I have painless swelling.\nPatient: I have pain.", "pain", False, False),
    ("I have painless swelling.\nPatient: I have pain.", "pain", True, True),
    ("I have painless swelling.", "painless swelling", False, True),
    ("I have left-sided headache.", "left-sided headache", False, True),
    ("I have left‑sided headache.", "left‑sided headache", False, True),
    ("I have swimmer’s ear.", "swimmer’s ear", False, True),
    ("I have cafe\u0301 spots.", "cafe\u0301 spots", False, True),
    ("I have (headache).", "headache", False, True),
    ("I have ‘headache’.", "headache", False, True),
    ("I have headache, mild today.", "headache", False, True),
    ("I have headache—especially in the morning.", "headache", False, True),
    ("I have headache/nausea.", "headache", False, True),
])
def test_complaint_source_word_and_offsets(report, value, last, supported):
    request = NanoInput("word-boundary", "Patient: " + report)
    source = transcript_source(request)
    start = (source.text.rindex if last else source.text.index)(value)
    span = EvidenceSpan(start, start + len(value), value)
    token = format_grounded_span_token(source, span)
    result, _ = compact_solver(lambda _: f"CC:S[{token}]|DUR:M|SEV:M|MED:M|ALG:M", binder="grounded").infer_grounded(request)
    complaint = next(f for f in result.fields if f.field is FieldName.CHIEF_COMPLAINT)
    expected = FieldState.SUPPORTED if supported else FieldState.UNCERTAIN

    # Inject a supported record independently so an adapter abstention cannot mask a broken note check.
    supplied = replace(complaint, state=FieldState.SUPPORTED, value=value, evidence=(span,))
    fields = tuple(supplied if f.field is complaint.field else f for f in result.fields)
    assessment = next(a for a in assess_fields(request, replace(result, fields=fields), context_policy="grounded")
                      if a.field is complaint.field)
    assert (complaint.state, assessment.state) == (expected, expected)

    compact = grounded_candidate_choices(request.transcript, complaint.field, set())
    english = build_grounded_english_menu(source, complaint.field, set())
    if not supported or (report == "I have " + value + "."):
        assert (f":S[{token}]" in compact) == supported
        assert any(code == "S" and span in citations for code, citations in english.values()) == supported
