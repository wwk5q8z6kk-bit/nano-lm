"""A current severity assertion must retain the time of its measurement."""
from dataclasses import replace

import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import assess_fields, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import build_grounded_english_menu, format_grounded_span_token, grounded_candidate_choices


@pytest.mark.parametrize("reply, quote, accepted", [
    ("Seven out of ten was the number just after the fall.", "Seven out of ten", False),
    ("Eight out of ten was the score at its worst.", "Eight out of ten", False),
    ("Four out of ten was my rating yesterday.", "Four out of ten", False),
    ("Six out of ten was the number before the visit.", "Six out of ten", False),
    ("Right now it is three out of ten.", "three out of ten", True),
    ("Right now, two out of ten.", "two out of ten", True),
    ("At the moment it is five out of ten.", "five out of ten", True),
    ("Seven out of ten.", "Seven out of ten", True),
    ("My wrist pain is four out of ten after exercise.", "four out of ten", True),
    ("My bus ride is three out of ten.", "three out of ten", False),
])
def test_current_severity_source_paths(reply, quote, accepted):
    field = FieldName.SEVERITY
    request = NanoInput("current-rating", "Doctor: How bad is the pain now?\nPatient: " + reply)
    source = transcript_source(request)
    start = request.transcript.index(quote)
    span = EvidenceSpan(start, start + len(quote), quote)
    token = format_grounded_span_token(source, span)
    raw = f"CC:M|DUR:M|SEV:S[{token}]|MED:M|ALG:M"
    result, _ = compact_solver(lambda _: raw, binder="grounded").infer_grounded(request)
    rating = next(entry for entry in result.fields if entry.field is field)
    assert (rating.state is FieldState.SUPPORTED) == accepted
    supplied = replace(rating, state=FieldState.SUPPORTED, value=quote, evidence=(span,))
    extraction = replace(result, fields=tuple(supplied if entry.field is field else entry for entry in result.fields))
    checked = next(entry for entry in assess_fields(request, extraction, context_policy="grounded") if entry.field is field)
    assert (checked.state is FieldState.SUPPORTED) == accepted
    menu = build_grounded_english_menu(source, field, set())
    assert any(code == "S" and span in spans for code, spans in menu.values()) == accepted
    assert (f":S[{quote}]" in grounded_candidate_choices(request.transcript, field, set())) == accepted
