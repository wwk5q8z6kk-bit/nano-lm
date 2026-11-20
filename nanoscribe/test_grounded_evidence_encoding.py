"""Every admitted state can carry exact patient text across the compact wire grammar."""
import pytest

from nano_ai.contract import EvidenceSpan, FieldName, NanoInput
from nanoscribe.check_note import transcript_source
from nanoscribe.compact import LocalCompactPredictor, compact_solver
from nanoscribe.grounded import (
    build_grounded_english_menu, format_grounded_span_token, grounded_candidate_choices,
    parse_grounded_summary, serialize_grounded_slot,
)
from nanoscribe.scribe import scribe


def selected(source, text, *, last=False):
    start = source.rindex(text) if last else source.index(text)
    return EvidenceSpan(start, start + len(text), text)


@pytest.mark.parametrize("quote", [
    "headache [left-sided]", "pain [left", "pain right]", "pain [[both]]", "pain | pressure",
    "pain || pressure", "pain [50%] | café", "@2..5:literal", "@2..5#literal",
])
@pytest.mark.parametrize("code", ["S", "A", "U", "C"])
def test_syntax_in_evidence_round_trips_with_original_offsets(quote, code):
    transcript = f"Doctor: {quote}\nPatient: {quote}\nPatient: {quote}\nPatient: other [item]"
    doc = transcript_source(NanoInput("wire", transcript))
    source_span = selected(transcript, quote, last=True)
    pair = code in ("U", "C")
    spans = (source_span, selected(transcript, "other [item]")) if pair else (source_span,)
    slot = serialize_grounded_slot(doc, FieldName.ALLERGY, code, spans)
    _, props, errors, strict_error = parse_grounded_summary("CC:M|DUR:M|SEV:M|MED:M|" + slot, transcript)
    assert strict_error is None and errors == [None] * 5
    assert props[-1].spans == spans


@pytest.mark.parametrize("value,expected", [
    ("headache", "headache"), ("left; right", "@19..30:left; right"),
    ("pollen", "@40..46:pollen"), ("50% better", "50% better"),
])
def test_existing_delimiter_safe_grammar_remains_verbatim(value, expected):
    text = "Patient: headache, left; right, pollen, pollen, 50% better"
    source = transcript_source(NanoInput("legacy-wire", text))
    span = selected(text, value, last=True)
    assert format_grounded_span_token(source, span) == expected


@pytest.mark.parametrize("suffix", ["different", "%5B", "%zz", "pain%255B"])
def test_encoded_quote_must_match_the_complete_source(suffix):
    text = "Patient: pain["
    raw = f"CC:S[@9..14:#{suffix}]|DUR:M|SEV:M|MED:M|ALG:M"
    _, props, errors, strict = parse_grounded_summary(raw, text)
    assert strict and errors[0] and props[0] is None
    assert all(e is None for e in errors[1:])


def test_encoded_evidence_still_requires_a_patient_turn():
    text = "Doctor: pain[\nPatient: yes"
    raw = "CC:S[@8..13:#pain%5B]|DUR:M|SEV:M|MED:M|ALG:M"
    _, props, _, error = parse_grounded_summary(raw, text)
    assert error and props[0] is None


@pytest.mark.parametrize("cache", [False, True])
def test_constrained_menu_recovers_the_complete_encounter_with_bracket_evidence(monkeypatch, tmp_path, cache):
    cc = "headache [left-sided]"
    transcript = f"Patient: I have {cc}.\nPatient: I took aspirin.\nPatient: No known allergies."
    answers = iter([f" REPORTED[{cc}]", " MISSING", " MISSING", " REPORTED[aspirin]", " DENIED[No known allergies.]"])
    def choose(messages, choices):
        answer = next(answers)
        assert answer in choices
        return answer, {}
    predictor = LocalCompactPredictor(str(tmp_path), decoding="constrained", candidate_policy="grounded", prefix_cache=cache)
    monkeypatch.setattr(predictor, "_choose", choose)
    raw = predictor(transcript)
    _, props, errors, strict = parse_grounded_summary(raw, transcript)
    assert not strict and errors == [None] * 5
    assert props[0].spans == (selected(transcript, cc),)
    assert raw.split("|")[0][2:] in grounded_candidate_choices(transcript, FieldName.CHIEF_COMPLAINT, set())
    doc = transcript_source(NanoInput("clinical-wire", transcript))
    assert build_grounded_english_menu(doc, FieldName.CHIEF_COMPLAINT, set())[f" REPORTED[{cc}]"][1] == props[0].spans
    note = scribe(transcript, solver=compact_solver(predictor, binder="grounded"), context_policy="grounded")
    assert note["verification"]["passed"]
    claims = {entry["field"]: entry for entry in note["note"]["claims"]}
    assert claims["chief_complaint"]["value"] == cc
    assert claims["medication"]["value"] == "aspirin"
    assert claims["allergy"]["state"] == "absent"


def test_offset_payload_that_looks_encoded_retains_its_literal_legacy_meaning():
    transcript = "Patient: #pain%5B"
    _, props, errors, _ = parse_grounded_summary("CC:S[@9..17:#pain%5B]|DUR:M|SEV:M|MED:M|ALG:M", transcript)
    assert errors == [None] * 5
    assert props[0].spans == (selected(transcript, "#pain%5B"),)
