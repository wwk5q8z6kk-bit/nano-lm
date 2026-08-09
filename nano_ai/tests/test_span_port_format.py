"""Unit tests for the route-(b) span-port parse/format helpers."""

from __future__ import annotations

from nano_ai.training.run_span_port_probe import (
    NO_MATCH_ACCEPT,
    NO_MATCH_FALSIFY,
    _decision,
)
from nano_ai.training.span_port_format import (
    PROMPT_WITH_SPANS,
    classify_span_relocation,
    format_span_answer,
    parse_label_and_spans,
)


def test_prompt_requests_quoted_span_line_not_one_word() -> None:
    assert "exactly one word" not in PROMPT_WITH_SPANS
    assert 'STATED: "exact patient words' in PROMPT_WITH_SPANS
    assert "NOT_MENTIONED" in PROMPT_WITH_SPANS


def test_format_span_answer_joins_quoted_spans() -> None:
    assert format_span_answer("DENIED", ("No allergy.",)) == 'DENIED: "No allergy."'
    assert format_span_answer("NOT_MENTIONED", ()) == "NOT_MENTIONED"
    assert (
        format_span_answer("STATED", ("penicillin", "amoxicillin"))
        == 'STATED: "penicillin" "amoxicillin"'
    )


def test_parse_label_and_spans_reads_final_label_and_quotes() -> None:
    label, spans = parse_label_and_spans('DENIED: "I take no medication."')
    assert label == "DENIED"
    assert spans == ("I take no medication.",)

    label, spans = parse_label_and_spans("NOT_MENTIONED")
    assert label == "NOT_MENTIONED"
    assert spans == ()

    label, spans = parse_label_and_spans(
        'thinking...\nSTATED: "penicillin"\n'
    )
    assert label == "STATED"
    assert spans == ("penicillin",)


def test_parse_returns_none_on_garbage() -> None:
    label, spans = parse_label_and_spans("I am not sure")
    assert label is None
    assert spans == ()


def test_classify_span_relocation_matches_locator_refusals() -> None:
    transcript = (
        "Doctor: Any allergies?\n"
        "Patient: No allergy.\n"
        "Doctor: Thanks.\n"
        "Patient: No allergy.\n"
    )
    assert classify_span_relocation(transcript, "No allergy.") == "ambiguous"
    assert classify_span_relocation(transcript, "penicillin") == "no_match"

    unique = (
        "Doctor: Any allergies?\n"
        "Patient: No allergy.\n"
    )
    assert classify_span_relocation(unique, "No allergy.") == "relocated"


def test_prereg_decision_thresholds_are_frozen() -> None:
    assert NO_MATCH_ACCEPT == 0.10
    assert NO_MATCH_FALSIFY == 0.25
    assert _decision(0.05, True) == "route_b_sufficient"
    assert _decision(0.11, True) == "gray_zone"
    assert _decision(0.26, True) == "route_b_falsified"
    assert _decision(0.0, False) == "uninterpretable"
    assert _decision(None, True) == "no_spans_emitted"
