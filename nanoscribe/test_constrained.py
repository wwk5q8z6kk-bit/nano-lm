"""Exercise source selection and grammar enforcement without loading weights."""
import pytest
from nanoscribe.compact import LocalCompactPredictor
from nano_ai.adapters.state_span import parse_state_span_summary


def test_source_constrained_fields_keep_all_states_and_exact_conflict(tmp_path, monkeypatch):
    source = "Doctor: What symptoms?\nPatient: I've been having toe pain.\nDoctor: How long?\nPatient: For about 13 days now.\nDoctor: How long?\nPatient: For about 2 weeks now.\nDoctor: How intense?\nPatient: I cannot rate it.\nDoctor: Any medicine?\nPatient: No, nothing yet."
    answers = iter([":S[toe pain]", ":C[13 days", ";2 weeks]", ":U[I cannot rate it.]", ":A[No, nothing yet.]", ":M"])
    calls = []
    def choose(messages, choices):
        answer = next(answers)
        assert answer in choices
        assert messages[1]["content"] == source
        assert messages[-1]["role"] == "assistant"
        assert not messages[-1]["content"].endswith(":")
        calls.append(answer)
        return answer, {"generated_tokens": 3, "generation_seconds": .1}
    predictor = LocalCompactPredictor(str(tmp_path), decoding="constrained")
    monkeypatch.setattr(predictor, "_choose", choose)
    raw = predictor(source)
    assert raw == "CC:S[toe pain]|DUR:C[13 days;2 weeks]|SEV:U[I cannot rate it.]|MED:A[No, nothing yet.]|ALG:M"
    assert len(parse_state_span_summary(raw, source)) == 5
    assert predictor(source) == raw
    assert len(calls) == 6


def test_constrained_span_menu_excludes_clinician_and_ambiguous_text(tmp_path):
    predictor = LocalCompactPredictor(str(tmp_path), decoding="constrained")
    spans, replies = predictor.source_choices("Doctor: invented\nPatient: Just zinc.\nDoctor: allergy?\nPatient: Just zinc.")
    assert "invented" not in spans
    assert "zinc" not in spans
    assert "Just zinc." not in replies


def test_invalid_constrained_choice_cannot_be_silently_repaired(tmp_path, monkeypatch):
    predictor = LocalCompactPredictor(str(tmp_path), decoding="constrained")
    monkeypatch.setattr(predictor, "_choose", lambda *_: ("S[invented]", {}))
    with pytest.raises(ValueError):
        predictor("Doctor: What symptoms?\nPatient: I've been having toe pain.")


def test_contextual_menu_cannot_choose_another_fields_reply(tmp_path, monkeypatch):
    source = "Doctor: Which medicine did you try?\nPatient: I've been taking diclofenac.\nDoctor: Which substances are you allergic to?\nPatient: I'm allergic to chromium."
    def choose(messages, choices):
        field = messages[-1]["content"].split("|")[-1]
        if field == "MED":
            assert ":S[diclofenac]" in choices
            assert ":S[chromium]" not in choices
            return ":S[diclofenac]", {}
        if field == "ALG":
            assert ":S[chromium]" in choices
            assert ":S[diclofenac]" not in choices
            return ":S[chromium]", {}
        assert choices == [":M", ":U[]"]
        return ":M", {}
    predictor = LocalCompactPredictor(str(tmp_path), decoding="constrained")
    monkeypatch.setattr(predictor, "_choose", choose)
    assert predictor(source) == "CC:M|DUR:M|SEV:M|MED:S[diclofenac]|ALG:S[chromium]"


def test_duration_and_severity_candidates_respect_field_types(tmp_path, monkeypatch):
    source = "Doctor: What is the duration of your symptoms?\nPatient: For about 13 days now.\nDoctor: Please describe how severe your discomfort is.\nPatient: I'd call it severe."
    def choose(messages, choices):
        field = messages[-1]["content"].split("|")[-1]
        if field == "DUR":
            assert ":S[13 days]" in choices
            assert ":S[about 13 days]" not in choices
            assert not any(c.startswith(":C[") for c in choices)
            return ":S[13 days]", {}
        if field == "SEV":
            assert ":S[severe]" in choices
            assert ":S[severe.]" not in choices
            return ":S[severe]", {}
        return ":M", {}
    predictor = LocalCompactPredictor(str(tmp_path), decoding="constrained")
    monkeypatch.setattr(predictor, "_choose", choose)
    assert "DUR:S[13 days]|SEV:S[severe]" in predictor(source)
