"""Exercise source selection and grammar enforcement without loading weights."""
import pytest
from nanoscribe.compact import LocalCompactPredictor
from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FieldName


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


def _constrained_menus(tmp_path, monkeypatch, source):
    """Observe the actual decoder menus without loading a model checkpoint."""
    menus = {}
    predictor = LocalCompactPredictor(str(tmp_path), decoding="constrained")
    def choose(messages, choices):
        menus[messages[-1]["content"].split("|")[-1]] = choices
        return ":M", {}
    monkeypatch.setattr(predictor, "_choose", choose)
    predictor(source)
    return menus


@pytest.mark.parametrize("label,question,reply,value", [
    ("MED", "Any medicine?", "I've been taking diclofenac.", "diclofenac"),
    ("MED", "Any medicine?", "Only zinc so far.", "zinc"),
    ("ALG", "Any allergies?", "I'm allergic to chromium.", "chromium"),
    ("ALG", "Any allergies?", "Yes, cobalt.", "cobalt"),
])
def test_positive_reply_admits_only_its_scalar_and_never_absence(tmp_path, monkeypatch, label, question, reply, value):
    choices = _constrained_menus(tmp_path, monkeypatch, f"Doctor: {question}\nPatient: {reply}")[label]
    assert [choice for choice in choices if choice.startswith(":S[")] == [f":S[{value}]"]
    assert not any(choice.startswith(":A[") for choice in choices)
    assert f":U[{reply}]" in choices


@pytest.mark.parametrize("label,question,reply", [
    ("MED", "Any medicine?", "No, nothing yet."),
    ("MED", "Any medicine?", "I haven't taken anything."),
    ("ALG", "Any allergies?", "No allergies."),
    ("ALG", "Any allergies?", "None whatsoever."),
])
def test_complete_denial_never_admits_supported_fragments(tmp_path, monkeypatch, label, question, reply):
    choices = _constrained_menus(tmp_path, monkeypatch, f"Doctor: {question}\nPatient: {reply}")[label]
    assert [choice for choice in choices if choice.startswith(":A[")] == [f":A[{reply}]"]
    assert not any(choice.startswith(":S[") for choice in choices)
    assert f":U[{reply}]" in choices


@pytest.mark.parametrize("label,question,reply", [
    ("MED", "Any medicine?", "No medication except zinc."),
    ("MED", "Any medicine?", "I've been taking no medication."),
    ("ALG", "Any allergies?", "No allergies except cobalt."),
    ("ALG", "Any allergies?", "I'm allergic to no substances."),
])
def test_qualified_denial_is_available_only_for_uncertainty(tmp_path, monkeypatch, label, question, reply):
    choices = _constrained_menus(tmp_path, monkeypatch, f"Doctor: {question}\nPatient: {reply}")[label]
    assert not any(choice.startswith((":S[", ":A[", ":C[")) for choice in choices)
    assert f":U[{reply}]" in choices


@pytest.mark.parametrize("label,question,reply", [
    ("MED", "Any medicine?", "I've been taking zinc and aspirin."),
    ("MED", "Any medicine?", "Only zinc with aspirin so far."),
    ("ALG", "Any allergies?", "I'm allergic to cobalt and nickel."),
    ("ALG", "Any allergies?", "Yes, cobalt reaction."),
])
def test_compound_reply_does_not_admit_a_scalar_or_its_fragments(tmp_path, monkeypatch, label, question, reply):
    choices = _constrained_menus(tmp_path, monkeypatch, f"Doctor: {question}\nPatient: {reply}")[label]
    assert not any(choice.startswith((":S[", ":A[", ":C[")) for choice in choices)
    assert f":U[{reply}]" in choices


@pytest.mark.parametrize("label,question,first,second", [
    ("MED", "Any medicine?", "I've been taking zinc.", "Just Zinc."),
    ("MED", "Any medicine?", "No, nothing yet.", "I haven't taken anything."),
    ("ALG", "Any allergies?", "I'm allergic to cobalt.", "Yes, Cobalt."),
    ("ALG", "Any allergies?", "No allergies.", "None whatsoever."),
    ("DUR", "How long?", "For about 13 days now.", "It started 13 days ago."),
    ("SEV", "How intense?", "I'd call it severe.", "Definitely severe."),
])
def test_equivalent_observations_do_not_offer_conflict(tmp_path, monkeypatch, label, question, first, second):
    source = f"Doctor: {question}\nPatient: {first}\nDoctor: {question}\nPatient: {second}"
    choices = _constrained_menus(tmp_path, monkeypatch, source)[label]
    assert not any(choice.startswith(":C[") for choice in choices)


@pytest.mark.parametrize("label,question,first,second,want_first,want_second", [
    ("MED", "Any medicine?", "I've been taking zinc.", "No, nothing yet.", "zinc", "No, nothing yet."),
    ("ALG", "Any allergies?", "I'm allergic to cobalt.", "No allergies.", "cobalt", "No allergies."),
])
def test_conflict_choices_are_incompatible_complete_observations(tmp_path, monkeypatch, label, question, first, second, want_first, want_second):
    source = f"Doctor: {question}\nPatient: {first}\nDoctor: {question}\nPatient: {second}"
    menus = _constrained_menus(tmp_path, monkeypatch, source)
    assert set(choice for choice in menus[label] if choice.startswith(":C[")) == {
        f":C[{want_first}", f":C[{want_second}",
    }
    predictor = LocalCompactPredictor(str(tmp_path), decoding="constrained")
    def choose(messages, choices):
        slot = messages[-1]["content"].split("|")[-1]
        if slot == label:
            return f":C[{want_first}", {}
        if slot.startswith(label + ":C["):
            assert choices == [f";{want_second}]"]
            return choices[0], {}
        return ":M", {}
    monkeypatch.setattr(predictor, "_choose", choose)
    assert f"{label}:C[{want_first};{want_second}]" in predictor(source)


@pytest.mark.parametrize("label,question,first,second", [
    ("MED", "Any medicine?", "I've been taking zinc.", "Just diclofenac."),
    ("ALG", "Any allergies?", "I'm allergic to cobalt.", "Yes, chromium."),
])
def test_distinct_medication_and_allergy_values_do_not_imply_conflict(tmp_path, monkeypatch, label, question, first, second):
    source = f"Doctor: {question}\nPatient: {first}\nDoctor: {question}\nPatient: {second}"
    choices = _constrained_menus(tmp_path, monkeypatch, source)[label]
    assert not any(choice.startswith(":C[") for choice in choices)


@pytest.mark.parametrize("doctor,patient,newline,trailing", [
    ("doctor: ", "patient: ", "\n", ""),
    (" \tDoCtOr \t:  ", " \tPaTiEnT \t:  ", "\r\n", " \t"),
    ("Doctor: ", "Patient: ", "\r\n", "  "),
])
def test_role_label_formatting_preserves_attribution_and_exact_offsets(tmp_path, monkeypatch, doctor, patient, newline, trailing):
    source = f"{doctor}Any medicine?{trailing}{newline}{patient}I've been taking zinc.{trailing}"
    spans, replies = LocalCompactPredictor.source_choices(source, field=FieldName.MEDICATION)
    assert "zinc" in spans
    assert replies == ["I've been taking zinc."]
    predictor = LocalCompactPredictor(str(tmp_path), decoding="constrained")
    def choose(messages, choices):
        slot = messages[-1]["content"].split("|")[-1]
        answer = ":S[zinc]" if slot == "MED" else ":M"
        assert answer in choices
        return answer, {}
    monkeypatch.setattr(predictor, "_choose", choose)
    raw = predictor(source)
    assert raw == "CC:M|DUR:M|SEV:M|MED:S[zinc]|ALG:M"
    medication = next(item for item in parse_state_span_summary(raw, source) if item.field is FieldName.MEDICATION)
    span, = medication.spans
    assert (span.start, span.end, span.text) == (source.index("zinc"), source.index("zinc") + 4, "zinc")


@pytest.mark.parametrize("boundary", [
    "Nurse: Any allergies?", "Unlabelled annotation", "Doctor:", "Patient:",
    " doctor :  ", " patient :  ",
])
def test_unknown_or_empty_turn_breaks_question_attribution(tmp_path, monkeypatch, boundary):
    source = (f"Doctor: Any medicine?\r\n{boundary}\r\nPatient: I've been taking zinc.\r\n"
              "Doctor: Any allergies?\r\nPatient: No allergies.")
    assert LocalCompactPredictor.source_choices(source, field=FieldName.MEDICATION) == ([], [])
    menus = _constrained_menus(tmp_path, monkeypatch, source)
    assert menus["MED"] == [":M", ":U[]"]
    assert ":A[No allergies.]" in menus["ALG"]
