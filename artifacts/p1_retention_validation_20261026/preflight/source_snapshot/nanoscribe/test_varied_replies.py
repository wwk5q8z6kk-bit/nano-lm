"""Exposed software development cases, with literal source-derived expectations."""
import pytest

from nano_ai.contract import FieldName, FieldState, NanoInput
from nanoscribe.context import extract_context_fields, question_field
from nanoscribe.compact import compact_solver, LocalCompactPredictor
from nanoscribe.scribe import scribe


@pytest.mark.parametrize("question,field", [
    ("What symptoms are you having?", FieldName.CHIEF_COMPLAINT),
    ("What is troubling you?", FieldName.CHIEF_COMPLAINT),
    ("How long has it hurt?", FieldName.DURATION),
    ("How painful is it now?", FieldName.SEVERITY),
    ("How strong is the ache?", FieldName.SEVERITY),
    ("What are you taking for it?", FieldName.MEDICATION),
])
def test_direct_requests_offer_the_requested_field(question, field):
    assert question_field(question) is field


@pytest.mark.parametrize("reply,value", [("For two days.", "two days"),
                                           ("Three weeks ago.", "Three weeks"),
                                           ("About four hours.", "four hours")])
def test_whole_duration_reply_preserves_exact_value(reply, value):
    source = "Doctor: How long?\nPatient: " + reply
    field = extract_context_fields(NanoInput("duration", source))[1]
    assert field.state is FieldState.SUPPORTED
    assert field.evidence[0].text == value


@pytest.mark.parametrize("denial", ["No medication.", "No meds!", "I deny taking medicine."])
def test_natural_blanket_denial_cannot_erase_medication_use(denial):
    source = ("Doctor: What symptoms?\nPatient: I have finger soreness.\n"
              "Doctor: What are you taking for it?\nPatient: I take celecoxib.\n"
              "Doctor: Any medicine?\nPatient: " + denial)
    raw = "CC:S[finger soreness]|DUR:M|SEV:M|MED:A[" + denial + "]|ALG:M"
    result = scribe(source, solver=compact_solver(lambda _: raw, binder="context"), context_policy="context")
    assert result["verification"]["passed"]
    assert result["disposition"] == "review"
    assert result["note"]["claims"][3]["state"] == "conflicting"
    assert {s["text"] for s in result["extraction"]["fields"][3]["evidence"]} == {"celecoxib", denial}
    assert not result["inference_diagnostics"]["fields"][3]["verified"]


@pytest.mark.parametrize("reply", ["No regular medication.", "No aspirin.", "No medication for my mother.",
                                   "I take celecoxib, but I stopped."])
def test_scoped_denials_and_qualified_use_remain_uncertain(reply):
    source = "Doctor: Any medicine?\nPatient: " + reply
    field = extract_context_fields(NanoInput("qualified", source))[3]
    assert field.state is FieldState.UNCERTAIN


def test_constrained_medication_conflict_is_available_with_natural_denial(tmp_path, monkeypatch):
    source = ("Doctor: Any medicine?\nPatient: I take celecoxib.\n"
              "Doctor: Any medicine?\nPatient: No medication.")
    predictor = LocalCompactPredictor(str(tmp_path), decoding="constrained")
    def choose(messages, choices):
        if messages[-1]["content"].endswith("MED"):
            assert ":C[celecoxib" in choices
            assert ":A[No medication.]" in choices
            return ":C[celecoxib", {}
        if messages[-1]["content"].endswith(":C[celecoxib"):
            assert ";No medication.]" in choices
            return ";No medication.]", {}
        return ":M", {}
    monkeypatch.setattr(predictor, "_choose", choose)
    assert "MED:C[celecoxib;No medication.]" in predictor(source)
