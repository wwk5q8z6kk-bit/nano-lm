"""Fresh synthetic development regressions, not held-out or clinical evidence.

Raw proposals are deliberately injected to test the checked path, not model
quality. Frozen pilots, adapters and experiment inputs are never loaded here.
"""
import pytest

from nano_ai.contract import FIELD_ORDER, EvidenceSpan, FieldName, FieldOutput, FieldState, NanoOutput
from nano_ai.solver import SolverDescriptor, SolverKind
from nanoscribe.compact import compact_solver
from nanoscribe.scribe import scribe


COMPLAINT = "Doctor: What symptoms brought you here?\nPatient: I've been having wrist soreness.\n"
MED_QUESTION = "Doctor: Which medicine did you try?\n"
PREFIX = "CC:S[wrist soreness]|DUR:M|SEV:M|MED:"


def draft(source, medication, allergy="M"):
    raw = PREFIX + medication + "|ALG:" + allergy
    result = scribe(source, context_policy="context", solver=compact_solver(lambda _: raw, binder="context"))
    assert result["verification"]["passed"]
    for span in result["encounter"]["evidence"]:
        assert source[span["start"]:span["end"]] == span["text"]
    return result


@pytest.mark.parametrize("medication", ["S[naproxen]", "A[Nothing at all.]", "M", "U[]", "C[I've been taking naproxen.;Nothing at all.]"])
@pytest.mark.parametrize("reverse", [False, True])
def test_source_conflict_survives_wrong_proposal_and_preserves_complaint(medication, reverse):
    replies = ["I've been taking naproxen.", "Nothing at all."]
    if reverse:
        replies.reverse()
    source = COMPLAINT + "".join(MED_QUESTION + "Patient: " + reply + "\n" for reply in replies)
    result = draft(source, medication)
    assert result["disposition"] == "review"
    assert result["note"]["claims"][0]["value"] == "wrist soreness"
    assert result["note"]["claims"][3]["state"] == "conflicting"
    assert result["note"]["claims"][3]["value"] is None
    conflict = result["encounter"]["conflicts"][0]
    evidence = {s["evidence_id"]: s["text"] for s in result["encounter"]["evidence"]}
    assert {evidence[ref] for ref in conflict["evidence_ids"]} == {"naproxen", "Nothing at all."}
    trace = result["inference_diagnostics"]["fields"][3]
    assert trace["verified"] is False
    assert trace["decision"] == "rejected_preserving_source_conflict"
    assert result["inference_diagnostics"]["raw_summary"] == PREFIX + medication + "|ALG:M"


def test_all_three_conflicting_observations_are_retained():
    source = COMPLAINT + "".join(MED_QUESTION + "Patient: " + reply + "\n" for reply in (
        "I've been taking naproxen.", "Only famotidine so far.", "Nothing at all."))
    result = draft(source, "C[naproxen;Nothing at all.]")
    assert result["disposition"] == "review"
    assert len(result["encounter"]["conflicts"][0]["evidence_ids"]) == 3


def test_compatible_raw_pair_is_not_credited_as_a_conflict_when_an_omitted_denial_conflicts():
    source = COMPLAINT + "".join(MED_QUESTION + "Patient: " + reply + "\n" for reply in (
        "I've been taking naproxen.", "Only famotidine so far.", "Nothing at all."))
    result = draft(source, "C[naproxen;famotidine]")
    assert result["note"]["claims"][3]["state"] == "conflicting"
    trace = result["inference_diagnostics"]["fields"][3]
    assert not trace["verified"]
    assert trace["reason"] == "proposed_pair_does_not_establish_conflict"
    assert len(trace["final_spans"]) == 3


@pytest.mark.parametrize("medication", ["M", "U[]", "U[I cannot remember what I took.]"])
def test_bounded_medication_uncertainty_is_local_even_when_model_omits_it(medication):
    source = COMPLAINT + MED_QUESTION + "Patient: I cannot remember what I took.\n"
    result = draft(source, medication)
    assert result["disposition"] == "review"
    assert result["note"]["claims"][0]["value"] == "wrist soreness"
    assert result["note"]["claims"][3]["state"] == "uncertain"
    assert any(s["text"] == "I cannot remember what I took." for s in result["encounter"]["evidence"])


@pytest.mark.parametrize("medication", ["A[Some naproxen, but it barely helps.]", "M", "S[naproxen]"])
def test_bounded_medication_use_cannot_be_denial_or_suppress_complaint(medication):
    source = COMPLAINT + MED_QUESTION + "Patient: Some naproxen, but it barely helps.\n"
    result = draft(source, medication)
    assert result["disposition"] == "review"
    assert result["note"]["claims"][0]["value"] == "wrist soreness"
    assert result["note"]["claims"][3]["state"] == ("supported" if medication.startswith("S[") else "uncertain")


def test_qualified_allergy_denial_is_local_uncertainty():
    source = COMPLAINT + "Doctor: Which substances are you allergic to?\nPatient: Not that I know of.\n"
    result = draft(source, "M", "A[Not that I know of.]")
    assert result["disposition"] == "review"
    assert result["note"]["claims"][4]["state"] == "uncertain"
    assert "patient denial" not in result["note"]["text"]


def test_wrong_field_proposal_remains_uncertain_without_erasing_valid_complaint():
    source = COMPLAINT + MED_QUESTION + "Patient: Only alverine so far.\n"
    result = draft(source, "M", "S[alverine]")
    assert result["disposition"] == "review"
    assert result["note"]["claims"][0]["value"] == "wrist soreness"
    assert result["note"]["claims"][3]["state"] == "uncertain"
    assert result["note"]["claims"][4]["state"] == "uncertain"


def test_omitted_repeated_same_value_is_preserved_without_inventing_conflict():
    source = COMPLAINT + MED_QUESTION + "Patient: Only Alverine so far.\n" + MED_QUESTION + "Patient: Just alverine.\n"
    result = draft(source, "S[Alverine]")
    assert result["disposition"] == "review"
    assert result["note"]["claims"][3]["state"] == "supported"
    assert len(result["note"]["claims"][3]["evidence_ids"]) == 2


@pytest.mark.parametrize("extra", [
    MED_QUESTION + "Patient: I cannot remember what I took, but I am allergic to nickel.\n",
    MED_QUESTION + "Patient: Just avoiding naproxen.\n",
    "Doctor: Which medicines is your father taking?\nPatient: Just famotidine.\n",
    "Doctor: Which medicines were you taking last year?\nPatient: Just famotidine.\n",
])
def test_field_local_or_unrelated_context_preserves_verified_complaint(extra):
    result = draft(COMPLAINT + extra, "M")
    assert result["disposition"] == "review"
    assert result["note"]["claims"][0]["state"] == "supported"
    assert result["note"]["claims"][3]["state"] not in ("supported", "absent")


@pytest.mark.parametrize("extra", [
    "Patient: Actually, those symptoms were my neighbor's.\n",
    "Nurse: Those answers were about her sister.\n",
])
def test_unbounded_context_still_quarantines_whole_note(extra):
    result = draft(COMPLAINT + extra, "M")
    assert result["disposition"] == "abstain"
    assert all(c["state"] not in ("supported", "absent") for c in result["note"]["claims"])


@pytest.mark.parametrize("reply", ["Nothing at all.", "I cannot remember what I took."])
def test_independent_checker_catches_same_field_observation_omitted_by_solver(reply):
    source = COMPLAINT + MED_QUESTION + "Patient: I've been taking naproxen.\n" + MED_QUESTION + "Patient: " + reply

    class IncompleteSolver:
        descriptor = SolverDescriptor("development/incomplete", SolverKind.REFERENCE)

        def infer(self, request):
            values = {FieldName.CHIEF_COMPLAINT: "wrist soreness", FieldName.MEDICATION: "naproxen"}
            fields = []
            for field in FIELD_ORDER:
                if field in values:
                    text = values[field]
                    start = request.transcript.index(text)
                    fields.append(FieldOutput(field, FieldState.SUPPORTED, text, (EvidenceSpan(start, start + len(text), text),)))
                else:
                    fields.append(FieldOutput(field, FieldState.MISSING))
            return NanoOutput(request.item_id, self.descriptor.solver_id, tuple(fields))

    result = scribe(source, context_policy="context", solver=IncompleteSolver())
    assert result["verification"]["passed"]
    assert result["disposition"] == "review"
    assert result["note"]["claims"][0]["state"] == "supported"
    assert result["note"]["claims"][3]["state"] == "uncertain"
    assert any(s["text"] == reply for s in result["encounter"]["evidence"])


@pytest.mark.parametrize("field,question,replies,raw", [
    (3, MED_QUESTION, ("Only alverine so far.", "Just mebeverine."), "C[alverine;mebeverine]"),
    (4, "Doctor: Which substances are you allergic to?\n", ("Yes, nickel.", "Just chromium."), "C[nickel;chromium]"),
])
def test_distinct_medicines_or_allergens_are_not_invented_conflicts(field, question, replies, raw):
    source = COMPLAINT + "".join(question + "Patient: " + reply + "\n" for reply in replies)
    result = draft(source, raw if field == 3 else "M", raw if field == 4 else "M")
    assert result["disposition"] == "review"
    assert result["note"]["claims"][field]["state"] == "uncertain"
    assert len(result["note"]["claims"][field]["evidence_ids"]) == 2
    assert result["encounter"]["conflicts"] == []


@pytest.mark.parametrize("field,question,replies", [
    (3, MED_QUESTION, ("Only alverine so far.", "Just mebeverine.")),
    (4, "Doctor: Which substances are you allergic to?\n", ("Yes, nickel.", "Just chromium.")),
])
@pytest.mark.parametrize("quote_kind,verified", [("whole", True), ("fragment", False), ("wrong_field", False)])
def test_aggregate_uncertainty_verifies_only_bound_complete_replies(field, question, replies, quote_kind, verified):
    source = COMPLAINT + "".join(question + "Patient: " + reply + "\n" for reply in replies)
    quote = {"whole": replies[0], "fragment": replies[0][:-1],
             "wrong_field": "I've been having wrist soreness."}[quote_kind]
    raw = "U[" + quote + "]"
    result = draft(source, raw if field == 3 else "M", raw if field == 4 else "M")
    trace = result["inference_diagnostics"]["fields"][field]
    assert trace["verified"] is verified
    assert result["disposition"] == "review"
    assert result["note"]["claims"][field]["state"] == "uncertain"
    if verified:
        assert trace["reason"] == "verified_uncertainty_reply"
        assert trace["raw_spans"][0]["text"] == replies[0]
        assert len(trace["final_spans"]) == 2


@pytest.mark.parametrize("empty_role", ["Doctor:", "Patient :  "])
def test_empty_role_boundary_is_not_erased(empty_role):
    source = COMPLAINT + MED_QUESTION + empty_role + "\nPatient: Only alverine so far.\n"
    result = draft(source, "S[alverine]")
    assert result["disposition"] == "review"
    assert result["note"]["claims"][0]["state"] == "supported"
    assert result["note"]["claims"][3]["state"] == "uncertain"
    assert any(s["text"] == empty_role.strip() for s in result["encounter"]["evidence"])


def test_context_conflict_keeps_unicode_crlf_offsets_and_runs_prediction_once():
    source = (COMPLAINT + MED_QUESTION + "Patient: Only médicament so far.\n" + MED_QUESTION + "Patient: Nothing at all.\n").replace("\n", "\r\n")
    seen = []
    def predict(transcript):
        seen.append(transcript)
        return PREFIX + "A[Nothing at all.]|ALG:M"
    result = scribe(source, context_policy="context", solver=compact_solver(predict, binder="context"))
    assert seen == [source]
    assert result["disposition"] == "review"
    assert result["note"]["claims"][3]["state"] == "conflicting"
    assert result["solver"]["version"].endswith("/context-source-scope-v4")
    for span in result["encounter"]["evidence"]:
        assert source[span["start"]:span["end"]] == span["text"]
