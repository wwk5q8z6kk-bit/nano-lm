"""New synthetic parser-recovery development cases; no model-quality evidence."""

import pytest

from nano_ai.adapters.state_span import StateSpanFormatError, parse_state_span_summary
from nano_ai.contract import FieldState, NanoInput
from nanoscribe.context_solver import ContextStateSpanSolver
from nanoscribe.scribe import scribe


SOURCE = (
    "Doctor: What brings you to the clinic today?\n"
    "Patient: I've been having palm soreness.\n"
    "Doctor: How long has this been going on?\n"
    "Patient: For about 6 days now.\n"
    "Doctor: How bad would you say it is?\n"
    "Patient: I'd call it moderate.\n"
    "Doctor: Did you try any medicine?\n"
    "Patient: Only naproxen so far.\n"
    "Doctor: Any allergies I should know about?\n"
    "Patient: I'm allergic to nickel.\n"
)
SLOTS = ["CC:S[palm soreness]", "DUR:S[6 days]", "SEV:S[moderate]", "MED:S[naproxen]", "ALG:S[nickel]"]


def run(raw, source=SOURCE):
    calls = []

    def predict(transcript):
        calls.append(transcript)
        return raw

    solver = ContextStateSpanSolver(predict)
    result = scribe(source, solver=solver, context_policy="context")
    assert calls == [source]
    return result


@pytest.mark.parametrize("slot", [
    "DUR:S[six days]",  # Normalized prose is not an exact source quotation.
    "DUR:S[6 days",     # An unterminated field cannot consume its neighbors.
    "DUR:X[6 days]",    # Unknown state code.
    "DUR:M[6 days]",    # Evidence on a missing state is invalid.
    "DUR:S[]",         # Empty positive evidence.
])
def test_one_invalid_slot_preserves_other_valid_fields(slot):
    slots = [*SLOTS]
    slots[1] = slot
    raw = "|".join(slots)
    with pytest.raises(StateSpanFormatError):
        parse_state_span_summary(raw, SOURCE)
    result = run(raw)
    assert result["status"] == "draft"
    assert result["disposition"] == "review"
    assert result["verification"]["passed"]
    assert [claim["state"] for claim in result["note"]["claims"]] == [
        "supported", "uncertain", "supported", "supported", "supported",
    ]
    trace = result["inference_diagnostics"]
    assert trace["raw_summary"] == raw
    assert trace["strict_parse_error"]
    assert trace["recovered_field_count"] == 4
    assert trace["protocol_version"] == "nanoscribe.context-state-diagnostics.v3"
    assert trace["fields"][1]["raw_slot"] == slot
    assert trace["fields"][1]["parse_error"]
    assert trace["fields"][1]["verified"] is False
    assert trace["fields"][1]["reason"] == "native_state_parse_failed"
    assert all(field["parse_error"] is None for i, field in enumerate(trace["fields"]) if i != 1)


def test_clinician_only_evidence_is_rejected_without_losing_patient_fields():
    slots = [*SLOTS]
    slots[3] = "MED:S[medicine]"
    result = run("|".join(slots))
    trace = result["inference_diagnostics"]["fields"][3]
    assert "not exact text inside a Patient turn" in trace["parse_error"]
    assert not trace["verified"]
    assert result["note"]["claims"][3]["state"] == "uncertain"
    assert result["note"]["claims"][0]["value"] == "palm soreness"


def test_ambiguous_patient_evidence_is_not_resolved_by_first_occurrence():
    source = SOURCE + "Doctor: Did you try any medicine?\nPatient: Just naproxen.\n"
    result = run("|".join(SLOTS), source)
    trace = result["inference_diagnostics"]["fields"][3]
    assert "ambiguous across Patient turns" in trace["parse_error"]
    assert trace["raw_spans"] == []
    assert result["note"]["claims"][3]["state"] == "uncertain"
    assert result["note"]["claims"][4]["value"] == "nickel"
    assert len(trace["final_spans"]) == 2


@pytest.mark.parametrize("indices", [(0, 3), (0, 3, 4)])
def test_duplicate_exact_evidence_quarantines_every_involved_slot(indices):
    slots = [*SLOTS]
    for index in indices:
        slots[index] = slots[index].split(":")[0] + ":S[palm soreness]"
    raw = "|".join(slots)
    with pytest.raises(StateSpanFormatError, match="duplicated across field proposals"):
        parse_state_span_summary(raw, SOURCE)
    result = run(raw)
    traces = result["inference_diagnostics"]["fields"]
    for index, trace in enumerate(traces):
        if index in indices:
            assert "duplicated across field proposals" in trace["parse_error"]
            assert trace["verified"] is False
            assert trace["final_state"] == "uncertain"
            assert trace["raw_spans"][0]["text"] == "palm soreness"
        else:
            assert trace["parse_error"] is None
            assert trace["verified"] is True
            assert trace["final_state"] == "supported"


@pytest.mark.parametrize("raw", [
    None,
    "|".join(SLOTS[:-1]),
    "|".join([*SLOTS, "ALG:M"]),
    "|".join([SLOTS[1], SLOTS[0], *SLOTS[2:]]),
    "|".join(["cc:S[palm soreness]", *SLOTS[1:]]),
    "|".join([" CC:S[palm soreness]", *SLOTS[1:]]),
    "|".join([*SLOTS[:-1], "MED:S[nickel]"]),
])
def test_unrecoverable_global_shape_still_fails(raw):
    result = run(raw)
    assert result["status"] == "failed"
    assert result["disposition"] == "abstain"
    assert result["note"] is None
    assert result["extraction"] is None


def test_malformed_conflict_proposal_preserves_checked_conflict_but_is_not_verified():
    source = SOURCE + "Doctor: Did you try any medicine?\nPatient: Nothing at all.\n"
    slots = [*SLOTS]
    slots[3] = "MED:C[naproxen;]"
    result = run("|".join(slots), source)
    trace = result["inference_diagnostics"]["fields"][3]
    assert trace["parse_error"]
    assert trace["verified"] is False
    assert trace["decision"] == "rejected_preserving_source_conflict"
    assert trace["reason"] == "native_state_parse_failed"
    assert trace["final_state"] == "conflicting"
    assert {span["text"] for span in trace["final_spans"]} == {"naproxen", "Nothing at all."}
    assert result["note"]["claims"][0]["state"] == "supported"


def test_recovery_keeps_unicode_crlf_exact_offsets_and_single_inference():
    source = SOURCE.replace("palm soreness", "café sensitivity").replace("\n", "\r\n")
    slots = [*SLOTS]
    slots[0] = "CC:S[café sensitivity]"
    slots[1] = "DUR:S[six days]"
    result = run("|".join(slots), source)
    assert result["disposition"] == "review"
    for span in result["encounter"]["evidence"]:
        assert source[span["start"]:span["end"]] == span["text"]
    assert result["note"]["claims"][0]["value"] == "café sensitivity"


def test_fully_valid_output_has_no_parse_recovery_and_identical_exact_evidence():
    raw = "|".join(SLOTS)
    proposals = parse_state_span_summary(raw, SOURCE)
    output, diagnostics = ContextStateSpanSolver(lambda _: raw).infer_with_diagnostics(NanoInput("new-case", SOURCE))
    assert diagnostics["strict_parse_error"] is None
    assert diagnostics["recovered_field_count"] == 0
    assert [field.evidence for field in output.fields] == [proposal.spans for proposal in proposals]
    assert all(field.state is FieldState.SUPPORTED for field in output.fields)
    assert all(trace["verified"] and trace["parse_error"] is None for trace in diagnostics["fields"])
