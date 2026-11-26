"""Independent toy annotations: no frozen screen or model execution."""

from dataclasses import replace

import pytest

from nano_ai.contract import FIELD_ORDER, FieldName, FieldOutput, FieldState, NanoInput, NanoOutput
from nano_ai.solver import SolverDescriptor, SolverKind
from nano_ai.training.state_span_data import StateSpanExample


def example():
    return StateSpanExample(
        split="toy", example_id="toy-1", world_id="world-1", variant="mixed",
        target_field=FieldName.ALLERGY, target_state=FieldState.CONFLICTING,
        transcript=("Doctor: Tell this story in your own words.\n"
                    "Patient: headache\nPatient: possibly mild\n"
                    "Patient: no medications\nPatient: penicillin\nPatient: latex"),
        target="CC:S[headache]|DUR:M|SEV:U[possibly mild]|MED:A[no medications]|ALG:C[penicillin;latex]",
    )


def quotas():
    states = (FieldState.SUPPORTED, FieldState.MISSING, FieldState.UNCERTAIN,
              FieldState.ABSENT, FieldState.CONFLICTING)
    return {field: {state: int(state is selected) for state in FieldState}
            for field, selected in zip(FIELD_ORDER, states, strict=True)}


def api():
    from nano_ai.training import annotated_evaluation
    return annotated_evaluation


def test_annotation_transport_ignores_question_vocabulary_and_preserves_spans():
    row = example()
    case, = api().annotated_cases([row], expected_counts=quotas())
    assert [f.state for f in case.gold.fields] == [
        FieldState.SUPPORTED, FieldState.MISSING, FieldState.UNCERTAIN,
        FieldState.ABSENT, FieldState.CONFLICTING]
    assert case.gold.fields[0].value == "headache"
    assert [s.text for s in case.gold.fields[-1].evidence] == ["penicillin", "latex"]
    for field in case.gold.fields:
        for span in field.evidence:
            assert row.transcript[span.start:span.end] == span.text


class ToySolver:
    descriptor = SolverDescriptor("toy-oracle", SolverKind.REFERENCE)

    def __init__(self, fields):
        self.fields = fields
        self.calls = 0

    def infer(self, request):
        assert isinstance(request, NanoInput)
        assert not hasattr(request, "target")
        self.calls += 1
        return NanoOutput(request.item_id, self.descriptor.solver_id, self.fields)


def test_oracle_scores_perfectly_through_standard_evaluator():
    case, = api().annotated_cases([example()], expected_counts=quotas())
    solver = ToySolver(case.gold.fields)  # test-only oracle, never a model result
    report = api().evaluate_annotated(solver, [example()], expected_counts=quotas())
    assert report.quality["state_accuracy"] == 1.0
    assert report.quality["grounded_exact_field_accuracy"] == 1.0
    assert solver.calls == 1


def test_all_missing_solver_cannot_receive_false_perfect_score():
    solver = ToySolver(tuple(FieldOutput(field, FieldState.MISSING) for field in FIELD_ORDER))
    report = api().evaluate_annotated(solver, [example()], expected_counts=quotas())
    assert report.quality["state_accuracy"] == 0.2
    assert report.quality["grounded_exact_field_accuracy"] == 0.2


@pytest.mark.parametrize("bad_target", [
    "CC:S[invented]|DUR:M|SEV:M|MED:M|ALG:M",
    "CC:S[headache]|DUR:M|SEV:M|MED:M|ALG:C[latex;latex]",
    "CC:M|DUR:M|SEV:M|MED:M",  # incomplete field set
    "CC:S[Tell this story]|DUR:M|SEV:M|MED:M|ALG:M",  # Doctor-only evidence
    "CC:M|DUR:M|SEV:M|MED:A[headache]|ALG:M",  # not a medication denial
])
def test_bad_annotation_stops_before_inference(bad_target):
    solver = ToySolver(())
    with pytest.raises(ValueError):
        api().evaluate_annotated(solver, [replace(example(), target=bad_target)],
                                 expected_counts=quotas())
    assert solver.calls == 0


def test_quota_mismatch_stops_before_inference():
    counts = quotas()
    counts[FieldName.CHIEF_COMPLAINT][FieldState.SUPPORTED] = 0
    solver = ToySolver(())
    with pytest.raises(ValueError, match="counts"):
        api().evaluate_annotated(solver, [example()], expected_counts=counts)
    assert solver.calls == 0


@pytest.mark.parametrize("rows", [[], [example(), example()]])
def test_empty_or_duplicate_corpus_rejected(rows):
    with pytest.raises(ValueError):
        api().annotated_cases(rows, expected_counts=quotas())


def test_incomplete_and_boolean_quotas_rejected():
    with pytest.raises(ValueError):
        api().annotated_cases([example()], expected_counts={})
    counts = quotas()
    counts[FieldName.CHIEF_COMPLAINT][FieldState.SUPPORTED] = True
    with pytest.raises(ValueError):
        api().annotated_cases([example()], expected_counts=counts)


def test_invalid_later_row_prevents_any_solver_calls():
    solver = ToySolver(())
    later = replace(example(), example_id="toy-2", target="invalid")
    with pytest.raises(ValueError):
        api().evaluate_annotated(solver, [example(), later], expected_counts=quotas())
    assert solver.calls == 0


def test_uncertainty_without_evidence_is_preserved():
    row = replace(example(), target=example().target.replace("U[possibly mild]", "U[]"))
    case, = api().annotated_cases([row], expected_counts=quotas())
    assert case.gold.fields[2].state is FieldState.UNCERTAIN
    assert case.gold.fields[2].evidence == ()


def test_annotation_metadata_must_agree_with_target():
    with pytest.raises(ValueError, match="target state"):
        api().annotated_cases(
            [replace(example(), target_state=FieldState.SUPPORTED)], expected_counts=quotas())


def test_supported_value_is_normalized_but_evidence_remains_verbatim():
    row = replace(example(), transcript=example().transcript.replace("headache", "Headache"),
                  target=example().target.replace("headache", "Headache"))
    case, = api().annotated_cases([row], expected_counts=quotas())
    assert case.gold.fields[0].value == "headache"
    assert case.gold.fields[0].evidence[0].text == "Headache"
