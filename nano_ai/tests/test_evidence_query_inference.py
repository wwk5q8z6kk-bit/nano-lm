from __future__ import annotations

import math

import pytest
import torch

from nano_ai.adapters.state_span import StateSpanProposal
from nano_ai.contract import FIELD_ORDER, EvidenceSpan, FieldState
from nano_ai.training.evidence_query_inference import (
    CALIBRATION_THRESHOLD_POLICY,
    CalibrationGold,
    EvidenceQueryInferenceResult,
    PointerInferenceInput,
    PointerPrediction,
    _masked_probabilities,
    apply_global_threshold,
    batched_evidence_query_inference,
    select_global_threshold,
)
from nano_ai.training.pointer_data import STATE_ORDER
from nano_ai.training.pointer_model import PointerModelOutput


def _proposal(
    index: int,
    state: FieldState,
    spans: tuple[EvidenceSpan, ...],
) -> StateSpanProposal:
    return StateSpanProposal(
        field=FIELD_ORDER[index],
        state_code={
            FieldState.SUPPORTED: "S",
            FieldState.ABSENT: "A",
            FieldState.MISSING: "M",
            FieldState.UNCERTAIN: "U",
            FieldState.CONFLICTING: "C",
        }[state],
        state=state,
        spans=spans,
    )


def _mixed_gold() -> tuple[StateSpanProposal, ...]:
    first = EvidenceSpan(start=9, end=13, text="pain", speaker="patient")
    second = EvidenceSpan(start=14, end=18, text="none", speaker="patient")
    third = EvidenceSpan(start=19, end=24, text="maybe", speaker="patient")
    fourth = EvidenceSpan(start=25, end=28, text="one", speaker="patient")
    fifth = EvidenceSpan(start=29, end=32, text="two", speaker="patient")
    return (
        _proposal(0, FieldState.SUPPORTED, (first,)),
        _proposal(1, FieldState.ABSENT, (second,)),
        _proposal(2, FieldState.MISSING, ()),
        _proposal(3, FieldState.UNCERTAIN, (third,)),
        _proposal(4, FieldState.CONFLICTING, (fourth, fifth)),
    )


def test_training_threshold_is_maximum_wrong_presented_confidence_and_inclusive() -> (
    None
):
    gold = _mixed_gold()
    wrong_span = EvidenceSpan(start=33, end=38, text="wrong", speaker="patient")
    raw = (_proposal(0, FieldState.SUPPORTED, (wrong_span,)), *gold[1:])
    inference = EvidenceQueryInferenceResult(
        example_ids=("calibration-1",),
        predictions=(PointerPrediction(proposals=raw),),
        field_joint_confidences=((0.6, 0.9, 0.8, 0.7, 0.5),),
    )

    selection = select_global_threshold(
        inference,
        (CalibrationGold(example_id="calibration-1", proposals=gold),),
    )

    assert selection.threshold == 0.6
    assert selection.to_dict()["threshold_policy"] == CALIBRATION_THRESHOLD_POLICY
    assert selection.uncalibrated_diagnostics["wrong_presented"] == {
        "numerator": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert selection.calibrated_diagnostics["wrong_presented"]["numerator"] == 0
    calibrated = selection.calibrated_predictions[0].proposals
    assert calibrated is not None
    assert calibrated[0].state is FieldState.UNCERTAIN
    assert calibrated[0].spans == (wrong_span,)
    assert calibrated[1] == gold[1]


def test_zero_threshold_leaves_positive_confidence_presentations_unchanged() -> None:
    gold = _mixed_gold()
    inference = EvidenceQueryInferenceResult(
        example_ids=("calibration-1",),
        predictions=(PointerPrediction(proposals=gold),),
        field_joint_confidences=((0.8, 0.7, 0.6, 0.5, 0.4),),
    )

    selection = select_global_threshold(
        inference,
        (CalibrationGold(example_id="calibration-1", proposals=gold),),
    )

    assert selection.threshold == 0.0
    assert selection.calibrated_predictions == inference.predictions
    assert apply_global_threshold(inference, 0.0) == inference.predictions


def test_pointer_confidence_mask_has_exactly_zero_probability_outside_mask() -> None:
    floor = torch.finfo(torch.float32).min

    probabilities = _masked_probabilities(
        torch.tensor((floor, floor), dtype=torch.float32),
        (True, False),
    )

    assert probabilities.tolist() == [1.0, 0.0]


def test_batched_inference_confidence_uses_selected_state_and_patient_boundaries() -> (
    None
):
    item = PointerInferenceInput(
        example_id="dev-1",
        transcript="Patient: pain",
        token_ids=(17,),
        attention_mask=(True,),
        pointer_mask=(True,),
        token_offsets=((9, 13),),
    )
    state_logits = torch.full((1, len(FIELD_ORDER), len(STATE_ORDER)), -4.0)
    state_logits[0, 0, STATE_ORDER.index(FieldState.SUPPORTED)] = 4.0
    state_logits[0, 1:, STATE_ORDER.index(FieldState.MISSING)] = 4.0
    pointer_logits = torch.zeros((1, 1, len(FIELD_ORDER), 2))

    class FixedModel:
        def eval(self):
            return self

        def __call__(self, _token_ids, *, attention_mask):
            assert attention_mask.tolist() == [[True]]
            return PointerModelOutput(
                hidden_states=torch.zeros((1, 1, 192)),
                state_logits=state_logits,
                start_logits=pointer_logits,
                end_logits=pointer_logits,
            )

    result = batched_evidence_query_inference(
        FixedModel(),  # type: ignore[arg-type]
        (item,),
        device="cpu",
        batch_size=1,
    )

    proposal = result.predictions[0].proposals
    assert proposal is not None
    assert proposal[0].state is FieldState.SUPPORTED
    assert proposal[0].spans[0].text == "pain"
    expected_state_probability = math.exp(4.0) / (math.exp(4.0) + 4 * math.exp(-4.0))
    assert result.field_joint_confidences[0][0] == pytest.approx(
        expected_state_probability
    )
