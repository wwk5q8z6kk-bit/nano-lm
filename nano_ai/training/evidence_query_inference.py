"""Shared H3 inference and training-only confidence calibration.

The evidence-query trainer and evaluator use this module as their single
decoding authority.  It preserves H2's constrained state/span grammar while
recording a field-level confidence for the exact boundaries selected by the
decoder.  Calibration may only turn a presented supported/absent proposal into
an uncertain abstention; it never changes evidence or supplies a value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from nano_ai.adapters.state_span import StateSpanProposal
from nano_ai.contract import FIELD_ORDER, EvidenceSpan, FieldState
from nano_ai.training.evaluate_pointer import (
    PointerDecodeError,
    PointerEvaluationError,
    PointerInferenceInput,
    PointerPrediction,
    batched_pointer_inference,
    build_pointer_inference_inputs,
    decode_pointer_logits,
    raw_pointer_diagnostics,
)
from nano_ai.training.evidence_query_model import NanoEvidenceQueryPointerModel
from nano_ai.training.pointer_data import STATE_ORDER, token_span_to_evidence

CALIBRATION_THRESHOLD_POLICY = "minimal_zero_wrong_presented_inclusive_v1"
_PRESENTED_STATES = frozenset({FieldState.SUPPORTED, FieldState.ABSENT})
_CALIBRATION_SLICES = (
    "absence",
    "missing_target",
    "uncertain_target",
    "conflicting_target",
)


@dataclass(frozen=True, slots=True)
class EvidenceQueryInferenceResult:
    """Ordered H3 predictions and confidence for each of their five fields."""

    example_ids: tuple[str, ...]
    predictions: tuple[PointerPrediction, ...]
    field_joint_confidences: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_ids", tuple(self.example_ids))
        object.__setattr__(self, "predictions", tuple(self.predictions))
        object.__setattr__(
            self,
            "field_joint_confidences",
            tuple(tuple(row) for row in self.field_joint_confidences),
        )
        if not self.example_ids or len(set(self.example_ids)) != len(self.example_ids):
            raise PointerEvaluationError(
                "evidence-query inference IDs must be non-empty and unique"
            )
        if not (
            len(self.example_ids)
            == len(self.predictions)
            == len(self.field_joint_confidences)
        ):
            raise PointerEvaluationError("evidence-query inference rows do not align")
        for example_id, prediction, confidences in zip(
            self.example_ids,
            self.predictions,
            self.field_joint_confidences,
            strict=True,
        ):
            if not isinstance(example_id, str) or not example_id:
                raise PointerEvaluationError("evidence-query inference ID is invalid")
            if not isinstance(prediction, PointerPrediction):
                raise PointerEvaluationError(
                    "evidence-query inference prediction is invalid"
                )
            if len(confidences) != len(FIELD_ORDER) or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= float(value) <= 1.0
                for value in confidences
            ):
                raise PointerEvaluationError(
                    "evidence-query field confidences are invalid"
                )


@dataclass(frozen=True, slots=True)
class CalibrationGold:
    """Training-only gold proposals for one calibration example."""

    example_id: str
    proposals: tuple[StateSpanProposal, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposals", tuple(self.proposals))
        if not isinstance(self.example_id, str) or not self.example_id:
            raise PointerEvaluationError("calibration example ID is invalid")
        if len(self.proposals) != len(FIELD_ORDER) or any(
            not isinstance(proposal, StateSpanProposal)
            or proposal.field is not expected
            for expected, proposal in zip(
                FIELD_ORDER,
                self.proposals,
                strict=True,
            )
        ):
            raise PointerEvaluationError(
                "calibration proposals are not canonically ordered"
            )


@dataclass(frozen=True, slots=True)
class GlobalThresholdSelection:
    """One training-only threshold and both sides of its measured effect."""

    threshold: float
    uncalibrated_diagnostics: Mapping[str, Any]
    calibrated_diagnostics: Mapping[str, Any]
    calibrated_predictions: tuple[PointerPrediction, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
            or not 0.0 <= float(self.threshold) <= 1.0
        ):
            raise PointerEvaluationError("global threshold is invalid")
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(
            self,
            "calibrated_predictions",
            tuple(self.calibrated_predictions),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the exact trainer/evaluator calibration contract."""

        return {
            "uncalibrated": dict(self.uncalibrated_diagnostics),
            "global_threshold": self.threshold,
            "calibrated": dict(self.calibrated_diagnostics),
            "threshold_policy": CALIBRATION_THRESHOLD_POLICY,
        }


def _selected_boundaries(
    item: PointerInferenceInput,
    evidence: EvidenceSpan,
    start_scores: Tensor,
    end_scores: Tensor,
) -> tuple[int, int]:
    """Recover the exact token boundaries chosen for one decoded evidence span."""

    best: tuple[float, int, int] | None = None
    allowed = tuple(index for index, value in enumerate(item.pointer_mask) if value)
    for start_offset, start_index in enumerate(allowed):
        for end_index in allowed[start_offset:]:
            if not all(item.pointer_mask[start_index : end_index + 1]):
                continue
            try:
                candidate = token_span_to_evidence(
                    item.transcript,
                    item.token_offsets,
                    start_index,
                    end_index,
                )
            except (TypeError, ValueError):
                continue
            if candidate != evidence:
                continue
            score = float(start_scores[start_index]) + float(end_scores[end_index])
            row = (score, -start_index, -end_index)
            if best is None or row > (best[0], -best[1], -best[2]):
                best = (score, start_index, end_index)
    if best is None:
        raise PointerDecodeError(
            "decoded evidence could not be mapped back to selected boundaries"
        )
    return best[1], best[2]


def _masked_probabilities(scores: Tensor, mask: Sequence[bool]) -> Tensor:
    if scores.ndim != 1 or len(scores) != len(mask):
        raise PointerDecodeError("pointer confidence scores have invalid shape")
    allowed = torch.tensor(tuple(mask), dtype=torch.bool, device=scores.device)
    if not bool(allowed.any()) or not bool(torch.isfinite(scores).all()):
        raise PointerDecodeError("pointer confidence scores are invalid")
    return torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=0)


def _field_confidences(
    item: PointerInferenceInput,
    proposals: Sequence[StateSpanProposal],
    state_logits: Tensor,
    start_logits: Tensor,
    end_logits: Tensor,
) -> tuple[float, ...]:
    state_probabilities = torch.softmax(state_logits, dim=-1)
    values: list[float] = []
    for field_index, proposal in enumerate(proposals):
        state_index = STATE_ORDER.index(proposal.state)
        components = [float(state_probabilities[field_index, state_index])]
        for slot, evidence in enumerate(proposal.spans):
            start_scores = start_logits[:, field_index, slot]
            end_scores = end_logits[:, field_index, slot]
            selected_start, selected_end = _selected_boundaries(
                item,
                evidence,
                start_scores,
                end_scores,
            )
            start_probabilities = _masked_probabilities(
                start_scores,
                item.pointer_mask,
            )
            end_probabilities = _masked_probabilities(
                end_scores,
                item.pointer_mask,
            )
            components.extend(
                (
                    float(start_probabilities[selected_start]),
                    float(end_probabilities[selected_end]),
                )
            )
        values.append(min(components))
    return tuple(values)


def batched_evidence_query_inference(
    model: NanoEvidenceQueryPointerModel,
    inputs: Sequence[PointerInferenceInput],
    *,
    device: str,
    batch_size: int,
) -> EvidenceQueryInferenceResult:
    """Run H3 once, decoding proposals and their selected-boundary confidence."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise PointerEvaluationError("batch_size must be a positive integer")
    frozen = tuple(inputs)
    if not frozen:
        raise PointerEvaluationError(
            "evidence-query inference requires at least one input"
        )

    predictions: list[PointerPrediction] = []
    confidences: list[tuple[float, ...]] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(frozen), batch_size):
            batch = frozen[offset : offset + batch_size]
            maximum = max(len(item.token_ids) for item in batch)
            token_ids = torch.tensor(
                [
                    [*item.token_ids, *([0] * (maximum - len(item.token_ids)))]
                    for item in batch
                ],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.tensor(
                [
                    [
                        *item.attention_mask,
                        *([False] * (maximum - len(item.token_ids))),
                    ]
                    for item in batch
                ],
                dtype=torch.bool,
                device=device,
            )
            outputs = model(token_ids, attention_mask=attention_mask)
            expected_state = (len(batch), len(FIELD_ORDER), len(STATE_ORDER))
            expected_pointer = (len(batch), maximum, len(FIELD_ORDER), 2)
            if (
                tuple(outputs.state_logits.shape) != expected_state
                or tuple(outputs.start_logits.shape) != expected_pointer
                or tuple(outputs.end_logits.shape) != expected_pointer
            ):
                raise PointerEvaluationError(
                    "evidence-query model returned invalid logit shapes"
                )
            state_rows = outputs.state_logits.detach().cpu()
            start_rows = outputs.start_logits.detach().cpu()
            end_rows = outputs.end_logits.detach().cpu()
            for row_index, item in enumerate(batch):
                length = len(item.token_ids)
                state_row = state_rows[row_index]
                start_row = start_rows[row_index, :length]
                end_row = end_rows[row_index, :length]
                try:
                    proposals = decode_pointer_logits(
                        item,
                        state_row,
                        start_row,
                        end_row,
                    )
                    row_confidences = _field_confidences(
                        item,
                        proposals,
                        state_row,
                        start_row,
                        end_row,
                    )
                except PointerDecodeError as exc:
                    predictions.append(PointerPrediction(error=str(exc)))
                    confidences.append((0.0,) * len(FIELD_ORDER))
                else:
                    predictions.append(PointerPrediction(proposals=proposals))
                    confidences.append(row_confidences)
    return EvidenceQueryInferenceResult(
        example_ids=tuple(item.example_id for item in frozen),
        predictions=tuple(predictions),
        field_joint_confidences=tuple(confidences),
    )


def apply_global_threshold(
    inference: EvidenceQueryInferenceResult,
    threshold: float,
) -> tuple[PointerPrediction, ...]:
    """Abstain on presented fields at or below one frozen global threshold."""

    if not isinstance(inference, EvidenceQueryInferenceResult):
        raise TypeError("inference must be an EvidenceQueryInferenceResult")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise PointerEvaluationError("global threshold is invalid")
    threshold = float(threshold)
    calibrated: list[PointerPrediction] = []
    for prediction, row_confidences in zip(
        inference.predictions,
        inference.field_joint_confidences,
        strict=True,
    ):
        if prediction.error is not None:
            calibrated.append(prediction)
            continue
        assert prediction.proposals is not None
        proposals = tuple(
            replace(
                proposal,
                state_code="U",
                state=FieldState.UNCERTAIN,
            )
            if proposal.state in _PRESENTED_STATES and confidence <= threshold
            else proposal
            for proposal, confidence in zip(
                prediction.proposals,
                row_confidences,
                strict=True,
            )
        )
        calibrated.append(PointerPrediction(proposals=proposals))
    return tuple(calibrated)


def _span_key(span: EvidenceSpan) -> tuple[int, int, str, str]:
    return (span.start, span.end, span.text, span.speaker)


def _proposal_exact(
    predicted: StateSpanProposal,
    gold: StateSpanProposal,
) -> bool:
    return predicted.state is gold.state and {
        _span_key(span) for span in predicted.spans
    } == {_span_key(span) for span in gold.spans}


def _rate_bucket(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else 0.0,
    }


def _calibration_diagnostics(
    predictions: Sequence[PointerPrediction],
    gold: Sequence[CalibrationGold],
) -> dict[str, Any]:
    if len(predictions) != len(gold):
        raise PointerEvaluationError("calibration prediction rows do not align")
    slice_counts = {
        "overall": [0, 0],
        **{name: [0, 0] for name in _CALIBRATION_SLICES},
    }
    wrong_presented = 0
    presented = 0
    state_slice = {
        FieldState.ABSENT: "absence",
        FieldState.MISSING: "missing_target",
        FieldState.UNCERTAIN: "uncertain_target",
        FieldState.CONFLICTING: "conflicting_target",
    }
    for prediction, gold_row in zip(predictions, gold, strict=True):
        if prediction.error is None:
            assert prediction.proposals is not None
            proposed = prediction.proposals
        else:
            proposed = ()
        for index, gold_proposal in enumerate(gold_row.proposals):
            exact = bool(proposed) and _proposal_exact(proposed[index], gold_proposal)
            slice_counts["overall"][1] += 1
            slice_counts["overall"][0] += int(exact)
            slice_name = state_slice.get(gold_proposal.state)
            if slice_name is not None:
                slice_counts[slice_name][1] += 1
                slice_counts[slice_name][0] += int(exact)
            if proposed:
                is_presented = proposed[index].state in _PRESENTED_STATES
                presented += int(is_presented)
                wrong_presented += int(is_presented and not exact)

    slices = {
        name: _rate_bucket(numerator, denominator)
        for name, (numerator, denominator) in slice_counts.items()
    }
    if any(slices[name]["denominator"] == 0 for name in _CALIBRATION_SLICES):
        raise PointerEvaluationError(
            "calibration selection slices must all be non-empty"
        )
    macro = sum(float(slices[name]["rate"]) for name in _CALIBRATION_SLICES) / len(
        _CALIBRATION_SLICES
    )
    return {
        "slices": slices,
        "selection": {
            "macro_joint": macro,
            "overall_joint": slices["overall"]["rate"],
        },
        "wrong_presented": _rate_bucket(wrong_presented, presented),
    }


def select_global_threshold(
    inference: EvidenceQueryInferenceResult,
    gold: Sequence[CalibrationGold],
) -> GlobalThresholdSelection:
    """Select the smallest inclusive threshold eliminating calibration errors."""

    if not isinstance(inference, EvidenceQueryInferenceResult):
        raise TypeError("inference must be an EvidenceQueryInferenceResult")
    frozen_gold = tuple(gold)
    if not frozen_gold or any(
        not isinstance(item, CalibrationGold) for item in frozen_gold
    ):
        raise PointerEvaluationError("calibration gold rows are invalid")
    if inference.example_ids != tuple(item.example_id for item in frozen_gold):
        raise PointerEvaluationError("calibration inference/gold IDs do not align")

    wrong_confidences: list[float] = []
    for prediction, confidences, gold_row in zip(
        inference.predictions,
        inference.field_joint_confidences,
        frozen_gold,
        strict=True,
    ):
        if prediction.error is not None:
            continue
        assert prediction.proposals is not None
        for proposal, confidence, gold_proposal in zip(
            prediction.proposals,
            confidences,
            gold_row.proposals,
            strict=True,
        ):
            if proposal.state in _PRESENTED_STATES and not _proposal_exact(
                proposal,
                gold_proposal,
            ):
                wrong_confidences.append(float(confidence))
    threshold = max(wrong_confidences, default=0.0)
    calibrated_predictions = apply_global_threshold(inference, threshold)
    uncalibrated = _calibration_diagnostics(inference.predictions, frozen_gold)
    calibrated = _calibration_diagnostics(calibrated_predictions, frozen_gold)
    if calibrated["wrong_presented"]["numerator"] != 0:
        raise RuntimeError("calibration threshold did not eliminate presented errors")
    return GlobalThresholdSelection(
        threshold=threshold,
        uncalibrated_diagnostics=uncalibrated,
        calibrated_diagnostics=calibrated,
        calibrated_predictions=calibrated_predictions,
    )


__all__ = [
    "CALIBRATION_THRESHOLD_POLICY",
    "CalibrationGold",
    "EvidenceQueryInferenceResult",
    "GlobalThresholdSelection",
    "PointerDecodeError",
    "PointerInferenceInput",
    "PointerPrediction",
    "apply_global_threshold",
    "batched_evidence_query_inference",
    "batched_pointer_inference",
    "build_pointer_inference_inputs",
    "decode_pointer_logits",
    "raw_pointer_diagnostics",
    "select_global_threshold",
]
