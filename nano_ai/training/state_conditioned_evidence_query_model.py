"""H6 state-conditioned boundary queries on Nano's frozen H5 model.

H6 preserves the complete H5 model and adds one deliberately small mechanism:
Nano's own soft semantic-state posterior selects a learned residual for each
start/end boundary query.  The residuals are zero-initialized, so under the
same seed H6 at step zero is exactly the function of a freshly initialized H5;
it does not resume a trained H5 checkpoint.  The posterior is detached on this
first mechanism test: state can condition boundary selection during the
forward pass, while pointer loss cannot change the state head through the new
path.

No gold state is teacher-forced.  The new parameters remain model-owned and do
not encode verifier rules or inject values after decoding.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nano_ai.training.evidence_query_model import (
    EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
    NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
    QUERY_WIDTH,
    NanoEvidenceQueryPointerModel,
)
from nano_ai.training.pointer_model import (
    FIELD_COUNT,
    MAX_SPANS_PER_FIELD,
    STATE_COUNT,
    PointerModelOutput,
)

ARCHITECTURE_VERSION = "nano_evidence_query_state_boundary_residual_v1"
BOUNDARY_COUNT = 2
STATE_CONDITIONING_PARAMETER_COUNT = STATE_COUNT * BOUNDARY_COUNT * QUERY_WIDTH
EVIDENCE_QUERY_HEAD_PARAMETER_COUNT_H6 = (
    EVIDENCE_QUERY_HEAD_PARAMETER_COUNT + STATE_CONDITIONING_PARAMETER_COUNT
)
NANO_EVIDENCE_QUERY_PARAMETER_COUNT_H6 = (
    NANO_EVIDENCE_QUERY_PARAMETER_COUNT + STATE_CONDITIONING_PARAMETER_COUNT
)


class NanoStateConditionedEvidenceQueryPointerModel(NanoEvidenceQueryPointerModel):
    """Nano evidence-query model with identity-preserving state query offsets."""

    architecture_version = ARCHITECTURE_VERSION

    def __init__(self) -> None:
        # Construct every H5 parameter first.  The explicit zeros below consume
        # no random draws and make the initial H6 function exactly equal to H5.
        super().__init__()
        self.state_boundary_query_offsets = nn.Parameter(
            torch.zeros(STATE_COUNT, BOUNDARY_COUNT, QUERY_WIDTH)
        )

        if self.head_parameter_count != EVIDENCE_QUERY_HEAD_PARAMETER_COUNT_H6:
            raise RuntimeError("H6 evidence-query head geometry is inconsistent")
        if self.parameter_count != NANO_EVIDENCE_QUERY_PARAMETER_COUNT_H6:
            raise RuntimeError("H6 evidence-query model geometry is inconsistent")

    def _state_conditioned_query_offsets(self, state_logits: Tensor) -> Tensor:
        """Return detached-posterior query offsets with shape ``[B,5,2,64]``."""

        if state_logits.ndim != 3 or tuple(state_logits.shape[1:]) != (
            FIELD_COUNT,
            STATE_COUNT,
        ):
            raise ValueError("state_logits must have shape [batch, 5, 5]")
        if state_logits.device != self.state_boundary_query_offsets.device:
            raise ValueError("state_logits and H6 query offsets must share a device")

        state_posterior = torch.softmax(state_logits.detach(), dim=-1)
        return torch.einsum(
            "bfc,cdq->bfdq",
            state_posterior,
            self.state_boundary_query_offsets,
        )

    def _state_conditioned_boundary_logits(
        self,
        hidden_states: Tensor,
        slot_states: Tensor,
        state_logits: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        expected_slots = (
            FIELD_COUNT,
            MAX_SPANS_PER_FIELD,
            self.config.model_width,
        )
        if slot_states.ndim != 4 or tuple(slot_states.shape[1:]) != expected_slots:
            raise ValueError("slot_states must have shape [batch, 5, 2, 192]")
        if slot_states.shape[0] != state_logits.shape[0]:
            raise ValueError("slot_states and state_logits batch sizes must match")

        offsets = self._state_conditioned_query_offsets(state_logits)
        start_queries = self.start_query(slot_states) + offsets[:, :, 0, None, :]
        end_queries = self.end_query(slot_states) + offsets[:, :, 1, None, :]
        token_keys = self.boundary_key(hidden_states)

        start_logits = torch.einsum(
            "btq,bfsq->btfs",
            token_keys,
            start_queries,
        ) * (QUERY_WIDTH**-0.5)
        end_logits = torch.einsum(
            "btq,bfsq->btfs",
            token_keys,
            end_queries,
        ) * (QUERY_WIDTH**-0.5)

        invalid_positions = ~valid_mask[:, :, None, None]
        minimum = torch.finfo(start_logits.dtype).min
        return (
            start_logits.masked_fill(invalid_positions, minimum),
            end_logits.masked_fill(invalid_positions, minimum),
        )

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> PointerModelOutput:
        hidden_states = self.encode_hidden_states(token_ids)
        valid_mask = self._valid_token_mask(hidden_states, attention_mask)
        slot_states = self._read_evidence_slots(hidden_states, valid_mask)
        field_states = self._field_states(slot_states)
        state_logits = self.classify_states(field_states)
        start_logits, end_logits = self._state_conditioned_boundary_logits(
            hidden_states,
            slot_states,
            state_logits,
            valid_mask,
        )
        return PointerModelOutput(
            state_logits=state_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=hidden_states,
        )


__all__ = [
    "ARCHITECTURE_VERSION",
    "BOUNDARY_COUNT",
    "EVIDENCE_QUERY_HEAD_PARAMETER_COUNT_H6",
    "NANO_EVIDENCE_QUERY_PARAMETER_COUNT_H6",
    "STATE_CONDITIONING_PARAMETER_COUNT",
    "NanoStateConditionedEvidenceQueryPointerModel",
]
