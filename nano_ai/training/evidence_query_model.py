"""Versioned H3 evidence-query heads on Nano's frozen v0.1 trunk.

The H2 pointer heads scored each causal token independently and learned a
separate state classifier for every field.  This bounded H3 repair preserves
Nano's exact trunk while adding three targeted mechanisms:

* field- and span-conditioned reads over the complete valid transcript;
* one shared five-state classifier applied to every field; and
* bilinear start/end scores whose queries contain full-transcript context.

The evidence-query heads only produce model-owned state and span scores.  They
do not contain verifier rules or inject values after decoding.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from nano_ai.training.model import (
    NANO_MODEL_CONFIG,
    NanoGPT,
    NanoModelConfig,
    load_frozen_nano_state_dict,
)
from nano_ai.training.pointer_model import (
    FIELD_COUNT,
    MAX_SPANS_PER_FIELD,
    NANO_TRUNK_PARAMETER_COUNT,
    STATE_COUNT,
    PointerModelOutput,
)

ARCHITECTURE_VERSION = "nano_evidence_query_pointer_v1"
QUERY_WIDTH = 64

EVIDENCE_QUERY_HEAD_PARAMETER_COUNT = 137_861
NANO_EVIDENCE_QUERY_PARAMETER_COUNT = 3_286_469

_FROZEN_GEOMETRY = (
    4_098,  # vocabulary size
    192,  # model width
    6,  # layers
    6,  # query heads
    2,  # key/value heads
    32,  # head width
    512,  # feed-forward width
    512,  # context length
)


def _geometry(config: NanoModelConfig) -> tuple[int, ...]:
    return (
        config.vocabulary_size,
        config.model_width,
        config.layer_count,
        config.attention_heads,
        config.kv_heads,
        config.head_width,
        config.feed_forward_width,
        config.sequence_length,
    )


class NanoEvidenceQueryPointerModel(nn.Module):
    """Nano v0.1 with shared-state, global-evidence pointer heads."""

    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, config: NanoModelConfig = NANO_MODEL_CONFIG) -> None:
        super().__init__()
        if not isinstance(config, NanoModelConfig):
            raise TypeError("config must be a NanoModelConfig")
        if _geometry(config) != _FROZEN_GEOMETRY:
            raise ValueError(
                "evidence-query candidates must use Nano v0.1's exact frozen geometry"
            )
        if config.parameter_count != NANO_TRUNK_PARAMETER_COUNT:
            raise ValueError("Nano trunk parameter count does not match v0.1")

        self.config = config
        self.trunk = NanoGPT(config)

        width = config.model_width
        self.field_embeddings = nn.Embedding(FIELD_COUNT, width)
        self.span_embeddings = nn.Embedding(MAX_SPANS_PER_FIELD, width)

        # Each field/span seed queries all valid causal token states.  Two slot
        # reads preserve evidence multiplicity for conflict classification.
        self.evidence_query = nn.Linear(width, QUERY_WIDTH, bias=False)
        self.evidence_key = nn.Linear(width, QUERY_WIDTH, bias=False)
        self.slot_norm = nn.RMSNorm(width)

        # The readout and classifier are shared across fields.  Field identity
        # enters through ``field_embeddings``, not through separate state heads.
        self.state_readout = nn.Linear(
            MAX_SPANS_PER_FIELD * width,
            width,
            bias=False,
        )
        self.state_norm = nn.RMSNorm(width)
        self.shared_state_head = nn.Linear(width, STATE_COUNT)

        # A causal token supplies the key, but each boundary query includes its
        # field/slot's complete-transcript readout.  Later context can therefore
        # alter a score assigned to an earlier evidence token.
        self.boundary_key = nn.Linear(width, QUERY_WIDTH, bias=False)
        self.start_query = nn.Linear(width, QUERY_WIDTH, bias=False)
        self.end_query = nn.Linear(width, QUERY_WIDTH, bias=False)

        if self.head_parameter_count != EVIDENCE_QUERY_HEAD_PARAMETER_COUNT:
            raise RuntimeError("evidence-query head parameter geometry is inconsistent")
        if self.parameter_count != NANO_EVIDENCE_QUERY_PARAMETER_COUNT:
            raise RuntimeError(
                "evidence-query model parameter geometry is inconsistent"
            )

    @property
    def head_parameter_count(self) -> int:
        """Number of parameters added on top of Nano's native trunk."""

        trunk_parameter_ids = {id(parameter) for parameter in self.trunk.parameters()}
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if id(parameter) not in trunk_parameter_ids
        )

    @property
    def parameter_count(self) -> int:
        """Total trainable parameter count, including the unchanged Nano trunk."""

        return sum(parameter.numel() for parameter in self.parameters())

    def load_base_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        """Strictly load an unprefixed Nano v0.1 trunk state dictionary."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        self.trunk.load_state_dict(state_dict, strict=True)

    def load_frozen_base(self, checkpoint_path: str | Path) -> None:
        """Hash-check the registered Nano v0.1 checkpoint and load its trunk."""

        self.load_base_state_dict(load_frozen_nano_state_dict(checkpoint_path))

    @classmethod
    def from_frozen_base(
        cls,
        checkpoint_path: str | Path,
    ) -> NanoEvidenceQueryPointerModel:
        """Construct H3 initialized from the frozen Nano anchor."""

        model = cls()
        model.load_frozen_base(checkpoint_path)
        return model

    def encode_hidden_states(self, token_ids: Tensor) -> Tensor:
        """Return Nano's final normalized token states with shape ``[B,T,192]``."""

        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        sequence_length = token_ids.shape[1]
        if sequence_length < 1:
            raise ValueError("token_ids must contain at least one token")
        if sequence_length > self.config.sequence_length:
            raise ValueError(
                f"token sequence length {sequence_length} exceeds Nano's "
                f"{self.config.sequence_length}-token context"
            )

        cosine, sine = self.trunk._rope(sequence_length, token_ids.device)
        hidden = self.trunk.emb(token_ids)
        for block in self.trunk.blocks:
            hidden = block(hidden, cosine, sine)
        return self.trunk.nf(hidden)

    def _valid_token_mask(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        if attention_mask is None:
            return torch.ones(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if not isinstance(attention_mask, Tensor):
            raise TypeError("attention_mask must be a torch.Tensor")
        if tuple(attention_mask.shape) != tuple(hidden_states.shape[:2]):
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if attention_mask.device != hidden_states.device:
            raise ValueError("attention_mask and token_ids must be on the same device")
        if torch.is_floating_point(attention_mask) or attention_mask.is_complex():
            raise TypeError("attention_mask must use a boolean or integer dtype")
        if not bool(torch.all((attention_mask == 0) | (attention_mask == 1))):
            raise ValueError("attention_mask values must be zero or one")

        mask = attention_mask.to(dtype=torch.bool)
        if not bool(torch.all(mask[:, 0])):
            raise ValueError("each attention_mask row must select at least one token")
        if mask.shape[1] > 1 and bool(torch.any(mask[:, 1:] & ~mask[:, :-1])):
            raise ValueError("attention_mask must describe contiguous right padding")
        return mask

    @staticmethod
    def _terminal_hidden_states(hidden_states: Tensor, valid_mask: Tensor) -> Tensor:
        terminal_indices = valid_mask.sum(dim=1, dtype=torch.long) - 1
        batch_indices = torch.arange(
            hidden_states.shape[0],
            device=hidden_states.device,
        )
        return hidden_states[batch_indices, terminal_indices]

    def _read_evidence_slots(
        self,
        hidden_states: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        terminal_hidden = self._terminal_hidden_states(hidden_states, valid_mask)
        slot_seeds = (
            terminal_hidden[:, None, None, :]
            + self.field_embeddings.weight[None, :, None, :]
            + self.span_embeddings.weight[None, None, :, :]
        )

        evidence_scores = torch.einsum(
            "bfsq,btq->bfst",
            self.evidence_query(slot_seeds),
            self.evidence_key(hidden_states),
        ) * (QUERY_WIDTH**-0.5)
        evidence_scores = evidence_scores.masked_fill(
            ~valid_mask[:, None, None, :],
            torch.finfo(evidence_scores.dtype).min,
        )
        evidence_weights = torch.softmax(evidence_scores, dim=-1)
        evidence_context = torch.einsum(
            "bfst,bth->bfsh",
            evidence_weights,
            hidden_states,
        )
        return self.slot_norm(slot_seeds + evidence_context)

    def classify_states(self, field_states: Tensor) -> Tensor:
        """Apply one shared semantic state map to every field representation."""

        expected_shape = (FIELD_COUNT, self.config.model_width)
        if field_states.ndim != 3 or tuple(field_states.shape[1:]) != expected_shape:
            raise ValueError("field_states must have shape [batch, 5, 192]")
        return self.shared_state_head(field_states)

    def _field_states(self, slot_states: Tensor) -> Tensor:
        batch_size = slot_states.shape[0]
        flattened_slots = slot_states.reshape(batch_size, FIELD_COUNT, -1)
        return self.state_norm(self.state_readout(flattened_slots))

    def _boundary_logits(
        self,
        hidden_states: Tensor,
        slot_states: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        token_keys = self.boundary_key(hidden_states)
        start_logits = torch.einsum(
            "btq,bfsq->btfs",
            token_keys,
            self.start_query(slot_states),
        ) * (QUERY_WIDTH**-0.5)
        end_logits = torch.einsum(
            "btq,bfsq->btfs",
            token_keys,
            self.end_query(slot_states),
        ) * (QUERY_WIDTH**-0.5)

        invalid_positions = ~valid_mask[:, :, None, None]
        start_logits = start_logits.masked_fill(
            invalid_positions,
            torch.finfo(start_logits.dtype).min,
        )
        end_logits = end_logits.masked_fill(
            invalid_positions,
            torch.finfo(end_logits.dtype).min,
        )
        return start_logits, end_logits

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
        start_logits, end_logits = self._boundary_logits(
            hidden_states,
            slot_states,
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
    "EVIDENCE_QUERY_HEAD_PARAMETER_COUNT",
    "NANO_EVIDENCE_QUERY_PARAMETER_COUNT",
    "QUERY_WIDTH",
    "NanoEvidenceQueryPointerModel",
]
