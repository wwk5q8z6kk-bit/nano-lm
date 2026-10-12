"""Native pointer heads on Nano's exact frozen v0.1 transformer trunk.

The pointer model keeps Nano's 3,148,608-parameter causal trunk unchanged and
replaces autoregressive text decoding with three small supervised heads:

* one five-state classifier for each of the five scribe fields; and
* start/end token classifiers for up to two evidence spans per field.

The frozen release checkpoint is accepted only through Nano's existing
hash-and-architecture-verified loader.  Loading it initializes the trunk; it
does not freeze the trunk's gradients during subsequent training.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from nano_ai.training.model import (
    NANO_MODEL_CONFIG,
    NanoGPT,
    NanoModelConfig,
    load_frozen_nano_state_dict,
)

FIELD_COUNT = 5
STATE_COUNT = 5
MAX_SPANS_PER_FIELD = 2

NANO_TRUNK_PARAMETER_COUNT = 3_148_608
POINTER_HEAD_PARAMETER_COUNT = 8_665
NANO_POINTER_PARAMETER_COUNT = 3_157_273

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


@dataclass(frozen=True, slots=True)
class PointerModelOutput:
    """Typed, shape-checked output from :class:`NanoPointerModel`."""

    state_logits: Tensor
    start_logits: Tensor
    end_logits: Tensor
    hidden_states: Tensor

    def __post_init__(self) -> None:
        for name, value in (
            ("state_logits", self.state_logits),
            ("start_logits", self.start_logits),
            ("end_logits", self.end_logits),
            ("hidden_states", self.hidden_states),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        if self.hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, width]")
        batch_size, sequence_length, model_width = self.hidden_states.shape
        if model_width != NANO_MODEL_CONFIG.model_width:
            raise ValueError("hidden_states width must equal Nano's frozen model width")
        if tuple(self.state_logits.shape) != (
            batch_size,
            FIELD_COUNT,
            STATE_COUNT,
        ):
            raise ValueError("state_logits must have shape [batch, 5, 5]")
        expected_span_shape = (
            batch_size,
            sequence_length,
            FIELD_COUNT,
            MAX_SPANS_PER_FIELD,
        )
        if tuple(self.start_logits.shape) != expected_span_shape:
            raise ValueError("start_logits must have shape [batch, sequence, 5, 2]")
        if tuple(self.end_logits.shape) != expected_span_shape:
            raise ValueError("end_logits must have shape [batch, sequence, 5, 2]")


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


class NanoPointerModel(nn.Module):
    """Nano's native trunk with bounded state and evidence-pointer heads."""

    def __init__(self, config: NanoModelConfig = NANO_MODEL_CONFIG) -> None:
        super().__init__()
        if not isinstance(config, NanoModelConfig):
            raise TypeError("config must be a NanoModelConfig")
        if _geometry(config) != _FROZEN_GEOMETRY:
            raise ValueError(
                "pointer candidates must use Nano v0.1's exact frozen geometry"
            )
        if config.parameter_count != NANO_TRUNK_PARAMETER_COUNT:
            raise ValueError("Nano trunk parameter count does not match v0.1")

        self.config = config
        self.trunk = NanoGPT(config)
        self.state_head = nn.Linear(
            config.model_width,
            FIELD_COUNT * STATE_COUNT,
        )
        self.start_head = nn.Linear(
            config.model_width,
            FIELD_COUNT * MAX_SPANS_PER_FIELD,
            bias=False,
        )
        self.end_head = nn.Linear(
            config.model_width,
            FIELD_COUNT * MAX_SPANS_PER_FIELD,
            bias=False,
        )

        if self.head_parameter_count != POINTER_HEAD_PARAMETER_COUNT:
            raise RuntimeError("pointer head parameter geometry is inconsistent")
        if self.parameter_count != NANO_POINTER_PARAMETER_COUNT:
            raise RuntimeError("pointer model parameter geometry is inconsistent")

    @property
    def head_parameter_count(self) -> int:
        """Number of parameters added on top of Nano's native trunk."""

        return sum(
            parameter.numel()
            for head in (self.state_head, self.start_head, self.end_head)
            for parameter in head.parameters()
        )

    @property
    def parameter_count(self) -> int:
        """Total trainable parameter count, including the Nano trunk."""

        return sum(parameter.numel() for parameter in self.parameters())

    def load_base_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        """Strictly load an unprefixed Nano v0.1 trunk state dictionary.

        This entry point is useful after an external caller has already
        established artifact identity.  It deliberately rejects missing,
        unexpected, or shape-incompatible tensors instead of partially loading
        a checkpoint.
        """

        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        self.trunk.load_state_dict(state_dict, strict=True)

    def load_frozen_base(self, checkpoint_path: str | Path) -> None:
        """Hash-check the registered Nano v0.1 checkpoint and load its trunk."""

        self.load_base_state_dict(load_frozen_nano_state_dict(checkpoint_path))

    @classmethod
    def from_frozen_base(cls, checkpoint_path: str | Path) -> NanoPointerModel:
        """Construct a pointer model initialized from the frozen Nano anchor."""

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

    def _terminal_hidden_states(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        if attention_mask is None:
            return hidden_states[:, -1]
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

        terminal_indices = mask.sum(dim=1, dtype=torch.long) - 1
        batch_indices = torch.arange(
            hidden_states.shape[0],
            device=hidden_states.device,
        )
        return hidden_states[batch_indices, terminal_indices]

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> PointerModelOutput:
        hidden_states = self.encode_hidden_states(token_ids)
        terminal_hidden = self._terminal_hidden_states(
            hidden_states,
            attention_mask,
        )
        batch_size, sequence_length, _ = hidden_states.shape

        state_logits = self.state_head(terminal_hidden).reshape(
            batch_size,
            FIELD_COUNT,
            STATE_COUNT,
        )
        start_logits = self.start_head(hidden_states).reshape(
            batch_size,
            sequence_length,
            FIELD_COUNT,
            MAX_SPANS_PER_FIELD,
        )
        end_logits = self.end_head(hidden_states).reshape(
            batch_size,
            sequence_length,
            FIELD_COUNT,
            MAX_SPANS_PER_FIELD,
        )
        return PointerModelOutput(
            state_logits=state_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=hidden_states,
        )


__all__ = [
    "FIELD_COUNT",
    "MAX_SPANS_PER_FIELD",
    "NANO_POINTER_PARAMETER_COUNT",
    "NANO_TRUNK_PARAMETER_COUNT",
    "POINTER_HEAD_PARAMETER_COUNT",
    "STATE_COUNT",
    "NanoPointerModel",
    "PointerModelOutput",
]
