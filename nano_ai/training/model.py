"""Training-capable implementation of Nano's frozen native architecture.

The parameter names and tensor geometry deliberately match the v0.1 scribe
checkpoint.  The only runtime change from the historical training script is
that attention uses the input's actual sequence length (up to the frozen
512-token context) instead of assuming every batch is padded to 512 tokens.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as torch_functional

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01


class FrozenNanoCheckpointError(RuntimeError):
    """The supplied file is not the registered frozen Nano checkpoint."""


@dataclass(frozen=True, slots=True)
class NanoModelConfig:
    """Native Nano transformer geometry encoded by the frozen checkpoint."""

    vocabulary_size: int
    model_width: int
    layer_count: int
    attention_heads: int
    kv_heads: int
    head_width: int
    feed_forward_width: int
    sequence_length: int

    def __post_init__(self) -> None:
        for name, value in (
            ("vocabulary_size", self.vocabulary_size),
            ("model_width", self.model_width),
            ("layer_count", self.layer_count),
            ("attention_heads", self.attention_heads),
            ("kv_heads", self.kv_heads),
            ("head_width", self.head_width),
            ("feed_forward_width", self.feed_forward_width),
            ("sequence_length", self.sequence_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.attention_heads % self.kv_heads:
            raise ValueError("attention_heads must be divisible by kv_heads")
        if self.attention_heads * self.head_width != self.model_width:
            raise ValueError("attention geometry must equal model_width")
        if self.head_width % 2:
            raise ValueError("head_width must be even for rotary embeddings")

    @property
    def parameter_count(self) -> int:
        """Count parameters without adding an untied output head."""

        width = self.model_width
        attention = (
            width * self.attention_heads * self.head_width
            + 2 * width * self.kv_heads * self.head_width
            + width * self.attention_heads * self.head_width
        )
        feed_forward = 3 * width * self.feed_forward_width
        block_norms = 2 * width
        return (
            self.vocabulary_size * width
            + self.layer_count * (attention + feed_forward + block_norms)
            + width
        )


NANO_MODEL_CONFIG = NanoModelConfig(
    vocabulary_size=FROZEN_NANO_V01.vocabulary_size,
    model_width=FROZEN_NANO_V01.model_width,
    layer_count=FROZEN_NANO_V01.layer_count,
    attention_heads=FROZEN_NANO_V01.attention_heads,
    kv_heads=FROZEN_NANO_V01.kv_heads,
    head_width=FROZEN_NANO_V01.head_width,
    feed_forward_width=FROZEN_NANO_V01.feed_forward_width,
    sequence_length=FROZEN_NANO_V01.sequence_length,
)


class _NanoBlock(nn.Module):
    """One grouped-query attention and SwiGLU block."""

    def __init__(self, config: NanoModelConfig) -> None:
        super().__init__()
        width = config.model_width
        heads = config.attention_heads
        kv_heads = config.kv_heads
        head_width = config.head_width
        feed_forward = config.feed_forward_width

        self.attention_heads = heads
        self.kv_heads = kv_heads
        self.head_width = head_width

        # Names are part of the frozen checkpoint's state_dict geometry.
        self.n1 = nn.RMSNorm(width)
        self.n2 = nn.RMSNorm(width)
        self.q = nn.Linear(width, heads * head_width, bias=False)
        self.k = nn.Linear(width, kv_heads * head_width, bias=False)
        self.v = nn.Linear(width, kv_heads * head_width, bias=False)
        self.o = nn.Linear(heads * head_width, width, bias=False)
        self.g = nn.Linear(width, feed_forward, bias=False)
        self.u = nn.Linear(width, feed_forward, bias=False)
        self.dn = nn.Linear(feed_forward, width, bias=False)

    def forward(self, values: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
        batch_size, sequence_length, _ = values.shape
        hidden = self.n1(values)
        query = (
            self.q(hidden)
            .view(
                batch_size,
                sequence_length,
                self.attention_heads,
                self.head_width,
            )
            .transpose(1, 2)
        )
        key = (
            self.k(hidden)
            .view(batch_size, sequence_length, self.kv_heads, self.head_width)
            .transpose(1, 2)
        )
        value = (
            self.v(hidden)
            .view(batch_size, sequence_length, self.kv_heads, self.head_width)
            .transpose(1, 2)
        )

        query = _apply_rope(query, cosine, sine)
        key = _apply_rope(key, cosine, sine)
        key = key.repeat_interleave(self.attention_heads // self.kv_heads, dim=1)
        value = value.repeat_interleave(self.attention_heads // self.kv_heads, dim=1)
        attention = torch_functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )
        values = values + self.o(
            attention.transpose(1, 2).reshape(
                batch_size,
                sequence_length,
                self.attention_heads * self.head_width,
            )
        )
        hidden = self.n2(values)
        return values + self.dn(torch_functional.silu(self.g(hidden)) * self.u(hidden))


def _apply_rope(values: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    first = values[..., 0::2]
    second = values[..., 1::2]
    return torch.stack(
        [first * cosine - second * sine, first * sine + second * cosine],
        dim=-1,
    ).flatten(-2)


class NanoGPT(nn.Module):
    """Nano's 3,148,608-parameter tied-head causal language model."""

    def __init__(self, config: NanoModelConfig = NANO_MODEL_CONFIG) -> None:
        super().__init__()
        if not isinstance(config, NanoModelConfig):
            raise TypeError("config must be a NanoModelConfig")
        self.config = config

        # These exact names match scribe/scribe_sft.py and the frozen checkpoint.
        self.emb = nn.Embedding(config.vocabulary_size, config.model_width)
        self.blocks = nn.ModuleList(
            _NanoBlock(config) for _ in range(config.layer_count)
        )
        self.nf = nn.RMSNorm(config.model_width)

    def _rope(
        self, sequence_length: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
        inverse = 1.0 / (
            10000
            ** (
                torch.arange(0, self.config.head_width, 2, device=device).float()
                / self.config.head_width
            )
        )
        frequency = torch.outer(positions, inverse)
        return frequency.cos()[None, None], frequency.sin()[None, None]

    def forward(self, token_ids: Tensor) -> Tensor:
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

        cosine, sine = self._rope(sequence_length, token_ids.device)
        hidden = self.emb(token_ids)
        for block in self.blocks:
            hidden = block(hidden, cosine, sine)
        # The output projection is tied to ``emb.weight`` and adds no parameters.
        return torch_functional.linear(self.nf(hidden), self.emb.weight)


def load_frozen_nano_state_dict(
    checkpoint_path: str | Path,
) -> Mapping[str, Tensor]:
    """Hash-check and strictly validate the frozen v0.1 state dictionary."""

    return load_verified_nano_state_dict(
        checkpoint_path,
        expected_sha256=FROZEN_NANO_V01.checkpoint_sha256,
    )


def load_verified_nano_state_dict(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
) -> Mapping[str, Tensor]:
    """Load one hash-identified state dictionary with Nano's exact geometry.

    The file is read once into an immutable byte snapshot.  Hash verification,
    deserialization, and geometry validation all operate on that same snapshot.
    """

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")

    path = Path(checkpoint_path)
    try:
        with path.open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise FrozenNanoCheckpointError(
                    "frozen Nano checkpoint is not a regular file"
                )
            checkpoint_bytes = handle.read()
    except FrozenNanoCheckpointError:
        raise
    except OSError as exc:
        raise FrozenNanoCheckpointError(
            "frozen Nano checkpoint is unavailable"
        ) from exc

    observed = hashlib.sha256(checkpoint_bytes).hexdigest()
    if not hmac.compare_digest(observed, expected_sha256):
        raise FrozenNanoCheckpointError(
            "Nano checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed}"
        )

    try:
        state_dict = torch.load(
            io.BytesIO(checkpoint_bytes),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state_dict, Mapping):
            raise TypeError("checkpoint payload is not a state dictionary")
        validation_model = NanoGPT()
        validation_model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise FrozenNanoCheckpointError(
            "frozen Nano checkpoint does not match the registered architecture"
        ) from exc
    return state_dict


__all__ = [
    "NANO_MODEL_CONFIG",
    "FrozenNanoCheckpointError",
    "NanoGPT",
    "NanoModelConfig",
    "load_frozen_nano_state_dict",
    "load_verified_nano_state_dict",
]


def initialize_weights(model: nn.Module, *, generator: torch.Generator | None = None) -> nn.Module:
    """Apply the recorded from-scratch initialization: normal(0, 0.02), depth-scaled.

    `pretrain/AUDIT.md` records the July scheme as "init 0.02 depth-scaled" and
    its step-1 loss of 8.35 ~= ln(4096) as the sanity check. That scheme lived
    only in the historical pretraining script; `NanoModel.__init__` performs no
    initialization at all, which is invisible today because every current path
    warm-starts from an anchor checkpoint and `load_state_dict` overwrites the
    PyTorch defaults. A from-scratch rung-1 pretrain would silently use those
    defaults instead.

    This is **explicit and opt-in on purpose**. It is not called from any
    constructor, so no warm-start path, no frozen recipe, and no SHA-pinned
    module changes behaviour. In particular H6's `state_boundary_query_offsets`
    is deliberately zero-initialized; callers that add such parameters must
    apply this before creating them, or exclude them explicitly.

    Residual output projections (`o`, `dn`) are additionally scaled by
    1/sqrt(2 * layer_count) so residual-stream variance does not grow with depth.
    """

    config = getattr(model, "config", None)
    layer_count = getattr(config, "layer_count", None)
    if not isinstance(layer_count, int) or layer_count <= 0:
        raise ValueError("model must expose config.layer_count to depth-scale init")
    depth_scale = (2.0 * layer_count) ** -0.5

    def _normal(tensor: Tensor, std: float) -> None:
        with torch.no_grad():
            tensor.normal_(mean=0.0, std=std, generator=generator)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            _normal(module.weight, 0.02)
            if module.bias is not None:
                with torch.no_grad():
                    module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            _normal(module.weight, 0.02)

    for block in getattr(model, "blocks", []):
        for name in ("o", "dn"):
            projection = getattr(block, name, None)
            if isinstance(projection, nn.Linear):
                _normal(projection.weight, 0.02 * depth_scale)

    return model
