"""Tiny forward-only Nano LM correctness reference.

NumPy float64, unoptimized, no training, no pretrained weights. This implements
fixed logical depth with optional weight sharing. It is not an APS result.
Run: python3 prototype/native_reference.py [--out NEW_FILE.json]
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Config:
    vocab_size: int = 32
    d_model: int = 16
    n_heads: int = 2
    d_ff: int = 32
    logical_layers: int = 4
    unique_layers: int = 2
    max_context: int = 128
    rms_eps: float = 1e-6
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        dimensions = (self.vocab_size, self.d_model, self.n_heads, self.d_ff,
                      self.logical_layers, self.unique_layers, self.max_context)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in dimensions):
            raise ValueError("Dimensions must be positive integers")
        if self.vocab_size < 2 or self.d_model % self.n_heads:
            raise ValueError("Vocabulary >=2 and integral head width required")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("Interleaved RoPE requires even head width")
        if self.unique_layers > self.logical_layers or self.logical_layers % self.unique_layers:
            raise ValueError("Logical depth must be a multiple of unique block count")
        if not np.isfinite(self.rms_eps) or self.rms_eps <= 0:
            raise ValueError("Positive finite RMS epsilon required")
        if not np.isfinite(self.rope_base) or self.rope_base <= 1:
            raise ValueError("Finite RoPE base greater than one required")


def parameter_inventory(config: Config) -> dict[str, int]:
    """Counts distinct learned scalars; the tied output head adds no new ones."""
    v, d, f, u = config.vocab_size, config.d_model, config.d_ff, config.unique_layers
    counts = {
        "embedding_and_tied_head": v * d,
        "unique_attention_projections": u * 4 * d * d,
        "unique_swiglu_projections": u * 3 * d * f,
        "unique_block_norm_gains": u * 2 * d,
        "final_norm_gain": d,
    }
    return {**counts, "total": sum(counts.values())}


def cache_bytes(config: Config, batch_size: int, prefix_tokens: int, element_bytes: int) -> int:
    """KV tensors only; excludes allocation overhead, activations and weights."""
    dimensions = (batch_size, prefix_tokens, element_bytes)
    if any(not isinstance(n, int) or isinstance(n, bool) for n in dimensions):
        raise ValueError("Cache dimensions and element bytes must be integers")
    if prefix_tokens < 0 or batch_size < 1 or element_bytes < 1:
        raise ValueError("Invalid cache dimensions")
    return 2 * config.logical_layers * batch_size * prefix_tokens * config.d_model * element_bytes


def rms_norm(x: np.ndarray, gain: np.ndarray, eps: float) -> np.ndarray:
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * gain


def silu(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid, avoiding overflow even on deliberate large-value inputs.
    exp_negative_abs = np.exp(-np.abs(x))
    sigmoid = np.where(x >= 0, 1.0 / (1.0 + exp_negative_abs),
                       exp_negative_abs / (1.0 + exp_negative_abs))
    return x * sigmoid


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def rotate_positions(x: np.ndarray, positions: np.ndarray, base: float) -> np.ndarray:
    """Interleaved RoPE: x is [batch, heads, time, head_width]."""
    width = x.shape[-1]
    inverse_frequencies = base ** (-np.arange(0, width, 2, dtype=np.float64) / width)
    angles = positions[:, None] * inverse_frequencies[None, :]
    cosine, sine = np.cos(angles)[None, None], np.sin(angles)[None, None]
    result = np.empty_like(x)
    even, odd = x[..., 0::2], x[..., 1::2]
    result[..., 0::2] = even * cosine - odd * sine
    result[..., 1::2] = even * sine + odd * cosine
    return result


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class KV:
    key: np.ndarray
    value: np.ndarray


@dataclass(frozen=True)
class DecodeState:
    model_version: str
    batch_size: int
    length: int
    layers: tuple[KV, ...]


class NativeReference:
    """Bias-free pre-norm causal Transformer; block l uses weights l % U."""

    def __init__(self, config: Config = Config(), seed: int = 20261121) -> None:
        self.config = config
        rng = np.random.default_rng(seed)
        d, f = config.d_model, config.d_ff
        self.embedding = _readonly(rng.normal(0.0, 0.02, (config.vocab_size, d)))
        self.blocks: list[dict[str, np.ndarray]] = []
        for _ in range(config.unique_layers):
            block = {
                "q": rng.normal(0.0, d ** -0.5, (d, d)),
                "k": rng.normal(0.0, d ** -0.5, (d, d)),
                "v": rng.normal(0.0, d ** -0.5, (d, d)),
                "o": rng.normal(0.0, d ** -0.5, (d, d)),
                "gate": rng.normal(0.0, d ** -0.5, (d, f)),
                "up": rng.normal(0.0, d ** -0.5, (d, f)),
                "down": rng.normal(0.0, f ** -0.5, (f, d)),
                "norm_attention": np.ones(d),
                "norm_ffn": np.ones(d),
            }
            self.blocks.append({name: _readonly(value) for name, value in block.items()})
        self.final_norm = _readonly(np.ones(d))
        self.model_version = self._parameter_digest()

    def named_parameters(self) -> list[tuple[str, np.ndarray]]:
        parameters = [("embedding", self.embedding)]
        for index, block in enumerate(self.blocks):
            parameters.extend((f"block_{index}.{name}", array) for name, array in block.items())
        return parameters + [("final_norm", self.final_norm)]

    def _parameter_digest(self) -> str:
        """Fingerprint configuration, tensor identities/layouts and exact bytes."""
        digest = hashlib.sha256(json.dumps(asdict(self.config), sort_keys=True).encode())
        for name, value in sorted(self.named_parameters(), key=lambda item: item[0]):
            if not isinstance(value, np.ndarray):
                raise ValueError("Reference parameters must be NumPy arrays")
            metadata = {"name": name, "shape": list(value.shape), "dtype": value.dtype.str}
            digest.update(json.dumps(metadata, sort_keys=True).encode() + b"\0")
            digest.update(value.tobytes())
        return digest.hexdigest()

    def _assert_unchanged(self) -> None:
        # This tiny correctness reference has no weight-update API. NumPy's
        # read-only flags alone cannot prevent replacing a dictionary entry.
        # Rehashing every call is deliberately unsuitable for performance use.
        if self._parameter_digest() != self.model_version:
            raise ValueError("Parameters/configuration changed; construct a fresh reference and cache")

    def _validate_tokens(self, tokens: np.ndarray, ndim: int) -> np.ndarray:
        tokens = np.asarray(tokens)
        if tokens.ndim != ndim or tokens.dtype.kind not in "iu" or any(n == 0 for n in tokens.shape):
            raise ValueError(f"Nonempty integer token array of rank {ndim} required")
        if np.any(tokens < 0) or np.any(tokens >= self.config.vocab_size):
            raise ValueError("Token outside vocabulary")
        return tokens

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        b, t, _ = x.shape
        return x.reshape(b, t, self.config.n_heads, -1).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        b, _, t, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, self.config.d_model)

    def _feedforward(self, x: np.ndarray, block: dict[str, np.ndarray]) -> np.ndarray:
        normed = rms_norm(x, block["norm_ffn"], self.config.rms_eps)
        return x + (silu(normed @ block["gate"]) * (normed @ block["up"])) @ block["down"]

    def _head(self, x: np.ndarray) -> np.ndarray:
        return rms_norm(x, self.final_norm, self.config.rms_eps) @ self.embedding.T

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """Fresh-request full-prefix forward; returns [batch, time, vocab]."""
        self._assert_unchanged()
        tokens = self._validate_tokens(tokens, ndim=2)
        time = tokens.shape[1]
        if time > self.config.max_context:
            raise ValueError("Context bound exceeded")
        x = self.embedding[tokens]
        positions = np.arange(time)
        forbidden = np.triu(np.ones((time, time), dtype=bool), k=1)
        head_width = self.config.d_model // self.config.n_heads
        for logical_depth in range(self.config.logical_layers):
            block = self.blocks[logical_depth % self.config.unique_layers]
            normed = rms_norm(x, block["norm_attention"], self.config.rms_eps)
            query = rotate_positions(self._split_heads(normed @ block["q"]), positions, self.config.rope_base)
            key = rotate_positions(self._split_heads(normed @ block["k"]), positions, self.config.rope_base)
            value = self._split_heads(normed @ block["v"])
            score = (query @ key.swapaxes(-1, -2)) / np.sqrt(head_width)
            probability = softmax(np.where(forbidden[None, None], -np.inf, score))
            x = x + self._merge_heads(probability @ value) @ block["o"]
            x = self._feedforward(x, block)
        return self._head(x)

    def new_state(self, batch_size: int) -> DecodeState:
        self._assert_unchanged()
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("Positive integer batch size required")
        shape = (batch_size, self.config.n_heads, 0,
                 self.config.d_model // self.config.n_heads)
        layers = tuple(KV(_readonly(np.empty(shape)), _readonly(np.empty(shape)))
                       for _ in range(self.config.logical_layers))
        return DecodeState(self.model_version, batch_size, 0, layers)

    def step(self, token_ids: np.ndarray, state: DecodeState) -> tuple[np.ndarray, DecodeState]:
        """Consume one observed token, without mutating the supplied state."""
        self._assert_unchanged()
        token_ids = self._validate_tokens(token_ids, ndim=1)
        if state.model_version != self.model_version:
            raise ValueError("Cache/checkpoint mismatch")
        if token_ids.shape[0] != state.batch_size:
            raise ValueError("Batch/cache mismatch")
        if not 0 <= state.length < self.config.max_context:
            raise ValueError("Context bound exceeded or negative cache length")
        if len(state.layers) != self.config.logical_layers:
            raise ValueError("One KV cache per logical layer required")
        expected = (state.batch_size, self.config.n_heads, state.length,
                    self.config.d_model // self.config.n_heads)
        if any(cache.key.shape != expected or cache.value.shape != expected for cache in state.layers):
            raise ValueError("Malformed logical-layer cache")

        x = self.embedding[token_ids][:, None, :]
        positions = np.array([state.length])
        head_width = self.config.d_model // self.config.n_heads
        updated: list[KV] = []
        for logical_depth, cache in enumerate(state.layers):
            block = self.blocks[logical_depth % self.config.unique_layers]
            normed = rms_norm(x, block["norm_attention"], self.config.rms_eps)
            query = rotate_positions(self._split_heads(normed @ block["q"]), positions, self.config.rope_base)
            new_key = rotate_positions(self._split_heads(normed @ block["k"]), positions, self.config.rope_base)
            new_value = self._split_heads(normed @ block["v"])
            key = np.concatenate((cache.key, new_key), axis=2)
            value = np.concatenate((cache.value, new_value), axis=2)
            score = (query @ key.swapaxes(-1, -2)) / np.sqrt(head_width)
            x = x + self._merge_heads(softmax(score) @ value) @ block["o"]
            x = self._feedforward(x, block)
            updated.append(KV(_readonly(key), _readonly(value)))
        next_state = DecodeState(self.model_version, state.batch_size,
                                 state.length + 1, tuple(updated))
        return self._head(x)[:, 0], next_state


def cross_entropy(logits: np.ndarray, targets: np.ndarray, valid_mask: np.ndarray) -> float:
    """Mean natural-log CE on explicitly aligned targets; no optimizer/training."""
    logits, targets, valid_mask = np.asarray(logits), np.asarray(targets), np.asarray(valid_mask)
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or valid_mask.shape != targets.shape:
        raise ValueError("Expected logits[B,T,V], aligned targets[B,T], mask[B,T]")
    if logits.dtype.kind != "f":
        raise ValueError("Floating-point logits required; integer subtraction can overflow")
    if targets.dtype.kind not in "iu" or valid_mask.dtype.kind != "b":
        raise ValueError("Integer targets and boolean mask required")
    if np.any(targets < 0) or np.any(targets >= logits.shape[-1]):
        raise ValueError("Target outside vocabulary")
    if not np.all(np.isfinite(logits)) or not np.any(valid_mask):
        raise ValueError("Finite logits and at least one scored target required")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=-1))
    chosen = np.take_along_axis(shifted, targets[..., None], axis=-1)[..., 0]
    return float(np.mean((log_normalizer - chosen)[valid_mask]))


def write_report(report: dict[str, Any], output: str | None) -> None:
    """Print by default; explicit output uses exclusive creation, never overwrite."""
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if output is not None:
        with Path(output).open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Optional NEW JSON path; parent must exist")
    args = parser.parse_args()
    config = Config()
    model = NativeReference(config)
    tokens = np.array([[1, 3, 2, 9, 4]], dtype=np.int64)
    logits = model.forward(tokens)
    report = {
        "scope": "Forward-only causal reference; no training or language-quality result",
        "configuration": asdict(config),
        "parameter_inventory": parameter_inventory(config),
        "checkpoint_sha256": model.model_version,
        "logits_shape": list(logits.shape),
        "toy_next_token_ce_nats": cross_entropy(logits[:, :-1], tokens[:, 1:],
                                               np.ones((1, 4), dtype=bool)),
        "numpy_version": np.__version__,
    }
    write_report(report, args.out)


if __name__ == "__main__":
    main()
