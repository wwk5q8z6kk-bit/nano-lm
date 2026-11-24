"""Tiny NumPy float64 training reference with explicit reverse-mode derivatives.

Matches native_reference.py, including tied embeddings and shared blocks.
This is an engineering instrument, not a language-quality or speed benchmark.
Rows are separate documents. No packing, dropout, sampler, external data or GPU.
Default CLI runs a synthetic learning smoke and prints a receipt; --out is NEW.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
from native_reference import (Config, NativeReference, cross_entropy,
                              parameter_inventory, rms_norm, rotate_positions,
                              silu, softmax, write_report)


def _matrix_gradient(x: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])


def _rms_backward(x: np.ndarray, gain: np.ndarray, dy: np.ndarray,
                  eps: float) -> tuple[np.ndarray, np.ndarray]:
    inverse = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    normalized = x * inverse
    axes = tuple(range(x.ndim - 1))
    gain_gradient = np.sum(dy * normalized, axis=axes)
    dg = dy * gain
    dx = inverse * (dg - normalized * np.mean(dg * normalized, axis=-1, keepdims=True))
    return dx, gain_gradient


def _sigmoid(x: np.ndarray) -> np.ndarray:
    negative_abs = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + negative_abs),
                    negative_abs / (1.0 + negative_abs))


class TrainableNative:
    """Mutable C/F-contiguous parameters; inference uses frozen snapshots."""

    def __init__(self, config: Config = Config(), seed: int = 20261121) -> None:
        if not isinstance(config, Config):
            raise ValueError("A native-reference Config is required")
        self.config = config
        self._initial_config = asdict(config)
        initial = NativeReference(config, seed)
        self.parameters = {name: value.copy() for name, value in initial.named_parameters()}
        self._shapes = {name: value.shape for name, value in self.parameters.items()}

    def _validate_parameters(self) -> None:
        if not isinstance(self.config, Config) or asdict(self.config) != self._initial_config:
            raise ValueError("Model configuration changed; construct another model")
        if set(self.parameters) != set(self._shapes):
            raise ValueError("Parameter names differ from the model definition")
        for name, array in self.parameters.items():
            if (not isinstance(array, np.ndarray) or array.dtype != np.float64
                    or array.shape != self._shapes[name] or not array.flags.writeable
                    or not np.all(np.isfinite(array))):
                raise ValueError(f"Invalid mutable float64 parameter: {name}")
            # A writable strided view can alias itself even when no other named
            # parameter shares its storage. Contiguity excludes that case and
            # makes each reported tensor coordinate a distinct writable value.
            if not (array.flags.c_contiguous or array.flags.f_contiguous):
                raise ValueError(f"Parameter storage must be C- or F-contiguous: {name}")
        # Every dictionary entry denotes a distinct learned tensor. Weight tying
        # and logical block reuse are expressed by reusing its one name, never
        # by aliases under different names. Check storage, not object identity:
        # transposes and partially overlapping slices can alias too.
        named = list(self.parameters.items())
        for index, (name, array) in enumerate(named):
            for other_name, other in named[:index]:
                if np.shares_memory(array, other):
                    raise ValueError(f"Overlapping parameter storage: {other_name}, {name}")

    def _tokens(self, tokens: np.ndarray, minimum_time: int = 1) -> np.ndarray:
        tokens = np.asarray(tokens)
        if (tokens.ndim != 2 or tokens.dtype.kind not in "iu"
                or tokens.shape[0] < 1 or tokens.shape[1] < minimum_time):
            raise ValueError("Nonempty integer token rows with sufficient length required")
        if np.any(tokens < 0) or np.any(tokens >= self.config.vocab_size):
            raise ValueError("Token outside vocabulary")
        return tokens

    def _split(self, value: np.ndarray) -> np.ndarray:
        b, t, _ = value.shape
        return value.reshape(b, t, self.config.n_heads, -1).transpose(0, 2, 1, 3)

    def _merge(self, value: np.ndarray) -> np.ndarray:
        b, _, t, _ = value.shape
        return value.transpose(0, 2, 1, 3).reshape(b, t, self.config.d_model)

    def _forward(self, tokens: np.ndarray, retain_tape: bool = False):
        self._validate_parameters()
        tokens = self._tokens(tokens)
        if tokens.shape[1] > self.config.max_context:
            raise ValueError("Context bound exceeded")
        p, c = self.parameters, self.config
        x = p["embedding"][tokens]
        positions = np.arange(tokens.shape[1])
        forbidden = np.triu(np.ones((tokens.shape[1], tokens.shape[1]), dtype=bool), 1)
        scale = np.sqrt(c.d_model // c.n_heads)
        tape = []
        for depth in range(c.logical_layers):
            prefix = f"block_{depth % c.unique_layers}."
            block = {name: p[prefix + name] for name in
                     ("q", "k", "v", "o", "gate", "up", "down",
                      "norm_attention", "norm_ffn")}
            na = rms_norm(x, block["norm_attention"], c.rms_eps)
            q = rotate_positions(self._split(na @ block["q"]), positions, c.rope_base)
            k = rotate_positions(self._split(na @ block["k"]), positions, c.rope_base)
            v = self._split(na @ block["v"])
            scores = (q @ k.swapaxes(-1, -2)) / scale
            probability = softmax(np.where(forbidden[None, None], -np.inf, scores))
            context = self._merge(probability @ v)
            attended = x + context @ block["o"]
            nf = rms_norm(attended, block["norm_ffn"], c.rms_eps)
            gate, up = nf @ block["gate"], nf @ block["up"]
            activated = silu(gate)
            product = activated * up
            updated = attended + product @ block["down"]
            if retain_tape:
                tape.append((prefix, x, na, q, k, v, probability, context,
                             attended, nf, gate, up, activated, product))
            x = updated
        final = rms_norm(x, p["final_norm"], c.rms_eps)
        logits = final @ p["embedding"].T
        if not np.all(np.isfinite(logits)):
            raise ValueError("Nonfinite forward logits")
        return logits, (tokens, positions, x, final, tape) if retain_tape else None

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """Fresh causal inference without training-state mutation or KV caching."""
        return self._forward(tokens)[0]

    def _documents(self, documents: np.ndarray, valid_mask: np.ndarray | None):
        documents = self._tokens(documents, minimum_time=2)
        if documents.shape[1] - 1 > self.config.max_context:
            raise ValueError("Shifted document exceeds context bound")
        targets = documents[:, 1:]
        mask = np.ones(targets.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask)
        if mask.dtype.kind != "b" or mask.shape != targets.shape or not np.any(mask):
            raise ValueError("Boolean target mask with at least one scored token required")
        return documents[:, :-1], targets, mask

    def loss(self, documents: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
        tokens, targets, mask = self._documents(documents, valid_mask)
        return cross_entropy(self.forward(tokens), targets, mask)

    def loss_and_grad(self, documents: np.ndarray,
                      valid_mask: np.ndarray | None = None) -> tuple[float, dict[str, np.ndarray]]:
        """Mean next-token CE and gradients of every distinct learned tensor.

        Each row is a separate document. A loss mask does not reset attention;
        it only omits explicitly aligned targets from the mean.
        """
        tokens, targets, mask = self._documents(documents, valid_mask)
        logits, saved = self._forward(tokens, retain_tape=True)
        loss = cross_entropy(logits, targets, mask)
        p, c = self.parameters, self.config
        grad = {name: np.zeros_like(array) for name, array in p.items()}
        derivative = softmax(logits)
        b, t = np.indices(targets.shape)
        derivative[b, t, targets] -= 1.0
        derivative *= mask[..., None] / int(mask.sum())
        tokens, positions, final_x, final_normed, tape = saved
        grad["embedding"] += derivative.reshape(-1, c.vocab_size).T @ final_normed.reshape(-1, c.d_model)
        dx, norm_gradient = _rms_backward(final_x, p["final_norm"],
                                         derivative @ p["embedding"], c.rms_eps)
        grad["final_norm"] += norm_gradient
        scale = np.sqrt(c.d_model // c.n_heads)
        for record in reversed(tape):
            (prefix, x, na, q, k, v, probability, context,
             attended, nf, gate, up, activated, product) = record
            # Repeated logical blocks add into the same distinct tensor's gradient.
            grad[prefix + "down"] += _matrix_gradient(product, dx)
            dproduct = dx @ p[prefix + "down"].T
            sigmoid = _sigmoid(gate)
            dgate = dproduct * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
            dup = dproduct * activated
            grad[prefix + "gate"] += _matrix_gradient(nf, dgate)
            grad[prefix + "up"] += _matrix_gradient(nf, dup)
            dnf = dgate @ p[prefix + "gate"].T + dup @ p[prefix + "up"].T
            dnormalized, dnorm_gain = _rms_backward(attended, p[prefix + "norm_ffn"], dnf, c.rms_eps)
            grad[prefix + "norm_ffn"] += dnorm_gain
            dattended = dx + dnormalized
            grad[prefix + "o"] += _matrix_gradient(context, dattended)
            dcontext = self._split(dattended @ p[prefix + "o"].T)
            dprobability = dcontext @ v.swapaxes(-1, -2)
            dv = probability.swapaxes(-1, -2) @ dcontext
            dscore = probability * (dprobability - np.sum(dprobability * probability, axis=-1, keepdims=True))
            dscore /= scale
            dq = dscore @ k
            dk = dscore.swapaxes(-1, -2) @ q
            # Orthogonal RoPE's transpose is the rotation at negative position.
            dq = self._merge(rotate_positions(dq, -positions, c.rope_base))
            dk = self._merge(rotate_positions(dk, -positions, c.rope_base))
            dv = self._merge(dv)
            grad[prefix + "q"] += _matrix_gradient(na, dq)
            grad[prefix + "k"] += _matrix_gradient(na, dk)
            grad[prefix + "v"] += _matrix_gradient(na, dv)
            dna = dq @ p[prefix + "q"].T + dk @ p[prefix + "k"].T + dv @ p[prefix + "v"].T
            dnormalized, dnorm_gain = _rms_backward(x, p[prefix + "norm_attention"], dna, c.rms_eps)
            grad[prefix + "norm_attention"] += dnorm_gain
            dx = dattended + dnormalized
        # Accumulate repeated token occurrences and the independent output-head path.
        np.add.at(grad["embedding"], tokens, dx)
        if any(not np.all(np.isfinite(array)) for array in grad.values()):
            raise ValueError("Nonfinite gradient")
        return loss, grad

    def snapshot(self) -> NativeReference:
        """Independent immutable inference copy with a fresh parameter digest."""
        self._validate_parameters()
        result = NativeReference(self.config, seed=0)
        def frozen(name):
            array = self.parameters[name].copy()
            array.flags.writeable = False
            return array
        result.embedding = frozen("embedding")
        result.final_norm = frozen("final_norm")
        for index, block in enumerate(result.blocks):
            for name in block:
                block[name] = frozen(f"block_{index}.{name}")
        result.model_version = result._parameter_digest()
        return result


@dataclass(frozen=True)
class AdamWConfig:
    learning_rate: float = 0.003
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    def __post_init__(self):
        values = asdict(self)
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not np.isfinite(v) for v in values.values()):
            raise ValueError("Finite numeric optimizer hyperparameters required")
        if (self.learning_rate <= 0 or self.epsilon <= 0 or self.weight_decay < 0
                or self.max_grad_norm <= 0 or not 0 <= self.beta1 < 1
                or not 0 <= self.beta2 < 1):
            raise ValueError("Optimizer hyperparameter out of range")


class AdamW:
    """Global raw-gradient clipping, Adam moments, then decoupled weight decay.

    Decay applies to every distinct tensor, including embeddings and norm gains.
    This explicit reference policy is not a tuned training recommendation.
    No nullspace projection or continual-learning guarantee is implemented.
    """

    def __init__(self, model: TrainableNative, config: AdamWConfig = AdamWConfig()):
        if not isinstance(config, AdamWConfig):
            raise ValueError("AdamWConfig required")
        model._validate_parameters()
        self.model, self.config, self.step_number = model, config, 0
        self.first = {k: np.zeros_like(v) for k, v in model.parameters.items()}
        self.second = {k: np.zeros_like(v) for k, v in model.parameters.items()}
        self.last_displacements: dict[str, dict[str, np.ndarray]] = {}

    def _validate_state(self):
        self.model._validate_parameters()
        parameters = self.model.parameters
        if type(self.step_number) is not int or self.step_number < 0:
            raise ValueError("Invalid optimizer step")
        if set(self.first) != set(parameters) or set(self.second) != set(parameters):
            raise ValueError("Optimizer state names differ")
        for state, nonnegative in ((self.first, False), (self.second, True)):
            for name, array in state.items():
                if (not isinstance(array, np.ndarray) or array.dtype != np.float64
                        or array.shape != parameters[name].shape
                        or not np.all(np.isfinite(array))
                        or (nonnegative and np.any(array < 0))
                        or (self.step_number == 0 and np.any(array))):
                    raise ValueError(f"Invalid optimizer tensor: {name}")

    def step(self, gradients: dict[str, np.ndarray]) -> dict:
        self._validate_state()
        parameters, c = self.model.parameters, self.config
        if set(gradients) != set(parameters):
            raise ValueError("One gradient per distinct parameter required")
        for name, value in gradients.items():
            if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                    or value.shape != parameters[name].shape or not np.all(np.isfinite(value))):
                raise ValueError(f"Invalid gradient: {name}")
        norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in gradients.values())))
        if not np.isfinite(norm):
            raise ValueError("Nonfinite global gradient norm")
        clipping = min(1.0, c.max_grad_norm / norm) if norm else 1.0
        step = self.step_number + 1
        first, second, proposals, candidates, actual = {}, {}, {}, {}, {}
        for name, parameter in parameters.items():
            g = gradients[name] * clipping
            first[name] = c.beta1 * self.first[name] + (1.0 - c.beta1) * g
            second[name] = c.beta2 * self.second[name] + (1.0 - c.beta2) * g * g
            corrected_m = first[name] / (1.0 - c.beta1 ** step)
            corrected_v = second[name] / (1.0 - c.beta2 ** step)
            proposals[name] = -c.learning_rate * (
                corrected_m / (np.sqrt(corrected_v) + c.epsilon) + c.weight_decay * parameter)
            candidates[name] = parameter + proposals[name]
            actual[name] = candidates[name] - parameter
        if any(not np.all(np.isfinite(x)) for group in
               (first, second, proposals, candidates, actual) for x in group.values()):
            raise ValueError("Nonfinite optimizer proposal; parameters unchanged")
        # Even finite arrays can overflow a diagnostic norm. Validate the full
        # receipt before committing parameters, moments or the step counter.
        with np.errstate(over="ignore", invalid="ignore"):
            per_parameter = {
                name: {"proposed_l2": float(np.linalg.norm(proposals[name])),
                       "actual_l2": float(np.linalg.norm(actual[name])),
                       "max_abs_rounding_difference": float(np.max(np.abs(actual[name] - proposals[name])))}
                for name in parameters
            }
            proposal_norm = float(np.sqrt(sum(float(np.sum(x*x)) for x in proposals.values())))
            actual_norm = float(np.sqrt(sum(float(np.sum(x*x)) for x in actual.values())))
        measurements = [proposal_norm, actual_norm]
        measurements.extend(x for row in per_parameter.values() for x in row.values())
        if not all(np.isfinite(x) for x in measurements):
            raise ValueError("Optimizer diagnostic norm overflow; state unchanged")
        receipt = {
            "step": step, "raw_gradient_l2": norm, "gradient_clip_scale": clipping,
            "proposed_displacement_l2": proposal_norm,
            "actual_displacement_l2": actual_norm,
            "per_parameter": per_parameter,
            "policy": "Global gradient clipping before Adam; decoupled decay on all tensors; no update projection.",
        }
        # All shapes, writability, candidates and receipt values are checked.
        for name, candidate in candidates.items():
            parameters[name][...] = candidate
        self.first, self.second, self.step_number = first, second, step
        self.last_displacements = {"proposed": proposals, "actual": actual}
        return receipt

    def train_step(self, documents: np.ndarray, valid_mask: np.ndarray | None = None) -> dict:
        loss, gradients = self.model.loss_and_grad(documents, valid_mask)
        return {"loss_before_nats": loss, **self.step(gradients)}

    def _state_arrays(self):
        return {**{"p::" + k: v for k, v in self.model.parameters.items()},
                **{"m::" + k: v for k, v in self.first.items()},
                **{"v::" + k: v for k, v in self.second.items()}}

    def _metadata(self):
        return {"schema_version": 1, "model_config": asdict(self.model.config),
                "optimizer_config": asdict(self.config), "step": self.step_number}

    def state_digest(self) -> str:
        self._validate_state()
        h = hashlib.sha256(json.dumps(self._metadata(), sort_keys=True, allow_nan=False).encode())
        for name, array in sorted(self._state_arrays().items()):
            h.update(json.dumps({"name": name, "shape": list(array.shape),
                                 "dtype": array.dtype.str}, sort_keys=True).encode() + b"\0")
            h.update(array.tobytes())
        return h.hexdigest()

    def save_checkpoint(self, path: str | Path) -> None:
        """Exclusive pickle-free NPZ; stores model, moments, step and hyperparameters.

        No sampler exists, so there is no sampler/RNG cursor to save. Exact resume
        requires the caller to supply the same subsequent documents in the same
        order. An interrupted file is rejected by load; no crash-atomic guarantee.
        """
        self.model._validate_parameters()
        metadata = {**self._metadata(), "state_sha256": self.state_digest()}
        arrays = self._state_arrays()
        arrays["metadata"] = np.frombuffer(json.dumps(metadata, sort_keys=True,
                                                     allow_nan=False).encode(), dtype=np.uint8)
        with Path(path).open("xb") as target:
            np.savez(target, **arrays)


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint metadata key: {key}")
        result[key] = value
    return result


def load_checkpoint(path: str | Path) -> tuple[TrainableNative, AdamW]:
    """Validate names, config, shapes, dtypes, finite state and internal checksum."""
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or "metadata" not in archive.files:
            raise ValueError("Missing or duplicate checkpoint arrays")
        metadata_array = archive["metadata"]
        if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1 or metadata_array.size > 65536:
            raise ValueError("Invalid checkpoint metadata encoding")
        metadata = json.loads(metadata_array.tobytes().decode(), object_pairs_hook=_unique_json)
        expected_fields = {"schema_version", "model_config", "optimizer_config", "step", "state_sha256"}
        if (not isinstance(metadata, dict) or set(metadata) != expected_fields
                or type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1
                or type(metadata["step"]) is not int or metadata["step"] < 0
                or not isinstance(metadata["state_sha256"], str)):
            raise ValueError("Invalid checkpoint metadata contract")
        if (not isinstance(metadata["model_config"], dict)
                or not isinstance(metadata["optimizer_config"], dict)
                or set(metadata["model_config"]) != set(asdict(Config()))
                or set(metadata["optimizer_config"]) != set(asdict(AdamWConfig()))):
            raise ValueError("Checkpoint configuration fields differ")
        model = TrainableNative(Config(**metadata["model_config"]), seed=0)
        optimizer = AdamW(model, AdamWConfig(**metadata["optimizer_config"]))
        expected_arrays = optimizer._state_arrays()
        if set(archive.files) != set(expected_arrays) | {"metadata"}:
            raise ValueError("Checkpoint tensor names differ")
        loaded = {}
        for name, template in expected_arrays.items():
            value = archive[name]
            if (value.dtype != np.float64 or value.shape != template.shape
                    or not np.all(np.isfinite(value))
                    or (name.startswith("v::") and np.any(value < 0))):
                raise ValueError(f"Invalid checkpoint tensor: {name}")
            if metadata["step"] == 0 and not name.startswith("p::") and np.any(value):
                raise ValueError("Nonzero moments at step zero")
            loaded[name] = value.copy()
        for name in model.parameters:
            model.parameters[name][...] = loaded["p::" + name]
            optimizer.first[name] = loaded["m::" + name]
            optimizer.second[name] = loaded["v::" + name]
        optimizer.step_number = metadata["step"]
        if optimizer.state_digest() != metadata["state_sha256"]:
            raise ValueError("Checkpoint state checksum mismatch")
    return model, optimizer


def learning_smoke(steps: int = 60) -> dict:
    """Fit a deterministic cyclic-token fixture; training-set behavior only."""
    if type(steps) is not int or not 1 <= steps <= 500:
        raise ValueError("Smoke step count must be an integer in [1,500]")
    documents = (np.arange(9)[None, :] + np.arange(4)[:, None]) % 8
    rows = []
    for unique in (2, 1):
        config = Config(vocab_size=8, d_model=8, n_heads=2, d_ff=16,
                        logical_layers=2, unique_layers=unique, max_context=8)
        model = TrainableNative(config, seed=9100)
        optimizer = AdamW(model, AdamWConfig(learning_rate=0.02, weight_decay=0.01))
        initial = model.loss(documents)
        for _ in range(steps):
            last = optimizer.train_step(documents)
        final = model.loss(documents)
        rows.append({"unique_layers": unique, "logical_layers": 2, "steps": steps,
                     "parameters": parameter_inventory(config)["total"],
                     "training_targets_per_step": int(documents[:, 1:].size),
                     "initial_training_ce_nats": initial, "final_training_ce_nats": final,
                     "last_actual_displacement_l2": last["actual_displacement_l2"],
                     "final_model_sha256": model.snapshot().model_version,
                     "optimizer_state_sha256": optimizer.state_digest()})
    return {"scope": "Synthetic cyclic-token learning smoke, not language quality, generalization, cost comparison or architecture selection.",
            "data": "Four rows of nine cyclic integer tokens modulo eight; same training rows reused each step.",
            "numpy_version": np.__version__, "arms": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Optional NEW JSON receipt, never overwrite")
    parser.add_argument("--steps", type=int, default=60, help="Synthetic fixture updates per arm, at most 500")
    args = parser.parse_args()
    if args.out is not None and Path(args.out).exists():
        raise FileExistsError(f"Existing receipt is preserved: {args.out}")
    write_report(learning_smoke(args.steps), args.out)


if __name__ == "__main__":
    main()
