"""Optional causal comparator; pure integer fixtures, no corpus or model loading.

The caller resets at *both* document and crop boundaries, observes the first
input token, then scores each next token before observing it. A masked target
is still observed, but contributes no loss. Missing/padded positions must not
be passed as observations. The comparator and native model must receive the
same crop prefix. This module cannot verify the native model's input.

The retained window contains observed tokens, not precomputed target rows.
Counting complete transitions inside that window deliberately drops transitions
that cross its left boundary. Prediction accepts no target or future suffix.
No-match prediction is uniform; mix_with_native uses weight zero on no match.
This is a transparent reference implementation, not a throughput claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class CacheConfig:
    vocab_size: int
    max_order: int = 3
    window: int = 512
    smoothing: float = 1.0

    def __post_init__(self):
        for name in ("vocab_size", "max_order", "window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive Python integer")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least two")
        if isinstance(self.smoothing, bool) or not isinstance(self.smoothing, Real):
            raise ValueError("smoothing must be finite and positive")
        try:
            alpha = float(self.smoothing)
        except (ValueError, OverflowError) as error:
            raise ValueError("smoothing must be representable as a positive float") from error
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("smoothing must be finite and positive")
        object.__setattr__(self, "smoothing", alpha)


@dataclass(frozen=True)
class ContextID:
    document: str
    crop: str

    def __post_init__(self):
        if not isinstance(self.document, str) or not self.document:
            raise ValueError("document must be a nonempty string")
        if not isinstance(self.crop, str) or not self.crop:
            raise ValueError("crop must be a nonempty string")


@dataclass(frozen=True)
class CacheSnapshot:
    config: CacheConfig
    context_id: ContextID
    observed_count: int
    retained_prefix: tuple[int, ...]
    state_sha256: str


@dataclass(frozen=True)
class CachePrediction:
    probabilities: tuple[float, ...]
    counts: tuple[int, ...]
    matched_order: int
    matches: int
    snapshot: CacheSnapshot


@dataclass(frozen=True)
class CacheScore:
    target: int
    scored: bool
    negative_log_likelihood: float | None
    prediction: CachePrediction


class CausalNgramCache:
    def __init__(self, config: CacheConfig, context_id: ContextID):
        if not isinstance(config, CacheConfig):
            raise TypeError("config must be CacheConfig")
        self._config = config
        self.reset(context_id)

    @property
    def config(self):
        return self._config

    def reset(self, context_id: ContextID):
        """Explicitly discard all observed context at a crop/document boundary."""
        if not isinstance(context_id, ContextID):
            raise TypeError("context_id must be ContextID")
        self._context_id = context_id
        self._prefix = ()
        self._observed_count = 0

    def _check_context(self, context_id):
        if not isinstance(context_id, ContextID) or context_id != self._context_id:
            raise ValueError("context identity changed; reset before observing/scoring")

    def _token(self, token):
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise ValueError("token must be an integer")
        token = int(token)
        if not 0 <= token < self.config.vocab_size:
            raise ValueError("token is outside vocabulary")
        return token

    def snapshot(self) -> CacheSnapshot:
        # Digest identifies retained *causal state*, not the full discarded
        # history, a model, a corpus, or an externally supplied document label.
        payload = {"schema": "causal-ngram-cache-v1", "config": asdict(self.config),
                   "context_id": asdict(self._context_id),
                   "observed_count": self._observed_count, "prefix": self._prefix}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False).encode("utf-8")).hexdigest()
        return CacheSnapshot(self.config, self._context_id, self._observed_count,
                             self._prefix, digest)

    def observe(self, token: int, *, context_id: ContextID):
        self._check_context(context_id)
        token = self._token(token)
        self._prefix = (self._prefix + (token,))[-self.config.window:]
        self._observed_count += 1

    def predict(self, *, context_id: ContextID) -> CachePrediction:
        self._check_context(context_id)
        prefix, vocabulary = self._prefix, self.config.vocab_size
        counts = [0] * vocabulary
        matched_order = 0
        # Every candidate follower has index start + order < len(prefix).
        # In particular, the unknown next token is never in the count table.
        for order in range(min(self.config.max_order, len(prefix)), 0, -1):
            context = prefix[-order:]
            current = [0] * vocabulary
            for start in range(len(prefix) - order):
                if prefix[start:start + order] == context:
                    current[prefix[start + order]] += 1
            if sum(current):
                counts, matched_order = current, order
                break
        matches = sum(counts)
        alpha = float(self.config.smoothing)
        probabilities = tuple((count + alpha / vocabulary) / (matches + alpha)
                              for count in counts)
        # Catch a requested smoothing scale too small/large for positive float64
        # probabilities; the module never silently emits an invalid distribution.
        if (not all(math.isfinite(p) and p > 0 for p in probabilities)
                or not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("smoothing is not numerically representable for this state")
        return CachePrediction(probabilities, tuple(counts), matched_order, matches,
                               self.snapshot())

    def score_then_observe(self, token: int, *, context_id: ContextID,
                           scored: bool = True) -> CacheScore:
        self._check_context(context_id)
        token = self._token(token)
        if not isinstance(scored, bool):
            raise ValueError("scored must be a boolean")
        prediction = self.predict(context_id=context_id)
        loss = -math.log(prediction.probabilities[token]) if scored else None
        self.observe(token, context_id=context_id)
        return CacheScore(token, scored, loss, prediction)


def mix_with_native(native_probabilities, prediction: CachePrediction,
                    weight: float) -> tuple[float, ...]:
    """Convex mixing at the same observed prefix; no fitting or hidden gate.

    The caller selects/fits weight using allowed training data only and freezes
    it for evaluation. With no observed match, or weight zero, return the validated
    Python-float native probabilities exactly. Every native probability must be
    positive. Inputs are converted to Python floats before validation/mixing.
    """
    if isinstance(weight, bool) or not isinstance(weight, Real):
        raise ValueError("weight must be finite and between zero and one")
    try:
        weight = float(weight)
    except (ValueError, OverflowError) as error:
        raise ValueError("weight must be representable as a finite float") from error
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be finite and between zero and one")
    # Predictions are public immutable records, so validate supplied records at
    # this boundary. This checks coherence, not provenance or document identity.
    if (not isinstance(prediction, CachePrediction)
            or not isinstance(prediction.snapshot, CacheSnapshot)
            or not isinstance(prediction.snapshot.config, CacheConfig)):
        raise ValueError("a cache prediction with a configuration is required")
    config = prediction.snapshot.config
    if (type(prediction.probabilities) is not tuple
            or len(prediction.probabilities) != config.vocab_size
            or any(isinstance(p, bool) or not isinstance(p, Real)
                   for p in prediction.probabilities)):
        raise ValueError("cache probabilities must be a numeric vocabulary-sized tuple")
    try:
        probabilities = tuple(float(p) for p in prediction.probabilities)
    except (ValueError, OverflowError) as error:
        raise ValueError("cache probabilities must be representable as finite floats") from error
    if (not all(math.isfinite(p) and p > 0 for p in probabilities)
            or not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0, abs_tol=1e-12)
            or type(prediction.counts) is not tuple
            or len(prediction.counts) != config.vocab_size
            or any(type(count) is not int or count < 0 for count in prediction.counts)
            or type(prediction.matches) is not int or prediction.matches < 0
            or sum(prediction.counts) != prediction.matches
            or type(prediction.matched_order) is not int
            or not 0 <= prediction.matched_order <= config.max_order
            or (prediction.matches == 0) != (prediction.matched_order == 0)):
        raise ValueError("cache probabilities, counts and matching metadata must be coherent")
    expected = tuple((count + config.smoothing / config.vocab_size)
                     / (prediction.matches + config.smoothing) for count in prediction.counts)
    if any(not math.isclose(p, q, rel_tol=0, abs_tol=1e-12)
           for p, q in zip(probabilities, expected)):
        raise ValueError("cache probabilities do not match the declared smoothed counts")
    native = tuple(float(p) for p in native_probabilities)
    if (len(native) != len(probabilities)
            or not all(math.isfinite(p) and p > 0 for p in native)
            or not math.isclose(math.fsum(native), 1.0, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("native probabilities must be a positive normalized distribution")
    if prediction.matches == 0 or weight == 0:
        return native
    # Anchor at the smaller positive operand. Two separately rounded products
    # can both underflow to zero even when their convex sum is representable.
    mixed = tuple(p + weight * (q - p) if q >= p
                  else q + (1 - weight) * (p - q)
                  for p, q in zip(native, probabilities))
    if (not all(math.isfinite(p) and p > 0 for p in mixed)
            or not math.isclose(math.fsum(mixed), 1.0, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("mixture is not a positive normalized Python-float distribution")
    return mixed
