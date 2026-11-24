"""Deterministic integer fixtures, independent count oracle, no language runs."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from causal_ngram_cache import CacheConfig, ContextID, CausalNgramCache, mix_with_native


def oracle(prefix, config):
    """Group all completed context/follower transitions, then query the suffix."""
    prefix = tuple(prefix[-config.window:])
    tables = {order: {} for order in range(1, config.max_order + 1)}
    for follower_at in range(1, len(prefix)):
        for order in range(1, min(follower_at, config.max_order) + 1):
            key = prefix[follower_at - order:follower_at]
            tables[order].setdefault(key, Counter())[prefix[follower_at]] += 1
    for order in range(min(config.max_order, len(prefix)), 0, -1):
        counts = tables[order].get(prefix[-order:])
        if counts:
            full = tuple(counts.get(v, 0) for v in range(config.vocab_size))
            total = sum(full)
            return order, full, tuple((n + config.smoothing / config.vocab_size)
                                     / (total + config.smoothing) for n in full)
    return 0, (0,) * config.vocab_size, (1 / config.vocab_size,) * config.vocab_size


def reject(operation):
    try:
        operation()
    except (TypeError, ValueError, FrozenInstanceError):
        return
    raise AssertionError("invalid operation was accepted")


def run_checks():
    rows = []
    cid = ContextID("document-A", "crop-0")
    config = CacheConfig(4, max_order=3, window=5, smoothing=1.0)

    # Exhaustive prefixes exercise overlap, longest-order backoff, empty state,
    # fixed-window eviction, and a separately grouped count implementation.
    checked = 0
    for length in range(7):
        for prefix in itertools.product(range(3), repeat=length):
            cache = CausalNgramCache(config, cid)
            for token in prefix:
                cache.observe(token, context_id=cid)
            prediction = cache.predict(context_id=cid)
            order, counts, expected = oracle(prefix, config)
            assert prediction.matched_order == order
            assert prediction.counts == counts
            assert prediction.probabilities == expected
            assert all(p > 0 for p in prediction.probabilities)
            assert math.isclose(math.fsum(prediction.probabilities), 1.0, abs_tol=1e-15)
            assert prediction.snapshot.retained_prefix == prefix[-config.window:]
            checked += 1
    rows.append({"check": "exhaustive_independent_oracle_and_positive_normalization",
                 "prefixes": checked, "passed": True})

    stream = [0, 1, 0, 2, 0, 1, 0, 3, 0, 1]
    cache = CausalNgramCache(config, cid)
    for at, token in enumerate(stream):
        before = cache.snapshot()
        sequential = cache.predict(context_id=cid)
        rebuilt = CausalNgramCache(config, cid)
        for known in stream[:at]:
            rebuilt.observe(known, context_id=cid)
        assert sequential == rebuilt.predict(context_id=cid)
        assert before == cache.snapshot()  # query is read-only
        receipt = cache.score_then_observe(token, context_id=cid)
        assert receipt.prediction == sequential
        assert receipt.negative_log_likelihood == -math.log(sequential.probabilities[token])
        assert receipt.prediction.snapshot.observed_count == at
        assert cache.snapshot().observed_count == at + 1
    rows.append({"check": "sequential_prefix_recompute_and_score_before_observe", "passed": True})

    # Complete vectors, not only the probability of a selected true target.
    def predictions(tokens):
        state = CausalNgramCache(config, cid)
        records = []
        for token in tokens:
            records.append(state.predict(context_id=cid))
            state.observe(token, context_id=cid)
        return records
    original = predictions(stream)
    alternatives = 0
    for cut in range(len(stream)):
        for replacement in range(config.vocab_size):
            changed = predictions(stream[:cut] + [replacement] * (len(stream) - cut))
            assert original[:cut + 1] == changed[:cut + 1]
            alternatives += 1
    rows.append({"check": "query_target_and_future_suffix_noninterference_full_vectors",
                 "suffix_variants": alternatives, "passed": True})

    # Different pre-crop histories are erased, even within the same document.
    left = CausalNgramCache(config, cid)
    right = CausalNgramCache(config, cid)
    for token in [0, 1, 0, 1, 0]: left.observe(token, context_id=cid)
    for token in [2, 3, 2, 3, 2]: right.observe(token, context_id=cid)
    crop = ContextID("document-A", "crop-1")
    left.reset(crop); right.reset(crop)
    for token in [0, 1, 0]:
        assert left.predict(context_id=crop) == right.predict(context_id=crop)
        left.observe(token, context_id=crop); right.observe(token, context_id=crop)
    assert left.snapshot().observed_count == 3
    reject(lambda: left.predict(context_id=cid))
    reject(lambda: left.observe(0, context_id=ContextID("document-B", "crop-1")))
    left.reset(ContextID("document-B", "crop-0"))
    assert left.snapshot().retained_prefix == ()
    rows.append({"check": "explicit_crop_document_reset_and_no_precrop_context", "passed": True})

    state = CausalNgramCache(config, cid)
    state.observe(0, context_id=cid)
    ignored = state.score_then_observe(1, context_id=cid, scored=False)
    assert ignored.negative_log_likelihood is None
    assert state.snapshot().retained_prefix == (0, 1)
    state.observe(0, context_id=cid)
    assert state.predict(context_id=cid).counts == (0, 1, 0, 0)
    rows.append({"check": "masked_loss_still_advances_observed_context", "passed": True})

    frozen = state.snapshot()
    reject(lambda: setattr(frozen, "observed_count", 999))
    state.observe(2, context_id=cid)
    assert frozen.retained_prefix == (0, 1, 0)
    assert frozen.state_sha256 != state.snapshot().state_sha256
    before = state.snapshot()
    for invalid in [-1, 4, 1.5, True]:
        reject(lambda: state.score_then_observe(invalid, context_id=cid))
        assert state.snapshot() == before
    reject(lambda: state.score_then_observe(1, context_id=cid, scored=1))
    assert state.snapshot() == before
    reject(lambda: state.reset("untyped identity"))
    assert state.snapshot() == before
    rows.append({"check": "immutable_causal_snapshot_invalid_input_is_atomic", "passed": True})

    for invalid in [dict(vocab_size=1), dict(vocab_size=4, window=0),
                    dict(vocab_size=4, max_order=True), dict(vocab_size=4, smoothing=0),
                    dict(vocab_size=4, smoothing=float("nan"))]:
        reject(lambda: CacheConfig(**invalid))
    for invalid in [("", "crop"), ("document", ""), (1, "crop")]:
        reject(lambda: ContextID(*invalid))
    rational = CacheConfig(4, smoothing=Fraction(1, 2))
    assert type(rational.smoothing) is float
    assert CausalNgramCache(rational, cid).predict(context_id=cid).probabilities == (0.25,) * 4
    reject(lambda: CacheConfig(4, smoothing=Fraction(1, 10 ** 1000)))
    rows.append({"check": "configuration_and_context_validation", "passed": True})

    native = (0.1, 0.2, 0.3, 0.4)
    no_match = CausalNgramCache(config, cid).predict(context_id=cid)
    assert mix_with_native(native, no_match, 1.0) == native
    match = CausalNgramCache(config, cid)
    for token in [0, 1, 0]: match.observe(token, context_id=cid)
    match = match.predict(context_id=cid)
    assert mix_with_native(native, match, 0.0) == native
    assert mix_with_native(native, match, 1.0) == match.probabilities
    mixed = mix_with_native(native, match, 0.25)
    assert all(math.isclose(value, 0.75 * p + 0.25 * q, rel_tol=0, abs_tol=1e-15)
               for value, p, q in zip(mixed, native, match.probabilities))
    assert math.isclose(math.fsum(mixed), 1.0, abs_tol=1e-15)
    for invalid in [-0.1, 1.1, float("nan"), True]:
        reject(lambda: mix_with_native(native, match, invalid))
    reject(lambda: mix_with_native((1, 0, 0, 0), match, 0.5))
    reject(lambda: mix_with_native((0.2,) * 4, match, 0.5))
    rational_mix = mix_with_native(native, match, Fraction(1, 2))
    assert all(type(p) is float for p in rational_mix)
    json.dumps(rational_mix)
    reject(lambda: mix_with_native(native, match, Fraction(10 ** 1000)))
    for malformed in [replace(match, probabilities=(-1.0, 2.0, 0.0, 0.0)),
                      replace(match, probabilities=(0.25,) * 4),
                      replace(match, counts=(0, 0, 0, 0)),
                      replace(match, matched_order=0),
                      replace(match, matches=True),
                      replace(match, probabilities=[0.25] * 4),
                      None]:
        reject(lambda: mix_with_native(native, malformed, 1.0))
    tiny = math.nextafter(0.0, 1.0)
    boundary_cache = CausalNgramCache(CacheConfig(2, smoothing=2 * tiny), cid)
    for token in [0, 1, 0]: boundary_cache.observe(token, context_id=cid)
    boundary_prediction = boundary_cache.predict(context_id=cid)
    assert boundary_prediction.probabilities == (tiny, 1.0)
    # Both inputs are positive and normalized in float64. The old two-product
    # formula returned zero in the first component at weight one half.
    assert 0.5 * tiny + 0.5 * tiny == 0.0
    assert mix_with_native((tiny, 1.0), boundary_prediction, 0.5) == (tiny, 1.0)
    for weight in (0.0, math.nextafter(0.0, 1.0), 0.5, math.nextafter(1.0, 0.0), 1.0):
        result = mix_with_native((1.0, tiny), boundary_prediction, weight)
        assert all(p > 0 for p in result)
        assert math.isclose(math.fsum(result), 1.0, rel_tol=0, abs_tol=1e-15)
    rows.append({"check": "proper_native_mixture_zero_weight_and_no_match_exact", "passed": True})

    return {"schema": "causal-cache-fixture-checks-v1", "scope": "synthetic integer fixtures only",
            "language_model_evaluation": False, "passed": all(row["passed"] for row in rows),
            "checks": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="optional new JSON path; existing files are never overwritten")
    args = parser.parse_args()
    report = run_checks()
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        with Path(args.out).open("x", encoding="utf-8") as output:
            output.write(rendered)
    print(rendered, end="")
