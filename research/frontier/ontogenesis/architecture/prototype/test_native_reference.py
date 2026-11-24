"""Behavioral checks for the tiny forward reference, not model-quality tests."""
from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

# Default runs do not create bytecode artifacts inside the research package.
sys.dont_write_bytecode = True
from native_reference import Config, NativeReference, cache_bytes, cross_entropy, parameter_inventory, write_report


def run_checks() -> dict:
    rng = np.random.default_rng(7501)
    tokens = rng.integers(0, 32, size=(2, 11), dtype=np.int64)
    outcomes = []
    for unique in (4, 2):
        config = Config(unique_layers=unique)
        model = NativeReference(config, seed=7502)
        whole = model.forward(tokens)
        state = model.new_state(batch_size=2)
        incremental = []
        for token in tokens.T:
            prior_length = state.length
            prior_bytes = [(cache.key.tobytes(), cache.value.tobytes()) for cache in state.layers]
            output, next_state = model.step(token, state)
            assert state.length == prior_length
            assert prior_bytes == [(cache.key.tobytes(), cache.value.tobytes()) for cache in state.layers]
            incremental.append(output)
            state = next_state
        decoded = np.stack(incremental, axis=1)
        error = float(np.max(np.abs(whole - decoded)))
        np.testing.assert_allclose(whole, decoded, rtol=1e-10, atol=1e-10)
        outcomes.append({"check": f"full_prefix_equals_incremental_unique_{unique}",
                         "max_abs_logit_error": error, "passed": True})

        changed = tokens.copy()
        changed[:, 6:] = (changed[:, 6:] + 7) % config.vocab_size
        np.testing.assert_array_equal(whole[:, :6], model.forward(changed)[:, :6])
        np.testing.assert_allclose(whole[:, :6], model.forward(tokens[:, :6]), rtol=1e-10, atol=1e-10)
        outcomes.append({"check": f"future_suffix_noninterference_unique_{unique}", "passed": True})

        # A separate fresh request must not inherit the first request's cache.
        fresh = model.new_state(batch_size=1)
        request_b = np.array([[4, 7, 19]], dtype=np.int64)
        request_b_steps = []
        for token in request_b.T:
            out, fresh = model.step(token, fresh)
            request_b_steps.append(out)
        np.testing.assert_allclose(np.stack(request_b_steps, axis=1),
                                   model.forward(request_b), rtol=1e-10, atol=1e-10)
        assert state.length == tokens.shape[1]
        np.testing.assert_allclose(whole[0:1], model.forward(tokens[0:1]), rtol=1e-10, atol=1e-10)
        outcomes.append({"check": f"fresh_request_and_batch_isolation_unique_{unique}", "passed": True})

        arrays = [array for _, array in model.named_parameters()]
        assert len({id(array) for array in arrays}) == len(arrays)
        assert sum(array.size for array in arrays) == parameter_inventory(config)["total"]
        assert len(state.layers) == config.logical_layers
        assert len({id(cache.key) for cache in state.layers}) == config.logical_layers
        assert sum(cache.key.nbytes + cache.value.nbytes for cache in state.layers) == cache_bytes(
            config, tokens.shape[0], tokens.shape[1], np.dtype(np.float64).itemsize)
        if unique < config.logical_layers:
            assert not np.allclose(state.layers[0].key, state.layers[unique].key)
        outcomes.append({"check": f"unique_parameters_distinct_logical_caches_unique_{unique}",
                         "parameters": parameter_inventory(config)["total"],
                         "kv_bytes": cache_bytes(config, 2, 11, 8), "passed": True})

    nominal = Config(vocab_size=32768, d_model=256, n_heads=4, d_ff=768,
                     logical_layers=8, unique_layers=8, max_context=1024)
    # Arithmetic only: no nominal-size model is allocated.
    assert parameter_inventory(nominal)["total"] == 15_208_704
    assert parameter_inventory(replace(nominal, unique_layers=4))["total"] == 11_798_784
    outcomes.append({"check": "nominal_parameter_counts_without_allocation", "passed": True})

    uniform_logits = np.zeros((1, 2, 32))
    targets = np.array([[2, 3]])
    mask = np.array([[True, False]])
    assert abs(cross_entropy(uniform_logits, targets, mask) - math.log(32)) < 1e-12
    uniform_logits[0, 1, 0] = 100
    assert abs(cross_entropy(uniform_logits, targets, mask) - math.log(32)) < 1e-12
    outcomes.append({"check": "masked_cross_entropy_known_answer", "passed": True})

    model = NativeReference(Config(max_context=1))
    _, state = model.step(np.array([1]), model.new_state(1))
    rejected = 0
    for tokens_invalid, state_invalid in [
        (np.array([2]), state),
        (np.array([32]), model.new_state(1)),
        (np.array([1]), NativeReference(Config(max_context=1), seed=99).new_state(1)),
    ]:
        try:
            model.step(tokens_invalid, state_invalid)
        except ValueError:
            rejected += 1
    assert rejected == 3
    outcomes.append({"check": "bounds_token_and_checkpoint_mismatch_rejected", "passed": True})

    # Replacing a parameter bypasses a NumPy array's read-only flag. The
    # initialized version must not silently bless the resulting stale cache.
    changed_model = NativeReference(Config(), seed=7)
    changed_tokens = np.array([[1, 2, 3, 4]])
    changed_state = changed_model.new_state(1)
    for token in changed_tokens[:, :3].T:
        _, changed_state = changed_model.step(token, changed_state)
    changed_model.blocks[0]["k"] = np.zeros_like(changed_model.blocks[0]["k"])
    mutation_rejections = 0
    for operation in [lambda: changed_model.step(changed_tokens[:, 3], changed_state),
                      lambda: changed_model.new_state(1),
                      lambda: changed_model.forward(changed_tokens)]:
        try:
            operation()
        except ValueError:
            mutation_rejections += 1
    assert mutation_rejections == 3

    invalid_dimensions = [(1, 1.5, 8), (True, 1, 8), (1, -1, 8), (1, 1, 0)]
    dimension_rejections = 0
    for batch, prefix, element in invalid_dimensions:
        try:
            cache_bytes(Config(), batch, prefix, element)
        except ValueError:
            dimension_rejections += 1
    assert dimension_rejections == len(invalid_dimensions)
    assert cache_bytes(Config(), 1, 0, 8) == 0

    # For [0,255], target0 has CE approximately255 nats. Unsigned integer
    # subtraction instead wrapped around and previously reported about0.3135.
    integer_logits = np.array([[[0, 255]]], dtype=np.uint8)
    integer_targets, integer_mask = np.array([[0]]), np.array([[True]])
    try:
        cross_entropy(integer_logits, integer_targets, integer_mask)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfloating logits silently changed the loss")
    expected_ce = cross_entropy(integer_logits.astype(np.float64), integer_targets, integer_mask)
    assert abs(expected_ce - 255.0) < 1e-12
    outcomes.append({"check": "parameter_mutation_and_invalid_numeric_contracts_rejected",
                     "mutation_operations_rejected": mutation_rejections,
                     "invalid_cache_dimensions_rejected": dimension_rejections,
                     "nonfloating_logits_rejected": True,
                     "known_float_ce_nats": expected_ce, "passed": True})

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "frozen.json"
        path.write_text("frozen", encoding="utf-8")
        try:
            write_report({"would_overwrite": True}, str(path))
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing output was overwritten")
        assert path.read_text(encoding="utf-8") == "frozen"
    outcomes.append({"check": "exclusive_output_preserves_existing_file", "passed": True})
    return {
        "suite": "Nano LM forward reference behavioral contracts",
        "scope": "Small deterministic correctness examples, not training, formal verification or quality benchmarks",
        "numpy_version": np.__version__,
        "passed": len(outcomes),
        "total": len(outcomes),
        "checks": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Optional NEW JSON path; parent must exist")
    args = parser.parse_args()
    write_report(run_checks(), args.out)


if __name__ == "__main__":
    main()
