"""Independent finite-difference and behavioral checks for tiny native training.

These are engineering checks using synthetic integers, not language benchmarks.
All filesystem fixtures live in temporary directories. Default output is JSON.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.dont_write_bytecode = True
from native_reference import Config, NativeReference, write_report
from trainable_native import (AdamW, AdamWConfig, TrainableNative,
                              learning_smoke, load_checkpoint)


def _reject(operation, message):
    try:
        operation()
    except (ValueError, TypeError):
        return
    raise AssertionError(message)


def run_checks():
    results = []
    config = Config(vocab_size=8, d_model=8, n_heads=2, d_ff=12,
                    logical_layers=2, unique_layers=2, max_context=8)
    rng = np.random.default_rng(9200)
    documents = rng.integers(0, 8, size=(2, 5))
    mask = np.array([[True, False, True, True], [False, True, True, False]])
    for unique in (2, 1):
        current = replace(config, unique_layers=unique)
        trainable = TrainableNative(current, seed=9201)
        reference = NativeReference(current, seed=9201)
        before = {name: value.tobytes() for name, value in trainable.parameters.items()}
        ours, expected = trainable.forward(documents[:, :-1]), reference.forward(documents[:, :-1])
        np.testing.assert_array_equal(ours, expected)
        results.append({"check": f"exact_initial_forward_unique_{unique}", "passed": True,
                        "max_abs_logit_difference": float(np.max(np.abs(ours - expected)))})

        loss, gradients = trainable.loss_and_grad(documents, mask)
        assert before == {name: value.tobytes() for name, value in trainable.parameters.items()}
        assert loss == trainable.loss(documents, mask)
        differences = []
        tested = []
        step = 1e-6
        # Central differences of the scalar loss are independent of backward code.
        # Two random unit directions cover every distinct learned tensor.
        for name, parameter in trainable.parameters.items():
            original = parameter.copy()
            for _ in range(2):
                direction = rng.normal(size=parameter.shape)
                direction /= np.linalg.norm(direction)
                try:
                    parameter[...] = original + step * direction
                    plus = trainable.loss(documents, mask)
                    parameter[...] = original - step * direction
                    minus = trainable.loss(documents, mask)
                finally:
                    parameter[...] = original
                numeric = (plus - minus) / (2 * step)
                analytic = float(np.sum(gradients[name] * direction))
                np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=3e-8,
                                           err_msg=name)
                differences.append(abs(numeric - analytic))
            tested.append(name)
        results.append({"check": f"directional_finite_differences_unique_{unique}",
                        "tensors_checked": tested, "directions": 2 * len(tested),
                        "maximum_absolute_error": max(differences), "passed": True})

    # An independently untied equivalent graph must sum into the reused block.
    shared = TrainableNative(replace(config, unique_layers=1), seed=9300)
    untied = TrainableNative(config, seed=9301)
    for name, value in untied.parameters.items():
        source_name = name.replace("block_1.", "block_0.")
        value[...] = shared.parameters[source_name]
    np.testing.assert_array_equal(shared.forward(documents[:, :-1]), untied.forward(documents[:, :-1]))
    _, shared_grad = shared.loss_and_grad(documents)
    _, untied_grad = untied.loss_and_grad(documents)
    for name, value in shared_grad.items():
        expected = untied_grad[name]
        if name.startswith("block_0."):
            expected = expected + untied_grad[name.replace("block_0.", "block_1.")]
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
    results.append({"check": "shared_gradient_equals_sum_of_untied_logical_uses", "passed": True})

    # Masking target positions never permits a later suffix to affect an earlier
    # scored token. The token embeddings are tied, so gradients are also checked.
    original_doc = np.array([[0, 1, 2, 3, 4]])
    changed_doc = np.array([[0, 1, 2, 6, 7]])
    early_mask = np.array([[True, True, False, False]])
    left_loss, left_grad = shared.loss_and_grad(original_doc, early_mask)
    right_loss, right_grad = shared.loss_and_grad(changed_doc, early_mask)
    assert left_loss == right_loss
    for name in left_grad:
        np.testing.assert_array_equal(left_grad[name], right_grad[name])
    np.testing.assert_array_equal(shared.forward(original_doc[:, :-1])[:, :3],
                                  shared.forward(changed_doc[:, :-1])[:, :3])
    results.append({"check": "masked_suffix_cannot_change_scored_loss_or_gradient", "passed": True})

    # Different document rows share parameters, not context or attention state.
    all_logits = shared.forward(documents[:, :-1])
    np.testing.assert_allclose(all_logits[:1], shared.forward(documents[:1, :-1]), rtol=1e-12, atol=1e-12)
    results.append({"check": "document_rows_have_independent_attention", "passed": True})

    # Closed-form first step tests clipping, bias correction and decoupled decay.
    model = TrainableNative(config, seed=9400)
    opt_config = AdamWConfig(learning_rate=0.003, weight_decay=0.1, max_grad_norm=0.7)
    optimizer = AdamW(model, opt_config)
    original = {name: array.copy() for name, array in model.parameters.items()}
    gradients = {name: np.ones_like(array) for name, array in original.items()}
    norm = np.sqrt(sum(array.size for array in gradients.values()))
    clipped = opt_config.max_grad_norm / norm
    receipt = optimizer.step(gradients)
    assert abs(receipt["gradient_clip_scale"] - clipped) < 1e-15
    for name, old in original.items():
        known = -opt_config.learning_rate * (
            clipped / (clipped + opt_config.epsilon) + opt_config.weight_decay * old)
        np.testing.assert_allclose(optimizer.last_displacements["proposed"][name], known,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(optimizer.last_displacements["actual"][name],
                                      model.parameters[name] - old)
    results.append({"check": "adamw_first_step_and_full_displacement_known_answer", "passed": True})

    # A bad late dictionary entry cannot partly update earlier parameters.
    before = optimizer.state_digest()
    bad = {name: np.zeros_like(value) for name, value in model.parameters.items()}
    bad["final_norm"][0] = np.nan
    _reject(lambda: optimizer.step(bad), "Nonfinite gradients were accepted")
    assert optimizer.state_digest() == before
    extreme_model = TrainableNative(config)
    extreme = AdamW(extreme_model, AdamWConfig(learning_rate=1e308, weight_decay=0))
    extreme_before = extreme.state_digest()
    _reject(lambda: extreme.step({k: np.ones_like(v) for k, v in extreme_model.parameters.items()}),
            "Overflowed displacement inventory was accepted")
    assert extreme.state_digest() == extreme_before and not extreme.last_displacements
    results.append({"check": "invalid_gradient_or_update_receipt_leaves_all_state_unchanged",
                    "rejections": ["nonfinite gradient", "finite-array displacement norm overflow"],
                    "passed": True})

    # Distinct names and distinct coordinates of a single tensor must never
    # designate overlapping storage. Otherwise assignment can invalidate its
    # displacement receipt; serialization also loses the alias and changes
    # future training. C/F contiguity is the bounded within-tensor contract.
    alias_rejections = 0
    with tempfile.TemporaryDirectory() as directory:
        for variant in ("same_array", "transpose", "overlapping_slices", "embedding_row",
                        "zero_stride", "overlapping_strides"):
            aliased_model = TrainableNative(config, seed=9450)
            aliased_optimizer = AdamW(aliased_model)
            aliased_optimizer.train_step(documents)
            if variant == "same_array":
                aliased_model.parameters["block_0.k"] = aliased_model.parameters["block_0.q"]
            elif variant == "transpose":
                aliased_model.parameters["block_0.k"] = aliased_model.parameters["block_0.q"].T
            elif variant == "overlapping_slices":
                backing = np.arange(config.d_model ** 2 + 1, dtype=np.float64)
                aliased_model.parameters["block_0.q"] = backing[:-1].reshape(config.d_model, config.d_model)
                aliased_model.parameters["block_0.k"] = backing[1:].reshape(config.d_model, config.d_model)
            elif variant == "embedding_row":
                aliased_model.parameters["block_0.norm_attention"] = aliased_model.parameters["embedding"][0]
            else:
                backing = np.full(2 * config.d_model - 1, .125, dtype=np.float64)
                strides = (0, 0) if variant == "zero_stride" else (backing.itemsize, backing.itemsize)
                aliased_model.parameters["block_0.q"] = np.lib.stride_tricks.as_strided(
                    backing, shape=(config.d_model, config.d_model), strides=strides, writeable=True)

            def record_array(array):
                return (id(array), array.dtype.str, array.shape, array.strides, array.tobytes())

            def capture():
                return {
                    "parameters": {k: record_array(v) for k, v in aliased_model.parameters.items()},
                    "first": {k: record_array(v) for k, v in aliased_optimizer.first.items()},
                    "second": {k: record_array(v) for k, v in aliased_optimizer.second.items()},
                    "step": aliased_optimizer.step_number,
                    "last_displacements": {
                        group: {k: record_array(v) for k, v in values.items()}
                        for group, values in aliased_optimizer.last_displacements.items()},
                }

            state_before = capture()
            assert state_before["step"] == 1 and state_before["last_displacements"]
            checkpoint = Path(directory) / (variant + ".npz")
            valid_gradients = {k: np.ones_like(v) for k, v in aliased_model.parameters.items()}
            operations = [
                lambda: aliased_model.forward(documents[:, :-1]),
                lambda: aliased_model.loss_and_grad(documents),
                lambda: aliased_optimizer.step(valid_gradients),
                lambda: aliased_optimizer.train_step(documents),
                lambda: aliased_model.snapshot(),
                lambda: aliased_optimizer.save_checkpoint(checkpoint),
                lambda: aliased_optimizer.state_digest(),
            ]
            for operation in operations:
                try:
                    operation()
                except ValueError as error:
                    expected = ("C- or F-contiguous" if variant in ("zero_stride", "overlapping_strides")
                                else "Overlapping parameter storage")
                    assert expected in str(error)
                    alias_rejections += 1
                else:
                    raise AssertionError(f"Accepted parameter overlap: {variant}")
                assert capture() == state_before
                assert not checkpoint.exists()
    fortran_model = TrainableNative(config, seed=9451)
    expected_logits = fortran_model.forward(documents[:, :-1])
    for name, value in fortran_model.parameters.items():
        fortran_model.parameters[name] = np.asfortranarray(value)
    np.testing.assert_allclose(fortran_model.forward(documents[:, :-1]), expected_logits,
                               rtol=1e-12, atol=1e-12)
    fortran_optimizer = AdamW(fortran_model)
    fortran_before = {name: value.copy() for name, value in fortran_model.parameters.items()}
    fortran_optimizer.step({name: np.ones_like(value) for name, value in fortran_model.parameters.items()})
    for name, old in fortran_before.items():
        np.testing.assert_array_equal(fortran_optimizer.last_displacements["actual"][name],
                                      fortran_model.parameters[name] - old)
    results.append({"check": "parameter_storage_cannot_overlap_between_or_within_tensors",
                    "alias_forms": ["same array", "transpose", "contiguous overlapping slices", "embedding row",
                                    "zero stride", "overlapping nonzero strides"],
                    "operations_rejected_without_state_mutation": alias_rejections,
                    "valid_fortran_contiguous_storage_accepted": True,
                    "passed": True})

    # Training updates create new inference snapshots, while an existing
    # snapshot and its cached prefix retain their old semantics.
    model = TrainableNative(config, seed=9500)
    optimizer = AdamW(model)
    old_snapshot = model.snapshot()
    old_logits = old_snapshot.forward(documents[:, :-1])
    state = old_snapshot.new_state(2)
    for token in documents[:, :2].T:
        _, state = old_snapshot.step(token, state)
    optimizer.train_step(documents)
    new_snapshot = model.snapshot()
    assert new_snapshot.model_version != old_snapshot.model_version
    np.testing.assert_array_equal(old_logits, old_snapshot.forward(documents[:, :-1]))
    _reject(lambda: new_snapshot.step(documents[:, 2], state), "Cross-update stale cache was accepted")
    old_step, _ = old_snapshot.step(documents[:, 2], state)
    np.testing.assert_allclose(old_step, old_logits[:, 2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new_snapshot.forward(documents[:, :-1]),
                               model.forward(documents[:, :-1]), rtol=1e-12, atol=1e-12)
    results.append({"check": "snapshots_are_independent_and_old_caches_rejected_by_new_weights", "passed": True})

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        path = directory / "checkpoint.npz"
        optimizer.train_step(documents[:, ::-1].copy())
        optimizer.save_checkpoint(path)
        loaded_model, loaded_optimizer = load_checkpoint(path)
        assert optimizer.state_digest() == loaded_optimizer.state_digest()
        for next_doc in [documents, documents[:, ::-1].copy(), np.roll(documents, 1, axis=1)]:
            original_receipt = optimizer.train_step(next_doc)
            resumed_receipt = loaded_optimizer.train_step(next_doc)
            assert original_receipt == resumed_receipt
        assert optimizer.state_digest() == loaded_optimizer.state_digest()
        for name in model.parameters:
            np.testing.assert_array_equal(model.parameters[name], loaded_model.parameters[name])
        results.append({"check": "checkpoint_resume_equals_uninterrupted_parameter_and_adam_trajectory",
                        "optimizer_steps": optimizer.step_number, "passed": True})

        original_bytes = path.read_bytes()
        try:
            optimizer.save_checkpoint(path)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Checkpoint was overwritten")
        assert path.read_bytes() == original_bytes
        report_path = directory / "receipt.json"
        report_path.write_text("prior receipt")
        try:
            write_report({"replacement": True}, str(report_path))
        except FileExistsError:
            pass
        else:
            raise AssertionError("Receipt was overwritten")
        assert report_path.read_text() == "prior receipt"
        results.append({"check": "checkpoint_and_receipt_exclusive_output_preserves_existing_file", "passed": True})

        with np.load(path, allow_pickle=False) as archive:
            saved = {name: archive[name].copy() for name in archive.files}
        failures = 0
        for variant in ("checksum", "dtype", "negative_second_moment", "extra_tensor", "duplicate_metadata", "object_array"):
            corrupted = {name: value.copy() for name, value in saved.items()}
            if variant == "checksum":
                corrupted["p::embedding"][0, 0] += 1e-4
            elif variant == "dtype":
                corrupted["p::embedding"] = corrupted["p::embedding"].astype(np.float32)
            elif variant == "negative_second_moment":
                corrupted["v::embedding"][0, 0] = -1
            elif variant == "extra_tensor":
                corrupted["unexpected"] = np.zeros(1)
            elif variant == "duplicate_metadata":
                text = corrupted["metadata"].tobytes().decode()
                corrupted["metadata"] = np.frombuffer((text[:-1] + ',"schema_version":1}').encode(), dtype=np.uint8)
            else:
                corrupted["p::embedding"] = np.array([{"not": "a tensor"}], dtype=object)
            bad_path = directory / (variant + ".npz")
            np.savez(bad_path, **corrupted)
            _reject(lambda: load_checkpoint(bad_path), f"Corrupted {variant} checkpoint accepted")
            failures += 1
        results.append({"check": "malformed_or_tampered_pickle_free_checkpoints_rejected",
                        "corruptions_rejected": failures, "passed": True})

    errors = [
        lambda: shared.loss(np.ones((2, 3), dtype=float)),
        lambda: shared.loss(np.array([[1]])),
        lambda: shared.loss(np.array([[1, 8]])),
        lambda: shared.loss(documents, np.zeros(documents[:, 1:].shape, dtype=bool)),
        lambda: shared.loss(documents, np.ones(documents[:, 1:].shape, dtype=int)),
        lambda: shared.loss(np.zeros((1, 10), dtype=int)),
        lambda: AdamWConfig(beta2=1.0),
        lambda: AdamWConfig(learning_rate=float("nan")),
    ]
    for failure in errors:
        _reject(failure, "Invalid training input accepted")
    results.append({"check": "invalid_document_mask_context_and_optimizer_contracts_rejected",
                    "cases": len(errors), "passed": True})

    model = TrainableNative(config)
    model.config = replace(config, n_heads=4)
    _reject(lambda: model.forward(documents[:, :-1]), "Changed model definition accepted")
    results.append({"check": "model_configuration_is_fixed_during_training", "passed": True})

    smoke = learning_smoke(steps=60)
    for arm in smoke["arms"]:
        assert arm["final_training_ce_nats"] < 0.1 * arm["initial_training_ce_nats"]
    results.append({"check": "both_parameterizations_learn_the_synthetic_training_fixture",
                    "scope": smoke["scope"], "arms": smoke["arms"], "passed": True})
    return {"suite": "Nano LM explicit-backward training contracts",
            "scope": "Synthetic engineering checks only; no held-out language evaluation, architecture win or performance claim.",
            "numpy_version": np.__version__, "passed": len(results), "total": len(results),
            "checks": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Optional NEW JSON receipt")
    args = parser.parse_args()
    if args.out is not None and Path(args.out).exists():
        raise FileExistsError(f"Existing receipt is preserved: {args.out}")
    write_report(run_checks(), args.out)


if __name__ == "__main__":
    main()
