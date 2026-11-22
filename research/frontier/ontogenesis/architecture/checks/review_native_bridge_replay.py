"""Independent replay of the fixed native correction integration recipe.

Checks execution and accounting only. No accuracy threshold, benchmark,
external data, saved receipt, or package mutation. Prints JSON by default.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import native_correction_smoke as bridge
from native_reference import Config
from trainable_native import AdamW, AdamWConfig, TrainableNative


def run():
    report = bridge.run_smoke()
    model = TrainableNative(Config(**report["recipe"]["model_config"]),
                            report["recipe"]["model_seed"])
    optimizer = AdamW(model, AdamWConfig(**report["recipe"]["optimizer_config"]))
    max_ce_error, max_logit_error = 0.0, 0.0
    parameter_deltas, state_checks, predictions = 0, 0, 0
    snapshots = {}
    for event in report["events"]:
        if optimizer.state_digest() != event["checkpoint_before"]["optimizer_state_sha256"]:
            raise AssertionError("Pre-event state mismatch")
        state_checks += 1
        query = np.array([[event["key"][0], 2 + event["key"][1], 4]], dtype=np.int64)
        target = 5 + event["target_value"]
        document = np.concatenate([query, np.array([[target]], dtype=np.int64)], axis=1)
        mask = np.array([[False, False, True]], dtype=bool)
        for receipt in event["optimizer_updates"]:
            logits = model.forward(query)[0, -1]
            maximum = float(np.max(logits))
            direct_ce = maximum + np.log(np.exp(logits - maximum).sum()) - logits[target]
            error = float(abs(direct_ce - receipt["loss_before_nats"]))
            max_ce_error = max(max_ce_error, error)
            if error >= 1e-12:
                raise AssertionError("Direct full-head CE disagrees with the training receipt")
            before = {name: value.copy() for name, value in model.parameters.items()}
            loss, gradients = model.loss_and_grad(document, mask)
            actual_receipt = {"loss_before_nats": loss, **optimizer.step(gradients)}
            if actual_receipt != receipt:
                raise AssertionError("Replayed optimizer receipt mismatch")
            for name, value in model.parameters.items():
                np.testing.assert_array_equal(value - before[name],
                                              optimizer.last_displacements["actual"][name])
                parameter_deltas += 1
        if optimizer.state_digest() != event["checkpoint_after"]["optimizer_state_sha256"]:
            raise AssertionError("Post-event state mismatch")
        state_checks += 1
        snapshot = model.snapshot()
        snapshots[event["event_id"]] = snapshot
        logits = snapshot.forward(query)[0, -1]
        error = float(np.max(np.abs(logits - np.asarray(event["prediction_after_write"]["logits"]))))
        max_logit_error = max(max_logit_error, error)
        if error >= 1e-12:
            raise AssertionError("Original-reference immediate logits disagree")
        predictions += 1
    for phase in report["phase_ends"]:
        prior = [event for event in report["events"] if event["phase"] <= phase["phase"]]
        snapshot = snapshots[prior[-1]["event_id"]] if prior else TrainableNative(
            Config(**report["recipe"]["model_config"]), report["recipe"]["model_seed"]
        ).snapshot()
        for prediction in phase["predictions"]:
            key = prediction["key"]
            query = np.array([[key[0], 2 + key[1], 4]], dtype=np.int64)
            logits = snapshot.forward(query)[0, -1]
            error = float(np.max(np.abs(logits - np.asarray(prediction["logits"]))))
            max_logit_error = max(max_logit_error, error)
            if error >= 1e-12:
                raise AssertionError("Original-reference phase-end logits disagree")
            token = int(np.argmax(logits))
            value = token - 5 if token in (5, 6, 7) else None
            if prediction["raw_argmax_token"] != token or prediction["prediction"] != value:
                raise AssertionError("Full-head decoded prediction mismatch")
            predictions += 1
    if optimizer.state_digest() != report["final_checkpoint"]["optimizer_state_sha256"]:
        raise AssertionError("Final optimizer state mismatch")
    return {
        "status": "PASS", "numpy_version": np.__version__,
        "optimizer_updates_replayed": optimizer.step_number,
        "event_boundary_state_checks": state_checks,
        "tensor_displacement_arrays_checked": parameter_deltas,
        "original_reference_predictions_checked": predictions,
        "maximum_direct_ce_absolute_error": max_ce_error,
        "maximum_snapshot_logit_absolute_error": max_logit_error,
        "counts": report["counts"],
        "implementation_sha256": report["implementation_sha256"],
        "review_replay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Independent native bridge execution/accounting review only; no effectiveness or language benchmark claim.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
