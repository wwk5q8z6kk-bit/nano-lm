"""Fixed CPU integration of native learning and causal revision measurement.

This is an engineering smoke, not a correction-effectiveness experiment or a
language benchmark. The recipe below was fixed before running it. It has no
accuracy target, tuning loop, replay, corpus, network call, GPU or LP solver.
Default stdout only; --out creates a new receipt and refuses to overwrite.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

sys.dont_write_bytecode = True
PROTOTYPE = Path(__file__).resolve().parents[1] / "prototype"
sys.path.insert(0, str(PROTOTYPE))
from native_reference import Config, parameter_inventory
from trainable_native import AdamW, AdamWConfig, TrainableNative
from correction_protocol import (
    Event, Manifest, VisiblePrefix, WorldSpec, deliveries, new_log,
    phase_end_queries, record_after_write, record_phase_end, summarize,
    write_report,
)


# These constants are an engineering recipe fixed before its first execution.
MODEL_CONFIG = Config(vocab_size=8, d_model=8, n_heads=2, d_ff=16,
                      logical_layers=2, unique_layers=1, max_context=3)
OPTIMIZER_CONFIG = AdamWConfig(learning_rate=0.003, beta1=0.9, beta2=0.95,
                              epsilon=1e-8, weight_decay=0.01, max_grad_norm=1.0)
MODEL_SEED = 20261122
STEPS_PER_EVENT = 2
HORIZONS = (0, 1, 5, 10, 20)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def fixed_manifest() -> Manifest:
    """Eight handwritten events; IDs denote assignments, including no-ops."""
    rows = (
        (0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 0, 2),  # initial observations
        (1, 0, 0, 1), (1, 0, 1, 2), (1, 0, 0, 2),  # within-phase supersession
        (2, 0, 0, 2),                               # same-value new revision
        (4, 1, 0, 0),                               # partial long follow-up
    )
    latest, events = {}, []
    for phase, entity, relation, value in rows:
        previous = latest.get((entity, relation))
        event = Event(f"native-smoke-{len(events):02d}", phase, len(events),
                      entity, relation, value,
                      None if previous is None else previous.event_id,
                      None if previous is None else previous.value)
        latest[event.key] = event
        events.append(event)
    spec = WorldSpec(phases=7, entities_per_phase=1, relations=2,
                     shown_relations=2, values=3, assignments_per_phase=2)
    return Manifest(MODEL_SEED, spec, tuple(events))


def query_tokens(key):
    """Two entity IDs, two relation IDs, then the query token; no label input."""
    if (type(key) is not tuple or len(key) != 2
            or any(type(x) is not int or not 0 <= x < 2 for x in key)):
        raise ValueError("This fixed codec accepts entity/relation IDs 0 or 1")
    return (key[0], 2 + key[1], 4)


def parameter_digest(model):
    """Hash model config and actual named tensor payloads, without a snapshot."""
    h = hashlib.sha256(_canonical(asdict(model.config)))
    for name, array in sorted(model.parameters.items()):
        h.update(_canonical({"name": name, "shape": list(array.shape),
                             "dtype": array.dtype.str}) + b"\0")
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


class NativeCorrectionLearner:
    """Learner surface: delivered prefix for writes; key only for predictions.

    This object receives no Manifest, future event list, evaluator or label
    table. It trains only on the current delivered event. Its observed history
    provides a receipt of causal delivery, and is not replayed. This is an API
    contract and audited call path, not a Python security boundary.
    """

    def __init__(self):
        self.model = TrainableNative(MODEL_CONFIG, seed=MODEL_SEED)
        self.optimizer = AdamW(self.model, OPTIMIZER_CONFIG)
        self.observed_events = ()
        self.target_token_exposures = 0
        self.inference_forward_calls = 0

    def checkpoint(self):
        return {
            "model_sha256": parameter_digest(self.model),
            "optimizer_state_sha256": self.optimizer.state_digest(),
            "optimizer_step": self.optimizer.step_number,
            "delivered_prefix_sha256": _digest([asdict(e) for e in self.observed_events]),
            "last_delivered_event_id": (self.observed_events[-1].event_id
                                        if self.observed_events else None),
        }

    def observe(self, visible: VisiblePrefix):
        if (not isinstance(visible, VisiblePrefix)
                or visible.observed_events != self.observed_events + (visible.current_event,)
                or visible.current_event.order != len(self.observed_events)):
            raise ValueError("Exactly the next delivered prefix is required")
        event = visible.current_event
        if event.value not in (0, 1, 2):
            raise ValueError("This fixed codec has three value tokens")
        query = query_tokens(event.key)
        # TrainableNative shifts the document: input is query only; the only
        # scored target is the final value token. No value token is an input.
        document = np.array([query + (5 + event.value,)], dtype=np.int64)
        mask = np.array([[False, False, True]], dtype=bool)
        before = self.checkpoint()
        steps = []
        for _ in range(STEPS_PER_EVENT):
            steps.append(self.optimizer.train_step(document, mask))
            self.target_token_exposures += int(mask.sum())
        self.observed_events = visible.observed_events
        after = self.checkpoint()
        return {
            "event_id": event.event_id, "phase": event.phase, "order": event.order,
            "kind": event.kind, "key": list(event.key), "target_value": event.value,
            "delivered_event_ids": [e.event_id for e in visible.observed_events],
            "visible_prefix_sha256": _digest(asdict(visible)),
            "trained_event_ids": [event.event_id],
            "training_document_tokens": document[0].tolist(),
            "scored_target_mask": mask[0].tolist(),
            "target_token_exposures": STEPS_PER_EVENT,
            "cumulative_target_token_exposures": self.target_token_exposures,
            "checkpoint_before": before, "checkpoint_after": after,
            "optimizer_updates": steps,
        }

    def predict(self, key):
        query = query_tokens(key)
        logits = self.model.forward(np.array([query], dtype=np.int64))[0, -1]
        self.inference_forward_calls += 1
        token = int(np.argmax(logits))
        # Argmax covers the entire native tied vocabulary. Non-value tokens
        # are explicit abstentions, measured as failures by the protocol.
        prediction = token - 5 if 5 <= token <= 7 else None
        return {
            "key": list(key), "query_tokens": list(query),
            "logits": logits.tolist(), "logits_dtype": logits.dtype.str,
            "raw_argmax_token": token, "prediction": prediction,
            "checkpoint": self.checkpoint(),
        }


def recipe():
    return {
        "model_config": asdict(MODEL_CONFIG),
        "optimizer_config": asdict(OPTIMIZER_CONFIG),
        "model_seed": MODEL_SEED, "steps_per_event": STEPS_PER_EVENT,
        "codec": {"entity_tokens": [0, 1], "relation_tokens": [2, 3],
                  "query_token": 4, "value_tokens": [5, 6, 7]},
        "training": "Current event only, one document, only final next-value target scored",
        "decoding": "Full-vocabulary argmax; lowest index wins ties; non-value token abstains",
        "replay": "None; no random stream consumed after model initialization",
        "censor_policy": "new_revision", "horizons": list(HORIZONS),
        "horizon_zero": "Phase end; immediately-after-write acquisition reported separately",
    }


def run_smoke(manifest=None):
    """Coordinator owns the manifest; learner calls never receive future data.

    The optional manifest exists for causal-invariance tests using the same
    fixed codec and recipe. The CLI always runs fixed_manifest().
    """
    manifest = fixed_manifest() if manifest is None else manifest
    if not isinstance(manifest, Manifest):
        raise ValueError("An immutable protocol Manifest is required")
    if (len(manifest.events) > 16 or manifest.spec.phases > 32
            or manifest.spec.values != 3):
        raise ValueError("Only a tiny three-value integration fixture is supported")
    for event in manifest.events:
        query_tokens(event.key)
    learner = NativeCorrectionLearner()
    initial = learner.checkpoint()
    log = new_log(manifest, censor_policy="new_revision")
    views = iter(deliveries(manifest))
    pending = next(views, None)
    event_receipts, phase_receipts = [], []
    for phase in range(manifest.spec.phases):
        while pending is not None and pending.current_event.phase == phase:
            receipt = learner.observe(pending)
            event = pending.current_event
            prediction = learner.predict(event.key)
            receipt["prediction_after_write"] = prediction
            event_receipts.append(receipt)
            if event.previous_revision is not None:
                log = record_after_write(manifest, log, event.event_id,
                                         prediction["prediction"])
            pending = next(views, None)
        # The evaluator extracts only keys; targets stay outside the learner.
        queries = phase_end_queries(manifest, phase)
        predictions = [learner.predict(key) for key in queries]
        log = record_phase_end(manifest, log, phase,
                               {tuple(p["key"]): p["prediction"] for p in predictions})
        phase_receipts.append({"phase": phase, "checkpoint": learner.checkpoint(),
                               "predictions": predictions})
    final = learner.checkpoint()
    exact_steps = sum(len(row["optimizer_updates"]) for row in event_receipts)
    source_paths = {
        "bridge": Path(__file__), "measurement": Path(__file__).with_name("correction_protocol.py"),
        "trainable": PROTOTYPE / "trainable_native.py", "native": PROTOTYPE / "native_reference.py",
    }
    return {
        "schema_version": 1, "instrument": "native-correction-integration-smoke-v1",
        "scope": "Engineering integration only; no effectiveness, benchmark or G03b pilot claim",
        "fixture_origin": "Handwritten bounded synthetic revision sequence; not the historical pilot stream",
        "optimizer_method": "AdamW; no LP solver or feasibility claim",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "dtype": "float64", "device": "CPU"},
        "implementation_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for name, path in source_paths.items()},
        "recipe": recipe(), "recipe_sha256": _digest(recipe()),
        "model_config_sha256": _digest(asdict(MODEL_CONFIG)),
        "parameter_inventory": parameter_inventory(MODEL_CONFIG),
        "manifest": json.loads(manifest.canonical_bytes()), "manifest_sha256": manifest.sha256,
        "initial_checkpoint": initial, "final_checkpoint": final,
        "counts": {
            "delivered_events": len(event_receipts),
            "initial_observations": sum(e.previous_revision is None for e in manifest.events),
            "assignment_events": sum(e.previous_revision is not None for e in manifest.events),
            "optimizer_steps": exact_steps,
            "training_documents": exact_steps, "training_input_tokens": 3 * exact_steps,
            "scored_target_token_exposures": learner.target_token_exposures,
            "masked_out_target_positions": 2 * exact_steps,
            "inference_forward_calls": learner.inference_forward_calls,
            "evaluation_query_tokens": 3 * learner.inference_forward_calls,
        },
        "events": event_receipts, "phase_ends": phase_receipts,
        "evaluation_log": asdict(log), "metrics": summarize(manifest, log, HORIZONS),
        "limitations": [
            "One fixed synthetic stream and seed; no statistical effectiveness inference.",
            "Initial observations are trained and queried, but correction denominators use assignments only.",
            "Same-value assignments consume training exposure and censor prior revisions under new_revision.",
            "No accuracy threshold, hyperparameter selection, external label lookup or replay is used.",
            "Checkpoint hashes identify exact in-memory states; checkpoint payloads are not saved by this CLI.",
            "Bitwise determinism is checked in this runtime; different NumPy/BLAS/platform builds can differ.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Optional NEW JSON receipt; refuses to overwrite")
    args = parser.parse_args()
    write_report(run_smoke(), args.out)


if __name__ == "__main__":
    main()
