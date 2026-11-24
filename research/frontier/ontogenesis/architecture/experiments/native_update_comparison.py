"""Fixed exploratory native update-allocation comparison on synthetic facts.

The adjacent protocol was written before first execution. This is not a
confirmatory G03b experiment, a language benchmark or an efficiency claim.
Defaults print JSON. --out exclusively creates a new receipt.
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))
from native_reference import Config, parameter_inventory
from trainable_native import AdamW, AdamWConfig, TrainableNative
from correction_protocol import (
    Manifest, VisiblePrefix, WorldSpec, deliveries, generate_manifest, new_log,
    phase_end_queries, record_after_write, record_phase_end, summarize,
    write_report,
)

PROTOCOL_PATH = Path(__file__).with_name("native_update_comparison_protocol.json")
PROTOCOL_BYTES = PROTOCOL_PATH.read_bytes()
PROTOCOL = json.loads(PROTOCOL_BYTES)
PROTOCOL_SHA256 = hashlib.sha256(PROTOCOL_BYTES).hexdigest()
EXPECTED_PROTOCOL_SHA256 = "4cd9bdabedd2b507663f5d5d368dab34c037c2b078b386d872d9238b23f18425"
SEEDS = tuple(PROTOCOL["seeds"])
ARM_NAMES = tuple(PROTOCOL["arms"])
MODEL_CONFIG = Config(**PROTOCOL["model"])
OPTIMIZER_CONFIG = AdamWConfig(**PROTOCOL["optimizer"])
WORLD_SPEC = WorldSpec(**PROTOCOL["world"])
HORIZONS = tuple(PROTOCOL["horizons"])


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(values):
    h = hashlib.sha256()
    for name, array in sorted(values.items()):
        h.update(_canonical({"name": name, "shape": list(array.shape),
                             "dtype": array.dtype.str}) + b"\0")
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


def parameter_digest(model):
    return _digest({"config": asdict(model.config),
                    "tensors": _array_digest(model.parameters)})


def query_tokens(key):
    if (type(key) is not tuple or len(key) != 2
            or any(type(n) is not int for n in key)
            or not 0 <= key[0] < 14 or not 0 <= key[1] < 3):
        raise ValueError("Fixed codec requires entity0..13 and relation0..2")
    return (key[0], 14 + key[1], 17)


def _document(event):
    if type(event.value) is not int or not 0 <= event.value < 4:
        raise ValueError("Fixed codec requires value0..3")
    return np.array([query_tokens(event.key) + (18 + event.value,)], dtype=np.int64)


def _metric(predictions, targets):
    """Complete fixed cohort; explicit abstentions count as incorrect."""
    if set(predictions) != set(targets):
        raise ValueError("Metric needs exactly one prediction for every target")
    correct = sum(predictions[key] == value for key, value in targets.items())
    n = len(targets)
    return {"correct": correct, "total": n,
            "accuracy": correct / n if n else None,
            "missing": 0}


def clone_training_state(source):
    """Copy the common weights and complete Adam state without resetting it."""
    model = TrainableNative(source.model.config, seed=0)
    for name, array in source.model.parameters.items():
        model.parameters[name][...] = array
    optimizer = AdamW(model, source.config)
    optimizer.first = {name: array.copy() for name, array in source.first.items()}
    optimizer.second = {name: array.copy() for name, array in source.second.items()}
    optimizer.step_number = source.step_number
    if optimizer.state_digest() != source.state_digest():
        raise AssertionError("Branch state differs from common weights/moments/step")
    return model, optimizer


def _checkpoint(model, optimizer, observed_events):
    return {
        "model_sha256": parameter_digest(model),
        "optimizer_state_sha256": optimizer.state_digest(),
        "optimizer_step": optimizer.step_number,
        "delivered_prefix_sha256": _digest([asdict(e) for e in observed_events]),
        "last_delivered_event_id": observed_events[-1].event_id if observed_events else None,
    }


def _step(optimizer, event, role):
    document = _document(event)
    mask = np.array([[False, False, True]], dtype=bool)
    report = optimizer.train_step(document, mask)
    return {
        "role": role, "source_event_id": event.event_id, "source_event_order": event.order,
        "key": list(event.key), "target_value": event.value,
        "training_document_tokens": document[0].tolist(),
        "scored_target_mask": mask[0].tolist(),
        "optimizer": report,
        "actual_tensor_displacements_sha256":
            _array_digest(optimizer.last_displacements["actual"]),
    }


def _latest(events):
    return {event.key: event for event in events}


def select_replay(visible):
    """Latest visible labels only, excluding the current key when possible."""
    current = visible.current_event
    candidates = [event for key, event in _latest(visible.observed_events).items()
                  if key != current.key]
    same_entity = [event for event in candidates if event.entity == current.entity]
    if same_entity:
        pool, category = same_entity, "same_entity_other_key"
    elif candidates:
        pool, category = candidates, "other_entity_fallback"
    else:
        return current, "current_event_fallback_no_other_key"
    return min(pool, key=lambda event: (event.order, event.key)), category


class NativeAllocationLearner:
    """Updates see delivered prefixes only; inference receives query keys only."""

    def __init__(self, common_optimizer, common_events, arm):
        if arm not in ARM_NAMES:
            raise ValueError("Unknown arm")
        self.model, self.optimizer = clone_training_state(common_optimizer)
        self.arm = arm
        self.observed_events = tuple(common_events)
        self.counts = {
            "post_prefix_events": 0, "optimizer_steps": 0,
            "current_target_presentations": 0, "replay_target_presentations": 0,
            "scored_target_presentations": 0, "training_documents": 0,
            "training_input_tokens": 0, "masked_out_target_positions": 0,
            "inference_forward_calls": 0, "evaluation_query_tokens": 0,
            "same_entity_replay_steps": 0, "other_entity_replay_steps": 0,
            "current_event_replay_fallback_steps": 0,
        }
        self.max_replay_payload_bytes = 0

    def checkpoint(self):
        return _checkpoint(self.model, self.optimizer, self.observed_events)

    def observe(self, visible):
        if (not isinstance(visible, VisiblePrefix)
                or visible.observed_events != self.observed_events + (visible.current_event,)
                or visible.current_event.order != len(self.observed_events)):
            raise ValueError("Exactly the next causal prefix is required")
        expected_targets = tuple((event.entity, event.relation, event.value, event.event_id)
                                 for _, event in sorted(_latest(visible.observed_events).items()))
        if visible.current_targets != expected_targets:
            raise ValueError("Current target table must equal the delivered prefix")
        event = visible.current_event
        _document(event)
        before = self.checkpoint()
        rule = PROTOCOL["arms"][self.arm]
        steps = []
        for _ in range(rule["current_steps_per_event"]):
            steps.append(_step(self.optimizer, event, "current"))
        replay_category, replay_event = None, None
        if rule["replay_steps_per_event"]:
            replay_event, replay_category = select_replay(visible)
            for _ in range(rule["replay_steps_per_event"]):
                steps.append(_step(self.optimizer, replay_event, "replay"))
            category_count = {
                "same_entity_other_key": "same_entity_replay_steps",
                "other_entity_fallback": "other_entity_replay_steps",
                "current_event_fallback_no_other_key": "current_event_replay_fallback_steps",
            }[replay_category]
            self.counts[category_count] += rule["replay_steps_per_event"]
            # Serialized semantic payload, not Python-object or peak RAM bytes.
            payload = [{"key": list(e.key), "value": e.value, "event_id": e.event_id,
                        "order": e.order} for _, e in sorted(_latest(visible.observed_events).items())]
            self.max_replay_payload_bytes = max(self.max_replay_payload_bytes, len(_canonical(payload)))
        self.observed_events = visible.observed_events
        self.counts["post_prefix_events"] += 1
        self.counts["optimizer_steps"] += len(steps)
        self.counts["current_target_presentations"] += rule["current_steps_per_event"]
        self.counts["replay_target_presentations"] += rule["replay_steps_per_event"]
        self.counts["scored_target_presentations"] += len(steps)
        self.counts["training_documents"] += len(steps)
        self.counts["training_input_tokens"] += 3 * len(steps)
        self.counts["masked_out_target_positions"] += 2 * len(steps)
        return {
            "event_id": event.event_id, "phase": event.phase, "order": event.order,
            "kind": event.kind, "key": list(event.key), "target_value": event.value,
            "checkpoint_before": before, "checkpoint_after": self.checkpoint(),
            "visible_prefix_sha256": _digest(asdict(visible)),
            "delivered_event_ids": [e.event_id for e in visible.observed_events],
            "current_target_presentations": rule["current_steps_per_event"],
            "replay_target_presentations": rule["replay_steps_per_event"],
            "replay_category": replay_category,
            "replay_event_id": replay_event.event_id if replay_event is not None else None,
            "optimizer_updates": steps,
        }

    def predict(self, key):
        query = query_tokens(key)
        logits = self.model.forward(np.array([query], dtype=np.int64))[0, -1]
        token = int(np.argmax(logits))
        self.counts["inference_forward_calls"] += 1
        self.counts["evaluation_query_tokens"] += len(query)
        return {
            "key": list(key), "query_tokens": list(query), "logits": logits.tolist(),
            "raw_argmax_token": token,
            "prediction": token - 18 if 18 <= token < 22 else None,
            "model_sha256": parameter_digest(self.model),
            "optimizer_step": self.optimizer.step_number,
        }


def negative_control_targets(manifest):
    """Coordinator-only labels; never passed to a learner or training RNG."""
    observed = {event.key for event in manifest.events}
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([manifest.seed, 5129033])))
    return {(entity, relation): int(rng.integers(4))
            for entity in range(14) for relation in range(3)
            if (entity, relation) not in observed}


def _common_training(seed, common_events):
    model = TrainableNative(MODEL_CONFIG, seed=seed + 10000)
    optimizer = AdamW(model, OPTIMIZER_CONFIG)
    documents = np.concatenate([_document(event) for event in common_events], axis=0)
    mask = np.tile(np.array([[False, False, True]], dtype=bool), (len(common_events), 1))
    initial = _checkpoint(model, optimizer, ())
    initial_loss = model.loss(documents, mask)
    updates = []
    for _ in range(PROTOCOL["common_prefix"]["steps"]):
        updates.append(optimizer.train_step(documents, mask))
    return optimizer, {
        "visible_event_ids": [e.event_id for e in common_events],
        "training_documents": documents.tolist(), "scored_target_mask": mask.tolist(),
        "initial_checkpoint": initial,
        "branch_checkpoint": _checkpoint(model, optimizer, common_events),
        "initial_training_ce_nats": initial_loss, "final_training_ce_nats": model.loss(documents, mask),
        "optimizer_updates": updates,
        "counts": {
            "executed_once_per_seed_optimizer_steps": 40,
            "scored_target_presentations": 160,
            "training_document_presentations": 160,
            "training_input_tokens": 480, "masked_out_target_positions": 320,
            "additional_diagnostic_loss_forward_calls": 2,
            "diagnostic_loss_document_queries": 8,
        },
    }


def run_seed(seed, manifest=None, arm_names=ARM_NAMES):
    """Fixed recipe; optional manifest/arm subset exists for causal tests."""
    if PROTOCOL_SHA256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Frozen recipe changed; create a new experiment identity")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only the frozen seed list is supported")
    if (type(arm_names) is not tuple or not arm_names or len(set(arm_names)) != len(arm_names)
            or any(name not in ARM_NAMES for name in arm_names)):
        raise ValueError("Explicit distinct known arm tuple required")
    manifest = generate_manifest(seed, WORLD_SPEC) if manifest is None else manifest
    if not isinstance(manifest, Manifest) or manifest.seed != seed or manifest.spec != WORLD_SPEC:
        raise ValueError("Manifest must match the fixed seed and world dimensions")
    common_events = tuple(event for event in manifest.events if event.phase == 0)
    if len(manifest.events) != 40 or len(common_events) != 4:
        raise ValueError("This comparison requires40 events and4 common-prefix observations")
    if any(event.previous_revision is not None for event in common_events):
        raise ValueError("Common prefix must contain initial observations")
    heldout = negative_control_targets(manifest)
    if (len(heldout) != 14
            or any(sum(key[0] == entity for key in heldout) != 1 for entity in range(14))):
        raise ValueError("Exactly one permanently unobserved relation per entity required")
    common_optimizer, common_receipt = _common_training(seed, common_events)
    result = {
        "seed": seed, "model_seed": seed + 10000,
        "manifest": json.loads(manifest.canonical_bytes()), "manifest_sha256": manifest.sha256,
        "negative_control": {
            "scope": "Permanently unobserved random labels, coordinator-only; no generalization claim.",
            "targets": [{"key": list(key), "value": value} for key, value in sorted(heldout.items())],
            "targets_sha256": _digest([[*key, value] for key, value in sorted(heldout.items())]),
        },
        "common_prefix_training": common_receipt, "arms": {},
    }
    common_targets = {event.key: event.value for event in common_events}
    for arm in arm_names:
        learner = NativeAllocationLearner(common_optimizer, common_events, arm)
        branch = learner.checkpoint()
        if branch != common_receipt["branch_checkpoint"]:
            raise AssertionError("Arm did not start at the exact common checkpoint")
        log = new_log(manifest, "new_revision")
        views = iter(deliveries(manifest))
        pending = next(views, None)
        while pending is not None and pending.current_event.phase == 0:
            pending = next(views, None)
        events, phases = [], []
        for phase in range(WORLD_SPEC.phases):
            while pending is not None and pending.current_event.phase == phase:
                receipt = learner.observe(pending)
                prediction = learner.predict(pending.current_event.key)
                receipt["prediction_after_write"] = prediction
                events.append(receipt)
                if pending.current_event.previous_revision is not None:
                    log = record_after_write(manifest, log, pending.current_event.event_id,
                                             prediction["prediction"])
                pending = next(views, None)
            current = _latest(learner.observed_events)
            introduced_entities = {entity for entity, _ in current}
            visible_heldout = {key: value for key, value in heldout.items()
                               if key[0] in introduced_entities}
            # Scoring targets stay in this coordinator; learner receives keys only.
            known_rows = [learner.predict(key) for key in sorted(current)]
            hidden_rows = [learner.predict(key) for key in sorted(visible_heldout)]
            known_predictions = {tuple(row["key"]): row["prediction"] for row in known_rows}
            hidden_predictions = {tuple(row["key"]): row["prediction"] for row in hidden_rows}
            log = record_phase_end(manifest, log, phase,
                                   {key: known_predictions[key] for key in phase_end_queries(manifest, phase)})
            unrevised = {key: value for key, value in common_targets.items()
                         if current[key].previous_revision is None}
            previous = {key: event.value for key, event in current.items() if event.phase < phase}
            phases.append({
                "phase": phase, "checkpoint": learner.checkpoint(),
                "known_predictions": known_rows, "negative_control_predictions": hidden_rows,
                "current_observed_facts": _metric(known_predictions, {k: e.value for k, e in current.items()}),
                "previously_observed_current_facts": _metric({k: known_predictions[k] for k in previous}, previous),
                "never_revised_common_facts": _metric({k: known_predictions[k] for k in unrevised}, unrevised),
                "unobserved_random_label_control": _metric(hidden_predictions, visible_heldout),
            })
        new_events = [row for row in events if row["kind"] == "initial_observation"]
        new_correct = sum(row["prediction_after_write"]["prediction"] == row["target_value"]
                          for row in new_events)
        counts = dict(learner.counts)
        counts.update({
            "parameter_array_bytes": sum(array.nbytes for array in learner.model.parameters.values()),
            "adam_moment_array_bytes": sum(array.nbytes for array in learner.optimizer.first.values())
                                       + sum(array.nbytes for array in learner.optimizer.second.values()),
            "maximum_algorithmic_replay_payload_json_bytes": learner.max_replay_payload_bytes,
            "instrumentation_delivered_history_json_bytes":
                len(_canonical([asdict(e) for e in learner.observed_events])),
        })
        result["arms"][arm] = {
            "branch_checkpoint": branch, "final_checkpoint": learner.checkpoint(),
            "counts": counts, "events": events, "phase_ends": phases,
            "evaluation_log": asdict(log), "correction_metrics": summarize(manifest, log, HORIZONS),
            "post_prefix_new_fact_acquisition": {
                "correct": new_correct, "total": len(new_events), "missing": 0,
                "accuracy": new_correct / len(new_events) if new_events else None},
            "memory_scope": "Array bytes and canonical replay/history payload sizes only; excludes Python overhead, temporary activations, kernel workspace and allocator peaks. Every arm retains audit history; replay payload is an algorithmic inventory, not incremental measured RAM.",
        }
    return result


def _summary(values):
    available = [value for value in values if value is not None]
    complete = len(available) == len(values)
    return {"values": values, "available_seed_count": len(available),
            "mean": float(np.mean(available)) if available and complete else None,
            "min": min(available) if available and complete else None,
            "max": max(available) if available and complete else None,
            "complete": complete}


def _endpoints(arm):
    cohort = arm["correction_metrics"]["cohorts"]["all_assignments"]
    last = arm["phase_ends"][-1]
    return {
        "assignment_immediate_accuracy": cohort["immediately_after_write"]["accuracy"],
        "assignment_horizon1_accuracy": cohort["phase_end_horizons"]["1"]["accuracy"],
        "new_fact_immediate_accuracy": arm["post_prefix_new_fact_acquisition"]["accuracy"],
        "final_current_observed_accuracy": last["current_observed_facts"]["accuracy"],
        "final_previous_current_accuracy": last["previously_observed_current_facts"]["accuracy"],
        "final_never_revised_common_accuracy": last["never_revised_common_facts"]["accuracy"],
        "final_unobserved_random_label_accuracy": last["unobserved_random_label_control"]["accuracy"],
    }


def run_comparison():
    runs = [run_seed(seed) for seed in SEEDS]
    metrics = tuple(_endpoints(runs[0]["arms"][ARM_NAMES[0]]))
    arms = {arm: {metric: _summary([_endpoints(run["arms"][arm])[metric] for run in runs])
                  for metric in metrics} for arm in ARM_NAMES}
    paired = {}
    for arm in ("replay_matched_total", "replay_additive", "frozen"):
        paired[arm + "_minus_current_only"] = {
            metric: _summary([
                None if (left := _endpoints(run["arms"][arm])[metric]) is None
                     or (right := _endpoints(run["arms"]["current_only"])[metric]) is None
                else left - right for run in runs]) for metric in metrics}
    sources = {
        "comparison": Path(__file__), "protocol": PROTOCOL_PATH,
        "measurement": Path(__file__).with_name("correction_protocol.py"),
        "trainable": ROOT / "prototype/trainable_native.py",
        "native": ROOT / "prototype/native_reference.py",
    }
    return {
        "schema_version": 1, "experiment_id": PROTOCOL["experiment_id"],
        "scope": PROTOCOL["classification"],
        "protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA256,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "dtype": "float64", "device": "CPU"},
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in sources.items()},
        "parameter_inventory": parameter_inventory(MODEL_CONFIG),
        "runs": runs, "descriptive_seed_summary": arms,
        "paired_seed_differences": paired,
        "total_executed": {
            "common_pretraining_optimizer_steps": 120,
            "post_branch_optimizer_steps": sum(arm["counts"]["optimizer_steps"]
                                              for run in runs for arm in run["arms"].values()),
            "common_pretraining_scored_targets": 480,
            "post_branch_scored_targets": sum(arm["counts"]["scored_target_presentations"]
                                              for run in runs for arm in run["arms"].values()),
        },
        "claim_boundaries": PROTOCOL["limits"] + [
            "Seed summaries are descriptive; undefined seed endpoints prevent a complete aggregate.",
            "Update/target matching is not a measured FLOP, latency, energy or peak-memory comparison.",
            "Arm-level common pretraining is shared once per seed, not re-executed four times.",
            "No failed acquisition or unfavorable seed is filtered from primary results.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional NEW receipt; existing output is preserved")
    args = parser.parse_args()
    if args.out is not None and (args.out.exists() or args.out.is_symlink()):
        raise FileExistsError(f"Existing receipt is preserved: {args.out}")
    write_report(run_comparison(), args.out)


if __name__ == "__main__":
    main()
