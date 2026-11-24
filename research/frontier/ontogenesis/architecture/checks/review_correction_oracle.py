"""Independent event-scan review of correction_protocol's cohort summaries.

Deterministic synthetic measurement fixtures only: no model, data benchmark,
architecture comparison, or external writes. Default execution prints a receipt.
The expected rows scan future event identities directly; they do not call the
implementation's successor or metric helpers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
from correction_protocol import (
    Event, Manifest, WorldSpec, new_log, phase_end_queries, record_after_write,
    record_phase_end, summarize,
)

SEED = 192734
TRIALS_PER_POLICY = 160
HORIZONS = (0, 1, 2, 5)


def expected_immediate(cohort, initial):
    row = {
        "events": len(cohort),
        "measured": sum(event.event_id in initial for event in cohort),
        "correct": sum(event.event_id in initial and initial[event.event_id] == event.value
                       for event in cohort),
    }
    row["missing"] = row["events"] - row["measured"]
    row["accuracy"] = row["correct"] / len(cohort) if cohort and not row["missing"] else None
    return row


def expected_horizon(events, cohort, initial, snapshots, policy, through, horizon):
    row = {
        "total_events": len(cohort), "eligible": 0, "censored": 0,
        "not_yet_due": 0, "measured": 0, "missing_measurement": 0,
        "correct": 0, "failed_initial_acquisition_eligible": 0,
        "initial_acquisition_unknown_eligible": 0, "acquired_only_eligible": 0,
        "acquired_only_measured": 0, "acquired_only_correct": 0,
    }
    for event in cohort:
        endpoint = event.phase + horizon
        later = [other for other in events
                 if other.order > event.order and other.key == event.key
                 and (policy == "new_revision" or other.value != event.value)]
        if later and later[0].phase <= min(endpoint, through):
            row["censored"] += 1
            continue
        if endpoint > through:
            row["not_yet_due"] += 1
            continue
        row["eligible"] += 1
        acquired = event.event_id in initial and initial[event.event_id] == event.value
        row["initial_acquisition_unknown_eligible"] += int(event.event_id not in initial)
        row["failed_initial_acquisition_eligible"] += int(event.event_id in initial and not acquired)
        row["acquired_only_eligible"] += int(acquired)
        if endpoint not in snapshots:
            row["missing_measurement"] += 1
            continue
        correct = snapshots[endpoint][event.key] == event.value
        row["measured"] += 1
        row["correct"] += int(correct)
        row["acquired_only_measured"] += int(acquired)
        row["acquired_only_correct"] += int(acquired and correct)
    row["accuracy"] = (row["correct"] / row["eligible"]
                       if row["eligible"] and not row["missing_measurement"] else None)
    row["acquired_only_accuracy"] = (
        row["acquired_only_correct"] / row["acquired_only_eligible"]
        if row["acquired_only_eligible"]
        and row["acquired_only_measured"] == row["acquired_only_eligible"]
        and not row["initial_acquisition_unknown_eligible"] else None
    )
    return row


def run():
    rng = np.random.default_rng(SEED)
    comparisons, histories = 0, 0
    for policy in ("new_revision", "value_change"):
        for trial in range(TRIALS_PER_POLICY):
            events, latest = [], {}

            def append(phase, key, value):
                prior = latest.get(key)
                event = Event(
                    "e" + str(len(events)), phase, len(events), *key, int(value),
                    None if prior is None else prior.event_id,
                    None if prior is None else prior.value,
                )
                events.append(event)
                latest[key] = event

            append(0, (0, 0), 0)
            append(0, (0, 1), 1)
            for phase in range(1, 5):
                for _ in range(int(rng.integers(1, 5))):
                    append(phase, (0, int(rng.integers(2))), rng.integers(3))
            manifest = Manifest(
                trial, WorldSpec(phases=5, entities_per_phase=1, relations=2,
                                 shown_relations=2, values=3, assignments_per_phase=2),
                tuple(events),
            )
            log = new_log(manifest, policy)
            initial, snapshots = {}, {}
            for phase in range(5):
                for event in events:
                    if event.phase == phase and event.previous_revision is not None and rng.random() < .7:
                        prediction = None if rng.random() < .2 else int(rng.integers(3))
                        log = record_after_write(manifest, log, event.event_id, prediction)
                        initial[event.event_id] = prediction
                if phase == 4 or rng.random() < .7:
                    predictions = {
                        key: None if rng.random() < .2 else int(rng.integers(3))
                        for key in phase_end_queries(manifest, phase)
                    }
                    log = record_phase_end(manifest, log, phase, predictions)
                    snapshots[phase] = predictions
            for through in range(5):
                report = summarize(manifest, log, horizons=HORIZONS, through_phase=through)
                observed = [event for event in events
                            if event.previous_revision is not None and event.phase <= through]
                cohorts = {
                    "all_assignments": observed,
                    "semantic_value_changes": [e for e in observed if e.value != e.previous_value],
                    "same_value_assignments": [e for e in observed if e.value == e.previous_value],
                }
                for name, cohort in cohorts.items():
                    actual = report["cohorts"][name]
                    expected = expected_immediate(cohort, initial)
                    if actual["immediately_after_write"] != expected:
                        raise AssertionError((policy, trial, through, name, "immediate", expected, actual))
                    comparisons += 1
                    for horizon in HORIZONS:
                        expected = expected_horizon(events, cohort, initial, snapshots,
                                                    policy, through, horizon)
                        if actual["phase_end_horizons"][str(horizon)] != expected:
                            raise AssertionError((policy, trial, through, name, horizon, expected, actual))
                        comparisons += 1
            histories += 1
    paths = {"implementation": EXPERIMENTS / "correction_protocol.py",
             "contract_tests": EXPERIMENTS / "test_correction_protocol.py",
             "review_oracle": Path(__file__).resolve()}
    return {
        "status": "PASS", "seed": SEED, "numpy_version": np.__version__,
        "histories": histories, "trials_per_policy": TRIALS_PER_POLICY,
        "cohort_rows_checked": comparisons,
        "policies": ["new_revision", "value_change"], "horizons": list(HORIZONS),
        "historical_cutoffs": [0, 1, 2, 3, 4],
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in paths.items()},
        "scope": "Independent event scan against measurement summaries; synthetic engineering fixtures only. No model or benchmark evaluation.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
