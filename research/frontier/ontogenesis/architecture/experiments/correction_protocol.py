"""G03b measurement foundation: event identity, causal views and explicit cohorts.

No model training, no benchmark result, and no claim that a solver success is a
valid model edit. The coordinator owns the complete manifest. A learner should
receive only VisiblePrefix objects and query keys, never the coordinator object.
This is an interface discipline, not a Python introspection/security boundary.

Default CLI prints a small manifest summary; --out exclusively creates a NEW
summary file. Existing registered experiments and result files are untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

Key = tuple[int, int]
MAX_EVENTS = 10_000
CENSOR_POLICIES = ("new_revision", "value_change")


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside bounds")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _nonnegative_finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a nonnegative finite scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative finite scalar")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _named_rng(seed: int, name: str) -> np.random.Generator:
    seed = _integer(seed, "seed", maximum=2**64 - 1)
    identity = hashlib.sha256(_canonical({"seed": seed, "stream": name})).digest()
    words = np.frombuffer(identity, dtype="<u4").tolist()
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


def replay_rng(seed: int, arm: str) -> np.random.Generator:
    """Replay consumption cannot change a separately generated world manifest."""
    if not isinstance(arm, str) or not arm:
        raise ValueError("A nonempty replay arm identity is required")
    return _named_rng(seed, "replay:" + arm)


@dataclass(frozen=True)
class WorldSpec:
    phases: int = 6
    entities_per_phase: int = 2
    relations: int = 3
    shown_relations: int = 2
    values: int = 5
    assignments_per_phase: int = 2

    def __post_init__(self):
        for name in ("phases", "entities_per_phase", "relations", "shown_relations"):
            value = _integer(getattr(self, name), name, minimum=1, maximum=MAX_EVENTS)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "values", _integer(self.values, "values", 2, 65_536))
        object.__setattr__(self, "assignments_per_phase",
                           _integer(self.assignments_per_phase, "assignments_per_phase"))
        if self.shown_relations > self.relations:
            raise ValueError("shown_relations exceeds relations")
        if self.assignments_per_phase > self.entities_per_phase * self.shown_relations:
            raise ValueError("The first correction phase has too few distinct old keys")
        bound = (self.phases * self.entities_per_phase * self.shown_relations
                 + (self.phases - 1) * self.assignments_per_phase)
        if bound > MAX_EVENTS:
            raise ValueError("Measurement reference is bounded to 10000 events")


@dataclass(frozen=True)
class Event:
    event_id: str
    phase: int
    order: int
    entity: int
    relation: int
    value: int
    previous_revision: str | None = None
    previous_value: int | None = None

    def __post_init__(self):
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("Nonempty event identity required")
        for name in ("phase", "order", "entity", "relation", "value"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.previous_revision is not None and (
                not isinstance(self.previous_revision, str) or not self.previous_revision):
            raise ValueError("Prior revision must be a nonempty ID or null")
        if self.previous_value is not None:
            object.__setattr__(self, "previous_value", _integer(self.previous_value, "previous_value"))

    @property
    def key(self) -> Key:
        return self.entity, self.relation

    @property
    def kind(self) -> str:
        if self.previous_revision is None:
            return "initial_observation"
        return "same_value_assignment" if self.value == self.previous_value else "value_change"


@dataclass(frozen=True)
class Manifest:
    seed: int
    spec: WorldSpec
    events: tuple[Event, ...]
    generator: str = "g03b-bounded-world-v1"

    def __post_init__(self):
        object.__setattr__(self, "seed", _integer(self.seed, "seed", maximum=2**64 - 1))
        if not isinstance(self.spec, WorldSpec) or type(self.events) is not tuple:
            raise ValueError("Manifest requires a WorldSpec and immutable event tuple")
        if self.generator != "g03b-bounded-world-v1" or len(self.events) > MAX_EVENTS:
            raise ValueError("Unsupported or oversized manifest")
        known, ids, last_phase = {}, set(), -1
        for order, event in enumerate(self.events):
            if not isinstance(event, Event):
                raise ValueError("Manifest entries must be Event objects")
            if not isinstance(event.event_id, str) or not event.event_id or event.event_id in ids:
                raise ValueError("Event IDs must be nonempty and unique")
            if _integer(event.order, "order") != order:
                raise ValueError("Global event order must be contiguous")
            phase = _integer(event.phase, "phase", maximum=self.spec.phases - 1)
            if phase < last_phase:
                raise ValueError("Event phases must be nondecreasing")
            _integer(event.entity, "entity", maximum=self.spec.phases * self.spec.entities_per_phase - 1)
            _integer(event.relation, "relation", maximum=self.spec.relations - 1)
            _integer(event.value, "value", maximum=self.spec.values - 1)
            prior = known.get(event.key)
            if prior is None:
                if event.previous_revision is not None or event.previous_value is not None:
                    raise ValueError("First observation cannot refer to a prior revision")
            elif (event.previous_revision != prior.event_id or event.previous_value != prior.value):
                raise ValueError("Assignment must name the latest prior revision and value")
            ids.add(event.event_id)
            known[event.key] = event
            last_phase = phase

    @property
    def sha256(self):
        """Digest of canonical JSON without a trailing newline, not a file hash."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def canonical_bytes(self):
        return _canonical({"schema_version": 1, "generator": self.generator,
                           "seed": self.seed, "spec": asdict(self.spec),
                           "events": [asdict(event) for event in self.events]})

    def write_new(self, path):
        with Path(path).open("xb") as target:
            target.write(self.canonical_bytes() + b"\n")

    @classmethod
    def read(cls, path):
        raw = json.loads(Path(path).read_text(), object_pairs_hook=_unique_object)
        if (not isinstance(raw, dict)
                or set(raw) != {"schema_version", "generator", "seed", "spec", "events"}
                or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
            raise ValueError("Unsupported manifest structure")
        try:
            return cls(raw["seed"], WorldSpec(**raw["spec"]),
                       tuple(Event(**event) for event in raw["events"]), raw["generator"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed manifest") from exc


def generate_manifest(seed=8001, spec=WorldSpec()) -> Manifest:
    """Precompute observed assignments; same-value assignments are retained."""
    if not isinstance(spec, WorldSpec):
        raise ValueError("WorldSpec required")
    rng = _named_rng(seed, "world")
    events, latest = [], {}

    def append(phase, entity, relation, value):
        prior = latest.get((entity, relation))
        event = Event(f"event-{len(events):08d}", phase, len(events), entity, relation, int(value),
                      None if prior is None else prior.event_id,
                      None if prior is None else prior.value)
        events.append(event)
        latest[event.key] = event

    for phase in range(spec.phases):
        old_keys = tuple(sorted(latest))
        for entity in range(phase * spec.entities_per_phase, (phase + 1) * spec.entities_per_phase):
            shown = sorted(rng.choice(spec.relations, spec.shown_relations, replace=False).tolist())
            for relation in shown:
                append(phase, entity, relation, rng.integers(spec.values))
        if phase:
            indices = rng.choice(len(old_keys), spec.assignments_per_phase, replace=False)
            for index in indices:
                entity, relation = old_keys[int(index)]
                append(phase, entity, relation, rng.integers(spec.values))
    return Manifest(seed, spec, tuple(events))


@dataclass(frozen=True)
class VisiblePrefix:
    """A learner-facing copy containing only delivered events and observed targets."""
    current_event: Event
    observed_events: tuple[Event, ...]
    current_targets: tuple[tuple[int, int, int, str], ...]


def deliveries(manifest: Manifest):
    latest = {}
    for index, event in enumerate(manifest.events):
        latest[event.key] = event
        targets = tuple((e.entity, e.relation, e.value, e.event_id)
                        for _, e in sorted(latest.items()))
        yield VisiblePrefix(event, manifest.events[:index + 1], targets)


@dataclass(frozen=True)
class Measurement:
    event_id: str
    endpoint: str
    phase: int
    prediction: int | None

    def __post_init__(self):
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("Measurement requires event identity")
        if self.endpoint not in ("after_write", "phase_end"):
            raise ValueError("Unknown measurement endpoint")
        object.__setattr__(self, "phase", _integer(self.phase, "phase"))
        if self.prediction is not None:
            object.__setattr__(self, "prediction", _integer(self.prediction, "prediction"))


@dataclass(frozen=True)
class EvaluationLog:
    manifest_sha256: str
    censor_policy: str = "new_revision"
    measurements: tuple[Measurement, ...] = ()
    completed_phases: tuple[int, ...] = ()
    last_position: tuple[int, int] = (-1, -1)

    def __post_init__(self):
        if self.censor_policy not in CENSOR_POLICIES:
            raise ValueError("Explicit supported censor policy required")
        if type(self.measurements) is not tuple or type(self.completed_phases) is not tuple:
            raise ValueError("Evaluation records must be immutable tuples")
        if not all(isinstance(sample, Measurement) for sample in self.measurements):
            raise ValueError("Measurement entries required")
        if any(type(phase) is not int or phase < 0 for phase in self.completed_phases):
            raise ValueError("Completed phases must be nonnegative integers")
        if tuple(sorted(set(self.completed_phases))) != self.completed_phases:
            raise ValueError("Completed phases must be strictly increasing")
        if (type(self.last_position) is not tuple or len(self.last_position) != 2
                or any(type(n) is not int or n < -1 for n in self.last_position)):
            raise ValueError("Malformed measurement clock")


def new_log(manifest, censor_policy="new_revision"):
    return EvaluationLog(manifest.sha256, censor_policy)


def _check_identity(manifest, log):
    if log.manifest_sha256 != manifest.sha256:
        raise ValueError("Evaluation log belongs to a different event manifest")
    event_by_id = {event.event_id: event for event in _assignments(manifest)}
    successors = _successors(manifest, log.censor_policy)
    known, last_sample = set(), (-1, -1)
    for sample in log.measurements:
        event = event_by_id.get(sample.event_id)
        identity = sample.event_id, sample.endpoint, sample.phase
        if event is None or identity in known:
            raise ValueError("Unknown or duplicate measurement identity")
        _prediction(sample.prediction, manifest.spec.values)
        if sample.endpoint == "after_write":
            if sample.phase != event.phase:
                raise ValueError("Immediate sample must match its write phase")
            position = sample.phase, event.order
        else:
            successor = successors[event.event_id]
            if (sample.phase < event.phase or sample.phase not in log.completed_phases
                    or (successor is not None and successor.phase <= sample.phase)):
                raise ValueError("Phase-end sample has invalid time or superseded target")
            position = sample.phase, len(manifest.events)
        if position < last_sample:
            raise ValueError("Measurement record clock moved backwards")
        last_sample = position
        known.add(identity)
    if any(phase >= manifest.spec.phases for phase in log.completed_phases):
        raise ValueError("Completed phase exceeds manifest")
    final_position = max(last_sample,
                         (max(log.completed_phases), len(manifest.events))
                         if log.completed_phases else (-1, -1))
    if final_position != log.last_position:
        raise ValueError("Measurement clock does not match recorded observations")


def _assignments(manifest):
    return tuple(event for event in manifest.events if event.previous_revision is not None)


def _successors(manifest, policy):
    """First superseding revision, or first later different value, by event ID."""
    latest, result = {}, {}
    for event in reversed(manifest.events):
        following = latest.get(event.key)
        if following is not None and policy == "value_change" and following.value == event.value:
            following = result[following.event_id]
        result[event.event_id] = following
        latest[event.key] = event
    return result


def _prediction(value, values):
    # None is an explicit abstention: a measured incorrect answer, not missing data.
    return None if value is None else _integer(value, "prediction", maximum=values - 1)


def record_after_write(manifest, log, event_id, prediction):
    """Record immediately after this individual assignment, before the next event."""
    _check_identity(manifest, log)
    events = {event.event_id: event for event in _assignments(manifest)}
    if event_id not in events:
        raise ValueError("after_write requires a known assignment event")
    event = events[event_id]
    position = event.phase, event.order
    if position <= log.last_position:
        raise ValueError("Measurements must advance the event clock without duplicates")
    item = Measurement(event_id, "after_write", event.phase,
                       _prediction(prediction, manifest.spec.values))
    return replace(log, measurements=log.measurements + (item,), last_position=position)


def phase_end_queries(manifest, phase):
    """Return observed query keys only; no target values or future-phase keys."""
    phase = _integer(phase, "phase", maximum=manifest.spec.phases - 1)
    return tuple(sorted({event.key for event in _assignments(manifest) if event.phase <= phase}))


def record_phase_end(manifest, log, phase, predictions: Mapping[Key, int | None]):
    """Record all current correction-key predictions at the end of one phase.

    A snapshot is explicit even when there are no correction queries. Every
    required key must be supplied; extra keys, including future keys, are rejected.
    """
    _check_identity(manifest, log)
    phase = _integer(phase, "phase", maximum=manifest.spec.phases - 1)
    position = phase, len(manifest.events)
    if position <= log.last_position:
        raise ValueError("Phase endpoints must advance the clock without duplicates")
    if not isinstance(predictions, Mapping):
        raise ValueError("Predictions must map integer query keys to answers")
    for key in predictions:
        if type(key) is not tuple or len(key) != 2:
            raise ValueError("Query keys must be entity/relation tuples")
        _integer(key[0], "query entity", maximum=manifest.spec.phases * manifest.spec.entities_per_phase - 1)
        _integer(key[1], "query relation", maximum=manifest.spec.relations - 1)
    if set(predictions) != set(phase_end_queries(manifest, phase)):
        raise ValueError("Predictions must contain exactly the observed correction query keys")
    normalized = {key: _prediction(value, manifest.spec.values)
                  for key, value in predictions.items()}
    successors = _successors(manifest, log.censor_policy)
    samples = []
    for event in _assignments(manifest):
        successor = successors[event.event_id]
        if event.phase <= phase and (successor is None or successor.phase > phase):
            samples.append(Measurement(event.event_id, "phase_end", phase, normalized[event.key]))
    return replace(log, measurements=log.measurements + tuple(samples),
                   completed_phases=log.completed_phases + (phase,), last_position=position)


def summarize(manifest, log, horizons=(0, 1, 5, 10, 20), through_phase=None):
    """Event-level counts. Missing eligible measurements make accuracy undefined.

    Known censoring by the horizon takes precedence over not-yet-due status.
    Future supersessions beyond through_phase are ignored. Primary accuracy
    includes failed initial acquisition; acquired-only retention is secondary.
    """
    _check_identity(manifest, log)
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("Distinct nonempty horizons required")
    horizons = tuple(_integer(h, "horizon") for h in horizons)
    completed = max(log.completed_phases, default=-1)
    through = completed if through_phase is None else _integer(through_phase, "through_phase")
    if through > completed:
        raise ValueError("Cannot summarize beyond the latest recorded phase endpoint")
    samples = {}
    for sample in log.measurements:
        key = sample.event_id, sample.endpoint, sample.phase
        if key in samples:
            raise ValueError("Duplicate measurement identity")
        samples[key] = sample
    events = tuple(event for event in _assignments(manifest) if event.phase <= through)
    successors = _successors(manifest, log.censor_policy)

    def metrics(cohort):
        immediate_known = [event for event in cohort
                           if (event.event_id, "after_write", event.phase) in samples]
        immediate_correct = sum(samples[(event.event_id, "after_write", event.phase)].prediction == event.value
                                for event in immediate_known)
        immediate = {
            "events": len(cohort), "measured": len(immediate_known),
            "correct": immediate_correct, "missing": len(cohort) - len(immediate_known),
            "accuracy": (immediate_correct / len(cohort)
                         if cohort and len(immediate_known) == len(cohort) else None),
        }
        results = {}
        for horizon in horizons:
            row = {"total_events": len(cohort), "eligible": 0, "censored": 0,
                   "not_yet_due": 0, "measured": 0, "missing_measurement": 0, "correct": 0,
                   "failed_initial_acquisition_eligible": 0, "initial_acquisition_unknown_eligible": 0,
                   "acquired_only_eligible": 0, "acquired_only_measured": 0, "acquired_only_correct": 0}
            for event in cohort:
                target_phase = event.phase + horizon
                successor = successors[event.event_id]
                if successor is not None and successor.phase <= min(target_phase, through):
                    row["censored"] += 1
                    continue
                if target_phase > through:
                    row["not_yet_due"] += 1
                    continue
                row["eligible"] += 1
                initial = samples.get((event.event_id, "after_write", event.phase))
                acquired = initial is not None and initial.prediction == event.value
                if initial is None:
                    row["initial_acquisition_unknown_eligible"] += 1
                elif not acquired:
                    row["failed_initial_acquisition_eligible"] += 1
                row["acquired_only_eligible"] += int(acquired)
                observed = samples.get((event.event_id, "phase_end", target_phase))
                if observed is None:
                    row["missing_measurement"] += 1
                    continue
                correct = observed.prediction == event.value
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
                and not row["initial_acquisition_unknown_eligible"] else None)
            results[str(horizon)] = row
        return {"immediately_after_write": immediate, "phase_end_horizons": results}

    cohorts = {
        "all_assignments": events,
        "semantic_value_changes": tuple(event for event in events if event.kind == "value_change"),
        "same_value_assignments": tuple(event for event in events if event.kind == "same_value_assignment"),
    }
    return {
        "manifest_sha256": manifest.sha256, "through_phase": through,
        "censor_policy": log.censor_policy,
        "cohorts": {name: metrics(cohort) for name, cohort in cohorts.items()},
        "scope": "Observed assignment-event measurements, not independent samples or a model-quality claim. Horizon0 means phase end; immediate acquisition is reported separately.",
    }


@dataclass(frozen=True)
class TargetProvenance:
    entity: int
    relation: int
    target: int
    source: str
    source_id: str

    def __post_init__(self):
        for name in ("entity", "relation", "target"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.source not in ("model_prediction", "observed_target", "new_assignment"):
            raise ValueError("Explicit prediction/observation/assignment provenance required")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("Source event or checkpoint identity required")


@dataclass(frozen=True)
class LinearConstraints:
    """A x <= b, with immutable values and provenance for every inequality."""
    matrix: tuple[tuple[float, ...], ...]
    upper: tuple[float, ...]
    targets: tuple[TargetProvenance, ...]

    def __post_init__(self):
        if (type(self.matrix) is not tuple or type(self.upper) is not tuple
                or type(self.targets) is not tuple or any(type(row) is not tuple for row in self.matrix)):
            raise ValueError("Constraint arrays must be immutable tuples")
        a, b = np.asarray(self.matrix, dtype=np.float64), np.asarray(self.upper, dtype=np.float64)
        if a.ndim != 2 or not all(a.shape) or b.shape != (a.shape[0],):
            raise ValueError("Nonempty matrix/vector dimensions required")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Finite constraint values required")
        if len(self.targets) != a.shape[0] or not all(isinstance(t, TargetProvenance) for t in self.targets):
            raise ValueError("Every inequality requires target provenance")

    @classmethod
    def from_arrays(cls, matrix, upper, targets):
        a, b = np.asarray(matrix, dtype=np.float64), np.asarray(upper, dtype=np.float64)
        if a.ndim != 2 or b.ndim != 1:
            raise ValueError("Matrix and vector expected")
        return cls(tuple(tuple(float(x) for x in row) for row in a),
                   tuple(float(x) for x in b), tuple(targets))

    @property
    def sha256(self):
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


def classify_solver_status(status):
    status = _integer(status, "solver status")
    if status == 0:
        return "candidate"
    if status == 2:
        return "reported_infeasible"
    return "unknown_failure"


def assess_linear_candidate(status, constraints: LinearConstraints, candidate=None,
                            deployed_value=None, atol=0.0, rtol=0.0):
    """Audit a solver candidate and actual deployed vector against original bounds.

    deployed_value must come from the real update application, including casting
    AND addition if an increment was applied. Its values are audited in float64
    against unchanged constraints. This does not certify all runtime operations
    or the nonlinear model; evaluate its actual logits separately.
    """
    atol = _nonnegative_finite(atol, "atol")
    rtol = _nonnegative_finite(rtol, "rtol")
    kind = classify_solver_status(status)
    result = {"solver_status": int(status), "solver_classification": kind,
              "validated": False, "atol": float(atol), "rtol": float(rtol),
              "constraints_sha256": constraints.sha256,
              "target_provenance": [asdict(t) for t in constraints.targets]}
    if kind != "candidate":
        result["verdict"] = kind
        return result
    a = np.asarray(constraints.matrix, dtype=np.float64)
    b = np.asarray(constraints.upper, dtype=np.float64)

    def audit(value):
        x = np.asarray(value)
        if x.shape != (a.shape[1],) or x.dtype.kind != "f" or not np.isfinite(x).all():
            return {"valid_input": False, "satisfies": False}
        # Retain the requested thresholds; do not round b down with the candidate.
        with np.errstate(over="ignore", invalid="ignore"):
            residual = a @ x.astype(np.float64) - b
            tolerance = atol + rtol * np.abs(b)
        finite = bool(np.isfinite(residual).all() and np.isfinite(tolerance).all())
        return {"valid_input": True, "dtype": str(x.dtype), "finite_residual": finite,
                "value_sha256": hashlib.sha256(
                    _canonical({"dtype": x.dtype.str, "shape": list(x.shape)}) + b"\0" + x.tobytes()
                ).hexdigest(),
                "satisfies": bool(finite and np.all(residual <= tolerance)),
                "maximum_violation": float(np.max(np.maximum(residual, 0))) if finite else None,
                "minimum_slack": float(np.min(-residual)) if finite else None,
                "residuals": residual.tolist() if finite else None}

    result["solver_candidate"] = audit(candidate)
    result["deployed_value"] = None if deployed_value is None else audit(deployed_value)
    if not result["solver_candidate"]["satisfies"]:
        result["verdict"] = "invalid_candidate"
    elif deployed_value is None:
        result["verdict"] = "candidate_only"
    elif not result["deployed_value"]["satisfies"]:
        result["verdict"] = "invalid_deployed_value"
    else:
        result["verdict"] = "validated_linear_candidate"
        result["validated"] = True
    return result


def check_logit_margin(logits, target, required_margin, atol=0.0):
    """Direct check on logits returned by the actual deployed model."""
    values = np.asarray(logits)
    if values.ndim != 1 or len(values) < 2 or values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ValueError("A finite floating logit vector is required")
    target = _integer(target, "target", maximum=len(values) - 1)
    required_margin = _nonnegative_finite(required_margin, "required_margin")
    atol = _nonnegative_finite(atol, "atol")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float64)
        achieved = float(values[target] - np.max(np.delete(values, target)))
    if not np.isfinite(achieved):
        raise ValueError("Margin overflowed the measurement precision")
    return {"target": target, "required_margin": float(required_margin),
            "achieved_margin": achieved, "atol": float(atol),
            "satisfies": bool(achieved + atol >= required_margin),
            "argmax_correct": bool(int(np.argmax(values)) == target)}


def write_report(report, output=None):
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output is not None:
        with Path(output).open("x", encoding="utf-8") as destination:
            destination.write(text)
    print(text, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=8001)
    parser.add_argument("--out", help="Optional NEW summary file; no overwrite")
    args = parser.parse_args()
    manifest = generate_manifest(args.seed)
    kinds = {kind: sum(event.kind == kind for event in manifest.events)
             for kind in ("initial_observation", "value_change", "same_value_assignment")}
    write_report({"scope": "Measurement-instrument manifest demonstration only; no model training or comparison.",
                  "manifest_sha256": manifest.sha256, "spec": asdict(manifest.spec),
                  "event_counts": kinds, "events_total": len(manifest.events),
                  "censor_policies": list(CENSOR_POLICIES)}, args.out)


if __name__ == "__main__":
    main()
