"""V6 temporal scoring reference: explicit validity, prediction time, and provenance.

Reuses correction_protocol's frozen event manifest and target provenance. Plans
belong to the coordinator, not to a learner. This is receipt validation, not a
security or authenticity boundary. No model training and no upstream result edits.

Default CLI prints the contract without writing. --out exclusively creates a new
contract receipt. See test_temporal_revision_scoring_v6.py for adversarial checks.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import sys

sys.dont_write_bytecode = True
from correction_protocol import CENSOR_POLICIES, Manifest, TargetProvenance, write_report

CONTRACT_JSON = r'''{"schema":"nanolm-temporal-scoring-v6-contract-v1","scope":"Bounded coordinator-side temporal scoring reference; no model training or G04/G06 result rewrite.","units":"Early adaptation uses assignment events; final-truth cohorts contain one anchor event per distinct key.","clock":"(phase, global event order); phase end uses order=len(manifest.events). Planned predictions occur at phase end.","validity":"Intervals are left-closed/right-open. new_revision ends at the next assignment; value_change ends at the first later different value. Same-value assignment IDs are never erased.","early_adaptation":"Query each observed cohort event at phase_end(event.phase+offset). A known superseding event by min(due,through) censors it; otherwise a future due phase is not_yet_due. Eligible queries use the latest manifest target valid at the prediction timestamp.","final_truth_by_cohort":"Query each distinct cohort key at through_phase end using the latest manifest event at that time, even when its anchor has been superseded. No historical anchor target is reused.","provenance":"Every eligible case has observed_target provenance with its latest source event ID. Plans bind canonical contract and manifest hashes. Model predictions contain only case ID, declared prediction time, value or explicit abstention, and plan hash.","denominators":"total=eligible+censored+not_yet_due. Primary accuracy=correct/eligible; missing eligible records and explicit abstentions count as wrong. Empty eligible denominator yields null.","validation":"Reject unknown or duplicate cases, foreign plan hashes, records for ineligible cases, invalid or wrong timestamps, out-of-domain values, duplicate cohort IDs, and duplicate final-cohort keys.","state":"Frozen objects and tuple records. Coordinator oracle/interval metadata is not a learner input. Receipt validation is not an authenticity/security guarantee.","storage":"Default CLI and checks do not create release artifacts. Explicit output uses exclusive creation and cannot overwrite an existing file.","checks":"A→B→C current/stale, B→C→B IDs, same-value policies, same-phase supersession, future-censor discipline, missing/duplicate/invalid timestamp/provenance inputs, exclusive output, read-only default."}'''


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def contract():
    """Fresh copy; callers cannot mutate the canonical contract."""
    return json.loads(CONTRACT_JSON)


def contract_sha256():
    return hashlib.sha256(_canonical(contract())).hexdigest()


def _integer(value, name, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be a bounded nonnegative Python integer")
    return value


@dataclass(frozen=True, order=True)
class Time:
    """After one event at (phase, global order), or phase end at order=N_events."""
    phase: int
    order: int

    def __post_init__(self):
        _integer(self.phase, "phase")
        _integer(self.order, "order")


def phase_end(manifest, phase):
    if not isinstance(manifest, Manifest):
        raise ValueError("Manifest required")
    _integer(phase, "phase", manifest.spec.phases - 1)
    return Time(phase, len(manifest.events))


def _validate_recorded_time(manifest, at):
    if not isinstance(at, Time) or at.phase >= manifest.spec.phases:
        raise ValueError("Recorded prediction time is outside the manifest")
    if at.order == len(manifest.events):
        return
    if at.order >= len(manifest.events) or manifest.events[at.order].phase != at.phase:
        raise ValueError("Time must identify an event in that phase or the phase end")


@dataclass(frozen=True)
class ValidityInterval:
    event_id: str
    key: tuple[int, int]
    value: int
    kind: str
    valid_from: Time
    valid_until: Time | None
    superseded_by: str | None
    censor_policy: str


def valid_intervals(manifest, censor_policy):
    """Left-closed/right-open intervals; repeated values retain distinct event IDs.

    value_change bridges same-value assignments only until the first different
    value. A later return to the old value never revives an earlier interval.
    """
    if not isinstance(manifest, Manifest) or censor_policy not in CENSOR_POLICIES:
        raise ValueError("Manifest and explicit supported censor policy required")
    latest, successors = {}, {}
    for event in reversed(manifest.events):
        following = latest.get(event.key)
        if (following is not None and censor_policy == "value_change"
                and following.value == event.value):
            following = successors[following.event_id]
        successors[event.event_id] = following
        latest[event.key] = event
    return tuple(
        ValidityInterval(
            event.event_id, event.key, event.value, event.kind,
            Time(event.phase, event.order),
            None if successors[event.event_id] is None else Time(
                successors[event.event_id].phase, successors[event.event_id].order),
            None if successors[event.event_id] is None else successors[event.event_id].event_id,
            censor_policy,
        ) for event in manifest.events
    )


@dataclass(frozen=True)
class Case:
    case_id: str
    anchor_event_id: str
    key: tuple[int, int]
    requested_time: Time
    status: str
    anchor_validity: ValidityInterval
    target_validity: ValidityInterval | None
    target_provenance: TargetProvenance | None


@dataclass(frozen=True)
class Plan:
    manifest: Manifest
    mode: str
    cohort_event_ids: tuple[str, ...]
    through_phase: int
    censor_policy: str
    offset: int | None

    def __post_init__(self):
        if not isinstance(self.manifest, Manifest):
            raise ValueError("Manifest required")
        _integer(self.through_phase, "through_phase", self.manifest.spec.phases - 1)
        if (type(self.cohort_event_ids) is not tuple
                or any(not isinstance(e, str) or not e for e in self.cohort_event_ids)
                or len(set(self.cohort_event_ids)) != len(self.cohort_event_ids)):
            raise ValueError("Unique immutable cohort event IDs required")
        by_id = {e.event_id: e for e in self.manifest.events}
        if any(e not in by_id for e in self.cohort_event_ids):
            raise ValueError("Unknown cohort event ID")
        if any(by_id[e].phase > self.through_phase for e in self.cohort_event_ids):
            raise ValueError("Cohort cannot contain future unobserved events")
        if self.mode == "early_adaptation":
            if self.censor_policy not in CENSOR_POLICIES:
                raise ValueError("Early adaptation requires an explicit censor policy")
            _integer(self.offset, "offset", 10000)
        elif self.mode == "final_truth_by_cohort":
            if self.offset is not None or self.censor_policy != "latest_truth":
                raise ValueError("Final-truth plans use latest_truth and no offset")
            keys = [by_id[e].key for e in self.cohort_event_ids]
            if len(keys) != len(set(keys)):
                raise ValueError("Final-truth cohorts require one anchor per distinct key")
        else:
            raise ValueError("Unknown temporal scoring mode")

    @property
    def cases(self):
        policy = self.censor_policy if self.mode == "early_adaptation" else "new_revision"
        intervals = {r.event_id: r for r in valid_intervals(self.manifest, policy)}
        revisions = {r.event_id: r for r in valid_intervals(self.manifest, "new_revision")}
        by_id = {e.event_id: e for e in self.manifest.events}
        by_key = {}
        for event in self.manifest.events:
            by_key.setdefault(event.key, []).append(event)
        times = {k: [Time(e.phase, e.order) for e in ee] for k, ee in by_key.items()}
        through = phase_end(self.manifest, self.through_phase)
        cases = []
        for index, identity in enumerate(self.cohort_event_ids):
            event, interval = by_id[identity], intervals[identity]
            requested = (through if self.mode == "final_truth_by_cohort" else
                         Time(event.phase + self.offset, len(self.manifest.events)))
            status = "eligible"
            if self.mode == "early_adaptation":
                if (interval.valid_until is not None and
                        interval.valid_until <= min(requested, through)):
                    status = "censored"
                elif requested > through:
                    status = "not_yet_due"
            target, target_interval = None, None
            if status == "eligible":
                position = bisect_right(times[event.key], requested) - 1
                active = by_key[event.key][position]
                if self.mode == "early_adaptation" and active.value != event.value:
                    raise AssertionError("Eligible historical target is not current")
                target = TargetProvenance(*event.key, active.value, "observed_target", active.event_id)
                target_interval = revisions[active.event_id]
            cases.append(Case(f"case-{index:06d}", identity, event.key, requested,
                              status, interval, target_interval, target))
        return tuple(cases)

    def canonical_payload(self):
        return {
            "schema": "nanolm-temporal-plan-v1",
            "contract_sha256": contract_sha256(),
            "manifest_sha256": self.manifest.sha256,
            "mode": self.mode,
            "cohort_event_ids": self.cohort_event_ids,
            "through_phase": self.through_phase,
            "censor_policy": self.censor_policy,
            "offset": self.offset,
            "cases": [asdict(case) for case in self.cases],
        }

    @property
    def sha256(self):
        return hashlib.sha256(_canonical(self.canonical_payload())).hexdigest()


def early_adaptation_plan(manifest, cohort_event_ids, *, offset, through_phase, censor_policy):
    return Plan(manifest, "early_adaptation", cohort_event_ids, through_phase, censor_policy, offset)


def final_truth_plan(manifest, cohort_event_ids, *, through_phase):
    return Plan(manifest, "final_truth_by_cohort", cohort_event_ids, through_phase, "latest_truth", None)


@dataclass(frozen=True)
class Prediction:
    case_id: str
    predicted_at: Time
    value: int | None
    plan_sha256: str

    def __post_init__(self):
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("Nonempty case ID required")
        if not isinstance(self.predicted_at, Time):
            raise ValueError("Explicit Time required")
        if self.value is not None:
            _integer(self.value, "prediction", 65535)
        if (not isinstance(self.plan_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.plan_sha256) is None):
            raise ValueError("Canonical plan SHA-256 required")


def score(plan, predictions):
    """Missing eligible predictions are wrong; censoring never hides them."""
    if not isinstance(plan, Plan) or type(predictions) is not tuple:
        raise ValueError("Plan and immutable Prediction tuple required")
    cases = plan.cases
    by_id = {case.case_id: case for case in cases}
    fingerprint = plan.sha256
    received = {}
    for prediction in predictions:
        if not isinstance(prediction, Prediction):
            raise ValueError("Prediction objects required")
        if prediction.plan_sha256 != fingerprint:
            raise ValueError("Prediction belongs to a foreign plan")
        case = by_id.get(prediction.case_id)
        if case is None or prediction.case_id in received:
            raise ValueError("Unknown or duplicate prediction case")
        if case.status != "eligible":
            raise ValueError("Predictions for censored/not-yet-due cases are invalid")
        _validate_recorded_time(plan.manifest, prediction.predicted_at)
        if prediction.predicted_at != case.requested_time:
            raise ValueError("Prediction timestamp does not match the requested endpoint")
        if prediction.value is not None and prediction.value >= plan.manifest.spec.values:
            raise ValueError("Prediction outside the manifest value domain")
        received[prediction.case_id] = prediction
    counts = {"total": len(cases), "eligible": 0, "censored": 0, "not_yet_due": 0,
              "measured": 0, "missing": 0, "abstained": 0, "correct": 0, "wrong": 0}
    result_rows = []
    for case in cases:
        counts[case.status] += 1
        prediction = received.get(case.case_id)
        row = asdict(case)
        row.update({"prediction": None, "prediction_time": None,
                    "prediction_status": "not_requested", "correct": None})
        if case.status == "eligible":
            if prediction is None:
                counts["missing"] += 1
                row.update(prediction_status="missing", correct=False)
            else:
                counts["measured"] += 1
                counts["abstained"] += int(prediction.value is None)
                row.update(prediction=prediction.value, prediction_time=asdict(prediction.predicted_at),
                           prediction_status="abstained" if prediction.value is None else "present",
                           correct=prediction.value == case.target_provenance.target)
            counts["correct"] += int(row["correct"])
            counts["wrong"] += int(not row["correct"])
        result_rows.append(row)
    assert counts["total"] == counts["eligible"] + counts["censored"] + counts["not_yet_due"]
    assert counts["eligible"] == counts["correct"] + counts["wrong"]
    assert counts["eligible"] == counts["measured"] + counts["missing"]
    counts["accuracy"] = counts["correct"] / counts["eligible"] if counts["eligible"] else None
    return {
        "schema": "nanolm-temporal-score-v1", "contract_sha256": contract_sha256(),
        "manifest_sha256": plan.manifest.sha256, "plan_sha256": fingerprint,
        "mode": plan.mode, "censor_policy": plan.censor_policy,
        "through_phase": plan.through_phase, "offset": plan.offset,
        "unit": "assignment_event" if plan.mode == "early_adaptation" else "distinct_cohort_key",
        "counts": counts, "cases": result_rows,
        "scope": "Coordinator-side temporal scoring; no model training, authenticity guarantee, or G04 result rewrite.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Optional NEW contract receipt; exclusive creation")
    args = parser.parse_args()
    write_report({"contract": contract(), "contract_sha256": contract_sha256(),
                  "scope": "Temporal-scoring contract only; no model training."}, args.out)


if __name__ == "__main__":
    main()

