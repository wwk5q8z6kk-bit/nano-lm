"""Reproducible P1 comparison harness: MODEL_TRACK × P1_TEST × SAME_EVALUATOR."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from nanoscribe.adapt import ModelInput, run_pipeline
from nanoscribe.adapters import AtomSpec, ModelAdapter
from nanoscribe.encounter import EncounterRecord
from nanoscribe.evaluate import EvalReport


class ModelTrack(str, Enum):
    FIXTURE = "fixture"
    COMPACT = "compact"
    FRONTIER = "frontier"


class P1TestSet(str, Enum):
    TINY_FIXTURE = "tiny_fixture"


@dataclass(frozen=True, slots=True)
class TrackConfig:
    track: ModelTrack
    model_id: str
    adapter_factory: Callable[[], ModelAdapter]
    cost_class: str
    notes: str = ""


@dataclass(frozen=True, slots=True)
class HarnessCase:
    test_set: P1TestSet
    encounter_id: str
    gold: EncounterRecord
    model_input: ModelInput
    atom_specs: tuple[AtomSpec, ...]


@dataclass(frozen=True, slots=True)
class FailureTaxonomy:
    invalid_span: int = 0
    wrong_source: int = 0
    wrong_mention: int = 0
    omission: int = 0
    unnecessary_abstention: int = 0
    malformed: int = 0
    critical_error: int = 0
    spurious_atom: int = 0
    ambiguity: int = 0

    @classmethod
    def from_report(cls, report: EvalReport) -> FailureTaxonomy:
        return cls(
            invalid_span=report.invalid_span,
            wrong_source=report.wrong_source,
            wrong_mention=report.wrong_mention,
            omission=report.omission,
            unnecessary_abstention=report.unnecessary_abstention,
            malformed=report.malformed,
            critical_error=report.critical_error,
            spurious_atom=report.spurious_atom,
            ambiguity=report.ambiguity,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "invalid_span": self.invalid_span,
            "wrong_source": self.wrong_source,
            "wrong_mention": self.wrong_mention,
            "omission": self.omission,
            "unnecessary_abstention": self.unnecessary_abstention,
            "malformed": self.malformed,
            "critical_error": self.critical_error,
            "spurious_atom": self.spurious_atom,
            "ambiguity": self.ambiguity,
        }


@dataclass(frozen=True, slots=True)
class HarnessResult:
    track: ModelTrack
    model_id: str
    test_set: P1TestSet
    encounter_id: str
    cost_class: str
    aggregate: dict[str, Any]
    failures: FailureTaxonomy
    per_atom: dict[str, dict[str, Any]]
    latency_s: float
    memory_bytes: int
    raw_lines: Mapping[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "track": self.track.value,
            "model_id": self.model_id,
            "test_set": self.test_set.value,
            "encounter_id": self.encounter_id,
            "cost_class": self.cost_class,
            "aggregate": self.aggregate,
            "failure_taxonomy": self.failures.to_dict(),
            "per_atom": self.per_atom,
            "latency_s": round(self.latency_s, 4),
            "memory_bytes": self.memory_bytes,
        }
        if self.raw_lines is not None:
            payload["raw_lines"] = dict(self.raw_lines)
        return payload


def _report_aggregate(report: EvalReport) -> dict[str, Any]:
    return {
        "exact_gold_span": report.exact_gold_span,
        "span_character_f1": round(report.span_character_f1, 4),
        "assertion_state_correct": report.assertion_state_correct,
        "support_direct_exact": report.support_direct_exact,
        "support_normalized": report.support_normalized,
        "support_review_required": report.support_review_required,
        "coverage": round(report.coverage, 4),
        "correct_abstention": report.correct_abstention,
    }


def _per_atom(report: EvalReport) -> dict[str, dict[str, Any]]:
    return {
        item.atom_id: {
            "exact_gold_span": item.exact_gold_span,
            "span_character_f1": round(item.span_character_f1, 4),
            "support_relation": (
                item.support_relation.value if item.support_relation else None
            ),
            "assertion_state_correct": item.assertion_state_correct,
            "abstained": item.abstained,
            "malformed": item.malformed,
            "omitted": item.omitted,
            "spurious_atom": item.spurious_atom,
        }
        for item in report.atom_results
    }


def run_case(
    track: TrackConfig,
    case: HarnessCase,
    *,
    capture_raw_lines: bool = False,
) -> HarnessResult:
    adapter = track.adapter_factory()
    batch = adapter.propose(case.model_input, case.atom_specs)
    predicted, report = run_pipeline(case.model_input, batch, gold=case.gold)
    assert report is not None
    raw_lines = None
    if capture_raw_lines:
        raw_lines = {
            atom.atom_id: (
                "NOT_MENTIONED"
                if atom.abstained
                else (
                    f"{atom.assertion_state.value if atom.assertion_state else 'MALFORMED'}: "
                    + " ".join(f'"{q}"' for q in atom.quotes)
                    if atom.quotes
                    else "MALFORMED"
                )
            )
            for atom in batch.atoms
        }
    return HarnessResult(
        track=track.track,
        model_id=track.model_id,
        test_set=case.test_set,
        encounter_id=case.encounter_id,
        cost_class=track.cost_class,
        aggregate=_report_aggregate(report),
        failures=FailureTaxonomy.from_report(report),
        per_atom=_per_atom(report),
        latency_s=batch.latency_s,
        memory_bytes=batch.memory_bytes,
        raw_lines=raw_lines,
    )


def run_matrix(
    tracks: Sequence[TrackConfig],
    cases: Sequence[HarnessCase],
    *,
    capture_raw_lines: bool = False,
) -> list[HarnessResult]:
    results: list[HarnessResult] = []
    for track in tracks:
        for case in cases:
            results.append(
                run_case(track, case, capture_raw_lines=capture_raw_lines)
            )
    return results


def write_results(
    results: Sequence[HarnessResult],
    output_path: Path,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "nano.p1.harness_result.v0",
        "results": [item.to_dict() for item in results],
    }
    if extra:
        payload["meta"] = dict(extra)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
