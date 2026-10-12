"""Dataset partition leakage validator — fail closed on eval/train overlap."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from nanoscribe.encounter import EncounterRecord, EvidenceSpan, assemble_source
from nanoscribe.harness import HarnessCase


class DatasetPartition(str, Enum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    INTERNAL_TEST = "INTERNAL_TEST"
    FROZEN_SCREENING_EVAL = "FROZEN_SCREENING_EVAL"


@dataclass(frozen=True, slots=True)
class LeakageViolation:
    kind: str
    value: str
    left: str
    right: str


def _source_hash(case: HarnessCase) -> str:
  payload = case.model_input.source.to_dict()
  blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(blob.encode()).hexdigest()


def _dialogue_hash(case: HarnessCase) -> str:
  turns = tuple(
    (turn.speaker.value, turn.text)
    for turn in case.model_input.source.turns
  )
  blob = json.dumps(turns, separators=(",", ":"))
  return hashlib.sha256(blob.encode()).hexdigest()


def _gold_span_keys(gold: EncounterRecord) -> frozenset[str]:
  keys: set[str] = set()
  for span in gold.evidence:
    if isinstance(span, EvidenceSpan):
      keys.add(f"{span.source_id}:{span.start}:{span.end}:{span.text}")
  return frozenset(keys)


def _case_fingerprints(case: HarnessCase) -> dict[str, str | frozenset[str]]:
  return {
    "case_id": case.encounter_id,
    "source_hash": _source_hash(case),
    "dialogue_hash": _dialogue_hash(case),
    "gold_spans": _gold_span_keys(case.gold),
  }


def validate_partition_disjointness(
    partitions: Mapping[DatasetPartition, Sequence[HarnessCase]],
) -> list[LeakageViolation]:
    """Fail on shared case IDs, source hashes, dialogue hashes, or gold spans."""
    violations: list[LeakageViolation] = []
    indexed: dict[DatasetPartition, list[dict[str, Any]]] = {}
    for partition, cases in partitions.items():
        indexed[partition] = [_case_fingerprints(case) for case in cases]

    frozen = DatasetPartition.FROZEN_SCREENING_EVAL
    train_like = {DatasetPartition.TRAIN, DatasetPartition.DEV, DatasetPartition.INTERNAL_TEST}
    for left_part in train_like:
        for right_part in (frozen,):
            left_cases = indexed.get(left_part, [])
            right_cases = indexed.get(right_part, [])
            for left in left_cases:
                for right in right_cases:
                    if left["case_id"] == right["case_id"]:
                        violations.append(
                            LeakageViolation("case_id", left["case_id"], left_part.value, right_part.value)
                        )
                    if left["source_hash"] == right["source_hash"]:
                        violations.append(
                            LeakageViolation(
                                "source_hash", left["source_hash"][:16], left_part.value, right_part.value
                            )
                        )
                    if left["dialogue_hash"] == right["dialogue_hash"]:
                        violations.append(
                            LeakageViolation(
                                "dialogue_hash", left["dialogue_hash"][:16], left_part.value, right_part.value
                            )
                        )
                    shared_spans = left["gold_spans"] & right["gold_spans"]
                    for span_key in shared_spans:
                        violations.append(
                            LeakageViolation("gold_span", span_key, left_part.value, right_part.value)
                        )
    return violations


def assert_no_leakage(partitions: Mapping[DatasetPartition, Sequence[HarnessCase]]) -> None:
    violations = validate_partition_disjointness(partitions)
    if violations:
        sample = violations[:5]
        detail = "; ".join(f"{item.kind}={item.value!r} ({item.left}↔{item.right})" for item in sample)
        raise ValueError(f"dataset leakage detected ({len(violations)} violations): {detail}")
