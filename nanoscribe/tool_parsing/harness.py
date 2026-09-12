"""Parser harness — run CapabilityToolParser against JSON fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoscribe.capabilities import CapabilityId, CapabilityToolParser
from nanoscribe.tool_calling import ToolCallParseOutcome

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "tool_calls"


@dataclass(frozen=True, slots=True)
class FixtureExpectation:
    outcome: ToolCallParseOutcome
    capability_id: CapabilityId | None = None
    atom_count: int | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class FixtureResult:
    name: str
    passed: bool
    actual_outcome: ToolCallParseOutcome
    message: str = ""


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"fixture must be an object: {path}")
    return data


def _expectation_from_fixture(data: dict[str, Any]) -> FixtureExpectation:
    outcome_raw = data.get("expected_outcome", "success")
    capability_raw = data.get("expected_capability")
    capability_id = CapabilityId(capability_raw) if capability_raw else None
    return FixtureExpectation(
        outcome=ToolCallParseOutcome(outcome_raw),
        capability_id=capability_id,
        atom_count=data.get("expected_atom_count"),
        error_code=data.get("expected_error_code"),
    )


def run_fixture(path: Path, parser: CapabilityToolParser | None = None) -> FixtureResult:
    parser = parser or CapabilityToolParser()
    data = _load_fixture(path)
    name = str(data.get("name", path.stem))
    expectation = _expectation_from_fixture(data)

    if "message" in data:
        result = parser.parse_message(data["message"])
    elif "tool_calls" in data:
        result = parser.parse_tool_calls(data["tool_calls"])
    elif "response" in data:
        result = parser.parse_openai_response(data["response"])
    else:
        return FixtureResult(name, False, ToolCallParseOutcome.EMPTY, "fixture missing message/tool_calls/response")

    if result.outcome != expectation.outcome:
        return FixtureResult(
            name,
            False,
            result.outcome,
            f"outcome {result.outcome.value} != expected {expectation.outcome.value}",
        )
    if expectation.capability_id is not None and result.capability_id != expectation.capability_id:
        return FixtureResult(
            name,
            False,
            result.outcome,
            f"capability {result.capability_id} != expected {expectation.capability_id}",
        )
    if expectation.atom_count is not None:
        count = len(result.candidate.atoms) if result.candidate else 0
        if count != expectation.atom_count:
            return FixtureResult(
                name,
                False,
                result.outcome,
                f"atom_count {count} != expected {expectation.atom_count}",
            )
    if expectation.error_code is not None:
        code = result.error.code if result.error else None
        if code != expectation.error_code:
            return FixtureResult(
                name,
                False,
                result.outcome,
                f"error_code {code} != expected {expectation.error_code}",
            )
    return FixtureResult(name, True, result.outcome)


def run_all_fixtures(
    fixtures_dir: Path | None = None,
    parser: CapabilityToolParser | None = None,
) -> tuple[FixtureResult, ...]:
    root = fixtures_dir or FIXTURES_DIR
    paths = sorted(root.glob("*.json"))
    return tuple(run_fixture(path, parser=parser) for path in paths)


def assert_all_fixtures_pass(fixtures_dir: Path | None = None) -> int:
    results = run_all_fixtures(fixtures_dir=fixtures_dir)
    failures = [result for result in results if not result.passed]
    if failures:
        lines = [f"{item.name}: {item.message}" for item in failures]
        raise AssertionError("fixture failures:\n" + "\n".join(lines))
    return len(results)
