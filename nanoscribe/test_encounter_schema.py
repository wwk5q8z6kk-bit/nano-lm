"""Encounter Representation v0 — JSON schema presence and wire-shape roundtrip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoscribe.campaign_datasets import SMOKE_SUITE_REVISION, campaign_cases
from nanoscribe.distill_train_suite import distill_train_cases
from nanoscribe.encounter import EncounterRecord, SCHEMA_VERSION
from nanoscribe.screening_suite import screening_core_cases

SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "encounter_v0.schema.json"


def test_encounter_schema_file_exists() -> None:
    assert SCHEMA_PATH.is_file()
    payload = json.loads(SCHEMA_PATH.read_text())
    assert payload["properties"]["schema_version"]["const"] == SCHEMA_VERSION


@pytest.mark.parametrize(
    "suite_loader",
    [
        lambda: campaign_cases(SMOKE_SUITE_REVISION),
        distill_train_cases,
        screening_core_cases,
    ],
)
def test_encounter_gold_roundtrip(suite_loader) -> None:
    for case in suite_loader():
        wire = case.gold.to_dict()
        assert wire["schema_version"] == SCHEMA_VERSION
        restored = EncounterRecord.from_dict(wire)
        assert restored.to_dict() == wire


def test_encounter_json_roundtrip() -> None:
    case = campaign_cases(SMOKE_SUITE_REVISION)[0]
    restored = EncounterRecord.from_json(case.gold.to_json())
    assert restored.encounter_id == case.gold.encounter_id
