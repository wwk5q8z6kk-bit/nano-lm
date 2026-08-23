# Campaign dataset partition tests.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.campaign_datasets import campaign_cases, suite_manifest


def test_campaign_v1_has_three_encounters() -> None:
    cases = campaign_cases("campaign_v1")
    ids = {c.encounter_id for c in cases}
    assert ids == {"enc-1", "enc-2", "enc-3"}


def test_suite_manifest_revision() -> None:
    manifest = suite_manifest()
    assert manifest["schema"] == "nano.campaign.dataset.v1"
    assert manifest["smoke_suite"] == "p1_contract_smoke_v1"
    assert manifest["screening_suite"] == "p1_screening_eval_v1"


def test_campaign_partitions_no_leakage() -> None:
    from nanoscribe.campaign_datasets import validate_campaign_partitions

    validate_campaign_partitions()


if __name__ == "__main__":
    test_campaign_v1_has_three_encounters()
    test_suite_manifest_revision()
    print("campaign_dataset pins: 2/2 PASS")
