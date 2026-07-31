"""Offline checks for E3 automated + agent-rubric committed artifacts (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

TRAJ = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def e3_auto():
    path = TRAJ / "results_e3_normalize_construct.json"
    assert path.is_file()
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def e3_agent():
    path = TRAJ / "results_e3_human.json"
    assert path.is_file()
    return json.loads(path.read_text())


def test_auto_primary_zero_rescues(e3_auto):
    p = e3_auto["primary_m0_voff"]
    assert p["norm_rescue_count"] == 0
    assert p["both_fail"] == 486
    assert p["gap_shrink_pts"] == 0.0


def test_auto_all_methods_zero_rescue_or_absent(e3_auto):
    for key, stats in e3_auto["stats_all_methods"].items():
        assert stats["norm_rescue_count"] == 0, key


def test_agent_rubric_not_clinician(e3_agent):
    rater = e3_agent["rater"]
    assert rater["id"] == "agent-rubric-pass-1"
    assert rater.get("iaa") is None
    assert e3_agent["rates"]["faithful_rate"] == 0.0
    assert e3_agent["counts"]["label_unfaithful"] == 100
    assert e3_agent["verdict_prereg"] == "EXACT_SURVIVES"
