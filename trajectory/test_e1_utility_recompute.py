"""Offline recompute of frozen E1 utility from committed result JSON (no GPU)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TRAJ = Path(__file__).resolve().parent
REPO = TRAJ.parent
sys.path.insert(0, str(TRAJ))

from e1.common import DEFAULT_WEIGHTS, normalize_value  # noqa: E402


@pytest.fixture(scope="module")
def utility():
    path = TRAJ / "results_e1_utility.json"
    assert path.is_file(), "committed E1 utility JSON missing"
    return json.loads(path.read_text())


def test_default_weights_match_prereg(utility):
    assert utility["weights_default"] == DEFAULT_WEIGHTS


def test_u_recomputes_from_components_verify_on(utility):
    w = utility["weights_default"]
    for row in utility["rows"]:
        if not row.get("verify_on"):
            continue
        u_hat = (
            w["alpha"] * row["P"]
            - w["beta"] * row["M"]
            - w["gamma"] * row["rho"]
            - w["lam"] * row["L_p50"]
            - w["kappa"] * row["C"]
        )
        assert abs(u_hat - row["U"]) < 1e-9, (row["method"], u_hat, row["U"])


def test_kill_m1_exceeds_official_m0(utility):
    d = utility["decision"]
    assert d["verdict"] == "KILL"
    assert d["best_nonlm"] == "M1_template"
    assert d["m0"] == "M0_pythia160m_lora"
    assert d["U_best_nonlm"] > d["U_m0"]
    assert abs(d["U_best_nonlm"] - 0.9989993963311425) < 1e-12
    assert abs(d["U_m0"] - 0.9252173639550433) < 1e-12
    # M2 does not dominate official M0 (may still be within delta)
    assert d["U_all"]["M2_dict_span"] < d["U_m0"]


def test_rho_is_review_load_not_hallucination_field(utility):
    """ρ component exists beside halluc; they are not the same column."""
    row = next(r for r in utility["rows"] if r["method"] == "M1_template" and r["verify_on"])
    assert "rho" in row and "halluc" in row
    assert row["rho"] == 0.0
    assert row["halluc"] == 0.0


def test_normalize_value_trivial_surface_rules():
    assert normalize_value("The Sulfa Drugs!") == "sulfa drug"
    assert normalize_value("  IBUPROFEN  ") == "ibuprofen"
    assert normalize_value("a headache") == "headache"
