# Offline pin for E1 frozen utility + kill/survive decision rule.
# Recomputes U from committed component rows and re-runs aggregate_decision
# against results_e1_utility.json. No model, no GPU, no network.
from __future__ import annotations

import json
import sys
from pathlib import Path

TRAJ = Path(__file__).resolve().parent
REPO = TRAJ.parent
sys.path.insert(0, str(REPO))

from trajectory.e1.common import (  # noqa: E402
    DEFAULT_WEIGHTS,
    SENSITIVITY,
    aggregate_decision,
)


def _load_utility():
    return json.loads((TRAJ / "results_e1_utility.json").read_text())


def _recompute_U(row: dict, weights=None) -> float:
    w = weights or DEFAULT_WEIGHTS
    return (
        w["alpha"] * row["P"]
        - w["beta"] * row["M"]
        - w["gamma"] * row["rho"]
        - w["lam"] * row["L_p50"]
        - w["kappa"] * row["C"]
    )


def test_default_weights_match_prereg():
    assert DEFAULT_WEIGHTS == dict(alpha=1.0, beta=0.5, gamma=0.3, lam=0.02, kappa=0.05)
    assert len(SENSITIVITY) == 3


def test_row_U_recomputes_from_components():
    data = _load_utility()
    for row in data["rows"]:
        got = _recompute_U(row)
        assert abs(got - row["U"]) < 1e-9, (row["method"], row["verify_on"], got, row["U"])


def test_official_m0_is_argmax_of_two_lora_arms():
    data = _load_utility()
    d = data["decision"]
    cands = d["official_m0_candidates"]
    assert set(cands) == {
        "M0_pythia160m_lora",
        "M0_ownstack_chinchilla_lora",
    }
    best = max(cands.items(), key=lambda kv: kv[1])
    assert d["m0"] == best[0]
    assert abs(d["U_m0"] - best[1]) < 1e-12
    assert d["venue"] == "runpod-cuda"


def test_aggregate_decision_reproduces_kill():
    """Rebuild verify-on method summaries from committed rows and re-decide."""
    data = _load_utility()
    by_method = {}
    for row in data["rows"]:
        if not row["verify_on"]:
            continue
        sens = []
        for w in SENSITIVITY:
            sens.append({"weights": w, "U_mean": _recompute_U(row, w)})
        by_method[row["method"]] = {
            "verify_on": True,
            "summary": {"U": row["U"], "U_sensitivity": sens},
        }

    m0 = data["decision"]["m0"]
    got = aggregate_decision(by_method, m0)
    gold = data["decision"]
    assert got["verdict"] == "KILL"
    assert got["verdict"] == gold["verdict"]
    assert got["m0"] == gold["m0"]
    assert abs(got["U_m0"] - gold["U_m0"]) < 1e-12
    assert got["best_nonlm"] == gold["best_nonlm"]
    assert abs(got["U_best_nonlm"] - gold["U_best_nonlm"]) < 1e-12
    assert abs(got["margin"] - gold["margin"]) < 1e-12
    assert got["sensitivity_flip"] is False
    assert got["U_best_nonlm"] >= got["U_m0"] - got["delta"]


def test_kill_stable_vs_m1_and_m2():
    data = _load_utility()
    u = data["decision"]["U_all"]
    u0 = data["decision"]["U_m0"]
    delta = data["decision"]["delta"]
    assert u["M1_template"] >= u0 - delta
    assert u["M2_dict_span"] >= u0 - delta


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in fns:
        f()
        print(f"  PASS {n}")
    print(f"e1 utility recompute: {len(fns)}/{len(fns)} PASS")
