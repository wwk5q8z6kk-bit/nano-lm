# Offline pin for E3 normalize construct (exact vs normalize-then-match).
# Unit-tests normalize_value + pins committed results_e3_normalize_construct.json.
# No model, no GPU, no network.
from __future__ import annotations

import json
import sys
from pathlib import Path

TRAJ = Path(__file__).resolve().parent
REPO = TRAJ.parent
sys.path.insert(0, str(REPO))

from trajectory.e1.common import PLURAL_MAP, normalize_value, score_fields  # noqa: E402


def _load_construct():
    return json.loads((TRAJ / "results_e3_normalize_construct.json").read_text())


def test_plural_map_frozen():
    assert PLURAL_MAP["drugs"] == "drug"
    assert PLURAL_MAP["pills"] == "pill"
    assert normalize_value("allergy pills") == "allergy pill"
    assert normalize_value("sulfa drugs") == "sulfa drug"


def test_normalize_strips_articles_and_punct():
    assert normalize_value("  The Peanuts! ") == "peanut"
    assert normalize_value("an aspirin") == "aspirin"
    assert normalize_value("IBUPROFEN") == "ibuprofen"


def test_normalize_does_not_invent_rescue_when_stems_differ():
    assert normalize_value("penicillin") != normalize_value("amoxicillin")
    assert normalize_value("peanuts") != normalize_value("tree nuts")


def test_score_fields_normalize_mode_matches_exact_when_no_morphology():
    pred = dict(cc="fever", dur="2 days", sev="mild", med="none", alg="peanuts")
    truth = dict(cc="fever", dur="2 days", sev="mild", med="none", alg="peanuts")
    assert score_fields(pred, truth, mode="exact") == score_fields(pred, truth, mode="normalize")


def test_score_fields_normalize_can_rescue_plural_only():
    pred = dict(cc="fever", dur="2 days", sev="mild", med="allergy pills", alg="none")
    truth = dict(cc="fever", dur="2 days", sev="mild", med="allergy pill", alg="none")
    exact = score_fields(pred, truth, mode="exact")
    norm = score_fields(pred, truth, mode="normalize")
    assert exact["med"] != "correct"
    assert norm["med"] == "correct"


def test_committed_construct_pins_zero_rescues():
    data = _load_construct()
    m0 = data["stats_all_methods"]["M0_scale|voff"]
    assert m0["norm_rescue_count"] == 0
    assert m0["both_fail"] == 486
    assert abs(m0["gap_shrink_pts"]) < 1e-12
    assert data["decision_auto"]["verdict_auto"] == "EXACT_NOT_OVERSTATING_BY_NORMALIZE"
    for key, st in data["stats_all_methods"].items():
        assert st["norm_rescue_count"] == 0, key
        assert abs(st["exact_rate"] - st["norm_rate"]) < 1e-12, key


def test_human_pack_verdict_exact_survives_if_present():
    path = TRAJ / "results_e3_human.json"
    assert path.exists()
    human = json.loads(path.read_text())
    verdict = (
        human.get("verdict_prereg")
        or human.get("verdict")
        or human.get("decision", {}).get("verdict")
    )
    assert verdict == "EXACT_SURVIVES"
    assert human.get("n") == 100
    counts = human.get("counts") or {}
    if "label_faithful" in counts:
        assert counts["label_faithful"] == 0
    if "label_unfaithful" in counts:
        assert counts["label_unfaithful"] == 100


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in fns:
        f()
        print(f"  PASS {n}")
    print(f"e3 normalize: {len(fns)}/{len(fns)} PASS")
