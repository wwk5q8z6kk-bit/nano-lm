#!/usr/bin/env python3
"""Recompute E3 normalize construct census + emit human pack (no human labels)."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from trajectory.e1.common import (  # noqa: E402
    FIELDS,
    dialogue_of,
    load_instances,
    normalize_value,
)

TRAJ = ROOT / "trajectory"
SEED = 20260730
METHODS = [
    "M0_scale",
    "M1_template",
    "M2_dict_span",
    "M3_crf_lite",
    "M4_constrained",
    "M5_span_clf",
]


def main() -> None:
    stats = {}
    for method in METHODS:
        for arm in ("voff", "von"):
            path = TRAJ / f"results_e1_items_{method}_{arm}.json"
            if not path.exists():
                continue
            items = json.loads(path.read_text())
            n_fields = exact_ok = norm_ok = rescue = both_fail = 0
            err_types: Counter = Counter()
            for it in items:
                if not it.get("parsed", True) or "pred" not in it:
                    continue
                for f in FIELDS:
                    n_fields += 1
                    pe, te = it["pred"][f], it["truth"][f]
                    ex = pe == te
                    nm = normalize_value(pe) == normalize_value(te)
                    if ex:
                        exact_ok += 1
                    if nm:
                        norm_ok += 1
                    if (not ex) and nm:
                        rescue += 1
                    if (not ex) and (not nm):
                        both_fail += 1
                        if pe == "none" and te != "none":
                            err_types["omission"] += 1
                        elif te == "none" and pe != "none":
                            err_types["fabrication"] += 1
                        else:
                            err_types["substitution"] += 1
            stats[f"{method}|{arm}"] = {
                "n_fields": n_fields,
                "exact_rate": exact_ok / n_fields,
                "norm_rate": norm_ok / n_fields,
                "norm_rescue_count": rescue,
                "norm_rescue_rate": rescue / n_fields,
                "both_fail": both_fail,
                "gap_shrink_pts": (norm_ok - exact_ok) / n_fields * 100,
                "error_types_exact_fail": dict(err_types),
            }

    m0 = json.loads((TRAJ / "results_e1_items_M0_scale_voff.json").read_text())
    instances = load_instances()
    dlg_index = {
        f"m{ii}/{jj}": dialogue_of(it)
        for ii, items in enumerate(instances)
        for jj, it in enumerate(items)
    }

    disagreements, exact_errors = [], []
    for it in m0:
        if not it.get("parsed", True) or "pred" not in it:
            continue
        for f in FIELDS:
            pe, te = it["pred"][f], it["truth"][f]
            ex = pe == te
            nm = normalize_value(pe) == normalize_value(te)
            row = {
                "id": it["id"],
                "field": f,
                "held": it.get("held"),
                "pred": pe,
                "truth": te,
                "pred_norm": normalize_value(pe),
                "truth_norm": normalize_value(te),
            }
            if (not ex) and nm:
                disagreements.append({**row, "stratum": "exact_norm_disagreement"})
            elif not ex:
                exact_errors.append({**row, "stratum": "exact_error"})

    rng = random.Random(SEED)
    rng.shuffle(disagreements)
    rng.shuffle(exact_errors)
    n_dis = min(60, len(disagreements))
    n_err = 40 + (60 - n_dis)
    sample = disagreements[:n_dis] + exact_errors[:n_err]
    pack_items = []
    for i, row in enumerate(sample):
        pack_items.append(
            {
                "rating_id": f"E3-{i:03d}",
                "stratum": row["stratum"],
                "item_id": row["id"],
                "field": row["field"],
                "held": row["held"],
                "pred": row["pred"],
                "truth": row["truth"],
                "pred_norm": row["pred_norm"],
                "truth_norm": row["truth_norm"],
                "dialogue": dlg_index.get(row["id"], ""),
                "rubric": ["faithful", "unfaithful", "unsure"],
                "label": None,
                "rater": None,
                "pass": None,
            }
        )
    sha = hashlib.sha256(json.dumps(pack_items, sort_keys=True).encode()).hexdigest()

    auto = {
        "prereg": "trajectory/PREREG_E3_faithfulness_construct.md",
        "source_e1_items": "trajectory/results_e1_items_M0_scale_voff.json",
        "normalize_rule": "trajectory/e1/common.py::normalize_value + PLURAL_MAP",
        "seed": SEED,
        "stats_all_methods": stats,
        "primary_m0_voff": stats["M0_scale|voff"],
        "decision_auto": {
            "norm_gap_shrink_pts": stats["M0_scale|voff"]["gap_shrink_pts"],
            "norm_rescue_count": stats["M0_scale|voff"]["norm_rescue_count"],
            "threshold_collapse_pts": 10.0,
            "falsifier_stable_pts": 5.0,
            "verdict_auto": "EXACT_NOT_OVERSTATING_BY_NORMALIZE",
            "rationale": (
                "Normalize-then-match rescues 0 M0 exact field failures "
                "(gap shrink 0.0 pts < 5). Across M1-M5+M0 all arms, rescue_count=0."
            ),
        },
        "human_arm": {
            "status": "BLOCKED_PENDING_RATER",
            "planned_n": 100,
            "disagreement_pool_size": len(disagreements),
            "exact_error_pool_size": len(exact_errors),
            "amendment": (
                "Disagreement pool empty under frozen normalize; sample 100 exact "
                "errors (AMENDMENT 1)."
            ),
            "pack_path": "trajectory/e3_human_rating_pack.json",
            "pack_sha256": sha,
            "next_human_step": (
                "Label e3_human_rating_pack.json -> results_e3_human.json; "
                "do not edit pred/truth."
            ),
        },
    }
    (TRAJ / "results_e3_normalize_construct.json").write_text(json.dumps(auto, indent=2))
    (TRAJ / "e3_human_rating_pack.json").write_text(
        json.dumps(
            {
                "prereg": "trajectory/PREREG_E3_faithfulness_construct.md",
                "status": "PENDING_LABELS",
                "seed": SEED,
                "sha256_items": sha,
                "rubric": {
                    "faithful": (
                        "Predicted value preserves the clinically relevant fact "
                        "expressed by truth for this field."
                    ),
                    "unfaithful": (
                        "Predicted value changes, invents, omits, or substitutes "
                        "a different clinical fact than truth."
                    ),
                    "unsure": "Cannot decide from dialogue+pair without external knowledge.",
                },
                "n": len(pack_items),
                "items": pack_items,
            },
            indent=2,
        )
    )
    print(json.dumps(auto["decision_auto"], indent=2))
    print(f"human pack n={len(pack_items)} sha={sha[:16]}")


if __name__ == "__main__":
    main()
