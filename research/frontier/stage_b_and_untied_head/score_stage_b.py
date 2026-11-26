"""Strict Mechanical Scorer for Stage B (`RESEARCH-PLAN §20.5a`: Criteria B1, B2, B3, and F2 Gate)."""
from __future__ import annotations
import hashlib
import json
import math
import sys
from pathlib import Path

SEEDS = (1000, 1001, 1002)
CHANCE = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "piqa": 0.50,
    "winogrande": 0.50,
    "openbookqa": 0.25,
    "lambada": 0.0,
}


def paired_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / max(1, n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    t_val = (mean / se) if se > 1e-12 else (math.copysign(999.0, mean) if abs(mean) > 0 else 0.0)
    return {
        "n": n,
        "mean": round(mean, 5),
        "sd": round(sd, 5),
        "se": round(se, 5),
        "t": round(t_val, 3),
        "sig_95_df2": bool(abs(t_val) > 4.302652729911275) if n == 3 else False,
        "all_positive": all(x > 0 for x in diffs),
        "per_seed": [round(x, 5) for x in diffs],
    }


def score_stage_b_bundle(bundle: dict) -> dict:
    required = ["stageb_P_parent", "stageb_A_all30_s1000", "stageb_C2_s1000", "stageb_R_s1000"] + [
        f"stageb_{arm}_s{s}" for arm in ("T", "C1") for s in SEEDS
    ]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise ValueError(f"Missing required Stage B receipts: {missing}")

    val_ref = bundle["stageb_P_parent"]["provenance"]["val_sha256"]
    tot_tok_ref = None
    for k in required:
        rec = bundle[k]
        if rec.get("completed") is not True:
            raise ValueError(f"{k}: completed != True")
        if rec["provenance"]["val_sha256"] != val_ref:
            raise ValueError(f"{k}: mismatched val_sha256")
        if rec.get("nonfinite_steps", 0) != 0:
            raise ValueError(f"{k}: had {rec['nonfinite_steps']} non-finite training steps")
        if rec["arm"] in ("T", "C1", "C2", "R"):
            tok_cnt = int(rec["recipe"]["total_student_tokens"])
            if tot_tok_ref is None:
                tot_tok_ref = tok_cnt
            elif tok_cnt != tot_tok_ref:
                raise ValueError(f"{k}: total_student_tokens {tok_cnt} != {tot_tok_ref}")

    p_loss = float(bundle["stageb_P_parent"]["final_heldout_nats"]["mix"])
    a30_loss = float(bundle["stageb_A_all30_s1000"]["final_heldout_nats"]["mix"])
    t_losses = [float(bundle[f"stageb_T_s{s}"]["final_heldout_nats"]["mix"]) for s in SEEDS]
    c1_losses = [float(bundle[f"stageb_C1_s{s}"]["final_heldout_nats"]["mix"]) for s in SEEDS]
    c2_loss = float(bundle["stageb_C2_s1000"]["final_heldout_nats"]["mix"])
    r_loss = float(bundle["stageb_R_s1000"]["final_heldout_nats"]["mix"])

    # B1: T has lower loss than C1 in all 3 seeds AND mean(C1 - T) >= 0.10 nats
    diffs_c1_minus_t = [c1 - t for c1, t in zip(c1_losses, t_losses)]
    b1_stats = paired_stats(diffs_c1_minus_t)
    b1_pass = bool(b1_stats["all_positive"] and b1_stats["mean"] >= 0.10)

    # B2: T closes >= 50% of the gap between layer-aligned all-converted (A_all30) and parent (P)
    t_mean = sum(t_losses) / len(t_losses)
    c1_mean = sum(c1_losses) / len(c1_losses)
    gap = a30_loss - p_loss
    target_50pct = p_loss + 0.50 * gap
    t_gap_closed = ((a30_loss - t_mean) / gap) if gap > 1e-6 else 0.0
    c1_gap_closed = ((a30_loss - c1_mean) / gap) if gap > 1e-6 else 0.0
    b2_pass = bool(t_mean <= target_50pct and gap > 0)

    # B3: Secondary 7-task zero-shot accuracies rescaled as (acc - chance) / (parent - chance)
    p_tasks = bundle["stageb_P_parent"]["downstream_7tasks"]
    b3_tasks = {}
    for task, p_info in p_tasks.items():
        metric = "acc_norm" if "acc_norm" in p_info else "acc"
        ch = CHANCE.get(task, 0.0)
        p_val = float(p_info[metric])
        denom = p_val - ch

        t_vals = [float(bundle[f"stageb_T_s{s}"]["downstream_7tasks"][task][metric]) for s in SEEDS]
        c1_vals = [float(bundle[f"stageb_C1_s{s}"]["downstream_7tasks"][task][metric]) for s in SEEDS]
        c2_val = float(bundle["stageb_C2_s1000"]["downstream_7tasks"][task][metric])
        r_val = float(bundle["stageb_R_s1000"]["downstream_7tasks"][task][metric])

        t_m = sum(t_vals) / len(t_vals)
        c1_m = sum(c1_vals) / len(c1_vals)
        b3_tasks[task] = {
            "metric": metric,
            "parent": round(p_val, 4),
            "T_mean": round(t_m, 4),
            "C1_mean": round(c1_m, 4),
            "C2": round(c2_val, 4),
            "R": round(r_val, 4),
            "T_minus_C1": paired_stats([tv - cv for tv, cv in zip(t_vals, c1_vals)]),
            "T_chance_adj_retention": round((t_m - ch) / denom, 4) if abs(denom) > 1e-4 else None,
            "C1_chance_adj_retention": round((c1_m - ch) / denom, 4) if abs(denom) > 1e-4 else None,
        }

    # Mechanical F2 decision (RESEARCH-PLAN §20.5a & §20.6)
    if b1_pass and b2_pass:
        f2 = "ADVANCE_TO_STAGE_C_WITH_INHERITED_PROJECTIONS"
    elif (not b2_pass) and (c1_mean > target_50pct):
        f2 = "NEITHER_RESOLVED_AT_250M_ELIGIBLE_FOR_500M_EXTENSION"
    elif (not b1_pass) and (t_mean <= target_50pct or c1_mean <= target_50pct):
        f2 = "HYBRID_DISTILLATION_SUCCEEDS_WEIGHT_INHERITANCE_NOT_NEEDED"
    else:
        f2 = "REJECT_INHERITANCE_ADVANTAGE_AT_135M"

    return {
        "student_tokens_verified": tot_tok_ref,
        "heldout_mix_nats": {
            "P_parent": round(p_loss, 5),
            "A_all30_stage1": round(a30_loss, 5),
            "midpoint_50pct_target": round(target_50pct, 5),
            "T_mean": round(t_mean, 5),
            "T_per_seed": [round(x, 5) for x in t_losses],
            "C1_mean": round(c1_mean, 5),
            "C1_per_seed": [round(x, 5) for x in c1_losses],
            "C2_no_align": round(c2_loss, 5),
            "R_all_random": round(r_loss, 5),
        },
        "criteria": {
            "B1_T_beats_C1_by_0_10_nats": {"pass": b1_pass, "stats": b1_stats},
            "B2_T_closes_50pct_gap": {
                "pass": b2_pass,
                "T_gap_fraction_closed": round(t_gap_closed, 4),
                "C1_gap_fraction_closed": round(c1_gap_closed, 4),
            },
            "B3_downstream_7tasks": b3_tasks,
        },
        "verdicts": {
            "B1": "PASS" if b1_pass else "FAIL",
            "B2": "PASS" if b2_pass else "FAIL",
            "F2_gate": f2,
        },
    }


if __name__ == "__main__":
    p = Path(sys.argv[1])
    data = json.loads(p.read_text()) if p.is_file() else {f.stem: json.loads(f.read_text()) for f in sorted(p.glob("stageb_*.json"))}
    print(json.dumps(score_stage_b_bundle(data), indent=2))
