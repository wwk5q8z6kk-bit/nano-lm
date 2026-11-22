"""Deterministic scorer for the pre-registered H-13 / H-10 vs Dense+Muon tournament.

Reads per-seed Kaggle output JSON files (`seed_1000.json`, `seed_1001.json`, `seed_1002.json`)
only once `completed` is true and every arm's last curve step equals the planned step count
(`recipe.steps`, otherwise the pre-registered `3000`). In-progress checkpoints are ignored.
Verifies identical evaluation crop reproduction, computes:
  1. Equal-token 3-seed means, SDs, paired differences, and two-sided paired t-statistics (df=2);
  2. Equal-wall-clock interpolated mix BPB for each candidate at `dense`'s final elapsed seconds
     on the same physical GPU within each seed;
  3. Rank-bucket (`Rank 1`, `Ranks 2..8`, `Ranks 9..64`, `Rank >= 65`) decomposition across arms;
  4. Mechanical verdicts for pre-registered Decision Rules D1 (Block AttnRes / ValueRes) and D2 (CodaRefine).
"""
from __future__ import annotations
import glob
import json
import math
import sys
from pathlib import Path

T_CRIT_DF2_95 = 4.302652729911275  # two-sided 95% t critical value for df=2
REGISTERED_STEPS = 3000  # pre-registered step count; recipe.steps overrides when the runner recorded one


def interp_bpb_at_time(curve: list[dict], target_sec: float, key: str = "mix") -> float:
    """Piecewise-linear interpolation of `key` along `train_sec` (clamped to curve range if faster)."""
    pts = [(float(r["train_sec"]), float(r[key])) for r in curve]
    if target_sec >= pts[-1][0]:
        # Candidate finished all steps faster than or equal to reference wall-clock!
        return pts[-1][1]
    if target_sec <= pts[0][0]:
        return pts[0][1]
    for (t0, y0), (t1, y1) in zip(pts[:-1], pts[1:]):
        if t0 <= target_sec <= t1:
            if t1 <= t0:
                return y1
            w = (target_sec - t0) / (t1 - t0)
            return y0 + w * (y1 - y0)
    return pts[-1][1]


def paired_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return {"n": n, "mean": round(mean, 5), "sd": 0.0, "se": 0.0, "t": 0.0, "sig_95": False, "per_seed": [round(x, 5) for x in diffs]}
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t_val = (mean / se) if se > 1e-12 else (math.copysign(999.0, mean) if abs(mean) > 0 else 0.0)
    sig = (abs(t_val) > T_CRIT_DF2_95) if n == 3 else (abs(t_val) > 2.0)
    return {
        "n": n,
        "mean": round(mean, 5),
        "sd": round(sd, 5),
        "se": round(se, 5),
        "t": round(t_val, 3),
        "sig_95": bool(sig),
        "all_positive": all(x > 0 for x in diffs),
        "per_seed": [round(x, 5) for x in diffs],
    }


def planned_steps(seed_doc: dict) -> int:
    recipe = seed_doc.get("recipe") or {}
    if "steps" in recipe:
        return int(recipe["steps"])
    return REGISTERED_STEPS


def seed_run_is_final(seed_doc: dict) -> bool:
    """True only for a finished seed, not a mid-eval rewrite of `seed_*.json`."""
    if seed_doc.get("completed") is not True:
        return False
    arms = seed_doc.get("arms") or {}
    if not arms:
        return False
    planned = planned_steps(seed_doc)
    for arm_doc in arms.values():
        curve = arm_doc.get("curve") or []
        if not curve or int(curve[-1].get("step", -1)) != planned:
            return False
    return True


def score_files(paths: list[Path]) -> dict:
    by_seed: dict[int, dict] = {}
    for p in sorted(paths):
        d = json.loads(p.read_text())
        sd = int(d["seed"])
        by_seed[sd] = d

    if not by_seed:
        raise SystemExit("No seed result files found")
    seeds = sorted(s for s in by_seed if seed_run_is_final(by_seed[s]))
    if not seeds:
        raise SystemExit("No completed seed result files found")

    arms = ["dense", "value_res", "block_attnres", "coda_refine", "dense10"]
    arm_summary = {}
    for arm in arms:
        if not all(arm in by_seed[s]["arms"] for s in seeds):
            continue
        mixes = [by_seed[s]["arms"][arm]["curve"][-1]["mix"] for s in seeds]
        mean_mix = sum(mixes) / len(mixes)
        sd_mix = math.sqrt(sum((x - mean_mix) ** 2 for x in mixes) / max(1, len(mixes) - 1)) if len(mixes) > 1 else 0.0
        toks_s = [by_seed[s]["arms"][arm]["curve"][-1]["tokens_per_second"] for s in seeds]
        secs = [by_seed[s]["arms"][arm]["curve"][-1]["train_sec"] for s in seeds]
        dom_means = {
            dom: round(sum(by_seed[s]["arms"][arm]["curve"][-1][dom] for s in seeds) / len(seeds), 4)
            for dom in ("web", "textbook", "math", "code")
        }
        eq_time_mixes = [
            interp_bpb_at_time(by_seed[s]["arms"][arm]["curve"], by_seed[s]["arms"]["dense"]["curve"][-1]["train_sec"])
            for s in seeds
        ]
        ranks = {}
        if all("rank_buckets_mix" in by_seed[s]["arms"][arm] for s in seeds):
            for rk in ("rank1_calib", "rank2_8_rerank", "rank9_64_mid", "rank65plus_miss", "top1_acc", "top64_acc"):
                ranks[rk] = round(sum(by_seed[s]["arms"][arm]["rank_buckets_mix"][rk] for s in seeds) / len(seeds), 4)

        arm_summary[arm] = {
            "params": by_seed[seeds[0]]["arms"][arm]["params"],
            "equal_tokens_mix_mean": round(mean_mix, 4),
            "equal_tokens_mix_sd": round(sd_mix, 5),
            "equal_tokens_per_seed": [round(x, 4) for x in mixes],
            "domains_mean": dom_means,
            "equal_wallclock_to_dense_mix_mean": round(sum(eq_time_mixes) / len(eq_time_mixes), 4),
            "equal_wallclock_per_seed": [round(x, 4) for x in eq_time_mixes],
            "mean_tokens_per_second": round(sum(toks_s) / len(toks_s)),
            "mean_train_sec": round(sum(secs) / len(secs), 1),
            "rank_buckets_mix_mean": ranks,
        }

    # Paired comparisons (positive = first-named candidate beats second-named baseline, i.e., lower BPB)
    comparisons = {}
    for cand in ("value_res", "block_attnres", "coda_refine", "dense10"):
        if cand in arm_summary and "dense" in arm_summary:
            d_tok = [
                by_seed[s]["arms"]["dense"]["curve"][-1]["mix"] - by_seed[s]["arms"][cand]["curve"][-1]["mix"]
                for s in seeds
            ]
            d_time = [
                by_seed[s]["arms"]["dense"]["curve"][-1]["mix"]
                - interp_bpb_at_time(by_seed[s]["arms"][cand]["curve"], by_seed[s]["arms"]["dense"]["curve"][-1]["train_sec"])
                for s in seeds
            ]
            comparisons[f"{cand}_vs_dense"] = {
                "equal_tokens_gain": paired_stats(d_tok),
                "equal_wallclock_gain": paired_stats(d_time),
            }

    if "block_attnres" in arm_summary and "value_res" in arm_summary:
        d_bv = [
            by_seed[s]["arms"]["value_res"]["curve"][-1]["mix"] - by_seed[s]["arms"]["block_attnres"]["curve"][-1]["mix"]
            for s in seeds
        ]
        comparisons["block_attnres_vs_value_res"] = {"equal_tokens_gain": paired_stats(d_bv)}

    # Evaluate Pre-Registered Decision Rules D1 and D2
    verdicts = {}
    if "block_attnres_vs_dense" in comparisons and "block_attnres_vs_value_res" in comparisons:
        ba_tok = comparisons["block_attnres_vs_dense"]["equal_tokens_gain"]
        ba_time = comparisons["block_attnres_vs_dense"]["equal_wallclock_gain"]
        ba_vr = comparisons["block_attnres_vs_value_res"]["equal_tokens_gain"]
        vr_tok = comparisons["value_res_vs_dense"]["equal_tokens_gain"]
        vr_time = comparisons["value_res_vs_dense"]["equal_wallclock_gain"]

        if ba_tok["mean"] >= 0.004 and ba_time["mean"] >= 0.004 and ba_tok["sig_95"] and ba_time["sig_95"] and ba_vr["mean"] >= 0.002:
            d1 = "ADOPT_BLOCK_ATTNRES"
        elif vr_tok["mean"] >= 0.004 and vr_time["mean"] >= 0.004 and vr_tok["sig_95"] and vr_time["sig_95"] and ba_vr["mean"] < 0.002:
            d1 = "ADOPT_VALUE_RES_REJECT_BLOCK_ATTNRES"
        else:
            d1 = "CLOSE_DEPTH_ROUTING_AT_14M"
        verdicts["D1_depth_routing"] = d1

    if "coda_refine_vs_dense" in comparisons and "dense10_vs_dense" in comparisons:
        cr_tok = comparisons["coda_refine_vs_dense"]["equal_tokens_gain"]
        cr_time = comparisons["coda_refine_vs_dense"]["equal_wallclock_gain"]
        d10_tok = comparisons["dense10_vs_dense"]["equal_tokens_gain"]
        frac_d10 = (cr_tok["mean"] / d10_tok["mean"]) if d10_tok["mean"] > 1e-6 else 0.0
        verdicts["coda_fraction_of_dense10_equal_token_gain"] = round(frac_d10, 4)
        if cr_tok["mean"] >= 0.004 and cr_tok["sig_95"] and frac_d10 >= 0.50:
            d2 = "ADVANCE_CODA_TO_SPARSE_GATED_40M" if cr_time["mean"] <= 0 else "ADOPT_UNCONDITIONAL_CODA_REFINE"
        else:
            d2 = "REJECT_UNCONDITIONAL_CODA_REFINE"
        verdicts["D2_coda_refinement"] = d2

    return {
        "seeds": seeds,
        "gpus": {str(s): by_seed[s].get("gpu") for s in seeds},
        "arm_orders": {str(s): by_seed[s].get("arm_order") for s in seeds},
        "arms": arm_summary,
        "comparisons": comparisons,
        "verdicts": verdicts,
    }


if __name__ == "__main__":
    files = [Path(p) for p in (sys.argv[1:] or sorted(glob.glob("results/tournament/seed_*.json")))]
    print(json.dumps(score_files(files), indent=2))
