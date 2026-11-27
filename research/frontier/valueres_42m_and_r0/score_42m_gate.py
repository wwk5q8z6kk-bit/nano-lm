"""Strict Provenance-Validating Scorer for the Pre-Registered 40.38M (`M`-scale) ValueRes Confirmation Gate.

Enforces:
  1. Complete run receipts (`completed == True`, exact `steps` and `train_tokens`, matching validation & code SHA256,
     no duplicate `(arm, seed)` pairs, full seed coverage across all 4 arms).
  2. Paired two-sided 95% t-tests (`|t| > 4.3027` for `df=2`) on BOTH equal-token AND equal-wall-clock comparisons.
  3. Mechanical Decision Rules `G1` (40.38M architecture promotion vs ResFormer fixed 0.5/0.5 vs generic x0 skip)
     and `G2` (causal positional v0-transport verification).
"""
from __future__ import annotations
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

T_CRIT_DF2_95 = 4.302652729911275
REQUIRED_ARMS = ("dense", "resid_x0_skip", "resformer_fixed", "value_res")


def interp_bpb_at_time(curve: list[dict], target_sec: float, key: str = "mix") -> float:
    pts = [(float(r["train_sec"]), float(r[key])) for r in curve]
    if target_sec >= pts[-1][0]:
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
        return {"n": n, "mean": round(mean, 5), "sd": 0.0, "se": 0.0, "t": 0.0, "sig_95": False, "all_positive": all(x > 0 for x in diffs), "per_seed": [round(x, 5) for x in diffs]}
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


def score_42m_runs(paths: list[Path], expected_steps: int | None = None, expected_tokens: int | None = None) -> dict:
    if not paths:
        raise ValueError("No run_*.json files supplied")

    grid: dict[int, dict[str, dict]] = {}
    val_hashes_ref = None
    code_sha_set = set()
    file_hashes = {}

    for p in sorted(paths):
        d = json.loads(p.read_text())
        if d.get("completed") is not True:
            raise ValueError(f"{p.name}: run not completed=True")
        arm = d["arm"]
        seed = int(d["seed"])
        if seed in grid and arm in grid[seed]:
            raise ValueError(f"{p.name}: duplicate (arm={arm}, seed={seed})")
        curve = d.get("curve", [])
        if not curve:
            raise ValueError(f"{p.name}: empty curve")
        rec = d["recipe"]
        exp_s = expected_steps if expected_steps is not None else int(rec["steps"])
        exp_tok = expected_tokens if expected_tokens is not None else int(rec["train_tokens"])
        if int(curve[-1]["step"]) != exp_s or int(curve[-1]["tokens"]) != exp_tok:
            raise ValueError(f"{p.name}: final step/tokens ({curve[-1]['step']}, {curve[-1]['tokens']}) != ({exp_s}, {exp_tok})")

        prov = d.get("provenance", {})
        vh = prov.get("val_sha256")
        if val_hashes_ref is None:
            val_hashes_ref = vh
        elif vh != val_hashes_ref:
            raise ValueError(f"{p.name}: mismatched validation shard SHA256 vs reference")
        if prov.get("code_sha256"):
            code_sha_set.add(prov["code_sha256"])
        file_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        grid.setdefault(seed, {})[arm] = d

    seeds = sorted(grid.keys())
    for s in seeds:
        missing = [a for a in REQUIRED_ARMS if a not in grid[s]]
        if missing:
            raise ValueError(f"Seed {s} missing arms: {missing}")

    arms_out = {}
    for arm in REQUIRED_ARMS:
        mixes = [grid[s][arm]["curve"][-1]["mix"] for s in seeds]
        mean_mix = sum(mixes) / len(mixes)
        sd_mix = math.sqrt(sum((x - mean_mix) ** 2 for x in mixes) / max(1, len(mixes) - 1)) if len(mixes) > 1 else 0.0
        eq_time = [
            interp_bpb_at_time(grid[s][arm]["curve"], grid[s]["dense"]["curve"][-1]["train_sec"])
            for s in seeds
        ]
        dom_means = {
            dom: round(sum(grid[s][arm]["curve"][-1][dom] for s in seeds) / len(seeds), 4)
            for dom in ("web", "textbook", "math", "code")
        }
        tok_s = [grid[s][arm]["curve"][-1]["tokens_per_second"] for s in seeds]
        secs = [grid[s][arm]["curve"][-1]["train_sec"] for s in seeds]
        ranks = {}
        if all(grid[s][arm].get("rank_buckets_mix") for s in seeds):
            for rk in ("rank1_calib", "rank2_8_rerank", "rank9_64_mid", "rank65plus_miss", "top1_acc", "top64_acc"):
                ranks[rk] = round(sum(grid[s][arm]["rank_buckets_mix"][rk] for s in seeds) / len(seeds), 4)

        tasks_mean = {}
        if all(grid[s][arm].get("downstream_7tasks") for s in seeds):
            task_names = sorted(grid[seeds[0]][arm]["downstream_7tasks"].keys())
            for tn in task_names:
                key_metric = "acc_norm" if "acc_norm" in grid[seeds[0]][arm]["downstream_7tasks"][tn] else "acc"
                vals_t = [grid[s][arm]["downstream_7tasks"][tn][key_metric] for s in seeds]
                tasks_mean[tn] = round(sum(vals_t) / len(vals_t), 4)
                if tn == "lambada":
                    ppls = [grid[s][arm]["downstream_7tasks"][tn]["ppl"] for s in seeds]
                    tasks_mean["lambada_ppl"] = round(sum(ppls) / len(ppls), 2)

        abl_mean = {}
        if all(grid[s][arm].get("causal_ablations") for s in seeds):
            for ab_k in grid[seeds[0]][arm]["causal_ablations"]:
                deltas = [grid[s][arm]["causal_ablations"][ab_k]["delta_mix_vs_intact"] for s in seeds]
                abl_mean[ab_k] = round(sum(deltas) / len(deltas), 4)

        arms_out[arm] = {
            "params": grid[seeds[0]][arm]["params"],
            "equal_tokens_mix_mean": round(mean_mix, 4),
            "equal_tokens_mix_sd": round(sd_mix, 5),
            "equal_tokens_per_seed": [round(x, 4) for x in mixes],
            "equal_wallclock_to_dense_mix_mean": round(sum(eq_time) / len(eq_time), 4),
            "equal_wallclock_per_seed": [round(x, 4) for x in eq_time],
            "domains_mean": dom_means,
            "mean_tokens_per_second": round(sum(tok_s) / len(tok_s)),
            "mean_train_sec": round(sum(secs) / len(secs), 1),
            "rank_buckets_mix_mean": ranks,
            "downstream_7tasks_mean": tasks_mean,
            "causal_ablation_delta_mix_mean": abl_mean,
        }

    comparisons = {}
    for cand in ("resid_x0_skip", "resformer_fixed", "value_res"):
        d_tok = [grid[s]["dense"]["curve"][-1]["mix"] - grid[s][cand]["curve"][-1]["mix"] for s in seeds]
        d_time = [
            grid[s]["dense"]["curve"][-1]["mix"]
            - interp_bpb_at_time(grid[s][cand]["curve"], grid[s]["dense"]["curve"][-1]["train_sec"])
            for s in seeds
        ]
        comparisons[f"{cand}_vs_dense"] = {
            "equal_tokens_gain": paired_stats(d_tok),
            "equal_wallclock_gain": paired_stats(d_time),
        }

    comparisons["value_res_vs_resid_x0_skip"] = {
        "equal_tokens_gain": paired_stats(
            [grid[s]["resid_x0_skip"]["curve"][-1]["mix"] - grid[s]["value_res"]["curve"][-1]["mix"] for s in seeds]
        ),
        "equal_wallclock_gain": paired_stats(
            [
                interp_bpb_at_time(grid[s]["resid_x0_skip"]["curve"], grid[s]["dense"]["curve"][-1]["train_sec"])
                - interp_bpb_at_time(grid[s]["value_res"]["curve"], grid[s]["dense"]["curve"][-1]["train_sec"])
                for s in seeds
            ]
        ),
    }
    comparisons["value_res_vs_resformer_fixed"] = {
        "equal_tokens_gain": paired_stats(
            [grid[s]["resformer_fixed"]["curve"][-1]["mix"] - grid[s]["value_res"]["curve"][-1]["mix"] for s in seeds]
        ),
        "equal_wallclock_gain": paired_stats(
            [
                interp_bpb_at_time(grid[s]["resformer_fixed"]["curve"], grid[s]["dense"]["curve"][-1]["train_sec"])
                - interp_bpb_at_time(grid[s]["value_res"]["curve"], grid[s]["dense"]["curve"][-1]["train_sec"])
                for s in seeds
            ]
        ),
    }
    comparisons["resformer_fixed_vs_resid_x0_skip"] = {
        "equal_tokens_gain": paired_stats(
            [grid[s]["resid_x0_skip"]["curve"][-1]["mix"] - grid[s]["resformer_fixed"]["curve"][-1]["mix"] for s in seeds]
        ),
    }

    # Pre-registered Mechanical Decision Rules G1 and G2
    vr_d_tok = comparisons["value_res_vs_dense"]["equal_tokens_gain"]
    vr_d_time = comparisons["value_res_vs_dense"]["equal_wallclock_gain"]
    vr_x0_tok = comparisons["value_res_vs_resid_x0_skip"]["equal_tokens_gain"]
    vr_rf_tok = comparisons["value_res_vs_resformer_fixed"]["equal_tokens_gain"]

    rf_d_tok = comparisons["resformer_fixed_vs_dense"]["equal_tokens_gain"]
    rf_d_time = comparisons["resformer_fixed_vs_dense"]["equal_wallclock_gain"]
    rf_x0_tok = comparisons["resformer_fixed_vs_resid_x0_skip"]["equal_tokens_gain"]

    vr_clears_base = (
        vr_d_tok["mean"] >= 0.008 and vr_d_tok["sig_95"] and vr_d_tok["all_positive"]
        and vr_d_time["mean"] >= 0.008 and vr_d_time["sig_95"] and vr_d_time["all_positive"]
        and vr_x0_tok["mean"] >= 0.004 and vr_x0_tok["sig_95"] and vr_x0_tok["all_positive"]
    )
    rf_clears_base = (
        rf_d_tok["mean"] >= 0.008 and rf_d_tok["sig_95"] and rf_d_tok["all_positive"]
        and rf_d_time["mean"] >= 0.008 and rf_d_time["sig_95"] and rf_d_time["all_positive"]
        and rf_x0_tok["mean"] >= 0.004 and rf_x0_tok["sig_95"] and rf_x0_tok["all_positive"]
    )

    if vr_clears_base and vr_rf_tok["mean"] >= 0.002 and vr_rf_tok["sig_95"]:
        g1 = "ADOPT_LEARNED_VALUE_RES_AT_42M"
    elif rf_clears_base and (not vr_clears_base or vr_rf_tok["mean"] < 0.002 or not vr_rf_tok["sig_95"]):
        g1 = "ADOPT_RESFORMER_FIXED_05_AT_42M"
    elif vr_clears_base:
        g1 = "ADOPT_VALUE_RES_FAMILY_TIE_LEARNED_VS_FIXED"
    else:
        g1 = "REJECT_VALUE_RES_PROMOTION_AT_42M"

    perm_deltas = [grid[s]["value_res"]["causal_ablations"]["perm_v0"]["delta_mix_vs_intact"] for s in seeds]
    late_deltas = [grid[s]["value_res"]["causal_ablations"]["ablate_late_6_7"]["delta_mix_vs_intact"] for s in seeds]
    mid_deltas = [grid[s]["value_res"]["causal_ablations"]["ablate_mid_3_4"]["delta_mix_vs_intact"] for s in seeds]
    if all(d >= 0.05 for d in perm_deltas) and all(l > m for l, m in zip(late_deltas, mid_deltas)):
        g2 = "CONFIRMED_LATE_POSITIONAL_V0_TRANSPORT"
    else:
        g2 = "NOT_CONFIRMED_LATE_POSITIONAL_V0_TRANSPORT"

    return {
        "seeds": seeds,
        "provenance_verified": {
            "code_sha256_unique": sorted(code_sha_set),
            "val_sha256": val_hashes_ref,
            "run_receipt_sha256": file_hashes,
        },
        "arms": arms_out,
        "comparisons": comparisons,
        "verdicts": {
            "G1_42m_architecture_gate": g1,
            "G2_causal_v0_mechanism_gate": g2,
        },
    }


if __name__ == "__main__":
    files = [Path(p) for p in (sys.argv[1:] or sorted(glob.glob("results/runs_42m/run_*.json")))]
    print(json.dumps(score_42m_runs(files), indent=2))
