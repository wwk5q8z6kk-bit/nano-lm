"""Strict Provenance-Validating Scorer for the Pre-Registered 40.38M / 57.2M Untied-Head x ValueRes Factorial.

Enforces:
  1. Complete run receipts (`completed == True`, exact `steps` and `train_tokens`, matching validation SHA256,
     no duplicate `(arm, seed)` pairs, full seed coverage across all 4 arms).
  2. Paired two-sided 95% t-tests (`|t| > 4.3027` for `df=2`) on BOTH equal-token AND equal-wall-clock comparisons
     (interpolated at `dense_tied`'s wall-clock time per seed).
  3. Pre-registered Mechanical Decision Rules `U_G1` (Untied Head Adoption Gate for Sub-4B Trunk) and
     `U_G2` (Tied-Head x ValueRes Sub-Additivity & Layer-7 Lambda Interaction Verdict).
"""
from __future__ import annotations
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

T_CRIT_DF2_95 = 4.302652729911275
REQUIRED_ARMS = ("dense_tied", "dense_untied", "dense_untied_vr", "dense_tied_57m")
PR214_TIED_VR_GAIN_MEAN = 0.00590
PR214_TIED_VR_LAM7_MEAN = 0.12300


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


def score_untied_runs(paths: list[Path], expected_steps: int | None = None, expected_tokens: int | None = None) -> dict:
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
            interp_bpb_at_time(grid[s][arm]["curve"], grid[s]["dense_tied"]["curve"][-1]["train_sec"])
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
        if all(grid[s][arm].get("causal_ablations") for s in seeds) and grid[seeds[0]][arm]["causal_ablations"]:
            for ab_k in grid[seeds[0]][arm]["causal_ablations"]:
                deltas = [grid[s][arm]["causal_ablations"][ab_k]["delta_mix_vs_intact"] for s in seeds]
                abl_mean[ab_k] = round(sum(deltas) / len(deltas), 4)

        arms_out[arm] = {
            "params": grid[seeds[0]][arm]["params"],
            "non_embed_params": grid[seeds[0]][arm].get("non_embed_params"),
            "equal_tokens_mix_mean": round(mean_mix, 4),
            "equal_tokens_mix_sd": round(sd_mix, 5),
            "equal_tokens_per_seed": [round(x, 4) for x in mixes],
            "equal_wallclock_to_dense_tied_mix_mean": round(sum(eq_time) / len(eq_time), 4),
            "equal_wallclock_per_seed": [round(x, 4) for x in eq_time],
            "domains_mean": dom_means,
            "mean_tokens_per_second": round(sum(tok_s) / len(tok_s)),
            "mean_train_sec": round(sum(secs) / len(secs), 1),
            "rank_buckets_mix_mean": ranks,
            "downstream_7tasks_mean": tasks_mean,
            "causal_ablation_delta_mix_mean": abl_mean,
        }

    comparisons = {}
    for cand in ("dense_untied", "dense_untied_vr", "dense_tied_57m"):
        d_tok = [grid[s]["dense_tied"]["curve"][-1]["mix"] - grid[s][cand]["curve"][-1]["mix"] for s in seeds]
        d_time = [
            grid[s]["dense_tied"]["curve"][-1]["mix"]
            - interp_bpb_at_time(grid[s][cand]["curve"], grid[s]["dense_tied"]["curve"][-1]["train_sec"])
            for s in seeds
        ]
        comparisons[f"{cand}_vs_dense_tied"] = {
            "equal_tokens_gain": paired_stats(d_tok),
            "equal_wallclock_gain": paired_stats(d_time),
        }

    comparisons["dense_untied_vr_vs_dense_untied"] = {
        "equal_tokens_gain": paired_stats(
            [grid[s]["dense_untied"]["curve"][-1]["mix"] - grid[s]["dense_untied_vr"]["curve"][-1]["mix"] for s in seeds]
        ),
        "equal_wallclock_gain": paired_stats(
            [
                interp_bpb_at_time(grid[s]["dense_untied"]["curve"], grid[s]["dense_tied"]["curve"][-1]["train_sec"])
                - interp_bpb_at_time(grid[s]["dense_untied_vr"]["curve"], grid[s]["dense_tied"]["curve"][-1]["train_sec"])
                for s in seeds
            ]
        ),
    }
    comparisons["dense_tied_57m_vs_dense_untied"] = {
        "equal_tokens_gain": paired_stats(
            [grid[s]["dense_untied"]["curve"][-1]["mix"] - grid[s]["dense_tied_57m"]["curve"][-1]["mix"] for s in seeds]
        ),
        "equal_wallclock_gain": paired_stats(
            [
                interp_bpb_at_time(grid[s]["dense_untied"]["curve"], grid[s]["dense_tied"]["curve"][-1]["train_sec"])
                - interp_bpb_at_time(grid[s]["dense_tied_57m"]["curve"], grid[s]["dense_tied"]["curve"][-1]["train_sec"])
                for s in seeds
            ]
        ),
    }

    # Mechanical Gate U_G1 (Untied Head Adoption Gate for Sub-4B Trunk)
    u_tok = comparisons["dense_untied_vs_dense_tied"]["equal_tokens_gain"]
    u_time = comparisons["dense_untied_vs_dense_tied"]["equal_wallclock_gain"]
    w57_tok = comparisons["dense_tied_57m_vs_dense_tied"]["equal_tokens_gain"]
    share_of_57m_equal_token_gain = round(u_tok["mean"] / w57_tok["mean"], 4) if abs(w57_tok["mean"]) > 1e-6 else None

    if (
        u_tok["mean"] >= 0.008 and u_tok["sig_95"] and u_tok["all_positive"]
        and u_time["mean"] >= 0.008 and u_time["sig_95"] and u_time["all_positive"]
    ):
        u_g1 = "ADOPT_UNTIED_HEAD_FOR_SUB4B_TRUNK"
    else:
        u_g1 = "REJECT_UNTIED_HEAD_FOR_SUB4B_TRUNK"

    # Mechanical Gate U_G2 (Tied-Head x ValueRes Sub-Additivity & Layer-7 Lambda Interaction)
    vr_given_untied = comparisons["dense_untied_vr_vs_dense_untied"]["equal_tokens_gain"]
    lam7_vals = [float(grid[s]["dense_untied_vr"]["diagnostics"]["lambdas"][-1]) for s in seeds]
    lam7_mean = sum(lam7_vals) / len(lam7_vals)
    cos_eh_untied = [float(grid[s]["dense_untied"]["diagnostics"]["mean_cos_emb_vs_lm_head"]) for s in seeds]

    if vr_given_untied["mean"] < PR214_TIED_VR_GAIN_MEAN and lam7_mean > 0.150:
        u_g2 = "CONFIRMED_TIED_HEAD_SUBADDITIVITY"
    else:
        u_g2 = "ADDITIVE_OR_INDEPENDENT_OF_HEAD_TYING"

    return {
        "seeds": seeds,
        "provenance_verified": {
            "code_sha256_unique": sorted(code_sha_set),
            "val_sha256": val_hashes_ref,
            "run_receipt_sha256": file_hashes,
        },
        "arms": arms_out,
        "comparisons": comparisons,
        "interaction_telemetry": {
            "untied_share_of_57m_tied_equal_token_gain": share_of_57m_equal_token_gain,
            "vr_marginal_gain_when_untied": vr_given_untied["mean"],
            "pr214_vr_marginal_gain_when_tied": PR214_TIED_VR_GAIN_MEAN,
            "untied_vr_final_layer_lambda_mean": round(lam7_mean, 5),
            "untied_vr_final_layer_lambda_per_seed": [round(x, 5) for x in lam7_vals],
            "pr214_tied_vr_final_layer_lambda_mean": PR214_TIED_VR_LAM7_MEAN,
            "dense_untied_mean_cos_emb_vs_lm_head": round(sum(cos_eh_untied) / len(cos_eh_untied), 5),
        },
        "verdicts": {
            "U_G1_untied_head_gate": u_g1,
            "U_G2_valueres_interaction": u_g2,
        },
    }


if __name__ == "__main__":
    files = [Path(p) for p in (sys.argv[1:] or sorted(glob.glob("results/runs_untied/run_*.json")))]
    print(json.dumps(score_untied_runs(files), indent=2))
