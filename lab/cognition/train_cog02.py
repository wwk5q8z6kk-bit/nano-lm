"""`LAB-COG-02`: Causal Ablation of Uncertainty-Gated Latent Belief Planning & Sym(12) Equivariance.

Evaluates (`n=3` seeds, `800` episodes, matched `~885K-940K` params) across THREE difficulty tiers:
  1. In-Distribution (`4-5` modules, `[100..999]` constants),
  2. Standard OOD (`7-9` modules, unseen `[1000..9999]` constants — identical to `LAB-COG-01`!),
  3. Extreme OOD (`10-12` modules, unseen `[1000..9999]` constants — 2.5x training graph size!).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import statistics as st
import time
from pathlib import Path
from typing import Any

import torch

from lab.cognition.train_cog01 import evaluate_arm, train_one_arm


def _run_cog02_job(job: tuple[str, int, int, int]) -> tuple[str, dict[str, Any], float]:
    arm, s, n_episodes, n_eval = job
    torch.set_num_threads(1)
    dev = torch.device("cpu")
    t0 = time.perf_counter()
    m = train_one_arm(arm, seed=s, n_episodes=n_episodes, dev=dev)
    n_params = sum(p.numel() for p in m.parameters())
    id_res = evaluate_arm(m, n_eval=n_eval, ood=False, dev=dev, seed=7000 + s)
    ood_res = evaluate_arm(m, n_eval=n_eval, ood=True, dev=dev, seed=8000 + s, extreme_ood=False)
    ext_res = evaluate_arm(m, n_eval=n_eval, ood=True, dev=dev, seed=9000 + s, extreme_ood=True)
    dt = time.perf_counter() - t0
    rec = {"seed": s, "params": n_params, "in_dist": id_res, "ood": ood_res, "extreme_ood": ext_res}
    print(
        f"[{arm} s={s}] ({dt:.1f}s) params={n_params:,} "
        f"ID={id_res['verified_resolution_full']:.3f} "
        f"OOD_full={ood_res['verified_resolution_full']:.3f} (val={ood_res['value_exact_match']:.3f}, prem={ood_res['premature_or_regression_rate']:.3f}) "
        f"EXT_OOD_full={ext_res['verified_resolution_full']:.3f} (val={ext_res['value_exact_match']:.3f})",
        flush=True,
    )
    return arm, rec, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--eval-episodes", type=int, default=180)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--out", default="/Users/mac/Projects/nano-lm/.worktrees/nanolm-lab/lab/reports/lab_cog_02_results.json")
    args = ap.parse_args()

    arms = ["C3_ebwm_counterfactual", "C4a_ulbp_latent_plan", "C4b_ulbp_equivariant"]
    jobs = [(arm, s, args.episodes, args.eval_episodes) for arm in arms for s in range(args.seeds)]
    grouped: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    wall_by_arm: dict[str, float] = {a: 0.0 for a in arms}

    t_global = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        futs = [ex.submit(_run_cog02_job, j) for j in jobs]
        for f in as_completed(futs):
            arm, rec, dt = f.result()
            grouped[arm].append(rec)
            wall_by_arm[arm] = max(wall_by_arm[arm], dt)

    all_results: dict[str, Any] = {
        "schema": "nanolm.lab_cog_02.v1",
        "episodes_train": args.episodes,
        "seeds": args.seeds,
        "arms": {},
    }
    for arm in arms:
        seed_runs = sorted(grouped[arm], key=lambda r: r["seed"])

        def _agg(split: str, key: str) -> dict[str, float]:
            vals = [r[split][key] for r in seed_runs]
            return {"mean": round(st.mean(vals), 4), "half_range": round((max(vals) - min(vals)) / 2, 4)}

        summary = {
            "params": seed_runs[0]["params"],
            "wall_s": round(wall_by_arm[arm], 2),
            "id_verified_resolution_full": _agg("in_dist", "verified_resolution_full"),
            "ood_verified_resolution_full": _agg("ood", "verified_resolution_full"),
            "ood_verified_resolution_policy_only": _agg("ood", "verified_resolution_oracle_val"),
            "ood_premature_or_regression_rate": _agg("ood", "premature_or_regression_rate"),
            "ood_unseen_value_exact_match": _agg("ood", "value_exact_match"),
            "ood_belief_revision_acc": _agg("ood", "belief_revision_after_contradiction"),
            "extreme_ood_verified_resolution_full": _agg("extreme_ood", "verified_resolution_full"),
            "extreme_ood_policy_only": _agg("extreme_ood", "verified_resolution_oracle_val"),
            "extreme_ood_unseen_value_exact": _agg("extreme_ood", "value_exact_match"),
            "extreme_ood_belief_revision_acc": _agg("extreme_ood", "belief_revision_after_contradiction"),
            "extreme_ood_premature_rate": _agg("extreme_ood", "premature_or_regression_rate"),
            "seed_runs": seed_runs,
        }
        all_results["arms"][arm] = summary
        print(f"=== SUMMARY {arm}: {json.dumps({k: v for k, v in summary.items() if k != 'seed_runs'})}", flush=True)

    all_results["total_wall_s"] = round(time.perf_counter() - t_global, 2)
    Path(args.out).write_text(json.dumps(all_results, indent=2))
    print("SAVED_LAB_COG_02", args.out, f"in {all_results['total_wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
