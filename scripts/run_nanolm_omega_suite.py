#!/usr/bin/env python3
"""Execute Local NANOLM Omega Suite ($0.00 Apple M3 Max):
  - OMEGA-EXP-01-MECH: 115k 3-seed mechanistic OOD & epistemic calibration comparison (X8 vs G4-Omega + ablations)
  - OMEGA-EXP-02: HardCausalSWEEnv ID(4-6) / OOD(9-12) / Super-OOD(16-24) EFE cognition tournament (3 seeds)
  - OMEGA-EXP-03: Hint-free 60-task Unified 10-Capability System benchmark (`OMEGA-BENCH-01`)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanolm_discovery.eval_benchmarks import (
    evaluate_epistemic_calibration,
    generate_multi_needle_batch,
    generate_overwrite_batch,
    generate_s3_batch,
)
from nanolm_omega.backbone import OmegaVariant, build_omega_model, compute_omega_census, make_omega_suite
from nanolm_omega.cognition_efe import run_efe_cognition_experiment
from nanolm_omega.unified_system import evaluate_omega_bench_01


def _mean_std(vals: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(vals)),
        "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def _masked_acc(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    denom = int(mask.sum().item())
    if denom == 0:
        return 0.0
    return float(((logits.argmax(dim=-1) == targets) & mask).sum().item()) / float(denom)


def run_mech_arm_single(cfg, *, seed: int, device: torch.device, steps: int = 360) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_omega_model(cfg).to(device)
    base_lr = 3.5e-3
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.01)

    model.train()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        task_id = step % 3
        if task_id == 0:
            x, y, m = generate_s3_batch(20, 32, rng=rng, device=device)
        elif task_id == 1:
            x, y, m = generate_overwrite_batch(20, 48, rng=rng, device=device)
        else:
            x, y, m = generate_multi_needle_batch(20, 96, n_needles=4, rng=rng, device=device)
        y_masked = y.masked_fill(~m, -1)
        out = model(x, targets=y_masked)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    eval_rng = np.random.default_rng(seed + 10_000)
    with torch.no_grad():
        s3_id = _masked_acc(model(generate_s3_batch(32, 32, rng=eval_rng, device=device)[0]).logits, *generate_s3_batch(32, 32, rng=np.random.default_rng(seed + 111), device=device)[1:])
        xs4, ys4, ms4 = generate_s3_batch(32, 128, rng=eval_rng, device=device)
        s3_4x = _masked_acc(model(xs4).logits, ys4, ms4)
        xo_id, yo_id, mo_id = generate_overwrite_batch(32, 48, rng=eval_rng, device=device)
        ow_id = _masked_acc(model(xo_id).logits, yo_id, mo_id)
        xo4, yo4, mo4 = generate_overwrite_batch(32, 192, rng=eval_rng, device=device)
        ow_4x = _masked_acc(model(xo4).logits, yo4, mo4)
        xn_id, yn_id, mn_id = generate_multi_needle_batch(32, 96, n_needles=4, rng=eval_rng, device=device)
        nd_id = _masked_acc(model(xn_id).logits, yn_id, mn_id)
        xn4, yn4, mn4 = generate_multi_needle_batch(32, 256, n_needles=8, rng=eval_rng, device=device)
        nd_ood = _masked_acc(model(xn4).logits, yn4, mn4)
        ep = evaluate_epistemic_calibration(model, rng=eval_rng, device=device)
    # Deterministic S3_ID paired eval
    xs_id, ys_id, ms_id = generate_s3_batch(32, 32, rng=np.random.default_rng(seed + 222), device=device)
    with torch.no_grad():
        s3_id = _masked_acc(model(xs_id).logits, ys_id, ms_id)

    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {
        "s3_id_t32": s3_id,
        "s3_ood4x_t128": s3_4x,
        "overwrite_id_t48": ow_id,
        "overwrite_ood4x_t192": ow_4x,
        "needle_id_t96_n4": nd_id,
        "needle_ood_t256_n8": nd_ood,
        "epistemic_auroc_internal": ep["auroc_internal_unc"],
        "epistemic_auroc_entropy": ep["auroc_predictive_entropy"],
        "state_dict": state_cpu,
    }


def _run_mech_variant(var: OmegaVariant, cfg, dev: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    census = compute_omega_census(cfg)
    runs = [run_mech_arm_single(cfg, seed=sd, device=dev, steps=360) for sd in (101, 202, 303)]
    agg: dict[str, Any] = {
        metric: _mean_std([r[metric] for r in runs])
        for metric in (
            "s3_id_t32",
            "s3_ood4x_t128",
            "overwrite_id_t48",
            "overwrite_ood4x_t192",
            "needle_id_t96_n4",
            "needle_ood_t256_n8",
            "epistemic_auroc_internal",
            "epistemic_auroc_entropy",
        )
    }
    agg["params"] = census.total_deployed_parameters
    agg["unembed_passes"] = census.unembed_passes
    agg["true_flops_per_token_4k"] = census.true_flops_per_token_4k
    agg["ood_composite_acc"] = float(
        (agg["s3_ood4x_t128"]["mean"] + agg["overwrite_ood4x_t192"]["mean"] + agg["needle_ood_t256_n8"]["mean"]) / 3.0
    )
    print(
        f"[OMEGA-01-MECH] {var.value:22s} S3_4x={agg['s3_ood4x_t128']['mean']:.3f} "
        f"OW_4x={agg['overwrite_ood4x_t192']['mean']:.3f} "
        f"Needle_OOD={agg['needle_ood_t256_n8']['mean']:.3f} "
        f"EpistAUROC={agg['epistemic_auroc_internal']['mean']:.3f} OOD_Comp={agg['ood_composite_acc']:.3f}",
        flush=True,
    )
    return agg, runs[0]["state_dict"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute Local NANOLM Omega Suite")
    parser.add_argument(
        "--recalibrate-g4-only",
        action="store_true",
        help="Re-run calibrated G4_OMEGA (3 seeds) + OMEGA-EXP-03 while preserving g4_uncalibrated_router",
    )
    args = parser.parse_args()

    out_dir = ROOT / "artifacts/nanolm_omega_v1"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dev_mps = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dev_cpu = torch.device("cpu")
    mech_suite = make_omega_suite("mech")
    t0_all = time.perf_counter()

    if args.recalibrate_g4_only:
        torch.set_num_threads(2)
        results_path = out_dir / "omega_local_results.json"
        existing = json.loads(results_path.read_text(encoding="utf-8"))
        mech_results = existing["omega_exp_01_mech"]
        if "g4_uncalibrated_router" not in mech_results and "g4_omega" in mech_results:
            mech_results["g4_uncalibrated_router"] = mech_results["g4_omega"]

        print("=== OMEGA-EXP-01-MECH (Recalibrated G4_OMEGA only, 115k, 3 seeds, 2 threads) ===", flush=True)
        g4_agg, g4_mech_ckpt = _run_mech_variant(
            OmegaVariant.G4_OMEGA, mech_suite[OmegaVariant.G4_OMEGA], dev_cpu
        )
        mech_results["g4_omega"] = g4_agg

        print("=== OMEGA-EXP-03: Hint-Free 60-Task Unified 10-Capability System (OMEGA-BENCH-01) ===", flush=True)
        g4_loaded = build_omega_model(mech_suite[OmegaVariant.G4_OMEGA]).eval()
        g4_loaded.load_state_dict(g4_mech_ckpt)
        sys_results = evaluate_omega_bench_01(backbone=g4_loaded)

        g4_ckpt_path = ckpt_dir / "g4_omega_mech_s101.pt"
        torch.save({"variant": "g4_omega", "seed": 101, "state_dict": g4_mech_ckpt}, g4_ckpt_path)
        manifest_path = out_dir / "checkpoint_manifest.json"
        ckpt_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        new_entry = {
            "name": "g4_omega_mech_s101.pt",
            "bytes": g4_ckpt_path.stat().st_size,
            "sha256": hashlib.sha256(g4_ckpt_path.read_bytes()).hexdigest(),
            "purpose": "Cycle Omega trained G4OmegaModel (115k, self-calibrated router) for reproducible inspection & unified router uncertainty",
        }
        entries = [e for e in ckpt_manifest.get("checkpoints", []) if e.get("name") != "g4_omega_mech_s101.pt"]
        ckpt_manifest["checkpoints"] = [new_entry] + entries
        manifest_path.write_text(json.dumps(ckpt_manifest, indent=2) + "\n", encoding="utf-8")

        existing["omega_exp_01_mech"] = mech_results
        existing["omega_exp_03_unified_system"] = sys_results
        existing["hardware"]["recalibrate_g4_seconds"] = time.perf_counter() - t0_all
        results_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        print(f"COMPLETED RECALIBRATED G4 + OMEGA-EXP-03 in {time.perf_counter() - t0_all:.1f}s -> {results_path}", flush=True)
        return 0

    # 1. OMEGA-EXP-01-MECH (3 seeds on local M3 Max)
    print("=== OMEGA-EXP-01-MECH: Mechanistic OOD & Dual-Unembed vs Latent-Refiner (115k, 3 seeds) ===", flush=True)
    mech_results: dict[str, Any] = {}
    g4_mech_ckpt: dict[str, torch.Tensor] = {}
    for var in (
        OmegaVariant.X8_DUAL_UNEMBED_ATRD,
        OmegaVariant.G4_OMEGA,
        OmegaVariant.G4_NO_VRES,
        OmegaVariant.G4_NO_SPARSE_REFINER,
    ):
        agg, ckpt0 = _run_mech_variant(var, mech_suite[var], dev_cpu)
        if var == OmegaVariant.G4_OMEGA:
            g4_mech_ckpt = ckpt0
        mech_results[var.value] = agg

    # 2. OMEGA-EXP-02: HardCausalSWEEnv (3 seeds, N=4..6 -> 9..12 -> 16..24 Super-OOD)
    print("=== OMEGA-EXP-02: Open-Entity Span-Graph & Expected Free-Energy Cognition (3 seeds) ===", flush=True)
    cog_results, efe_ckpt = run_efe_cognition_experiment([0, 1, 2], device=dev_cpu, train_episodes=420, eval_episodes=120)

    # 3. OMEGA-EXP-03: Unified 10-Capability System Benchmark wired with trained G4OmegaModel backbone
    print("=== OMEGA-EXP-03: Hint-Free 60-Task Unified 10-Capability System (OMEGA-BENCH-01) ===", flush=True)
    g4_loaded = build_omega_model(mech_suite[OmegaVariant.G4_OMEGA]).eval()
    g4_loaded.load_state_dict(g4_mech_ckpt)
    sys_results = evaluate_omega_bench_01(backbone=g4_loaded)

    # Save reproducible local checkpoints into gitignored directory + digest manifest
    g4_ckpt_path = ckpt_dir / "g4_omega_mech_s101.pt"
    efe_ckpt_path = ckpt_dir / "omega_efe_planner_s0.pt"
    torch.save({"variant": "g4_omega", "seed": 101, "state_dict": g4_mech_ckpt}, g4_ckpt_path)
    torch.save({"arm": "omega_efe_planner", "seed": 0, "state_dict": efe_ckpt}, efe_ckpt_path)
    ckpt_manifest = {
        "schema": "nanolm.omega.checkpoints.v1",
        "checkpoints": [
            {
                "name": "g4_omega_mech_s101.pt",
                "bytes": g4_ckpt_path.stat().st_size,
                "sha256": hashlib.sha256(g4_ckpt_path.read_bytes()).hexdigest(),
                "purpose": "Cycle Omega trained G4OmegaModel (115k) for reproducible inspection & unified router uncertainty",
            },
            {
                "name": "omega_efe_planner_s0.pt",
                "bytes": efe_ckpt_path.stat().st_size,
                "sha256": hashlib.sha256(efe_ckpt_path.read_bytes()).hexdigest(),
                "purpose": "Cycle Omega trained OmegaEFEPlanner Super-OOD cognitive checkpoint",
            },
        ],
    }
    (out_dir / "checkpoint_manifest.json").write_text(json.dumps(ckpt_manifest, indent=2) + "\n", encoding="utf-8")

    local_report = {
        "schema": "nanolm.omega.local_suite.v1",
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "device_used": dev_mps.type,
            "wall_seconds_total": time.perf_counter() - t0_all,
            "paid_compute_usd": 0.0,
        },
        "omega_exp_01_mech": mech_results,
        "omega_exp_02_cognition_efe": cog_results,
        "omega_exp_03_unified_system": sys_results,
    }
    (out_dir / "omega_local_results.json").write_text(json.dumps(local_report, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETED LOCAL OMEGA SUITE in {time.perf_counter() - t0_all:.1f}s -> {out_dir / 'omega_local_results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
