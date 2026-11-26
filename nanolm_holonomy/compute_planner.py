"""Hardware-aware Roofline Compute, VRAM, Latency & Cost Planner for NanoLM.

Provides pre-flight estimation before launching local MPS, free Kaggle, or
paid RunPod instances, plus retrospective calibration against measured Cycle-2
L40S telemetry.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GPUSpec:
    name: str
    backend: str
    vram_gb: float
    bf16_tflops: float
    mem_bw_gb_s: float
    usd_per_hr: float
    max_seq_supported: int = 65_536
    startup_overhead_s: float = 45.0


GPU_CATALOG: dict[str, GPUSpec] = {
    "apple_m_max_local": GPUSpec(
        name="Apple Silicon Unified (Local)",
        backend="mps",
        vram_gb=64.0,
        bf16_tflops=28.0,
        mem_bw_gb_s=400.0,
        usd_per_hr=0.0,
        startup_overhead_s=5.0,
    ),
    "kaggle_t4_x2": GPUSpec(
        name="Kaggle NVIDIA T4 (16 GB per device, dual T4 host)",
        backend="kaggle",
        vram_gb=16.0,
        bf16_tflops=65.0,  # T4 FP16 tensor-core proxy on the single device used by the runners
        mem_bw_gb_s=320.0,
        usd_per_hr=0.0,
        startup_overhead_s=110.0,
    ),
    "runpod_a40_48gb": GPUSpec(
        name="RunPod NVIDIA A40 48GB",
        backend="runpod",
        vram_gb=48.0,
        bf16_tflops=149.7,
        mem_bw_gb_s=696.0,
        usd_per_hr=0.40,
        startup_overhead_s=65.0,
    ),
    "runpod_l40s_48gb": GPUSpec(
        name="RunPod NVIDIA L40S 48GB",
        backend="runpod",
        vram_gb=48.0,
        bf16_tflops=362.0,
        mem_bw_gb_s=864.0,
        usd_per_hr=1.09,
        startup_overhead_s=65.0,
    ),
    "runpod_h100_80gb": GPUSpec(
        name="RunPod NVIDIA H100 SXM 80GB",
        backend="runpod",
        vram_gb=80.0,
        bf16_tflops=989.0,
        mem_bw_gb_s=3350.0,
        usd_per_hr=2.99,
        startup_overhead_s=75.0,
    ),
    "runpod_h200_141gb": GPUSpec(
        name="RunPod NVIDIA H200 SXM 141GB",
        backend="runpod",
        vram_gb=141.0,
        bf16_tflops=989.0,
        mem_bw_gb_s=4800.0,
        usd_per_hr=5.29,
        startup_overhead_s=80.0,
    ),
}


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    name: str
    total_params: int
    trainable_params: int
    n_layers: int = 32
    d_model: int = 2048
    n_kv_heads: int = 4
    d_head: int = 128
    batch_size: int = 1
    seq_len: int = 1024
    train_tokens: int = 0
    decode_tokens: int = 0
    weight_bytes: int = 2
    state_bytes_override: int | None = None
    eval_chunk_size: int | None = None
    activation_checkpointing: bool = False
    mfu_assume: float = 0.28
    bw_util_assume: float = 0.55
    download_seconds: float = 0.0


def estimate_workload(spec: WorkloadSpec, gpu: GPUSpec) -> dict[str, Any]:
    """Analytical Roofline compute, VRAM, latency, and USD cost model."""
    weights_mb = (spec.total_params * spec.weight_bytes) / (1024.0 * 1024.0)
    # Full AdamW states when training weights natively: fp32 master + grad + m + v (~16 B/param)
    opt_bytes = 16.0 if spec.trainable_params > 0 else 0.0
    trainable_grad_opt_mb = (spec.trainable_params * opt_bytes) / (1024.0 * 1024.0)

    if spec.state_bytes_override is not None:
        state_or_kv_mb = (spec.state_bytes_override * spec.batch_size) / (1024.0 * 1024.0)
    else:
        kv_bytes = 2 * spec.n_layers * spec.n_kv_heads * spec.d_head * 2 * spec.seq_len
        state_or_kv_mb = (kv_bytes * spec.batch_size) / (1024.0 * 1024.0)

    if spec.trainable_params > 0:
        act_factor = 6.0 if spec.activation_checkpointing else 28.0
        activation_mb = (
            spec.batch_size * spec.seq_len * spec.n_layers * spec.d_model * act_factor * 2
        ) / (1024.0 * 1024.0)
    else:
        # No-grad eval only materializes active chunk layers/workspace (chunked prefill caps peak)
        eff_seq = min(spec.seq_len, spec.eval_chunk_size) if spec.eval_chunk_size else spec.seq_len
        activation_mb = (
            spec.batch_size * eff_seq * spec.d_model * 36.0 * 2
        ) / (1024.0 * 1024.0)

    cuda_context_mb = 180.0 if gpu.backend != "mps" else 128.0
    estimated_peak_vram_mb = round(
        weights_mb + trainable_grad_opt_mb + state_or_kv_mb + activation_mb + cuda_context_mb,
        2,
    )
    safe_vram_required_mb = round(estimated_peak_vram_mb * 1.18, 2)
    gpu_vram_mb = gpu.vram_gb * 1024.0
    fits_vram = safe_vram_required_mb <= gpu_vram_mb

    flops_per_tok = 6.0 * spec.total_params if spec.trainable_params > 0 else 2.0 * spec.total_params
    effective_flops_s = gpu.bf16_tflops * 1e12 * spec.mfu_assume
    compute_s = (flops_per_tok * spec.train_tokens) / max(effective_flops_s, 1.0)

    bytes_per_decode_step = (weights_mb + state_or_kv_mb) * (1024.0 * 1024.0)
    effective_bw_s = gpu.mem_bw_gb_s * 1e9 * spec.bw_util_assume
    decode_ms_per_tok = (bytes_per_decode_step / max(effective_bw_s, 1.0)) * 1000.0
    decode_s = (decode_ms_per_tok / 1000.0) * spec.decode_tokens

    total_s = compute_s + decode_s + spec.download_seconds
    usd = (total_s / 3600.0) * gpu.usd_per_hr

    return {
        "workload": spec.name,
        "gpu": gpu.name,
        "fits_vram": fits_vram,
        "weights_mb": round(weights_mb, 2),
        "trainable_grad_opt_mb": round(trainable_grad_opt_mb, 2),
        "state_or_kv_mb": round(state_or_kv_mb, 3),
        "activation_mb": round(activation_mb, 2),
        "estimated_peak_vram_mb": estimated_peak_vram_mb,
        "safe_vram_required_mb": safe_vram_required_mb,
        "gpu_vram_mb": round(gpu_vram_mb, 1),
        "estimated_decode_ms_per_tok": round(decode_ms_per_tok, 3),
        "compute_seconds": round(compute_s, 2),
        "decode_seconds": round(decode_s, 2),
        "total_seconds": round(total_s, 2),
        "estimated_usd": round(usd, 4),
    }


def select_optimal_gpu(
    workloads: list[WorkloadSpec],
    *,
    max_wall_minutes: float = 25.0,
    allow_kaggle: bool = True,
    allow_local: bool = False,
) -> dict[str, Any]:
    """Select the cheapest GPU tier that fits peak VRAM + satisfies wall-clock SLA."""
    candidates: list[dict[str, Any]] = []
    for key, gpu in GPU_CATALOG.items():
        if gpu.backend == "mps" and not allow_local:
            continue
        if gpu.backend == "kaggle" and not allow_kaggle:
            continue
        items = [estimate_workload(w, gpu) for w in workloads]
        all_fit = all(it["fits_vram"] for it in items)
        peak_req = max((it["safe_vram_required_mb"] for it in items), default=0.0)
        tot_s = gpu.startup_overhead_s + sum(it["total_seconds"] for it in items)
        tot_min = round(tot_s / 60.0, 2)
        tot_usd = round((tot_s / 3600.0) * gpu.usd_per_hr, 4)
        feasible = all_fit and (tot_min <= max_wall_minutes)
        candidates.append(
            {
                "gpu_key": key,
                "gpu_name": gpu.name,
                "backend": gpu.backend,
                "feasible": feasible,
                "fits_vram": all_fit,
                "max_safe_vram_mb": peak_req,
                "gpu_vram_mb": gpu.vram_gb * 1024.0,
                "total_minutes": tot_min,
                "estimated_usd": tot_usd,
                "workloads": items,
            }
        )

    feasible_list = [c for c in candidates if c["feasible"]]
    feasible_list.sort(key=lambda c: (c["estimated_usd"], c["total_minutes"]))
    chosen = feasible_list[0] if feasible_list else min(
        [c for c in candidates if c["fits_vram"]] or candidates,
        key=lambda c: (c["estimated_usd"], c["total_minutes"]),
    )
    return {"recommended": chosen, "all_candidates": candidates}


def build_cycle2_retrospective_calibration(v2_dir: str | Path) -> dict[str, Any]:
    """Validate analytical VRAM & decode model against actual Cycle-2 L40S measurements."""
    v2_path = Path(v2_dir)
    gpu = GPU_CATALOG["runpod_l40s_48gb"]
    comparisons: list[dict[str, Any]] = []

    x3_file = v2_path / "c2_x3_32k_scaling.json"
    if x3_file.exists():
        x3 = json.loads(x3_file.read_text(encoding="utf-8"))
        for m in x3.get("models", []):
            cfg = m["config"]
            p_cnt = int(cfg["total_parameters"])
            for row in m.get("scaling", []):
                if row.get("status") != "ok":
                    continue
                slen = int(row["seq_len"])
                is_tx = "transformer" in cfg["name"]
                spec = WorkloadSpec(
                    name=f"{cfg['name']}@seq{slen}",
                    total_params=p_cnt,
                    trainable_params=0,
                    n_layers=48,
                    d_model=2560,
                    n_kv_heads=4,
                    d_head=128,
                    seq_len=slen,
                    state_bytes_override=None if is_tx else int(cfg["state_bytes_at_seq"]),
                    eval_chunk_size=None if is_tx else 512,
                    decode_tokens=16,
                )
                est = estimate_workload(spec, gpu)
                act_vram = float(row["peak_vram_mb"])
                comparisons.append(
                    {
                        "benchmark": spec.name,
                        "predicted_vram_mb": est["estimated_peak_vram_mb"],
                        "actual_vram_mb": act_vram,
                        "vram_rel_error_pct": round(100.0 * abs(est["estimated_peak_vram_mb"] - act_vram) / max(act_vram, 1.0), 2),
                        "actual_decode_ms": row["decode_ms_per_tok"],
                    }
                )

    x4_file = v2_path / "c2_x4_giant_killer.json"
    if x4_file.exists():
        x4 = json.loads(x4_file.read_text(encoding="utf-8"))
        for arm in x4.get("arms", [])[:2]:
            p_cnt = int(arm["total_parameters"])
            is_opp = arm.get("is_external_opponent", False)
            spec = WorkloadSpec(
                name=arm["arm"],
                total_params=p_cnt,
                trainable_params=0 if is_opp else 26_574_848,
                n_layers=36 if is_opp else 28,
                d_model=4096 if is_opp else 2048,
                n_kv_heads=8,
                d_head=128,
                seq_len=512,
                decode_tokens=4000,
            )
            est = estimate_workload(spec, gpu)
            act_vram = float(arm["by_dimension"]["inference_efficiency"]["peak_vram_mb"])
            comparisons.append(
                {
                    "benchmark": arm["arm"],
                    "predicted_vram_mb": est["estimated_peak_vram_mb"],
                    "actual_vram_mb": act_vram,
                    "vram_rel_error_pct": round(100.0 * abs(est["estimated_peak_vram_mb"] - act_vram) / max(act_vram, 1.0), 2),
                }
            )

    mean_err = round(
        sum(c["vram_rel_error_pct"] for c in comparisons) / max(len(comparisons), 1),
        2,
    )
    return {
        "gpu": gpu.name,
        "comparisons": comparisons,
        "mean_vram_rel_error_pct": mean_err,
    }


def plan_omega_campaign(*, allow_kaggle: bool = False) -> dict[str, Any]:
    """Build exact pre-flight compute/VRAM/cost estimate for NANOLM Omega EXP-1..EXP-3."""
    workloads = [
        WorkloadSpec(
            name="EXP_Omega1_S4_LieExp_SO3_3seeds_512",
            total_params=650_000,
            trainable_params=650_000,
            n_layers=2,
            d_model=192,
            batch_size=64,
            seq_len=512,
            train_tokens=15 * 600 * 64 * 32,
            decode_tokens=0,
        ),
        WorkloadSpec(
            name="EXP_Omega2_NullspaceLoRA_Continual_3seeds",
            total_params=4_500_000,
            trainable_params=4_500_000,
            n_layers=6,
            d_model=256,
            batch_size=16,
            seq_len=256,
            train_tokens=3 * (400 + 750) * 16 * 256,
            decode_tokens=0,
        ),
        WorkloadSpec(
            name="EXP_Omega3_Unified_1.7B_Nullspace_Cognitive",
            total_params=1_747_149_824,
            trainable_params=26_574_848,
            n_layers=28,
            d_model=2048,
            n_kv_heads=8,
            d_head=128,
            batch_size=1,
            seq_len=512,
            train_tokens=2 * 62 * 256 + 32 * 4 * 128,
            decode_tokens=11_500,
            download_seconds=95.0,
        ),
    ]
    sel = select_optimal_gpu(workloads, max_wall_minutes=20.0, allow_kaggle=allow_kaggle, allow_local=False)
    return {"campaign": "nanolm_omega_v1", "selection": sel}


def plan_omega_v2_campaign(*, allow_kaggle: bool = True) -> dict[str, Any]:
    """Pre-flight compute, VRAM, and USD cost plan for NANOLM Omega-v2 (EXP-Omega4..Omega7)."""
    from nanolm_holonomy.architecture import HSPA_OMEGA_4B_CONFIG

    workloads = [
        WorkloadSpec(
            name="EXP_Omega4_FullLM_S4_8xOOD",
            total_params=2_470_000,
            trainable_params=2_470_000,
            n_layers=6,
            d_model=192,
            batch_size=64,
            seq_len=256,
            train_tokens=4 * 3 * 1_000 * 64 * 32,
            decode_tokens=0,
        ),
        WorkloadSpec(
            name="EXP_Omega5_Gen4_LM_3seed",
            total_params=14_000_000,
            trainable_params=14_000_000,
            n_layers=8,
            d_model=384,
            batch_size=16,
            seq_len=256,
            train_tokens=3 * 3 * 800 * 16 * 256,
            decode_tokens=0,
        ),
        WorkloadSpec(
            name="EXP_Omega6_CUDAGraph_4B_Decode",
            total_params=HSPA_OMEGA_4B_CONFIG.parameter_count,
            trainable_params=0,
            n_layers=48,
            d_model=2560,
            n_kv_heads=4,
            d_head=128,
            batch_size=1,
            seq_len=512,
            state_bytes_override=HSPA_OMEGA_4B_CONFIG.state_bytes_per_sequence(512, 2),
            eval_chunk_size=512,
            decode_tokens=96,
        ),
        WorkloadSpec(
            name="EXP_Omega7_Unified_v2_1.7B_Merged",
            total_params=1_747_444_768,
            trainable_params=26_869_792,
            n_layers=28,
            d_model=2048,
            n_kv_heads=8,
            d_head=128,
            batch_size=1,
            seq_len=512,
            train_tokens=2 * 62 * 256 + 32 * 4 * 128,
            decode_tokens=14_000,
            download_seconds=95.0,
        ),
    ]
    sel = select_optimal_gpu(workloads, max_wall_minutes=90.0, allow_kaggle=allow_kaggle, allow_local=False)
    return {"campaign": "nanolm_omega_v2", "selection": sel}


def plan_fp_v3_campaign(
    *,
    allow_kaggle: bool = True,
    live_runpod_balance_usd: float | None = 23.0448,
    owner_reserve_usd: float = 20.0,
) -> dict[str, Any]:
    """Pre-flight compute, VRAM, and financial guard plan for NANOLM FP-v3 Cycle 4 ($0.00 Kaggle)."""
    workloads = [
        WorkloadSpec(
            name="EXP_FP1_Relational_Microcosm_3seeds",
            total_params=50_000,
            trainable_params=50_000,
            n_layers=4,
            d_model=48,
            batch_size=32,
            seq_len=192,
            train_tokens=5 * 3 * 400 * 32 * 24,
            decode_tokens=0,
        ),
        WorkloadSpec(
            name="EXP_FP2_Gen5_S4_And_14M_3seed_Muon",
            total_params=14_000_000,
            trainable_params=14_000_000,
            n_layers=8,
            d_model=384,
            batch_size=16,
            seq_len=256,
            train_tokens=6 * 3 * 800 * 16 * 256,
            decode_tokens=0,
        ),
        WorkloadSpec(
            name="EXP_FP3_8Primitives_And_VoI_Pareto",
            total_params=50_000,
            trainable_params=50_000,
            n_layers=4,
            d_model=48,
            batch_size=32,
            seq_len=64,
            train_tokens=9 * 3 * 300 * 32 * 24,
            decode_tokens=0,
        ),
        WorkloadSpec(
            name="EXP_FP4_1.7B_WorldController_v3_86tasks",
            total_params=1_747_444_768,
            trainable_params=26_869_792,
            n_layers=28,
            d_model=2048,
            n_kv_heads=8,
            d_head=128,
            batch_size=1,
            seq_len=512,
            train_tokens=2 * 62 * 256 + 32 * 4 * 128,
            decode_tokens=18_000,
            download_seconds=95.0,
        ),
    ]
    runpod_eligible = bool(
        live_runpod_balance_usd is not None and live_runpod_balance_usd > owner_reserve_usd + 8.0
    )
    sel = select_optimal_gpu(workloads, max_wall_minutes=120.0, allow_kaggle=allow_kaggle, allow_local=False)
    return {
        "campaign": "nanolm_fp_v3",
        "financial_guard": {
            "live_runpod_balance_usd": live_runpod_balance_usd,
            "owner_reserve_usd": owner_reserve_usd,
            "runpod_eligible": runpod_eligible,
            "enforce_zero_runpod_spend": not runpod_eligible,
        },
        "selection": sel,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NanoLM Roofline Compute & VRAM Pre-Flight Planner")
    parser.add_argument("--preflight", action="store_true", help="Print Omega pre-flight GPU/VRAM/cost estimate")
    parser.add_argument("--calibrate-v2", type=str, default="", help="Optional artifacts/nanolm_holonomy_v2 dir")
    parser.add_argument("--allow-kaggle", action="store_true")
    args = parser.parse_args()

    out: dict[str, Any] = {}
    if args.preflight or not args.calibrate_v2:
        out["omega_preflight"] = plan_omega_campaign(allow_kaggle=args.allow_kaggle)
        out["omega_v2_preflight"] = plan_omega_v2_campaign(allow_kaggle=args.allow_kaggle)
        out["fp_v3_preflight"] = plan_fp_v3_campaign(allow_kaggle=args.allow_kaggle)
    if args.calibrate_v2:
        out["cycle2_calibration"] = build_cycle2_retrospective_calibration(args.calibrate_v2)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
