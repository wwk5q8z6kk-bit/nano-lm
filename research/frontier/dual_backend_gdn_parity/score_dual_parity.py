"""Strict Mechanical Scorer for Dual-Backend `PyTorch/CUDA (fla 0.6.0) <-> JAX/XLA (Cloud TPU)` Parity & `R0-TPU`."""
from __future__ import annotations
import json
import sys
from pathlib import Path

# Verified Hopper H100 reference from Draft PR #214 (`r0_hopper_result.json`)
HOPPER_H100_REF = {
    "r0_547m_gdn_1xH100_2k": {
        "hardware": "1x NVIDIA H100 80GB HBM3 (fla 0.6.0 + triton 3.7.1)",
        "tokens_per_second": 56133.0,
        "mfu_of_datasheet_peak": 0.1862,
        "usd_per_hr_community": 2.69,
        "usd_per_hr_secure": 4.39,
    },
    "r0_547m_gdn_kd32_1xH100_2k": {
        "hardware": "1x NVIDIA H100 80GB HBM3 + Cached Top-32 KD",
        "tokens_per_second": 48261.0,
        "mfu_of_datasheet_peak": 0.1602,
        "usd_per_hr_community": 2.69,
        "usd_per_hr_secure": 4.39,
    },
}

TPU_CHIP_PRICING_USD_HR = {
    "v5e": {"spot_per_chip": 0.36, "ondemand_per_chip": 1.20},
    "v6e": {"spot_per_chip": 1.30, "ondemand_per_chip": 3.24},
}


def usd_per_1b_tokens(tok_s: float, usd_per_hr: float) -> float:
    return round((1e9 / max(tok_s, 1e-6)) / 3600.0 * usd_per_hr, 2)


def score_dual_parity(tpu_res: dict) -> dict:
    par = tpu_res["parity"]
    fp32 = par["fp32_vs_cuda_fla"]
    bf16 = par["bf16_vs_cuda_fla"]
    stab = tpu_res["stability_100"]
    benches = {r["label"]: r for r in tpu_res["benchmarks"]}

    r0_gdn_row = benches.get("r0_547m_gdn_1chip_2k") or next(iter(benches.values()))
    r0_att_row = benches.get("r0_527m_attn_1chip_2k") or next(
        (v for k, v in benches.items() if "attn" in k), r0_gdn_row
    )

    p1_pass = bool(
        par["roundtrip_max_abs_diff"] == 0.0
        and par["smollm2_135m_hybrid_params_jax"] == 151_928_140
        and par["smollm2_135m_hybrid_params_fla"] == 151_928_140
        and (tpu_res.get("smoke") or r0_gdn_row["params"] == 546_865_360)
    )
    p2_pass = bool(
        fp32["logit_max_abs_err"] <= 1e-4
        and fp32["loss_abs_err"] <= 1e-5
        and fp32["grad_max_rel_err"] <= 2e-3
    )
    p3_pass = bool(
        bf16["loss_abs_err"] <= 2e-3
        and bf16["step10_loss_abs_err"] <= 5e-3
    )

    speed_ratio_vs_attn = r0_gdn_row["tokens_per_second_total"] / max(r0_att_row["tokens_per_second_total"], 1e-6)
    p4_pass = bool(
        stab["nonfinite_steps"] == 0
        and (r0_gdn_row["mfu_of_datasheet_peak"] >= 0.15 or speed_ratio_vs_attn >= 0.60)
    )

    all_pass = p1_pass and p2_pass and p3_pass and p4_pass
    verdict = (
        "DUAL_BACKEND_PARITY_VERIFIED_READY_FOR_STAGE_C"
        if all_pass
        else "INVESTIGATE_PARITY_OR_TPU_KERNEL_BEFORE_STAGE_C"
    )

    dev_kind = tpu_res.get("device_kind", "TPU v5 lite")
    gen = "v6e" if "v6" in dev_kind.lower() else "v5e"
    prices = TPU_CHIP_PRICING_USD_HR[gen]

    routing_table = []
    for r in tpu_res["benchmarks"]:
        dp = r["dp_chips"]
        spot_hr = prices["spot_per_chip"] * dp
        od_hr = prices["ondemand_per_chip"] * dp
        routing_table.append({
            "backend": f"GCP {dev_kind} ({dp} chip{'s' if dp > 1 else ''}, JAX WY-chunk)",
            "workload": r["label"],
            "seq": r["seq"],
            "tokens_per_second": r["tokens_per_second_total"],
            "mfu": r["mfu_of_datasheet_peak"],
            "peak_hbm_gib_per_chip": r["peak_chip_hbm_gib"],
            "usd_per_1b_tokens_spot": usd_per_1b_tokens(r["tokens_per_second_total"], spot_hr),
            "usd_per_1b_tokens_ondemand": usd_per_1b_tokens(r["tokens_per_second_total"], od_hr),
        })

    for k, href in HOPPER_H100_REF.items():
        routing_table.append({
            "backend": href["hardware"],
            "workload": k,
            "seq": 2048,
            "tokens_per_second": href["tokens_per_second"],
            "mfu": href["mfu_of_datasheet_peak"],
            "usd_per_1b_tokens_community": usd_per_1b_tokens(href["tokens_per_second"], href["usd_per_hr_community"]),
            "usd_per_1b_tokens_secure": usd_per_1b_tokens(href["tokens_per_second"], href["usd_per_hr_secure"]),
        })

    if par.get("kaggle_gpu_meta"):
        km = par["kaggle_gpu_meta"]
        routing_table.append({
            "backend": "Kaggle Free GPU (Tesla T4, PyTorch fla 0.6.0)",
            "workload": "stageb_152m_hybrid_1xT4_1k",
            "seq": 1024,
            "tokens_per_second": km["tokens_per_second"],
            "peak_hbm_gib": km["peak_gpu_gib"],
            "usd_per_1b_tokens_free_quota": 0.0,
        })

    return {
        "device_kind": dev_kind,
        "criteria": {
            "P1_checkpoint_bridge_bit_exact": {
                "pass": p1_pass,
                "roundtrip_max_abs_diff": par["roundtrip_max_abs_diff"],
                "smollm2_135m_hybrid_params": par["smollm2_135m_hybrid_params_jax"],
                "r0_547m_gdn_params": r0_gdn_row["params"],
            },
            "P2_cuda_fla_vs_jax_fp32_parity": {
                "pass": p2_pass,
                **fp32,
            },
            "P3_cuda_fla_vs_jax_bf16_and_multistep_parity": {
                "pass": p3_pass,
                **bf16,
            },
            "P4_tpu_r0_stability_and_throughput_gate": {
                "pass": p4_pass,
                "nonfinite_steps_100": stab["nonfinite_steps"],
                "r0_gdn_mfu": r0_gdn_row["mfu_of_datasheet_peak"],
                "r0_gdn_vs_attn_speed_ratio_2k": round(speed_ratio_vs_attn, 4),
                "stability_loss_start_to_end": [stab["loss_step1"], stab["loss_final"]],
            },
        },
        "unified_hardware_routing_table": routing_table,
        "verdict": verdict,
    }


if __name__ == "__main__":
    p = Path(sys.argv[1])
    print(json.dumps(score_dual_parity(json.loads(p.read_text())), indent=2))
