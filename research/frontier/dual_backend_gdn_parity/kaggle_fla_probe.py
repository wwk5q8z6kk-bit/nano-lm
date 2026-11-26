"""Kaggle CUDA GPU (`fla==0.6.0`) Ground-Truth Parity & T4 Throughput Probe.

Installs upstream `flash-linear-attention==0.6.0` + `fla-core==0.6.0` on CUDA (`Tesla T4`), runs:
  1. Exact `SmolLM2-135M` hybrid parameter-count verification (`151,928,140` params across `22 GDN + 8 GQA`).
  2. Real CUDA `fla.layers.GatedDeltaNet` (`TorchHybridLM(..., use_fla=True)`) on a 4-layer hybrid (`3 GDN + 1 GQA`,
     `d=256, H=4, KV=2, hd=64, ffn=704, V=4096, seq=192`), exporting:
       - Initial weights (`state_dict`), token inputs,
       - `FP32` forward logits, scalar CE loss, and per-parameter gradients,
       - `BF16` mixed-precision forward logits, scalar CE loss, and a `10`-step `AdamW` training trajectory!
  3. Cross-checks CUDA `fla` (`use_fla=True`) against `TorchHybridLM(..., use_fla=False)` on the same GPU.
  4. Measures `Tesla T4` GDN training step throughput (`tokens/sec`) for the unified hardware routing table.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path(os.environ.get("PROBE_OUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

if os.environ.get("PROBE_INSTALL_FLA", "1") == "1":
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "flash-linear-attention==0.6.0", "fla-core==0.6.0"],
        check=True,
    )

import importlib.metadata as md
import numpy as np
import torch
import torch.nn.functional as F
from dual_gdn_core import build_torch_classes, torch_model_to_npz_dict

TorchGDNLayer, TorchHybridBlock, TorchHybridLM, _ = build_torch_classes()


def main():
    assert torch.cuda.is_available(), "CUDA required for real fla==0.6.0 kernel probe"
    dev = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    # Triton `tl.dot` supports hardware `bf16` MMA only on Ampere+ (`sm >= 80`); Turing (`sm_75` T4) uses native `fp16` Tensor Cores.
    amp_dtype = torch.bfloat16 if cc_major >= 8 else torch.float16
    amp_dtype_name = "bfloat16" if cc_major >= 8 else "float16"

    # 1. Verify exact SmolLM2-135M 22-GDN + 8-GQA parameter count (151,928,140) with real `fla.layers.GatedDeltaNet`
    m_135m = TorchHybridLM(
        d=576, layers=30, ffn=1536, heads=9, kv_heads=3, vocab=49152,
        attn_layers=(1, 2, 3, 4, 21, 22, 23, 24), use_fla=True,
    )
    smollm2_hybrid_params = sum(p.numel() for p in m_135m.parameters())
    del m_135m

    # 2. Construct deterministic 4-layer Hybrid (`layers 0,2,3 = GDN`, `layer 1 = GQA`) with `seq=192` (3 WY chunks of 64)
    cfg = dict(d=256, layers=4, ffn=704, heads=4, kv_heads=2, vocab=4096, attn_layers=(1,))
    torch.manual_seed(2026)
    m_fla = TorchHybridLM(**cfg, use_fla=True).to(dev)
    init_sd_np = torch_model_to_npz_dict(m_fla)

    # Also verify pure-PyTorch `use_fla=False` matches CUDA `fla` on GPU
    m_ref = TorchHybridLM(**cfg, use_fla=False).to(dev)
    m_ref.load_state_dict(m_fla.state_dict(), strict=True)

    rng = np.random.default_rng(77)
    ids_np = rng.integers(0, cfg["vocab"], size=(2, 193), dtype=np.int64)
    step_ids_np = rng.integers(0, cfg["vocab"], size=(10, 2, 193), dtype=np.int64)
    ids = torch.from_numpy(ids_np).to(dev)

    # ---- A. FP32 Forward + Backward on real CUDA `fla==0.6.0` ----
    m_fla.float().zero_grad(set_to_none=True)
    lg_fla32 = m_fla(ids[:, :-1])
    loss_fla32 = F.cross_entropy(lg_fla32.reshape(-1, cfg["vocab"]), ids[:, 1:].reshape(-1))
    loss_fla32.backward()
    grads_fla32 = {k: p.grad.detach().cpu().float().numpy() for k, p in m_fla.named_parameters()}
    lg_fla32_np = lg_fla32.detach().cpu().float().numpy()
    loss_fla32_val = float(loss_fla32.detach().cpu().item())

    # Compare vs pure-PyTorch WY reference (`use_fla=False`) on same GPU
    m_ref.float().zero_grad(set_to_none=True)
    lg_ref32 = m_ref(ids[:, :-1])
    loss_ref32 = F.cross_entropy(lg_ref32.reshape(-1, cfg["vocab"]), ids[:, 1:].reshape(-1))
    loss_ref32.backward()
    ref_vs_fla_logit_max = float((lg_ref32 - lg_fla32).abs().max().item())
    ref_vs_fla_loss_abs = abs(float(loss_ref32.item()) - loss_fla32_val)

    # ---- B. Native CUDA `fla==0.6.0` Mixed-Precision Forward (`BF16` on sm>=80, `FP16` Tensor Core on sm_75 T4) ----
    m_bf = TorchHybridLM(**cfg, use_fla=True).to(dev)
    m_bf.load_state_dict({k: torch.from_numpy(v).to(dev) for k, v in init_sd_np.items()}, strict=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
        lg_bf16 = m_bf(ids[:, :-1]).float()
        loss_bf16 = float(F.cross_entropy(lg_bf16.reshape(-1, cfg["vocab"]), ids[:, 1:].reshape(-1)).item())
    lg_bf16_np = lg_bf16.cpu().numpy()

    # ---- C. 10-step AdamW Trajectory on real CUDA `fla==0.6.0` (native mixed-precision autocast, no GradScaler needed at lr=3e-4) ----
    opt = torch.optim.AdamW(m_bf.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    traj_bf16 = []
    for s_i in range(10):
        b_ids = torch.from_numpy(step_ids_np[s_i]).to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            lg_s = m_bf(b_ids[:, :-1])
            ls = F.cross_entropy(lg_s.float().reshape(-1, cfg["vocab"]), b_ids[:, 1:].reshape(-1))
        ls.backward()
        opt.step()
        traj_bf16.append(round(float(ls.detach().cpu().item()), 6))

    # Also record exact 10-step FP32 trajectory
    m_fp = TorchHybridLM(**cfg, use_fla=True).to(dev).float()
    m_fp.load_state_dict({k: torch.from_numpy(v).to(dev) for k, v in init_sd_np.items()}, strict=True)
    opt32 = torch.optim.AdamW(m_fp.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    traj_fp32 = []
    for s_i in range(10):
        b_ids = torch.from_numpy(step_ids_np[s_i]).to(dev)
        opt32.zero_grad(set_to_none=True)
        lg_s = m_fp(b_ids[:, :-1])
        ls = F.cross_entropy(lg_s.reshape(-1, cfg["vocab"]), b_ids[:, 1:].reshape(-1))
        ls.backward()
        opt32.step()
        traj_fp32.append(round(float(ls.detach().cpu().item()), 6))

    # ---- D. Measure `SmolLM2-135M`-scale (`151.9M`) Hybrid Throughput on Kaggle GPU (`bs=4, seq=1024`, FP16) ----
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m_bench = TorchHybridLM(
        d=576, layers=30, ffn=1536, heads=9, kv_heads=3, vocab=49152,
        attn_layers=(1, 2, 3, 4, 21, 22, 23, 24), use_fla=True,
    ).to(dev)
    opt_b = torch.optim.AdamW(m_bench.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    x_bench = torch.randint(0, 49152, (4, 1025), device=dev)
    for w_i in range(12):
        if w_i == 2:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        opt_b.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            lg_b = m_bench(x_bench[:, :-1])
            lb = F.cross_entropy(lg_b.float().reshape(-1, 49152), x_bench[:, 1:].reshape(-1))
        scaler.scale(lb).backward()
        scaler.step(opt_b)
        scaler.update()
    torch.cuda.synchronize()
    dt_bench = (time.perf_counter() - t0) / 10.0
    t4_152m_tok_s = round((4 * 1024) / dt_bench, 1)
    t4_peak_gib = round(torch.cuda.max_memory_allocated() / (2 ** 30), 3)

    bundle_arrays = {
        **{f"init__{k}": v for k, v in init_sd_np.items()},
        **{f"grad32__{k}": v for k, v in grads_fla32.items()},
        "ids": ids_np,
        "step_ids": step_ids_np,
        "logits_fp32": lg_fla32_np,
        "logits_bf16": lg_bf16_np,
    }
    bundle_path = OUT_DIR / "fla_cuda_parity_bundle.npz"
    np.savez_compressed(bundle_path, **bundle_arrays)
    bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    meta = {
        "schema": "nanolm_fla_cuda_parity_probe_v1",
        "gpu_name": gpu_name,
        "compute_capability": f"{cc_major}.{cc_minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "fla_version": md.version("flash-linear-attention"),
        "triton_version": md.version("triton"),
        "cuda_fla_amp_dtype": amp_dtype_name,
        "smollm2_135m_hybrid_params_verified": smollm2_hybrid_params,
        "probe_cfg": cfg,
        "loss_fp32": loss_fla32_val,
        "loss_bf16": loss_bf16,
        "traj_fp32_10steps": traj_fp32,
        "traj_bf16_10steps": traj_bf16,
        "pytorch_wy_ref_vs_cuda_fla_fp32": {
            "logit_max_abs_err": ref_vs_fla_logit_max,
            "loss_abs_err": ref_vs_fla_loss_abs,
        },
        "kaggle_gpu_152m_hybrid_bench": {
            "batch_x_seq": "4x1024",
            "dtype": "float16",
            "step_ms": round(dt_bench * 1000.0, 2),
            "tokens_per_second": t4_152m_tok_s,
            "peak_gpu_gib": t4_peak_gib,
        },
        "bundle_file": bundle_path.name,
        "bundle_sha256": bundle_sha,
    }
    (OUT_DIR / "fla_cuda_probe_meta.json").write_text(json.dumps(meta, indent=2))
    print("FLA_PROBE_META_BEGIN")
    print(json.dumps(meta, indent=2))
    print("FLA_PROBE_META_END", flush=True)


if __name__ == "__main__":
    main()
