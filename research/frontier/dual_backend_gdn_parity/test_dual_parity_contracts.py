"""Local CPU PyTorch <-> JAX Gated DeltaNet Parity, Causality, Round-Trip, and Scorer Contract Tests."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG_DIR))

from dual_gdn_core import build_torch_classes, torch_model_to_npz_dict
from score_dual_parity import score_dual_parity


class DualBackendGDNContractTests(unittest.TestCase):
    def test_pytorch_vs_jax_exact_parity_causality_and_scorer(self):
        import torch
        import torch.nn.functional as F

        jax_env = os.environ.get("NANOLM_JAX_PYTHON", "")
        jax_py = Path(jax_env) if (jax_env and Path(jax_env).exists()) else Path(sys.executable)
        probe = subprocess.run([str(jax_py), "-c", "import jax, optax"], capture_output=True)
        if probe.returncode != 0:
            self.skipTest("JAX/Optax interpreter not configured in NANOLM_JAX_PYTHON")

        _, _, TorchHybridLM, _ = build_torch_classes()

        # 1. Verify exact R0 546,865,360 and SmolLM2-135M 151,928,140 parameter counts in PyTorch
        m_135 = TorchHybridLM(576, 30, 1536, 9, 3, 49152, attn_layers=(1, 2, 3, 4, 21, 22, 23, 24), use_fla=False)
        self.assertEqual(sum(p.numel() for p in m_135.parameters()), 151_928_140)
        del m_135

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            cfg = dict(d=128, layers=4, ffn=352, heads=4, kv_heads=2, vocab=1024, attn_layers=(1,))
            torch.manual_seed(2026)
            m = TorchHybridLM(**cfg, use_fla=False).float()
            init_sd = torch_model_to_npz_dict(m)

            rng = np.random.default_rng(77)
            ids_np = rng.integers(0, cfg["vocab"], size=(2, 193), dtype=np.int64)  # 192 tokens = 3 chunks of 64!
            step_ids_np = rng.integers(0, cfg["vocab"], size=(10, 2, 193), dtype=np.int64)
            ids = torch.from_numpy(ids_np)

            # Causal prefix invariance across chunk boundaries (token 90 > chunk_size=64!) in PyTorch WY GDN
            ids2 = ids.clone()
            ids2[:, 90:] = torch.from_numpy(rng.integers(0, cfg["vocab"], size=(2, 103), dtype=np.int64))
            with torch.no_grad():
                o1 = m(ids[:, :-1])
                o2 = m(ids2[:, :-1])
            self.assertLess(float((o1[:, :90] - o2[:, :90]).abs().max().item()), 1e-5)

            # FP32 fwd + bwd
            m.zero_grad(set_to_none=True)
            lg32 = m(ids[:, :-1])
            loss32 = F.cross_entropy(lg32.reshape(-1, cfg["vocab"]), ids[:, 1:].reshape(-1))
            loss32.backward()
            grads32 = {k: p.grad.detach().numpy() for k, p in m.named_parameters()}
            lg32_np = lg32.detach().numpy()
            loss32_val = float(loss32.item())

            # BF16 fwd
            m_bf = TorchHybridLM(**cfg, use_fla=False)
            m_bf.load_state_dict(m.state_dict(), strict=True)
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                lg_bf = m_bf(ids[:, :-1]).float()
                loss_bf_val = float(F.cross_entropy(lg_bf.reshape(-1, cfg["vocab"]), ids[:, 1:].reshape(-1)).item())

            # 10-step trajectories (FP32 and BF16)
            def torch_10_steps(use_bf16: bool):
                mt = TorchHybridLM(**cfg, use_fla=False).float()
                mt.load_state_dict({k: torch.from_numpy(v) for k, v in init_sd.items()}, strict=True)
                opt = torch.optim.AdamW(mt.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
                tr = []
                for s_i in range(10):
                    bi = torch.from_numpy(step_ids_np[s_i])
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
                        lg_s = mt(bi[:, :-1])
                        ls = F.cross_entropy(lg_s.float().reshape(-1, cfg["vocab"]), bi[:, 1:].reshape(-1))
                    ls.backward()
                    opt.step()
                    tr.append(round(float(ls.item()), 6))
                return tr

            traj32 = torch_10_steps(False)
            trajbf = torch_10_steps(True)

            bundle_p = td_p / "bundle.npz"
            np.savez_compressed(
                bundle_p,
                **{f"init__{k}": v for k, v in init_sd.items()},
                **{f"grad32__{k}": v for k, v in grads32.items()},
                ids=ids_np,
                step_ids=step_ids_np,
                logits_fp32=lg32_np,
                logits_bf16=lg_bf.numpy(),
            )
            meta_p = td_p / "meta.json"
            meta_p.write_text(json.dumps({
                "smollm2_135m_hybrid_params_verified": 151_928_140,
                "probe_cfg": cfg,
                "loss_fp32": loss32_val,
                "loss_bf16": loss_bf_val,
                "traj_fp32_10steps": traj32,
                "traj_bf16_10steps": trajbf,
            }))
            out_p = td_p / "r0_tpu_smoke.json"

            # Execute JAX `r0_tpu_bench_and_parity.py --smoke` + causal prefix invariance across chunk boundaries!
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "cpu"
            subprocess.run(
                [str(jax_py), str(PKG_DIR / "r0_tpu_bench_and_parity.py"), "--fla-bundle", str(bundle_p), "--fla-meta", str(meta_p), "--out", str(out_p), "--smoke"],
                env=env,
                check=True,
                timeout=180,
            )

            # Also verify JAX causal prefix invariance across chunk boundary (pos 90 > 64) & exact 546,865,360 R0 param count
            jax_extra = f"""
import sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "{PKG_DIR}")
from dual_gdn_core import init_jax_hybrid_params, jax_hybrid_logits, jax_n_params
p_r0 = init_jax_hybrid_params(jax.random.PRNGKey(0), dict(d=896, layers=24, ffn=4864, heads=7, kv_heads=7, vocab=151936, attn_layers=()))
assert jax_n_params(p_r0) == 546_865_360, jax_n_params(p_r0)
del p_r0
cfg = dict(d=64, layers=2, ffn=176, heads=2, kv_heads=2, vocab=256, attn_layers=())
p = init_jax_hybrid_params(jax.random.PRNGKey(3), cfg)
x1 = jnp.asarray(np.random.default_rng(1).integers(0, 256, (2, 160), dtype=np.int32))
x2 = x1.at[:, 90:].set(jnp.asarray(np.random.default_rng(2).integers(0, 256, (2, 70), dtype=np.int32)))
y1 = jax_hybrid_logits(p, x1, cfg, jnp.float32, chunk_size=64)
y2 = jax_hybrid_logits(p, x2, cfg, jnp.float32, chunk_size=64)
assert float(jnp.max(jnp.abs(y1[:, :90] - y2[:, :90]))) < 1e-5
print("JAX_EXTRA_OK")
"""
            ex_out = subprocess.check_output([str(jax_py), "-c", jax_extra], env=env, text=True)
            self.assertIn("JAX_EXTRA_OK", ex_out)

            res = json.loads(out_p.read_text())
            scored_cpu = score_dual_parity(res)
            self.assertTrue(scored_cpu["criteria"]["P1_checkpoint_bridge_bit_exact"]["pass"], msg=scored_cpu)
            self.assertTrue(scored_cpu["criteria"]["P2_cuda_fla_vs_jax_fp32_parity"]["pass"], msg=scored_cpu)
            self.assertTrue(scored_cpu["criteria"]["P3_cuda_fla_vs_jax_bf16_and_multistep_parity"]["pass"], msg=scored_cpu)
            self.assertEqual(scored_cpu["criteria"]["P4_tpu_r0_stability_and_throughput_gate"]["nonfinite_steps_100"], 0)
            # On laptop CPU (`mfu=0.0003` vs 197 TFLOPS TPU peak), P4 must fail closed:
            self.assertFalse(scored_cpu["criteria"]["P4_tpu_r0_stability_and_throughput_gate"]["pass"])
            self.assertEqual(scored_cpu["verdict"], "INVESTIGATE_PARITY_OR_TPU_KERNEL_BEFORE_STAGE_C")

            # Verify positive `P4` gate transition when TPU MFU threshold (`>= 0.15`) is met:
            res_tpu_mock = json.loads(json.dumps(res))
            res_tpu_mock["device_kind"] = "TPU v5 lite"
            res_tpu_mock["benchmarks"][0]["mfu_of_datasheet_peak"] = 0.21
            scored_tpu = score_dual_parity(res_tpu_mock)
            self.assertTrue(scored_tpu["criteria"]["P4_tpu_r0_stability_and_throughput_gate"]["pass"], msg=scored_tpu)
            self.assertEqual(scored_tpu["verdict"], "DUAL_BACKEND_PARITY_VERIFIED_READY_FOR_STAGE_C", msg=scored_tpu)


if __name__ == "__main__":
    unittest.main(verbosity=2)
