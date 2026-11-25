"""Unit, causality, zero-init equivalence, and Optax-Muon numerical parity tests."""
from __future__ import annotations
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_and_muon import ARM_SPECS, ModelCfg, NanoMechanismLM, OptaxParityMuon
from score_tournament import score_files
from audit_and_strict_score import strict_score_files, validate_seed_receipts


class MechanismAndMuonContractTests(unittest.TestCase):
    def test_exact_parameter_counts(self):
        expected = {
            "dense": 14_292_224,
            "value_res": 14_292_231,       # +7 scalar lambdas (layers 1..7)
            "block_attnres": 14_300_928,   # +17 DepthRouters * (256 norm + 256 query) = +8,704 scalars (+0.061%)
            "coda_refine": 14_292_480,     # +256 zero-init channel gate (+0.0018%)
            "dense10": 15_768_064,         # 10 untied layers (+2 * 737,920 = +1,475,840; equal 10 layer passes to coda_refine)
        }
        for arm, exp_p in expected.items():
            m = NanoMechanismLM(ModelCfg(arm=arm))
            actual = sum(p.numel() for p in m.parameters())
            self.assertEqual(actual, exp_p, f"{arm}: {actual} != {exp_p}")

    def test_coda_refine_step0_bit_identical_to_dense(self):
        torch.manual_seed(42)
        m_dense = NanoMechanismLM(ModelCfg(arm="dense", vocab=512, d=64, layers=4, n_head=4, n_kv=2, ffn=176))
        torch.manual_seed(42)
        m_coda = NanoMechanismLM(ModelCfg(arm="coda_refine", vocab=512, d=64, layers=4, n_head=4, n_kv=2, ffn=176))
        m_dense.eval()
        m_coda.eval()
        x = torch.randint(0, 512, (3, 24))
        with torch.no_grad():
            lg_d = m_dense.forward_logits(x)
            lg_c = m_coda.forward_logits(x)
        self.assertEqual(float((lg_d - lg_c).abs().max()), 0.0)

    def test_strict_causal_prefix_invariance_before_and_after_updates(self):
        """Every arm must be strictly causal both at step 0 AND after training steps activate zero-init gates."""
        for arm in ARM_SPECS:
            torch.manual_seed(101)
            cfg = ModelCfg(arm=arm, vocab=256, d=64, layers=4, n_head=4, n_kv=2, ffn=176)
            m = NanoMechanismLM(cfg)
            opt = OptaxParityMuon(m.named_parameters(), lr=4e-3, wd=0.1)

            # Check causality at initialization
            m.eval()
            x1 = torch.randint(0, 256, (2, 20))
            x2 = x1.clone()
            x2[:, 11:] = torch.randint(0, 256, (2, 9))
            with torch.no_grad():
                out1 = m.forward_logits(x1)
                out2 = m.forward_logits(x2)
            self.assertLess(float((out1[:, :11] - out2[:, :11]).abs().max()), 1e-6, f"Init leak in {arm}")

            # Run 3 training steps so q_depth / refine_gate / lam become nonzero
            m.train()
            for _ in range(3):
                bx = torch.randint(0, 256, (4, 17))
                opt.zero_grad(set_to_none=True)
                loss = m(bx[:, :-1], bx[:, 1:])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()

            if arm == "block_attnres":
                max_q = max(float(b.route_attn.q_depth.detach().abs().max()) for b in m.blocks)
                self.assertGreater(max_q, 1e-5, "q_depth did not receive gradients")
            if arm == "coda_refine":
                self.assertGreater(float(m.refine_gate.detach().abs().max()), 1e-5, "refine_gate did not receive gradients")

            m.eval()
            with torch.no_grad():
                out1_post = m.forward_logits(x1)
                out2_post = m.forward_logits(x2)
            self.assertLess(
                float((out1_post[:, :11] - out2_post[:, :11]).abs().max()),
                1e-6,
                f"Post-update causal leak in {arm}",
            )

    def test_optax_muon_numerical_parity(self):
        """Compare 5 steps of OptaxParityMuon against JAX optax.contrib.muon when venv-jax is present."""
        import os, tempfile
        jax_py = Path(os.environ.get("NANOLM_JAX_PYTHON", "/Volumes/Express1M2/nanolm-lab/venv-jax/bin/python"))
        if not jax_py.exists():
            self.skipTest("venv-jax not mounted")

        rng = np.random.default_rng(777)
        emb0 = rng.normal(0, 0.02, (64, 32)).astype(np.float32)
        w2d0 = rng.normal(0, 0.02, (32, 48)).astype(np.float32)
        w2d_tall0 = rng.normal(0, 0.02, (48, 32)).astype(np.float32)
        n1d0 = np.ones((32,), dtype=np.float32)
        grads = [
            {
                "emb": rng.normal(0, 0.05, emb0.shape).astype(np.float32),
                "w2d": rng.normal(0, 0.05, w2d0.shape).astype(np.float32),
                "wtall": rng.normal(0, 0.05, w2d_tall0.shape).astype(np.float32),
                "n1d": rng.normal(0, 0.05, n1d0.shape).astype(np.float32),
            }
            for _ in range(5)
        ]
        lrs = [0.001, 0.0025, 0.004, 0.003, 0.0005]

        # Run PyTorch OptaxParityMuon
        p_emb = torch.nn.Parameter(torch.from_numpy(emb0.copy()))
        p_w2d = torch.nn.Parameter(torch.from_numpy(w2d0.copy()))
        p_wtall = torch.nn.Parameter(torch.from_numpy(w2d_tall0.copy()))
        p_n1d = torch.nn.Parameter(torch.from_numpy(n1d0.copy()))
        named = [("emb.weight", p_emb), ("blocks.0.gu.weight", p_w2d), ("blocks.0.dn.weight", p_wtall), ("blocks.0.n1.w", p_n1d)]
        opt = OptaxParityMuon(named, lr=lrs[0], wd=0.1)
        for lr, g in zip(lrs, grads):
            for grp in opt.param_groups:
                grp["lr"] = lr
            p_emb.grad = torch.from_numpy(g["emb"].copy())
            p_w2d.grad = torch.from_numpy(g["w2d"].copy())
            p_wtall.grad = torch.from_numpy(g["wtall"].copy())
            p_n1d.grad = torch.from_numpy(g["n1d"].copy())
            torch.nn.utils.clip_grad_norm_([p_emb, p_w2d, p_wtall, p_n1d], 1.0)
            opt.step()

        # Run JAX optax.contrib.muon on identical inputs
        with tempfile.TemporaryDirectory() as td:
            tmp_npz = Path(td) / "muon_parity_in.npz"
            np.savez(
                tmp_npz,
                emb0=emb0, w2d0=w2d0, wtall0=w2d_tall0, n1d0=n1d0, lrs=np.array(lrs, dtype=np.float32),
                **{f"g_{k}_{i}": grads[i][k] for i in range(5) for k in ("emb", "w2d", "wtall", "n1d")},
            )
            jax_code = f"""
import numpy as np, jax, jax.numpy as jnp, optax
z = np.load("{tmp_npz}")
params = {{"emb": jnp.asarray(z["emb0"]), "blocks": [{{"w2d": jnp.asarray(z["w2d0"]), "wtall": jnp.asarray(z["wtall0"]), "n1d": jnp.asarray(z["n1d0"])}}]}}
lrs = jnp.asarray(z["lrs"])
sched = lambda n: lrs[n]
dims = lambda p: {{"emb": None, "blocks": [{{"w2d": optax.contrib.MuonDimensionNumbers(), "wtall": optax.contrib.MuonDimensionNumbers(), "n1d": None}}]}}
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.contrib.muon(sched, weight_decay=0.1, adam_b1=0.9, adam_b2=0.95, adam_weight_decay=0.1, consistent_rms=0.2, muon_weight_dimension_numbers=dims))
st = opt.init(params)
for i in range(5):
    g = {{"emb": jnp.asarray(z[f"g_emb_{{i}}"]), "blocks": [{{"w2d": jnp.asarray(z[f"g_w2d_{{i}}"]), "wtall": jnp.asarray(z[f"g_wtall_{{i}}"]), "n1d": jnp.asarray(z[f"g_n1d_{{i}}"])}}]}}
    u, st = opt.update(g, st, params)
    params = optax.apply_updates(params, u)
np.savez("{tmp_npz}.out.npz", emb=np.asarray(params["emb"]), w2d=np.asarray(params["blocks"][0]["w2d"]), wtall=np.asarray(params["blocks"][0]["wtall"]), n1d=np.asarray(params["blocks"][0]["n1d"]))
"""
            subprocess.run([str(jax_py), "-c", jax_code], check=True)
            out = np.load(f"{tmp_npz}.out.npz")
            self.assertLess(float(np.max(np.abs(p_emb.detach().numpy() - out["emb"]))), 1e-5)
            self.assertLess(float(np.max(np.abs(p_w2d.detach().numpy() - out["w2d"]))), 1e-5)
            self.assertLess(float(np.max(np.abs(p_wtall.detach().numpy() - out["wtall"]))), 1e-5)
            self.assertLess(float(np.max(np.abs(p_n1d.detach().numpy() - out["n1d"]))), 1e-5)

    def test_scorer_paired_rules_and_wallclock_interpolation(self):
        import json, tempfile
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for idx, sd in enumerate((1000, 1001, 1002)):
                noise = idx * 0.0002
                payload = {
                    "seed": sd,
                    "gpu": "TestGPU",
                    "arm_order": list(ARM_SPECS),
                    "arms": {
                        "dense": {"params": 14292224, "curve": [{"step": 1500, "train_sec": 100.0, "web": 1.3, "textbook": 1.0, "math": 1.4, "code": 1.1, "mix": 1.25 + noise, "tokens_per_second": 150000}, {"step": 3000, "train_sec": 200.0, "web": 1.24, "textbook": 0.95, "math": 1.34, "code": 1.04, "mix": 1.19 + noise, "tokens_per_second": 150000}]},
                        "value_res": {"params": 14292231, "curve": [{"step": 1500, "train_sec": 101.0, "web": 1.298, "textbook": 0.998, "math": 1.398, "code": 1.098, "mix": 1.248 + noise, "tokens_per_second": 148500}, {"step": 3000, "train_sec": 202.0, "web": 1.238, "textbook": 0.948, "math": 1.338, "code": 1.038, "mix": 1.187 + noise, "tokens_per_second": 148500}]},
                        "block_attnres": {"params": 14300928, "curve": [{"step": 1500, "train_sec": 104.0, "web": 1.29, "textbook": 0.99, "math": 1.39, "code": 1.09, "mix": 1.24 + noise, "tokens_per_second": 144000}, {"step": 3000, "train_sec": 208.0, "web": 1.23, "textbook": 0.94, "math": 1.33, "code": 1.03, "mix": 1.181 + noise, "tokens_per_second": 144000}]},
                        "coda_refine": {"params": 14292480, "curve": [{"step": 1500, "train_sec": 125.0, "web": 1.292, "textbook": 0.992, "math": 1.392, "code": 1.092, "mix": 1.242 + noise, "tokens_per_second": 120000}, {"step": 3000, "train_sec": 250.0, "web": 1.232, "textbook": 0.942, "math": 1.332, "code": 1.032, "mix": 1.184 + noise, "tokens_per_second": 120000}]},
                        "dense10": {"params": 15768064, "curve": [{"step": 1500, "train_sec": 125.0, "web": 1.288, "textbook": 0.988, "math": 1.388, "code": 1.088, "mix": 1.238 + noise, "tokens_per_second": 120000}, {"step": 3000, "train_sec": 250.0, "web": 1.228, "textbook": 0.938, "math": 1.328, "code": 1.028, "mix": 1.180 + noise, "tokens_per_second": 120000}]},
                    },
                }
                p = Path(td) / f"seed_{sd}.json"
                p.write_text(json.dumps(payload))
                paths.append(p)
            res = score_files(paths)
            self.assertEqual(res["verdicts"]["D1_depth_routing"], "ADOPT_BLOCK_ATTNRES")
            # coda_refine takes 250s vs 200s dense, so at 200s its interpolated mix is ~1.203 (> 1.19 dense) -> sparse gated only!
            self.assertEqual(res["verdicts"]["D2_coda_refinement"], "ADVANCE_CODA_TO_SPARSE_GATED_40M")

    def test_kaggle_tournament_end_to_end_smoke(self):
        import json, os, tempfile
        pkg_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            data_dir = td_p / "data"
            out_dir = td_p / "out"
            data_dir.mkdir()
            out_dir.mkdir()
            rng = np.random.default_rng(123)
            (data_dir / "train.bin").write_bytes(rng.integers(0, 256, 8192, dtype=np.uint16).tobytes())
            manifest = {"mix": {"web": 0.62, "textbook": 0.13, "math": 0.13, "code": 0.12}, "domains": {}}
            for dom in ("web", "textbook", "math", "code"):
                (data_dir / f"val_{dom}.bin").write_bytes(rng.integers(0, 256, 4096, dtype=np.uint16).tobytes())
                manifest["domains"][dom] = {"val": {"bytes_per_token": 4.0}}
            (data_dir / "manifest.json").write_text(json.dumps(manifest))

            env = os.environ.copy()
            env.update({
                "NANOLM_DATA": str(data_dir),
                "NANOLM_OUT": str(out_dir),
                "NANOLM_SEED": "1001",
                "NANOLM_STEPS": "2",
                "NANOLM_EVAL_EVERY": "2",
                "NANOLM_EVAL_BATCHES": "2",
                "NANOLM_WARMUP": "1",
                "NANOLM_BS": "2",
                "NANOLM_SEQ": "32",
                "NANOLM_VOCAB": "256",
                "NANOLM_D": "64",
                "NANOLM_LAYERS": "4",
                "NANOLM_FFN": "176",
            })
            subprocess.run([sys.executable, str(pkg_dir / "kaggle_tournament.py")], env=env, check=True)
            seed_json = json.loads((out_dir / "seed_1001.json").read_text())
            self.assertTrue(seed_json["completed"])
            self.assertEqual(set(seed_json["arms"].keys()), set(ARM_SPECS))
            for arm in ARM_SPECS:
                self.assertIn("rank_buckets_mix", seed_json["arms"][arm])
            scored = score_files([out_dir / "seed_1001.json"])
            self.assertIn("D1_depth_routing", scored["verdicts"])
            self.assertIn("D2_coda_refinement", scored["verdicts"])

    def test_strict_receipt_validator_and_real_receipts(self):
        import copy, json, tempfile
        pkg_dir = Path(__file__).resolve().parent
        real_paths = sorted((pkg_dir / "results" / "tournament").glob("seed_*.json"))
        self.assertEqual(len(real_paths), 3)
        res = strict_score_files(real_paths)
        self.assertEqual(res["historic_verdicts_unchanged"]["D1_depth_routing"], "CLOSE_DEPTH_ROUTING_AT_14M")
        self.assertEqual(res["historic_verdicts_unchanged"]["D2_coda_refinement"], "REJECT_UNCONDITIONAL_CODA_REFINE")
        self.assertEqual(res["strict_post_audit_verdicts"]["D1_strict"], "QUALIFY_VALUE_RES_FOR_42M_GATE_REJECT_BLOCK_ATTNRES")

        sample = json.loads(real_paths[0].read_text())
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            # Negative case 1: completed=False
            bad1 = copy.deepcopy(sample)
            bad1["completed"] = False
            p1 = td_p / "bad_incomplete.json"
            p1.write_text(json.dumps(bad1))
            with self.assertRaisesRegex(ValueError, "completed=True"):
                validate_seed_receipts([p1])

            # Negative case 2: truncated final step
            bad2 = copy.deepcopy(sample)
            bad2["arms"]["dense"]["curve"][-1]["step"] = 2500
            p2 = td_p / "bad_step.json"
            p2.write_text(json.dumps(bad2))
            with self.assertRaisesRegex(ValueError, "final step"):
                validate_seed_receipts([p2])

            # Negative case 3: duplicate seed ID
            p3a = td_p / "dup_a.json"
            p3b = td_p / "dup_b.json"
            p3a.write_text(json.dumps(sample))
            p3b.write_text(json.dumps(sample))
            with self.assertRaisesRegex(ValueError, "duplicate seed"):
                validate_seed_receipts([p3a, p3b])


if __name__ == "__main__":
    unittest.main(verbosity=2)
