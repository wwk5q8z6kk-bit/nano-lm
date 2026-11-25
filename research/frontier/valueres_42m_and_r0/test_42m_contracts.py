"""Unit, causality, parameter-count, ablation, and strict scorer tests for the 40.38M ValueRes package."""
from __future__ import annotations
import copy
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
from score_42m_gate import REQUIRED_ARMS, score_42m_runs


class Gate42MContractTests(unittest.TestCase):
    def test_scorer_strict_provenance_and_decision_rules(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            paths = []
            val_hashes = {d: "a" * 64 for d in ("web", "textbook", "math", "code")}
            for idx, seed in enumerate((1000, 1001, 1002)):
                eps = idx * 0.0001
                arm_specs = {
                    "dense": (40_379_904, 1.0210 + eps, 1200.0, {}),
                    "resid_x0_skip": (40_379_911, 1.0185 + eps, 1202.0, {"no_skip": 0.002, "ablate_late_6_7": 0.001, "ablate_mid_3_4": 0.001, "perm_v0": 0.01}),
                    "resformer_fixed": (40_379_904, 1.0105 + eps, 1201.0, {"no_skip": 0.04, "ablate_late_6_7": 0.03, "ablate_mid_3_4": 0.015, "perm_v0": 0.08}),
                    "value_res": (40_379_911, 1.0065 + eps, 1201.5, {"no_skip": 0.06, "ablate_late_6_7": 0.045, "ablate_mid_3_4": 0.012, "perm_v0": 0.11}),
                }
                for arm, (n_p, final_mix, sec, abls) in arm_specs.items():
                    rec = {
                        "schema": "nanolm_valueres_42m_confirm_v1",
                        "completed": True,
                        "arm": arm,
                        "size": "M",
                        "seed": seed,
                        "params": n_p,
                        "recipe": {"steps": 24414, "train_tokens": 799997952},
                        "provenance": {"code_sha256": "b" * 64, "val_sha256": val_hashes},
                        "curve": [
                            {"step": 12000, "tokens": 393216000, "train_sec": sec * 0.5, "tokens_per_second": 327000, "web": final_mix + 0.06, "textbook": final_mix - 0.2, "math": final_mix + 0.1, "code": final_mix - 0.15, "mix": final_mix + 0.04},
                            {"step": 24414, "tokens": 799997952, "train_sec": sec, "tokens_per_second": 327000, "web": final_mix + 0.05, "textbook": final_mix - 0.22, "math": final_mix + 0.08, "code": final_mix - 0.18, "mix": final_mix},
                        ],
                        "rank_buckets_mix": {"rank1_calib": 0.088, "rank2_8_rerank": 0.258, "rank9_64_mid": 0.285, "rank65plus_miss": 0.375, "top1_acc": 0.43, "top64_acc": 0.86},
                        "causal_ablations": {k: {"mix": final_mix + v, "delta_mix_vs_intact": v} for k, v in abls.items()},
                        "downstream_7tasks": {"arc_easy": {"acc": 0.45, "acc_norm": 0.44}, "lambada": {"acc": 0.25, "ppl": 65.0}},
                    }
                    p = td_p / f"run_{arm}_M_s{seed}.json"
                    p.write_text(json.dumps(rec))
                    paths.append(p)

            res = score_42m_runs(paths, expected_steps=24414, expected_tokens=799997952)
            self.assertEqual(res["verdicts"]["G1_42m_architecture_gate"], "ADOPT_LEARNED_VALUE_RES_AT_42M")
            self.assertEqual(res["verdicts"]["G2_causal_v0_mechanism_gate"], "CONFIRMED_LATE_POSITIONAL_V0_TRANSPORT")

            # Occam razor test: if value_res and resformer_fixed tie (<0.002 diff), prefer 0-extra-param resformer_fixed
            for seed in (1000, 1001, 1002):
                p_vr = td_p / f"run_value_res_M_s{seed}.json"
                d_vr = json.loads(p_vr.read_text())
                d_rf = json.loads((td_p / f"run_resformer_fixed_M_s{seed}.json").read_text())
                d_vr["curve"][-1]["mix"] = round(d_rf["curve"][-1]["mix"] - 0.0008, 5)
                p_vr.write_text(json.dumps(d_vr))
            res_tie = score_42m_runs(paths, expected_steps=24414, expected_tokens=799997952)
            self.assertEqual(res_tie["verdicts"]["G1_42m_architecture_gate"], "ADOPT_RESFORMER_FIXED_05_AT_42M")

            # Provenance mismatch rejection
            p_bad = td_p / "run_dense_M_s1000.json"
            d_bad = json.loads(p_bad.read_text())
            d_bad["provenance"]["val_sha256"]["code"] = "c" * 64
            p_bad.write_text(json.dumps(d_bad))
            with self.assertRaisesRegex(ValueError, "mismatched validation shard SHA256"):
                score_42m_runs(paths)

    def test_jax_params_causality_and_end_to_end_smoke(self):
        jax_env = os.environ.get("NANOLM_JAX_PYTHON", "")
        jax_py = Path(jax_env) if jax_env else Path(sys.executable)
        probe = subprocess.run([str(jax_py), "-c", "import jax, optax"], capture_output=True)
        if probe.returncode != 0:
            self.skipTest("JAX/Optax interpreter not configured in NANOLM_JAX_PYTHON")

        check_code = f"""
import sys, json, tempfile
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "{PKG_DIR}")
import valueres_42m_jax as m

# 1. Exact M-scale parameter counts
expected = {{"dense": 40_379_904, "resid_x0_skip": 40_379_911, "resformer_fixed": 40_379_904, "value_res": 40_379_911}}
for arm, exp in expected.items():
    p = m.init_params(jax.random.PRNGKey(0), m.SIZES["M"], arm)
    got = m.n_params(p)
    assert got == exp, f"{{arm}}: {{got}} != {{exp}}"

# 2. Step-0 identity of resid_x0_skip (alpha=1.0) to dense & strict causal prefix invariance on T-scale
cfg = m.SIZES["T"]
x1 = jnp.asarray(np.random.default_rng(7).integers(0, 256, (2, 16), dtype=np.int32))
x2 = x1.at[:, 9:].set(jnp.asarray(np.random.default_rng(8).integers(0, 256, (2, 7), dtype=np.int32)))

p_d = m.init_params(jax.random.PRNGKey(42), cfg, "dense", vocab=256)
p_x0 = m.init_params(jax.random.PRNGKey(42), cfg, "resid_x0_skip", vocab=256)
lg_d = m.forward_with_diag(p_d, x1, cfg, "dense", jnp.float32)
lg_x0 = m.forward_with_diag(p_x0, x1, cfg, "resid_x0_skip", jnp.float32)
assert float(jnp.max(jnp.abs(lg_d - lg_x0))) < 1e-6

for arm in m.ARMS:
    p = m.init_params(jax.random.PRNGKey(11), cfg, arm, vocab=256)
    o1 = m.forward_with_diag(p, x1, cfg, arm, jnp.float32)
    o2 = m.forward_with_diag(p, x2, cfg, arm, jnp.float32)
    assert float(jnp.max(jnp.abs(o1[:, :9] - o2[:, :9]))) < 1e-5, f"Causal leak in {{arm}}"

# 3. Full end-to-end execution of main() across all 4 arms + ablations + 7-task scorer + strict gate scorer
with tempfile.TemporaryDirectory() as td:
    td_p = Path(td)
    data_d = td_p / "data"
    out_d = td_p / "out"
    data_d.mkdir(); out_d.mkdir()
    rng = np.random.default_rng(5)
    (data_d / "train.bin").write_bytes(rng.integers(0, 32768, 8192, dtype=np.uint16).tobytes())
    manifest = {{"mix": {{"web": 0.62, "textbook": 0.13, "math": 0.13, "code": 0.12}}, "domains": {{}}}}
    for dom in ("web", "textbook", "math", "code"):
        (data_d / f"val_{{dom}}.bin").write_bytes(rng.integers(0, 32768, 4096, dtype=np.uint16).tobytes())
        manifest["domains"][dom] = {{"val": {{"bytes_per_token": 4.0}}}}
    (data_d / "manifest.json").write_text(json.dumps(manifest))
    tasks = {{
        "arc_easy": [{{"ctx": [10, 11, 12], "conts": [[13, 14], [15, 16], [17, 18], [19, 20]], "cont_chars": [5, 5, 5, 5], "gold": 1}}],
        "winogrande": [{{"ctxs": [[21, 22], [23, 24]], "cont": [25, 26], "gold": 0}}],
        "lambada": [{{"ctx": [30, 31, 32], "conts": [[33]], "gold": 0}}],
    }}
    tasks_p = td_p / "tasks.json"
    tasks_p.write_text(json.dumps(tasks))
    for arm in m.ARMS:
        res = m.run_single_job(
            arm=arm, seed=1000, size="T", data_dir=data_d, val_dir=data_d, out_dir=out_d,
            tasks_path=tasks_p, gcs_prefix="", device_idx=0, train_tokens=128,
            batch_size=2, micro=2, seq=32, warmup=1, chunk_steps=2, eval_every=2, eval_batches=1,
            provenance_base={{"val_sha256": {{d: "x"*64 for d in ("web","textbook","math","code")}}}},
        )
        assert res["completed"] is True
        assert res["downstream_7tasks"]["arc_easy"]["n"] == 1
        if arm in ("value_res", "resformer_fixed"):
            assert len(res["diagnostics"]["layer_value_telemetry"]) == 3
            assert "perm_v0" in res["causal_ablations"]
print("JAX_CONTRACT_OK")
"""
        out = subprocess.check_output([str(jax_py), "-c", check_code], text=True)
        self.assertIn("JAX_CONTRACT_OK", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
