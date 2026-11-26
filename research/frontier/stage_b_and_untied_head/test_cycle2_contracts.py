"""End-to-end contract, smoke, causality, parameter-count, and strict scorer tests for Cycle 2:
  - Part I: Stage B (`stage_b_runner.py` in `STAGEB_SMOKE=1` + `score_stage_b.py` + receipt extractor)
  - Part II: Untied-Head x ValueRes x 57M Tied Factorial (`untied_head_jax.py` + `score_untied_head.py`)
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG_DIR))

from run_stage_b_controller import REQUIRED_JOBS, extract_receipts
from score_stage_b import score_stage_b_bundle
from score_untied_head import score_untied_runs


class Cycle2ContractTests(unittest.TestCase):
    def test_stage_b_smoke_runner_and_strict_scorer(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            out_d = td_p / "out_sb"
            data_d = td_p / "data_sb"
            extract_d = td_p / "extracted"
            extract_d.mkdir()

            env = os.environ.copy()
            env["STAGEB_SMOKE"] = "1"
            env["STAGEB_OUT"] = str(out_d)
            env["STAGEB_DATA"] = str(data_d)
            proc = subprocess.run(
                [sys.executable, str(PKG_DIR / "stage_b_runner.py")],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(proc.returncode, 0, msg=f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout[-2000:]}")
            self.assertIn("STAGEB_ALL_DONE", proc.stdout)

            # Verify controller log extraction parses all 10 receipts cleanly
            captured: set[str] = set()
            extract_receipts(proc.stdout, extract_d, captured, lambda _m: None)
            self.assertEqual(captured, set(REQUIRED_JOBS))

            bundle = {f"stageb_{jn}": json.loads((extract_d / f"stageb_{jn}.json").read_text()) for jn in REQUIRED_JOBS}
            scored = score_stage_b_bundle(bundle)
            self.assertEqual(scored["student_tokens_verified"], 256)
            self.assertIn(scored["verdicts"]["B1"], ("PASS", "FAIL"))
            self.assertIn(scored["verdicts"]["B2"], ("PASS", "FAIL"))

            # Verify mismatched student-token budget is strictly rejected
            bad = json.loads(json.dumps(bundle))
            bad["stageb_C2_s1000"]["recipe"]["total_student_tokens"] = 999
            with self.assertRaisesRegex(ValueError, "total_student_tokens"):
                score_stage_b_bundle(bad)

    def test_untied_scorer_decision_rules_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            paths = []
            val_hashes = {d: "a" * 64 for d in ("web", "textbook", "math", "code")}
            for idx, seed in enumerate((1000, 1001, 1002)):
                eps = idx * 0.0001
                specs = {
                    "dense_tied": (40_379_904, 1.0163 + eps, 3050.0, {}, {}),
                    "dense_untied": (57_157_120, 1.0035 + eps, 3110.0, {"mean_cos_emb_vs_lm_head": 0.12}, {}),
                    "dense_untied_vr": (
                        57_157_127,
                        1.0010 + eps,
                        3130.0,
                        {"mean_cos_emb_vs_lm_head": 0.11, "lambdas": [0.95, 0.96, 0.97, 0.96, 0.90, 0.72, 0.34]},
                        {"no_skip": 0.02, "ablate_late_6_7": 0.015, "ablate_mid_3_4": 0.002, "perm_v0": 0.25},
                    ),
                    "dense_tied_57m": (57_355_904, 0.9960 + eps, 4520.0, {}, {}),
                }
                for arm, (n_p, final_mix, sec, diag, abls) in specs.items():
                    rec = {
                        "schema": "nanolm_untied_head_57m_factorial_v1",
                        "completed": True,
                        "arm": arm,
                        "size": "M",
                        "seed": seed,
                        "params": n_p,
                        "non_embed_params": 23_602_176 if arm != "dense_tied_57m" else 36_383_744,
                        "recipe": {"steps": 24400, "train_tokens": 799539200},
                        "provenance": {"code_sha256": "b" * 64, "val_sha256": val_hashes},
                        "curve": [
                            {"step": 12000, "tokens": 393216000, "train_sec": sec * 0.5, "tokens_per_second": 250000, "web": final_mix + 0.05, "textbook": final_mix - 0.15, "math": final_mix + 0.1, "code": final_mix - 0.18, "mix": final_mix + 0.025},
                            {"step": 24400, "tokens": 799539200, "train_sec": sec, "tokens_per_second": 250000, "web": final_mix + 0.04, "textbook": final_mix - 0.16, "math": final_mix + 0.09, "code": final_mix - 0.19, "mix": final_mix},
                        ],
                        "diagnostics": diag,
                        "causal_ablations": {k: {"mix": final_mix + v, "delta_mix_vs_intact": v} for k, v in abls.items()},
                        "downstream_7tasks": {"arc_easy": {"acc": 0.43, "acc_norm": 0.42}, "lambada": {"acc": 0.26, "ppl": 130.0}},
                    }
                    p = td_p / f"run_{arm}_M_s{seed}.json"
                    p.write_text(json.dumps(rec))
                    paths.append(p)

            res = score_untied_runs(paths, expected_steps=24400, expected_tokens=799539200)
            self.assertEqual(res["verdicts"]["U_G1_untied_head_gate"], "ADOPT_UNTIED_HEAD_FOR_SUB4B_TRUNK")
            self.assertEqual(res["verdicts"]["U_G2_valueres_interaction"], "CONFIRMED_TIED_HEAD_SUBADDITIVITY")

    def test_jax_untied_head_params_causality_and_smoke(self):
        jax_env = os.environ.get("NANOLM_JAX_PYTHON", "")
        jax_py = Path(jax_env) if (jax_env and Path(jax_env).exists()) else Path(sys.executable)
        probe = subprocess.run([str(jax_py), "-c", "import jax, optax"], capture_output=True)
        if probe.returncode != 0:
            self.skipTest("JAX/Optax interpreter not available")

        check_code = f"""
import sys, json, tempfile
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "{PKG_DIR}")
import untied_head_jax as m
from score_untied_head import score_untied_runs

# 1. Exact pre-registered parameter counts on M / M_WIDE
expected = {{
    "dense_tied": 40_379_904,
    "dense_untied": 57_157_120,
    "dense_untied_vr": 57_157_127,
    "dense_tied_57m": 57_355_904,
}}
for arm, exp in expected.items():
    p = m.init_params(jax.random.PRNGKey(1000), m.resolve_cfg("M", arm), arm)
    got = m.n_params(p)
    assert got == exp, f"{{arm}}: {{got}} != {{exp}}"

# 2. Exact initial emb & blocks[0] equality between dense_tied, dense_untied, dense_untied_vr
p_t = m.init_params(jax.random.PRNGKey(1000), m.resolve_cfg("T", "dense_tied"), "dense_tied", vocab=256)
p_u = m.init_params(jax.random.PRNGKey(1000), m.resolve_cfg("T", "dense_untied"), "dense_untied", vocab=256)
p_uv = m.init_params(jax.random.PRNGKey(1000), m.resolve_cfg("T", "dense_untied_vr"), "dense_untied_vr", vocab=256)
assert float(jnp.max(jnp.abs(p_t["emb"] - p_u["emb"]))) == 0.0
assert float(jnp.max(jnp.abs(p_t["blocks"][0]["q"] - p_u["blocks"][0]["q"]))) == 0.0
assert float(jnp.max(jnp.abs(p_u["lm_head"] - p_uv["lm_head"]))) == 0.0

# 3. Strict causal prefix invariance across all 4 arms
x1 = jnp.asarray(np.random.default_rng(7).integers(0, 256, (2, 16), dtype=np.int32))
x2 = x1.at[:, 9:].set(jnp.asarray(np.random.default_rng(8).integers(0, 256, (2, 7), dtype=np.int32)))
for arm in m.ARMS:
    cfg = m.resolve_cfg("T", arm)
    p = m.init_params(jax.random.PRNGKey(11), cfg, arm, vocab=256)
    o1 = m.forward_with_diag(p, x1, cfg, arm, jnp.float32)
    o2 = m.forward_with_diag(p, x2, cfg, arm, jnp.float32)
    assert float(jnp.max(jnp.abs(o1[:, :9] - o2[:, :9]))) < 1e-5, f"Causal leak in {{arm}}"

# 4. End-to-end execution across all 4 arms x 3 seeds on T-scale + strict scorer validation
with tempfile.TemporaryDirectory() as td:
    td_p = Path(td)
    data_d = td_p / "data"; out_d = td_p / "out"
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
    paths = []
    for sd in (1000, 1001, 1002):
        for arm in m.ARMS:
            m.run_single_job(
                arm=arm, seed=sd, size="T", data_dir=data_d, val_dir=data_d, out_dir=out_d,
                tasks_path=tasks_p, gcs_prefix="", device_idx=0, train_tokens=128,
                batch_size=2, micro=2, seq=32, warmup=1, chunk_steps=2, eval_every=2, eval_batches=1,
                provenance_base={{"val_sha256": {{d: "x"*64 for d in ("web","textbook","math","code")}}}},
            )
            paths.append(out_d / f"run_{{arm}}_T_s{{sd}}.json")
    summary = score_untied_runs(paths, expected_steps=2, expected_tokens=128)
    assert "U_G1_untied_head_gate" in summary["verdicts"]
print("UNTIED_JAX_CONTRACT_OK")
"""
        out = subprocess.check_output([str(jax_py), "-c", check_code], text=True)
        self.assertIn("UNTIED_JAX_CONTRACT_OK", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
