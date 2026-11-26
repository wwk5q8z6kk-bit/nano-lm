# NANOLM Ω — Reproducibility & Contamination Verification Report

## 1. Exact Reproduction Commands

All tests and experiments run cleanly from the root of this worktree (`PYTHONPATH=.` ensures isolated module loading):

```bash
# 1. Local unit, mathematical-identity, zero-oracle & CPU smoke suite (24 tests)
PYTHONPATH=. python3 -m pytest -q nanolm_holonomy/ nanolm_posttrain/ nanolm_frontier/

# 2. Pre-flight Roofline compute/VRAM planning & Cycle-2 calibration
PYTHONPATH=. python3 -m nanolm_holonomy.compute_planner --preflight --calibrate-v2 artifacts/nanolm_holonomy_v2

# 3. Repository integrity & protected-evidence gates
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py

# 4. RunPod Cycle-2 suite (C2-X0 Triton vs PyTorch through C2-X4 1.75B vs 8.19B)
./scripts/run_holonomy_c2_runpod.sh

# 5. Free Kaggle T4x2 equivalent launcher ($0.00 USD)
python3 scripts/run_holonomy_c2_kaggle.py --stage-and-push

# 6. RunPod NANOLM Omega suite (EXP-Omega1..EXP-Omega3)
./scripts/run_omega_runpod.sh

# 7. Free Kaggle T4x2 NANOLM Omega-v2 suite ($0.00 USD: EXP-Omega4..EXP-Omega7)
python3 scripts/run_omega_v2_kaggle.py --stage-and-push --run-mode all
python3 scripts/run_omega_v2_kaggle.py --stage-and-push --run-mode o7

# 8. Cycle 4 (NANOLM Omega-v3 First-Principles: EXP-FP1..EXP-FP4)
python3 scripts/run_fp_v3_kaggle.py --stage-and-push --run-mode fp23
PYTHONPATH=. python3 -m nanolm_holonomy.cycle4_first_principles --mode fp4 --output-dir artifacts/nanolm_fp_v3
```

---

## 2. Cryptographic Checksums, Seeds & Decontamination Audits

| Verification Axis | Measured Status | Artifacts / Witnesses |
|---|---|---|
| **Curriculum Decontamination (`C2` & `Ω` / `Ω-v2` / `Ω-v3`)** | **`overlaps = 0`** across all `62` `curriculum_v2` SFT + `24` `rl_cur` (`86` total) vs `56` `benchmark_15d` & `30` `FP-5Axis-30` tasks | Exact substring/13-gram check in `verify_zero_contamination` & `verify_fp_zero_contamination` (`exp_fp4_1p7b_integration.json`) |
| **Strict Zero-Oracle Cognitive & World Controllers (`v1..v3`)** | Verified programmatically via `inspect.signature(omega_generate)` & `inspect.signature(omega_fp_generate)` | `omega_generate` & `omega_fp_generate` receive only prompts/budgets; **`36/36`** local & Kaggle CUDA pytest pass |
| **Double-Precision & CUDA Triton Backward** | `torch.autograd.gradcheck` (`float64`, `atol=1e-5`) + CUDA forward/backward `rel_err < 1e-4` (`L=31, 300`) | Passed in `tests/test_holonomy.py`, `tests/test_omega.py`, `tests/test_first_principles.py` (`36/36` CUDA pytest) |
| **Step-0 KL / Logit Preservation** | `step0_max_abs_logit_diff == 0.0` | Verified pre/post `apply_omega_graft` on live `Qwen/Qwen3-1.7B` |
| **Zero-Overhead Nullspace Weight Merge (`EXP-Ω7`, `EXP-FP4`)** | `merged_layers = 196`, `exact_greedy_output_matches = 6/6`, `1.61x` speedup | `[2048, 2048]` ortho err `5e-7`, `[2048, 6144]` ortho err `7e-7` (`exp_omega6_cudagraph_merge.json`, `exp_fp4_1p7b_integration.json`) |
| **Trained `NanoLM` 1.7B Adapter Manifests (`v1` / `v2` / `fp-v3`)** | `total_parameters = 1,747,444,768` (`<= 4B`), `trainable = 26,869,792` (`1.5377%`), `fp-v3` `l2_norm = 156.8447` | `v1` `75458a98c76cb65e`, `v2` `7629198766e33997`, `fp-v3` `66f2ce64e47ff701` across `456` tensors ([nanolm_fp_v3_1.7b_manifest.json](artifacts/nanolm_fp_v3/nanolm_fp_v3_1.7b_manifest.json)) |
| **Canonical Freeze Integrity** | Untouched `papers/`, `trajectory/`, `freeze/`, `POST_ALPHA_EVIDENCE_FREEZE.*` | `check_active_now.py` -> `ACTIVE_NOW_OK`; `check_docs_integrity.py` -> `DOCS_INTEGRITY_OK` |
