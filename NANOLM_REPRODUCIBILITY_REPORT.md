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
```

---

## 2. Cryptographic Checksums, Seeds & Decontamination Audits

| Verification Axis | Measured Status | Artifacts / Witnesses |
|---|---|---|
| **Curriculum Decontamination (`C2` & `Ω`)** | **`overlaps = 0`** across all `62` `curriculum_v2` SFT examples & `24` `rl_cur` prompts vs all `56` held-out `benchmark_15d` tasks | Exact substring/13-gram check in `verify_zero_contamination` (`c2_x4_giant_killer.json`, `exp_omega3_unified_1p7b.json`) |
| **Strict Zero-Oracle Cognitive Controller** | Verified programmatically via `inspect.signature(omega_generate)` | `omega_generate` receives strictly `(model, tokenizer, system_prompt, user_prompt, max_new_tokens, device)`; `test_omega_cognitive_controller_zero_oracle_contract` passes |
| **Double-Precision & CUDA Triton Backward** | `torch.autograd.gradcheck` (`float64`, `atol=1e-5`) + CUDA forward/backward `rel_err < 1e-4` (`L=31, 300`) | Passed in `tests/test_holonomy.py` on both `NVIDIA L40S` & `NVIDIA A40` (`12/12` CUDA tests pass) |
| **Step-0 KL / Logit Preservation** | `step0_max_abs_logit_diff == 0.0` | Verified pre/post `apply_omega_graft` on live `Qwen/Qwen3-1.7B` |
| **Trained `NanoLM-Ω-1.7B` Adapter Manifest** | `total_parameters = 1,747,444,768` (`<= 4B`), `trainable = 26,869,792` (`1.5377%`), `l2_norm = 156.8438` | `adapter_sha256_prefix = 75458a98c76cb65e` across `456` tensors ([nanolm_omega_1.7b_manifest.json](artifacts/nanolm_omega_v1/nanolm_omega_1.7b_manifest.json)) |
| **Canonical Freeze Integrity** | Untouched `papers/`, `trajectory/`, `freeze/`, `POST_ALPHA_EVIDENCE_FREEZE.*` | `check_active_now.py` -> `ACTIVE_NOW_OK`; `check_docs_integrity.py` -> `DOCS_INTEGRITY_OK` |
