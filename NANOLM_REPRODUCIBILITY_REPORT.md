# NANOLM Ω — Reproducibility, Integrity & Verification Report

Every codebase claim, model census check, multi-seed experiment, JSON output, and `.pt` checkpoint on branch `nanolm/architecture-invention-v1` is independently reproducible from committed source files at `$0.00` cloud spend.

---

## 1. Environment & Interpreter Provenance

- **Branch / Worktree Isolation:** Developed strictly inside `.worktrees/nanolm-discovery` (`nanolm/architecture-invention-v1`); zero edits to canonical `origin/master`, root workspace, `PROTECTED_EVIDENCE_PREFIXES`, `docs/ACTIVE_NOW*`, sibling `.worktrees/`, or live RunPod pods.
- **Local Apple Silicon Host:** `macOS-26.7.3-arm64`, Apple M3 Max (`64 GB` UMA), `Python 3.14` (`python3.14`), `PyTorch 2.11.0` (`mps` + `cpu`), `OMP_NUM_THREADS=2` recommended for deterministic CPU timing without 16-thread OpenMP contention.
- **Kaggle Free GPU Host:** `Linux-6.18.55+-x86_64`, `PyTorch 2.11.0+cu130` (`CUDA 13.0`), `2x NVIDIA Tesla T4` (`14.56 GiB`/GPU), dataset `hassaneljesr/nanolm-discovery-corpus-v1` (`train_sha256 = 867beb0d8d4c3977acaaf97fb91da39be29fb9fb040d86ea0bfdf78c9de72ac4`). Running native `float32` on Turing `sm_75` avoids `sm_75` `float16` kernel NaNs (`F-Ω05`).

---

## 2. Cryptographic Verification (`SHA256SUMS` & Preserved Checkpoints)

### 2.1 Committed JSON Manifests
1. `artifacts/nanolm_discovery_v1/SHA256SUMS`:
   - `3f878c079643a7a54b6afd8b6e3479ee3e09b2ee4c9b55e79b55775601b31104  discovery_results.json`
   - `47a007a702fe157a7e538385fbc1612e12f0df4b43edaae39afc0cad93011669  kaggle_cuda_results.json`
   - `442aa811c34ee037c3784ef3fb27fade09b69c3f0d040e01936ddf725a5deeab  lineage_registry.json`
2. `artifacts/nanolm_omega_v1/SHA256SUMS`:
   - `2b0bff99b1014c30728a252eccc9d1052fbf83cb8f0aed9f4d7e940c6a5ffaa9  checkpoint_manifest.json`
   - `d3259761b1bc03133f34f1d2db0babe148e6ac8f5c9839e8abb32d6322c48458  omega_kaggle_cuda_results.json`
   - `bc458993a436da46c6ae55100624bae5bd387ff7c6e187c1d749677eac458253  omega_local_results.json`

### 2.2 Preserved `.pt` Checkpoints (Gitignored `artifacts/nanolm_omega_v1/checkpoints/`)
- `g4_omega_s42.pt` (`82,336,525` bytes) — `sha256: 90f1155347f9d4daf6cc1cc851a69bd70cca5537723b20368a9676adeeeaca47` *(Kaggle Dual-T4 `1,000`-step `20.57M` checkpoint)*
- `g4_omega_mech_s101.pt` (`485,189` bytes) — `sha256: b6713c1a710de7b4e946eff6ebb5b7cbbab1962c9b04cc7b3a19e9cca9700f2c` *(Local `115k` calibrated mechanistic checkpoint)*
- `omega_efe_planner_s0.pt` (`1,007,943` bytes) — `sha256: 78a900d44847bcb0d15bb64bc5249626dafdcc86bc10aace9881c64418351c4c` *(Local `248,328`-param `OmegaEFEPlanner` checkpoint)*

---

## 3. Deterministic Reproduction Commands

```bash
# Canonical workspace & CI invariants (synthetic-only unit tests)
python3.14 scripts/check_active_now.py
python3.14 scripts/check_docs_integrity.py
python3.14 fabric/test_fabric.py && python3.14 trajectory/test_recompute_c3.py
OMP_NUM_THREADS=2 python3.14 -m pytest -ra nanolm_discovery/tests/test_discovery.py nanolm_omega/tests/test_omega.py
OMP_NUM_THREADS=2 python3.14 -m pytest -rs nano nanoscribe nano_ai/tests benchmarks/

# Verify artifact SHA-256 digests
(cd artifacts/nanolm_discovery_v1 && shasum -a 256 -c SHA256SUMS)
(cd artifacts/nanolm_omega_v1 && shasum -a 256 -c SHA256SUMS)

# Reproduce Omega local suite (OMEGA-EXP-01-MECH, EXP-02, EXP-03) and Kaggle CUDA tournament
OMP_NUM_THREADS=2 python3.14 scripts/run_nanolm_omega_suite.py
python3.14 scripts/kaggle_omega_backbone_experiment.py --data-dir <CORPUS_DIR> --steps 1000 --seeds 42 137
```

---

## 4. Explicit Scope Boundaries
- All CI unit tests (`nanolm_discovery/tests/test_discovery.py`, `nanolm_omega/tests/test_omega.py`) run strictly on tiny synthetic tensors (`< 2s` per test) with zero dataset/checkpoint dependencies.
- `4.10M`-token (`1,000`-step) runs test early/intermediate crossover, not `50B+`-token Chinchilla pretraining; `OMEGA-EXP-02/03` measure synthetic causal diagnosis and structured 10-capability integration, not external clinical dual-IAA validation.
