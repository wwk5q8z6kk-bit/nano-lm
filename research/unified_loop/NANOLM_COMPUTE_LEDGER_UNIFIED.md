# Nano LM — Unified Research Loop Compute Ledger

- **Worktree / Branch:** `/Users/mac/Projects/nano-lm/.worktrees/nanolm-unified-loop` (`research/unified-discovery-loop-v1`)
- **Session Paid Cloud Compute Spend:** **`$0.00`**
- **Hardware Used:** Local Apple Silicon M3 Max (`cpu` single-threaded workers + `mps`)
- **Dataset:** Local disk `/Volumes/Express1M2/nanolm-lab/data/v1/` and cached local weights (`S0_125m`, `S1_125m`, `X4_46m`, `Qwen2.5-1.5B-Instruct`)
- **Peer Cloud Resources:** Four active peer RunPod pods (`4srqwah0chgn82`, `7lgtpru2bcyczr`, `y2w6enx8kkw2fx`, `ydlvfq3siahi4d`) and Kaggle history recorded in `PARALLEL_PROCESS_AND_COMPUTE_REGISTRY.json` were **left strictly untouched**.

| Experiment | Backend | Wall-Clock Runtime | Paid Cost (USD) | Verified Artifact |
|---|---|---:|---:|---|
| `EXP-U01` (`125M`/`46M` Autopsy) | Local CPU float32 | 4,303.35s | `$0.00` | `experiments/U01_real_125m_autopsy/results.json` |
| `EXP-U03` (`E23` & `E24` Audit) | Local CPU | 1,315.53s | `$0.00` | `experiments/U03_scaffold_routing_audit/results.json` |
| `EXP-U04` (Verifier + Qwen `C14`) | Local CPU + MPS | 200.61s | `$0.00` | `experiments/U04_nanoscribe_verifier_qwen/results.json` |
| `EXP-U05` (4-Domain Modular Eval) | Local CPU | 11.10s | `$0.00` | `experiments/U05_cross_domain_hierarchy_denom/results.json` |
| `EXP-U02 Part A` (`10×5×3` Micro-Lab) | Local CPU (5 workers) | 426.70s | `$0.00` | `experiments/U02_cross_paradigm_showdown/results_part_a.json` |
| `EXP-U02 Part B` (`data/v1`, `5×1000×2`) | Local MPS + 4 CPU workers | 819.88s | `$0.00` | `experiments/U02_cross_paradigm_showdown/results_part_b.json` |
| **Total Loop** | **Local Only** | **7,077.17s** | **`$0.00`** | **6 Complete Experimental Suites** |
