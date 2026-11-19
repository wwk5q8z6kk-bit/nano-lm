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
| `EXP-U06 Part A` (`6×5×3` Micro-Lab) | Local CPU (5 workers) | 329.52s | `$0.00` | `experiments/U06_spinor_pointer_synthesis/results_part_a.json` |
| `EXP-U06 Part B` (`data/v1`, `6×1000×2`, `1×/2×/4×`) | Local MPS | 1,960.85s | `$0.00` | `experiments/U06_spinor_pointer_synthesis/results_part_b.json` |
| `EXP-U07 Part A` (`5×5×3` Micro-Lab) | Local CPU (5 workers) | 221.85s | `$0.00` | `experiments/U07_first_principles_enhancements/results_part_a.json` |
| `EXP-U07 Part A2` (`5×2×2`, 600-step Algebra) | Local CPU (5 workers) | 138.53s | `$0.00` | `experiments/U07_first_principles_enhancements/results_part_a2_algebra.json` |
| `EXP-U07 Part B` (`data/v1`, `5×1000×2`, `1×/2×/4×`) | Local MPS | 1,076.53s | `$0.00` | `experiments/U07_first_principles_enhancements/results_part_b.json` |
| `EXP-U08 Part A` (`5×6×3` Micro-Lab + Alias) | Local CPU (5 workers) | 337.16s | `$0.00` | `experiments/U08_unified_speed_aux_hybrid/results_part_a.json` |
| `EXP-U08 Part A2` (`5×3×2`, 600-step Algebra+Alias) | Local CPU (5 workers) | 262.76s | `$0.00` | `experiments/U08_unified_speed_aux_hybrid/results_part_a2_algebra.json` |
| `EXP-U08 Part B` (`data/v1`, `5×1000×2`, `1×/2×/4×` + Rank Audit) | Local MPS | 1,201.01s | `$0.00` | `experiments/U08_unified_speed_aux_hybrid/results_part_b.json` |
| **Total Loop** | **Local Only** | **12,605.38s** | **`$0.00`** | **9 Complete Experimental Suites** |
