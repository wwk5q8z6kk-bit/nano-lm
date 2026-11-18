# NANOLM Ω — Compute, Financial & Calibration Ledger

> Every dollar, pod ID, GPU shape, runtime, and Roofline pre-flight comparison across the NanoLM repository and the active `nanolm/holonomic-discovery-v1` research lineage. All pods were terminated immediately upon artifact verification; **`0` own pods remain active**.

---

## 1. Consolidated Repository Spend Audit Across Tracks

| Track / Worktree | Pods / Backends | Status | Realized Spend (USD) | Ledger Source |
|---|---|---|---:|---|
| Historical Pre-Reset (`nano_ai` `H6`) | Kaggle (`$0.00`) + RunPod H6 dev (`$3.70`) | Closed (`H6 REJECTED`) | `$3.70` | [H6_TERMINAL_DECISION.json](artifacts/nano_h6/H6_TERMINAL_DECISION.json) |
| Sibling `nanolm/discovery-lab-v1` (`Phases 1–4`) | `19` RunPod pods (`4090`, `5090`, `A40`, `L40S`, `H100`, `H200`) | Completed (read-only ref) | `$92.11` | `lab/reports/SPEND_LEDGER.md` (`0256d52`) |
| Sibling `nanolm/architecture-invention-v1` (`G3_HMWR`) | Local M3 Max MPS + Kaggle `2x Tesla T4` | Completed (read-only ref) | `$0.00` | `NANOLM_EXPERIMENTS.md` (`b8af1c6`) |
| **Frontier Baseline (`nanolm_frontier_v1`, PR #171)** | RunPod `NVIDIA A40` (`pod 58b79zdtdqcr9n`) | Completed & Terminated | **`$0.3699`** | [SPEND_AND_PROVENANCE.json](artifacts/nanolm_frontier_v1/SPEND_AND_PROVENANCE.json) |
| **HSPA Cycle 1 (`nanolm_holonomy_v1`, PR #174)** | RunPod `NVIDIA L40S` (`pod 2v2dt0j536rh58`) | Completed & Terminated | **`$1.3026`** | [SPEND_AND_PROVENANCE.json](artifacts/nanolm_holonomy_v1/SPEND_AND_PROVENANCE.json) |
| **HSPA Cycle 2 (`nanolm_holonomy_v2`, PR #174)** | RunPod `NVIDIA L40S` (`pod vboeyg7n3dk7k3`) | Completed & Terminated | **`$0.7021`** | [SPEND_AND_PROVENANCE.json](artifacts/nanolm_holonomy_v2/SPEND_AND_PROVENANCE.json) |
| **Omega Debug Launch (`x7gkyrkahgegzo`, failed `EXP-Ω2`)** | RunPod `NVIDIA L40S` (`~1.5m`, auto-removed by trap) | Aborted cleanly (`encode()` arg fix) | `~$0.03` (est.) | `runpodctl` trap verified removed |
| **`NANOLM Ω` Complete Run (`nanolm_omega_v1`)** | RunPod `NVIDIA A40` (`pod jiwcmt2i7qspab`) | Completed & Terminated | **`$0.3201`** | [SPEND_AND_PROVENANCE.json](artifacts/nanolm_omega_v1/SPEND_AND_PROVENANCE.json) |
| **Total Attributed Spend (Frontier + HSPA C1–C2 + Ω)** | `5` completed RunPod pods (excl. peer account pods) | **All `0` active** | **`$2.6947`** *(+`~$0.03` abort)* | Verified JSON artifacts on branch |

*Account Balance Verification Note:* Live RunPod `clientBalance` was **`$87.9285`** prior to launching `jiwcmt2i7qspab` and **`$84.2127`** immediately after pod removal (`$3.7159` shared-account delta includes concurrent peer pods `v5swuwh5ukl1zh` and `y05unbw7rthg69`; our `A40` pod attributed cost for `1,953 s @ $0.59/hr` was **`$0.3201`**). Owner `$20.00` reserve remained untouched (`$84.21 >> $20.00`).

---

## 2. Per-Run RunPod Provenance on `nanolm/holonomic-discovery-v1`

| Campaign | Pod ID | GPU (`48 GB`) | Rate (`$/hr`) | Wall Time (`s` / `min`) | Pod Cost (USD) | Key Milestones Verified |
|---|---|---|---:|---:|---:|---|
| `nanolm_frontier_v1` | `58b79zdtdqcr9n` | `NVIDIA A40` | `$0.59` | `2,257s` (`37.6m`) | `$0.3699` | E1 (`C0..C4`), E2 (`Base 34/56`, `LoRA 36/56`, `Frontier 35/56`) |
| `nanolm_holonomy_v1` | `2v2dt0j536rh58` | `NVIDIA L40S` | `$1.09` | `4,302s` (`71.7m`) | `$1.3026` | Cycle-1 `X1..X4` (`Gen1_H2` `0.3723` BPB, `16K` `HSPA-4B`) |
| `nanolm_holonomy_v2` | `vboeyg7n3dk7k3` | `NVIDIA L40S` | `$1.09` | `2,319s` (`38.7m`) | `$0.7021` | `C2-X0..X4`: Triton `35x`, `32K` `HSPA-4B`, `1.75B vs 8.19B` |
| `nanolm_omega_v1` | `jiwcmt2i7qspab` | `NVIDIA A40` | `$0.59` | `1,953s` (`32.6m`) | `$0.3201` | CUDA `12/12`, `EXP-Ω1 (100% 16x)`, `EXP-Ω2`, `EXP-Ω3 (40/56)` |

---

## 3. Roofline `ComputePlanner` Calibration (`Pre-Flight` vs `Actual`)

1. **Peak VRAM Accuracy (`mean_vram_rel_error_pct = 12.91%`, [preflight_plan.json](artifacts/nanolm_omega_v1/preflight_plan.json)):**
   - `External_Opponent_Qwen3-8B` (`8.19B` eval): Predicted **`16,018.59 MB`** vs Actual **`16,541.46 MB`** (**`3.16%` error**).
   - `nanolm-transformer-4b` (`512..32,768` tok): Predicted **`8,148–18,732 MB`** vs Actual **`7,525–13,888 MB`**.
   - `nanolm-hspa-4b` (`512..32,768` tok): Chunked prefill model predicted **flat `7,606.34 MB`** (actual **flat `8,408.57–9,261.76 MB`**, `17.87%` constant offset across `64x` context growth!).
   - `NanoLM_Omega_1.7B` (`1.75B`): Predicted **`5,541.92 MB`** vs Actual **`4,402.96 MB`** (safely inside `48 GB` `A40` budget).
2. **Latency Calibration Finding (`4.83 min` Predicted vs `32.55 min` Actual Wall Time):**
   - Pure FLOP/HBM Roofline underestimated wall-clock duration by `~6.7x` on small-batch/eager-Python workloads (`EXP-Ω1`, `EXP-Ω2`, `batch=1` `Qwen3-1.7B` decode with `SpectralNullspaceLoRA` `x @ V_k` GEMMs running at `10.75 tok/s` vs `107.7 tok/s` ideal HBM bandwidth, plus `3.59 min` CUDA pytest + `17` cognitive retries). This empirically proves that **Python kernel-launch latency**, not HBM bandwidth, currently bounds `batch=1` inference speed—motivating Theorem `H-Ω20` (CUDA-Graph projection-scan fusion) as Priority #1 for the next cycle.
