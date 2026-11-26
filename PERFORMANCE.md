# Performance, Hardware Scaling & Spend Efficiency (`Cycles 1–2` + `Ω`)

The published X3 peak-memory values predate isolation of each context-length sample. They can include logits retained from the preceding sample or architecture. The corrected runner requires fresh GPU measurements for comparable peaks.

See [NANOLM_COMPUTE_LEDGER.md](NANOLM_COMPUTE_LEDGER.md) for the complete pod-by-pod cost and Roofline calibration table.

---

## 1. Kernel Optimization Impact (Cycle 1 PyTorch $\to$ Cycle 2 OpenAI Triton)

- **Microbenchmark (`C2-X0`, `L40S`):** Replacing naive Blelloch scans with `_triton_holonomy_fwd_kernel` + $O(1)$-tensor conjugate adjoint accelerated `fwd+bwd` **`11.08x` (`256`) $\to$ `23.48x` (`1,024`) $\to$ `35.00x` (`4,096`: `529.04 ms -> 15.12 ms`)** and cut activation VRAM **`2.33x` (`3,126 -> 1,344 MB`)**.
- **End-to-End 14M Training (`X2 -> C2-X2`, `L40S`):** Training speed rose **`3.77x`** (`13,159 -> 49,633 tok/s`), train peak VRAM fell **`43.1%`** (`6,700 -> 3,812 MB`), and decode latency halved **`2.05x`** (`25.01 -> 12.20 ms/tok`).

---

## 2. `3.845B <= 4B` `32,768`-Context Crossover (`L40S` BF16, `C2-X3`)

- At `seq_len = 32,768`:
  - **`nanolm-hspa-4b` (`3,845,419,056`)**: **`9,261.76 MB`** flat peak VRAM, **`1.781 MB`** analytical state, **`74.264 ms/tok`** decode.
  - **`nanolm-transformer-4b` (`3,843,291,648`)**: `13,887.97 MB` peak VRAM (`+4,626.21 MB`), `3,072.0 MB` KV cache (`1,725x` larger), `75.026 ms/tok` decode.
- **Finding:** `HSPA-4B` crosses below `Transformer-4B` in **both memory (`> 2.5K`) and incremental decode latency (`<= 32K`)**, without Flash-KV quantization tricks.
- **Cycle-3 CUDA-Graph + Nullspace Merge Resolution (`EXP-Ω6`, `EXP-Ω7` on `Tesla T4` FP16):**
  - `StaticGraphHSPADecoder` (`seq=512`, `dec=32`): `nanolm-hspa-omega-2b` (`1.808B`, `1.00 MB` state) improved from `158.413 -> 51.023 ms/tok` (**`3.10x`**, `greedy_parity=True`, `4,221.25 MB` VRAM); `nanolm-hspa-omega-4b` (`3.841B <= 4B`, `1.50 MB` state) improved from `236.105 -> 100.291 ms/tok` (**`2.35x`**, `greedy_parity=True`, `8,409.83 MB` VRAM).
  - `SpectralNullspaceLoRALinear.merge()`: `[2048, 2048]` linear GEMM accelerated **`7.55x`** (`252.24 -> 33.42 us`, ortho err `5e-7`), `[2048, 6144]` **`2.66x`** (`267.62 -> 100.47 us`, ortho err `7e-7`), and end-to-end HF `1.747B` decode **`1.61x`** (`8.73 -> 14.09 tok/s`, `6/6` exact match across `196` merged layers).

---

## 3. Intelligence-per-Parameter & VRAM Efficiency (`Qwen3-8B` vs `NanoLM-Ω-1.7B` & `Ω-v2`)

| Metric | External Opponent `Qwen3-8B` (`L40S`) | `NanoLM-Ω-1.7B Naked_Greedy` (`A40`) | `NanoLM-Ω-v2-1.7B Cognitive_v2` (`T4` FP16) |
|---|---:|---:|---:|
| **Parameters** | `8,190,735,360` | **`1,747,444,768`** (`4.69x` smaller) | **`1,747,444,768`** (`4.69x` smaller) |
| **56-Task Suite Accuracy** | `39 / 56` (`69.64%`) | `40 / 56` (`71.43%`) | **`41 / 56` (`73.21%`)** (`+3.57 pp` vs `8B`) |
| **Capability / Billion Params** | `0.0850` | `0.4088` (`4.81x`) | **`0.4190`** (**`4.93x` higher**) |
| **Evaluation Peak VRAM** | `16,541.46 MB` | **`4,402.96 MB`** (`3.76x` less) | `6,501.37 MB` (`2.54x` less) |
| **4K KV Cache Allocation** | `576.0 MB` | **`448.0 MB`** (`-22.2%`) | **`448.0 MB`** (`-22.2%`) |

---

## 4. Cumulative Autonomous Compute Spend (`nanolm/holonomic-discovery-v1` + `frontier`)

| Campaign | GPU & Backend | Wall Time | Realized Cost (USD) |
|---|---|---:|---:|
| `nanolm_frontier_v1` (`E1`, `E2`) | `A40` (`pod 58b79zdtdqcr9n`) | `2,257s` | `$0.3699` |
| `nanolm_holonomy_v1` (`X1..X4`) | `L40S` (`pod 2v2dt0j536rh58`) | `4,302s` | `$1.3026` |
| `nanolm_holonomy_v2` (`C2-X0..X4`) | `L40S` (`pod vboeyg7n3dk7k3`) | `2,319s` | `$0.7021` |
| `nanolm_omega_v1` (`EXP-Ω1..Ω3`) | `A40` (`pod jiwcmt2i7qspab`) | `1,953s` | `$0.3201` |
| `nanolm_omega_v2` (`EXP-Ω4..Ω7`) | Free Kaggle `2x Tesla T4` (`v1+v2`) | `6,602.54s` | **`$0.0000`** |
| `nanolm_fp_v3` Stage A (`EXP-FP1..FP3`) | Free Kaggle `2x Tesla T4` (`v2`) | `8,372.52s` | **`$0.0000`** |
| `nanolm_fp_v3` Stage B (`EXP-FP4`) | Local Apple M3 Max `MPS` (`64 GB`, `FP16`) | `~2,400s` | **`$0.0000`** |
| **Total Attributed Verified Spend** | `4` Completed Pods + `4` Kaggle Runs + Local MPS (`0` active) | `~28,206s` | **`$2.6947` USD** *(+`$0.31` aborted pod-init retries)* |
