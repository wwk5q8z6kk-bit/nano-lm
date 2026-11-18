# Performance, Hardware Scaling & Spend Efficiency (`Cycles 1–2` + `Ω`)

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
- **Finding:** `HSPA-4B` crosses below `Transformer-4B` in **both memory (`> 2.5K`) and incremental decode latency (`<= 32K`)**, without Flash-KV quantization tricks. Remaining fixed latency (`~74 ms/tok`) stems from eager Python per-layer `hamilton_product` launch overhead (`batch=1`), targeted next via CUDA Graphs.

---

## 3. Intelligence-per-Parameter & VRAM Efficiency (`Qwen3-8B` vs `NanoLM-Ω-1.7B`)

| Metric | External Opponent `Qwen3-8B` (`L40S`) | `NanoLM-Ω-1.7B Naked_Greedy` (`A40`) | Ratio (`Ω` vs `8B`) |
|---|---:|---:|---|
| **Parameters** | `8,190,735,360` | **`1,747,444,768`** (`<= 4B`) | **`4.69x` smaller** |
| **56-Task Suite Accuracy** | `39 / 56` (`69.64%`) | **`40 / 56` (`71.43%`)** | **`+1.79 pp` higher** |
| **Capability / Billion Params** | `0.0850` | **`0.4088`** | **`4.81x` higher** |
| **Evaluation Peak VRAM** | `16,541.46 MB` | **`4,402.96 MB`** | **`3.76x` lower** |
| **4K KV Cache Allocation** | `576.0 MB` | **`448.0 MB`** | `-22.2%` |

---

## 4. Cumulative Autonomous Compute Spend (`nanolm/holonomic-discovery-v1` + `frontier`)

| Campaign | GPU & Pod ID | Wall Time | Realized Cost (USD) |
|---|---|---:|---:|
| `nanolm_frontier_v1` (`E1`, `E2`) | `A40` (`58b79zdtdqcr9n`) | `2,257s` | `$0.3699` |
| `nanolm_holonomy_v1` (`X1..X4`) | `L40S` (`2v2dt0j536rh58`) | `4,302s` | `$1.3026` |
| `nanolm_holonomy_v2` (`C2-X0..X4`) | `L40S` (`vboeyg7n3dk7k3`) | `2,319s` | `$0.7021` |
| `nanolm_omega_v1` (`EXP-Ω1..Ω3`) | `A40` (`jiwcmt2i7qspab`) | `1,953s` | `$0.3201` |
| **Total Attributed Verified Spend** | `4` Completed Pods (`0` active) | `10,831s` (`3.01h`) | **`$2.6947` USD** *(+`~$0.03` aborted launch)* |
