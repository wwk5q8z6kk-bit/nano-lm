# Performance & Efficiency Benchmarks

All performance metrics reported below are directly measured on **RunPod `NVIDIA L40S` 48GB (`CUDA 13.2`, BF16)** across [x2_lm_evolution.json](artifacts/nanolm_holonomy_v1/x2_lm_evolution.json), [x3_4b_scaling.json](artifacts/nanolm_holonomy_v1/x3_4b_scaling.json), and [x4_foundation_holonomy.json](artifacts/nanolm_holonomy_v1/x4_foundation_holonomy.json).

---

## 1. Intelligence-per-Parameter & Throughput Tradeoffs (`~14.0M` Scale, X2)

| Architecture | Params | Mean Val BPB | State Probe Acc | Train Throughput (`tok/s`) | Train Peak VRAM (MB) | Decode (`ms/tok`) | State Memory @ 8K |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b0-transformer-micro` | 13,966,464 | `0.3794` | 1.04% | **107,437.9** | **1,189.9** | **8.936** | 24,576.0 KB |
| `b1-abelian-ssm-micro` | 14,010,240 | `0.3826` | 1.04% | 20,359.1 | 6,555.1 | 13.805 | **24.0 KB** |
| `gen1-h1-hspa-spinor-micro` | 14,019,456 | `0.3800` | 1.04% | 13,803.3 | 6,312.5 | 19.077 | **24.0 KB** |
| **`gen1-h2-hspa-predcode-micro` (`BEST`)** | 14,022,536 | **`0.3723`** | **13.54%** | 13,158.8 | 6,700.3 | 25.008 | **24.0 KB** |
| `gen2-h3-hspa-evolved-micro` | 14,027,648 | `0.3803` | 2.08% | 13,188.0 | 6,727.0 | 21.373 | 33.0 KB |

### Key Takeaways
- **`Gen1_H2_HSPA_PredCode` is Pareto-superior in bits-per-byte (`-0.0071 BPB`), `state` reasoning (`13.54%` vs `1.04%`), and inference state compression (`1,024x`) at fixed parameter count.**
- **Pure-PyTorch bottleneck:** Without a fused Triton/CUDA `hamilton_product` Hillis-Steele kernel, `HSPA` spends most wall-clock time launching elementwise scan operations, making eager training `8.2x` slower and using `~5.6x` higher activation memory during backward passes.

---

## 2. 4B-Class (`~3.84B`) Context Memory & Latency Scaling (`L40S` BF16, X3)

| Context Length `T` | `TRANSFORMER_4B` Prefill (`tok/s`) | `HSPA_4B` Prefill (`tok/s`) | `TRANSFORMER_4B` Peak VRAM | `HSPA_4B` Peak VRAM | VRAM Delta (`HSPA - Trans`) | Analytical State (`HSPA` vs `Trans`) |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 8,006.2 | 834.2 | **7,622.35 MB** | 10,877.26 MB (unchunked) | +3,254.9 MB | `1.781 MB` vs `48.0 MB` (`27x`) |
| 2,048 | 17,972.6 | 779.4 | **8,220.44 MB** | 8,308.06 MB (chunk 1024) | +87.6 MB | `1.781 MB` vs `192.0 MB` (`108x`) |
| 8,192 | 16,772.6 | 751.3 | 10,607.49 MB | **8,317.38 MB** | **-2,290.1 MB** | `1.781 MB` vs `768.0 MB` (`431x`) |
| **16,384** | 13,277.0 | 735.3 | 13,824.34 MB | **8,318.09 MB** | **-5,506.3 MB** | **`1.781 MB` vs `1,536.0 MB` (`862x`)** |

- **Crossover point:** Above `~2,200` tokens, `HSPA_4B` uses strictly *less* GPU VRAM than `TRANSFORMER_4B` when chunked (`1024`), locking total inference memory at **~8.32 GB** regardless of context length up to `16,384` tokens (and analytically beyond).
- **Speed target:** Compiling/fusing `SpinorRegisterField` in Triton is the #1 prerequisite to close the `3.2x` decode gap (`135 ms/tok` -> target `< 35 ms/tok`) and `18x` prefill gap.

---

## 3. 1.75B Foundation Suite Efficiency & Capability (`56`-Task 15-Dimension Evaluation, X4 vs `frontier_v1`)

| Arm (`Qwen/Qwen3-1.7B`) | Total Params | Acc | Cap / B-Params | Tok/s (`batch=1` eval) | Latency p50 / p95 (ms) | Peak VRAM (MB) | KV @ 4K (MB) |
|---|---:|---:|---:|---:|---|---:|---:|
| `Base` (`frontier_v1`) | 1.721B | 60.71% | 0.3528 | **33.53** | 1,158 / 7,662 | **3,899.07** | 448.0 |
| `LoRA_SFT` (`frontier_v1`) | 1.738B | **64.29%** | **0.3699** | 20.18 | 1,649 / 5,765 | 4,157.09 | 448.0 |
| `Frontier_v1` (`DexDiff + VAnchor`) | 1.753B | 62.50% | 0.3565 | 16.58 | 1,629 / 8,172 | 4,280.93 | 448.0 |
| `Holonomic_SFT2_GRPO` (`X4`) | 1.747B | 62.50% | 0.3577 | 12.96 | 2,570 / 10,965 | 4,246.31 | 448.0 |
