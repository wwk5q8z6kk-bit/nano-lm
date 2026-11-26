# NanoLM Empirical Findings, Architecture Decision & Responsible Scaling Plan

## 1. Summary of Executed Foundation Experiments (`EMPIRICAL_BASELINE_V1.json`)

All experiments were executed locally on Apple M3 Max (64 GB unified memory, PyTorch MPS `bfloat16`) with 13-gram benchmark decontamination and SHA-256 provenance tracking.

### 1.1 `E-TOK-01`: Multi-Domain Tokenizer & Embedding-Budget Study

| Tokenizer | Vocab Size | Overall `B/tok` | Code `B/tok` | Digit Split | Tied Emb Share of `2.04B` |
|---|---:|---:|---:|:---:|---:|
| `legacy-nano-4k` | 4,098 | 2.935 | 2.053 | Yes | 0.4% |
| `nanolm-bpe-16k` | 16,384 | 4.088 | 3.707 | Yes | 1.6% |
| `nanolm-bpe-32k` | 32,768 | **4.272** | **3.707** | Yes | **3.3%** |
| `qwen2.5-152k-ref` | 151,665 | 4.421 | 3.834 | Yes | 15.2% |

**Decision**: Adopt `32,768–65,536` Byte-Level BPE with individual digit splitting and 32 special tokens (`<|bos|>`, `<|eos|>`, `<|im_start|>`, `<|im_end|>`, `<|think|>`, `<|end_think|>`, `<|code|>`, `<|json|>`, `<|diagram|>`, plus 22 reserved slots). This captures `96.6%` of a 152k vocabulary's compression efficiency (`+80.6%` code density over legacy 4k) while keeping tied embeddings to `3.3%–6.6%` of a 2B model budget—saving ~175M–240M parameters for deep transformer reasoning blocks.

### 1.2 `E-INFRA-2B4B`: Full-Scale 2.04B–3.89B Live Hardware Profiling (MPS `bfloat16`)

| Candidate Preset | Deployed Params | Active Params / Tok | Effective Depth | Prefill (`tok/s`) | Decode (`tok/s`) | KV Cache @ 32k (`bf16`) | Weight RAM (`bf16` / `int4`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nanolm-2b-dense` | **2.041B** | 2.041B | 36 | 1,131.2 | 37.35 | 2.25 GB | 3.80 GB / 1.05 GB |
| **`nanolm-2b-hybrid-swa`** | **2.041B** | **2.041B** | **36** | **1,502.1** | **47.10** | **0.62 GB** | **3.80 GB / 1.05 GB** |
| `nanolm-2b-mla` | **2.121B** | 2.121B | 36 | 759.7 | 11.09 | 1.27 GB | 3.95 GB / 1.09 GB |
| `nanolm-2b-loop` | **2.041B** | 2.676B | 48 | 1,256.5 | 33.69 | 0.82 GB | 3.80 GB / 1.05 GB |
| `nanolm-3.3b-dense` | **3.314B** | 3.314B | 40 | 1,021.2 | 20.12 | 0.74 GB | 6.17 GB / 1.70 GB |
| `nanolm-3.8b-moe` | **3.895B** | 0.989B | 22 | 83.3 | 2.74 | 0.35 GB | 7.26 GB / 2.00 GB |

**Systems Findings**:
- `nanolm-2b-hybrid-swa` (`3:1` interleaved Sliding-Window `W=1024` + Global full causal attention, `QK-RMSNorm`, Logit Softcap) dominates the latency-memory Pareto frontier: **+32.8% faster prefill (`1,502 tok/s`), +26.1% faster decode (`47.1 tok/s`), and -72.4% KV-cache footprint at 32k context (`0.62 GB` vs `2.25 GB`)**.
- `nanolm-2b-mla` reduces short-context KV cache (`10.76 MB` vs `19.12 MB` at `T=272`) but incurs a `4.2x` decode throughput penalty (`11.09 tok/s`) without custom weight-absorbed matrix-multiplication kernels.
- `nanolm-3.8b-moe` (`3.895B` total / `0.989B` active) halves theoretical training FLOPs (`5.93e20` vs `1.22e21` at `100B` tokens) but requires fused grouped-GEMM (`torch._grouped_mm` / MegaBlocks on CUDA H100/B200) to avoid eager-dispatch bottlenecks.

### 1.3 `E-ARCH-01`: 6-Arm Controlled Architecture & Objective Ablation (`983k` Packed Tokens, Tokenizer-Invariant `BPB`)

| Arm | Architecture & Schedule | Held-Out FineWeb-Edu Prose `BPB` ↓ | Held-Out Multi-Domain `BPB` ↓ | Composite `BPB` ↓ | Decision |
|---|---|---:|---:|---:|---|
| `A3_mla_latent_kv_wsd` | MLA + WSD | **2.1748** | 0.1573 | **1.1660** | Retain as CUDA-fused research track |
| `A1_dense_qknorm_wsd` | Dense GQA + QK-Norm + WSD | 2.1971 | **0.1555** | 1.1763 | Confirmed (`-0.0258` BPB vs `A0`) |
| **`A2_hybrid_swa_mtp_wsd`** | **3:1 HybridSWA + 1-Step MTP + QK-Norm + WSD** | **2.1866** | **0.1664** | **1.1765** | **Selected Primary 2B Baseline** |
| `A4_moe_finegrained_wsd` | Aux-Loss-Free 8-Expert MoE (`CV=0.336`) + WSD | 2.2048 | 0.1705 | 1.1876 | Deferred to `>=100B` token CUDA runs |
| `A0_legacy_no_qknorm_cosine` | Unnormalized GQA + Cosine (Control) | 2.2264 | 0.1778 | 1.2021 | Superseded |
| `A5_depth_loop_wsd` | Recurrent Middle-Layer Tying (`12 -> 16` steps) | 2.2698 | 0.2197 | 1.2448 | **Rejected** (hurts early layer specialization) |

**Primary Architecture Selection**: **`nanolm-2b-hybrid-swa` (`2.041B` deployed parameters)** with 1-step auxiliary Multi-Token Prediction (`61.4M` training-only parameters stripped at export) is selected as NanoLM's primary baseline architecture: it beats full global `Dense+QKNorm` on real FineWeb-Edu prose (`2.1866` vs `2.1971` BPB), matches composite BPB within `0.0002`, cuts 32k KV cache by `72.4%`, and runs `26%–33%` faster on local hardware.

### 1.4 `E-CAP-01` → `E-CAP-02`: Controlled Capability Improvement on Held-Out 6-Suite Benchmark

- **`E-CAP-01` (Initial SFT Baseline — Constant LR + Narrow Slot Literals)**: `0.0%` (`0/16`) overall pass rate. Raw generation inspection localized two causes: constant-LR underfitting and slot-literal memorization.
- **`E-CAP-02` (Targeted Intervention — 1,099 Decontaminated Open-Vocabulary Compositional Examples + Warmup-Cosine Annealing)**: Overall held-out pass rate rose from **`0.0%` → `25.0%` (`4/16`)**, achieving **`75%` (`3/4`) on isolated subprocess Python unit-test execution** (surpassing the local 1.54B reference `Qwen2.5-1.5B-Instruct` at `25%` coding / `31.2%` overall), `50%` valid JSON syntax, `100%` valid chart JSON syntax, and `0%` hallucination on summarization.
- **Bottleneck Localized for 2B Scaling**: Remaining errors on `summarization`, `structured_json`, and `long_context` trace strictly to unseen multi-digit / open-vocabulary token copying (`fact_recall` / `semantic_check_failed`)—confirming that scaling pretraining from `1M` smoke tokens to multi-billion-token `FineWeb-Edu + StarCoder2/Stack-v2 + OpenWebMath` on `nanolm-2b-hybrid-swa` is the critical next lever.

## 2. Responsible Compute & Scaling Roadmap (`nanolm/infra/preflight.py`)

| Stage | Model | Token Horizon | Compute Venue | Estimated Hours | Estimated Cost | Gate Criteria |
|---|---|---:|---|---:|---:|---|
| **Stage A (Complete)** | `25M` Rungs + `2.04B–3.89B` Preflight | `1M–5M` | Local Apple M3 Max (64GB) | < 1 hr | **$0** | 12/12 unit tests, exact param parity, `BPB` & `E-CAP-02` lift verified |
| **Stage B (Next Pilot)** | `nanolm-350m` -> `nanolm-2b-hybrid-swa` | `5B` (Chinchilla Pilot) | RunPod `4x B200` or `8x H100` (prereg + budget auth) | `4.2h` (`4xB200`) | **~$91** (within `$180` envelope) | Verify FSDP/DDP MFU >= `42%`, monotonic `BPB` scaling curve, zero grad spikes |
| **Stage C (2B Pretrain)** | `nanolm-2b-hybrid-swa` (`2.041B`) | `100B` (Stage 1+2 WSD) | RunPod `4x B200` / `8x H100` (owner-authorized) | `84.0h` (`4xB200`) | **~$1,814** | Beat external 1.5B–2B baselines on held-out `BPB`, HumanEval+/MBPP, GSM8K, JSON/Mermaid |
| **Stage D (2B -> 3.3B Check)** | `nanolm-3.3b-dense` (`3.314B`) | `100B` | RunPod `4x B200` (conditional on Stage C) | `136.4h` | **~$2,946** | Retain only if capability/dollar exceeds extending `2.04B` to `200B` tokens |
