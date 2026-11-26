# NanoLM Architecture, Data, and Training Research Specification

## 1. First-Principles Bottlenecks at 2B–4B Scale

1. **Vocabulary vs. Embedding Budget Trade-off**: Over-sized vocabularies (`150k+`) consume >15–30% of a 2B model's parameters in embeddings alone, starving deep reasoning layers. Under-sized vocabularies (`4k`) inflate code/math sequence lengths by ~45%, wasting $O(T^2)$ attention FLOPs and effective context window. NanoLM targets `32k–65k` Byte-Level BPE with individual digit splitting and tied embeddings (`6.6%` of 2B budget at `65,536`).
2. **Depth vs. Width & Logit Stability**: Deep-and-narrow stacks (`L=36`, `d=2048`) outperform shallow wide stacks on compositional code and multi-step reasoning at 2B, provided attention entropy collapse and outlier activations are prevented via **QK-RMSNorm** (computed in fp32) and **z-loss** (`1e-4`).
3. **KV-Cache Bandwidth at Long Context**: Full global attention across 36 layers requires `2.25 GB` of bf16 KV cache at 32k context (`18 GB` at 256k), bottlenecking local/edge decode throughput by memory bandwidth.

## 2. Controlled Architectural Hypotheses (`nanolm/model/`)

Every major candidate is implemented in [`nanolm/model/`](../../nanolm/model/__init__.py) with explicit falsification criteria:

| Hypothesis | Mechanism | Problem Addressed | Rejection Criterion |
|---|---|---|---|
| **H-QKNorm-WSD** (`A1`) | Per-head fp32 `QK-RMSNorm` + Warmup-Stable-Decay (`WSD`) LR schedule | Attention logit spikes at high LR; premature cosine LR decay before multi-stage midtraining | Reject if held-out `BPB` fails to beat unnormalized Cosine baseline (`A0`) at matched FLOPs |
| **H-HybridSWA-MTP** (`A2`) | 3:1 interleaved Sliding-Window (`W=1024`) + Global causal attention + Logit Softcap + 1-step Multi-Token Prediction (`MTP`) head | `72%` KV-cache reduction (`2.25 GB -> 0.62 GB` @ 32k) + denser lookahead representation gradients per token | Reject if long-range/prose `BPB` degrades > 0.015 vs full global `GQA` or if decode latency regresses |
| **H-MLA** (`A3`) | Multi-Head Latent Attention (`kv_lora_rank=512`, decoupled `qk_rope_head_dim=64`) | Compresses KV cache (`1.27 GB` @ 32k) via learned low-rank latent state | Reject if latent up-projection FLOP/latency overhead yields worse quality-per-second than `HybridSWA` |
| **H-MoE-AuxFree** (`A4`) | 1 shared SwiGLU expert + 16 fine-grained routed experts (`top-2`), DeepSeek-style aux-loss-free dynamic bias balancing (`3.89B` total / `0.99B` active) | Increases knowledge/code capacity within the strict `<=4B` parameter cap at sub-1B active FLOPs | Reject if expert collapse (`CV > 0.5`) occurs or if total-memory cost fails to justify `BPB` gain over 2B dense |
| **H-DepthLoop** (`A5`) | Middle-stack parameter sharing (`L=12..24` repeated `2x` -> `48` effective layers in `2.04B` params) | Expands sequential reasoning depth without increasing weight memory | Reject if extra `1.31x` FLOPs/latency per token yields smaller `BPB` gain than non-recurrent capacity |

## 3. Data & Multi-Stage Training Lifecycle (`nanolm/data/`, `nanolm/train/`)

1. **Intra-Document Sequence Packing (`nanolm/data/packing.py`)**: Concatenates documents with `<|bos|>`/`<|eos|>` delimiters while resetting RoPE `position_ids` per document and enforcing block-diagonal intra-document causal masks (`doc_ids`) plus `-100` cross-document target masking.
2. **Stage 1 — Foundation Pretraining (`WSD` Stable Phase)**: Broad web/educational prose (`FineWeb-Edu`), multi-language code, mathematics, and technical documentation under constant peak LR after warmup, enabling checkpoint branching at arbitrary token horizons.
3. **Stage 2 — High-Density Midtraining (`WSD` Decay Phase + Context Extension)**: Anneals LR while up-weighting repository-level code, mathematical proofs, long-context synthesis (YaRN RoPE scaling `4k -> 32k`), structured JSON, and diagram/chart representations.
4. **Stage 3 — Post-Training (`SFT` + Verifiable `RLVR`)**: Assistant-masked multi-domain instruction tuning with `<|think|>...<|end_think|>` reasoning traces, followed by programmatic reinforcement learning on verifiable unit-test, math, JSON-schema, and Mermaid-topology checkers (`nanolm/eval/validators.py`).
