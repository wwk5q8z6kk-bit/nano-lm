# NanoLM (2B–4B) — High-Density Architecture, Post-Training & Multi-Domain Verification

## 1. Identity & Parameter Budget

**NanoLM** is an independent general-purpose foundation language model program. It is **not** a medical-scribing product and is **not** coupled to DevX or any external IDE/application runtime. Historical Scribe/H6/H7 artifacts remain preserved solely as empirical provenance.

- **Preferred development target:** **~2.0B parameters** (`nanolm-2b`: `1,978,345,602` parameters, 36 physical layers / 54 effective depth layers via recurrent thought loops).
- **Absolute upper limit:** **4.0B parameters** (`ABSOLUTE_MAX_PARAMETERS = 4_000_000_000`; `nanolm-4b`: `3,864,791,042` parameters, 48 physical layers / 72 effective depth layers). Configurations exceeding `4,000,000,000` parameters raise `ParameterBudgetExceededError`.

## 2. Architectural Levers for Maximum Capability per Parameter (`nanolm_posttrain/architecture.py`)

To rival 10–30× larger models within 2B–4B parameters, NanoLM concentrates synaptic capacity on reusable compositional circuits rather than long-tail lexical memorization:

1. **Tied Input/Output Embeddings:** Eliminates a separate output projection matrix (`~205M–256M` saved parameters at `V=100,352`), reallocating 10–13% of total model weights directly into deep attention and SwiGLU blocks.
2. **Grouped-Query Attention (GQA) + Per-Head QK-Norm + RoPE (`theta=500,000`, 64k context):** Stabilizes high-learning-rate optimization without attention-logit blowup while keeping FP16 KV-cache footprint at `72 KiB/token` (`nanolm-2b`) and `96 KiB/token` (`nanolm-4b`).
3. **Adaptive Recurrent Thought Loops (`thought_loops >= 2`):** Weight-shared middle layers execute multiple latent reasoning passes modulated by learned step embeddings (`loop_step_emb`) and zero-initialized residual tanh gates (`loop_gates`), increasing sequential reasoning depth by `1.5×` for `< 0.01%` additional parameter bytes.

## 3. Six-Pillar Deterministic Verification & Post-Training (`nanolm_posttrain/eval_suite.py`, `lora.py`)

Every core NanoLM capability is evaluated and post-trained against deterministic execution/structural verifiers rather than subjective LLM judges:

| Domain | Deterministic Verifier |
| :--- | :--- |
| **Coding (`coding`)** | Isolated `python3 -I` subprocess AST + unit-assertion execution (`pass@1`) |
| **Math & Logic (`reasoning`)** | Multi-step derivation structure + exact numerical/symbolic answer verification |
| **Summarization (`summarization`)** | Hard word-compression ceiling + salient fact recall + zero hallucinated numbers |
| **Structured Extraction (`structured_extraction`)** | Strict JSON parsing + exact schema key set + source-value conservation |
| **Diagrams & Charts (`diagrams_and_charts`)** | Mermaid header/edge topology + quoted special-character node labels + JSON chart series/total conservation |
| **Long Context (`long_context`)** | Multi-needle retrieval and arithmetic synthesis across 30-section documents |

Pure-PyTorch `LoRALinear` (`nanolm_posttrain/lora.py`) attaches FP32 low-rank `A`/`B` adapters across all 7 attention/MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with prompt-masked causal SFT (`ignore_index=-100`), running identically across Apple Silicon (`mps`), Kaggle GPUs, and RunPod CUDA nodes.
