# NanoLM — Official Project Charter

## 1. Identity and Mission

**NanoLM** is an independent general-purpose foundation language model research and engineering program.

**Primary Objective**: Build the most capable, reliable, and efficient general-purpose language model achievable within a strict **2.0–4.0 billion parameter** deployed budget—maximizing useful intelligence per parameter, per training token, per FLOP, and per byte of inference memory.

NanoLM is **the foundation model itself**. It is independent of DevX, independent of historical clinical/medical Scribe workflows, and free of domain-specific schema constraints.

## 2. Strict Parameter-Budget Policy

1. **Preferred Production Target**: ~2.0 billion deployed parameters (`nanolm-2b-*` family: `2.04B`).
2. **Permitted Expansion Band**: `2.0B` to `4.0B` deployed parameters (`nanolm-3.3b-dense` at `3.31B`; `nanolm-3.8b-moe` at `3.89B` total / `0.99B` active).
3. **Hard Ceiling**: **4,000,000,000 parameters (`HARD_MAX_PARAMETERS`)**. Enforced programmatically in `nanolm/config.py` and `NanoLMForCausalLM.__init__` (`ParameterBudgetExceededError`).
4. **Full Transparency Rule**: Every parameter required in inference memory—input embeddings, untied output heads (if any), all routed and shared experts, routers, norms, and deployed Multi-Token Prediction heads—is counted in `total_deployed_parameters`. Conditional-computation models report `total_deployed_parameters`, `active_parameters_per_token`, `kv_cache_gb`, `forward_flops_per_token`, and `routing_flops_per_token` side by side.

## 3. Core Capability Pillars

NanoLM develops unified general-purpose intelligence across six integrated axes:

1. **Software Engineering & Programming**: Multi-language code generation, debugging, refactoring, data structures, algorithms, and repository-level reasoning—validated by isolated unit-test execution.
2. **Mathematical & Logical Reasoning**: Multi-step quantitative reasoning, algebra, deductive constraint satisfaction, contradiction detection, and calibrated uncertainty.
3. **Information Processing & Faithful Summarization**: Compression and multi-document synthesis that preserve exact figures, causal relationships, qualifications, and exceptions.
4. **Structured Generation, Charts, and Diagrams**: Native reasoning into valid JSON schemas, chart specifications (correct chart type, axes, units, and exact data series), and Mermaid diagrams (`flowchart`, `sequenceDiagram`, `stateDiagram`) with verified graph topology and edge directionality.
5. **General Language & Knowledge**: Clear technical writing, instruction following, explanation, and adaptation to unfamiliar domains.
6. **Genuine Long-Context Intelligence**: Reliable retrieval, cross-passage synthesis, and distant contradiction detection across the entire context window—not merely an advertised RoPE limit.

## 4. Scientific Honesty & Engineering Discipline

- **Tokenizer-Invariant Comparisons**: Architecture and vocabulary experiments are evaluated on held-out **Bits-Per-Byte (`BPB`)**, preventing vocabulary changes from masquerading as perplexity gains.
- **Zero-Contamination Guarantee**: Training and SFT corpora undergo automated exact-hash deduplication and 13-gram decontamination against all held-out evaluation prompts (`nanolm/data/decontaminate.py`).
- **Evidence Over Complexity**: Architectural mechanisms are retained only when controlled ablations demonstrate measurable gains in BPB, capability pass rates, memory footprint, or throughput.
