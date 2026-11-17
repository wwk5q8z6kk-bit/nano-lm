# NanoLM

**NanoLM is an independent general-purpose foundation language model research and development project targeting maximum practical intelligence within a strict 2.0–4.0 billion parameter budget.**

NanoLM is the foundation model itself—not a clinical Scribe wrapper, IDE subsystem, or hardcoded application runtime. Earlier medical-scribing and 3.15M–160M closed-world copying experiments are preserved intact as historical evidence ([`papers/EVIDENCE_LEDGER.md`](papers/EVIDENCE_LEDGER.md), [`docs/archive/legacy/`](docs/archive/legacy/)), while active architecture, tokenization, data, training, and evaluation research lives in [`nanolm/`](nanolm/).

## Hard Parameter & Architecture Policy

- **Preferred target:** ~`2.0B` deployed parameters (`nanolm-2b-hybrid-swa` & `nanolm-2b-dense` at `2.041B`; `nanolm-2b-mla` at `2.121B`).
- **Permitted expansion range:** `2.0B – 4.0B` deployed parameters (`nanolm-3.3b-dense` at `3.314B`; `nanolm-3.8b-moe` at `3.895B` total / `0.989B` active).
- **Hard ceiling:** **4,000,000,000 parameters** across all inference-resident weights (embeddings, output projections, all routed/shared experts, routers, norms, and deployed MTP heads), enforced programmatically in [`nanolm/config.py`](nanolm/config.py).

## Package Architecture (`nanolm/`)

| Module | Role |
|--------|------|
| [`nanolm/config.py`](nanolm/config.py) | Strict parameter/FLOP/KV-cache accounting and verified 2B–4B + research-rung presets |
| [`nanolm/model/`](nanolm/model/__init__.py) | First-principles PyTorch implementations of `GQA+QKNorm`, `3:1 HybridSWA`, `MLA`, `Aux-Loss-Free Fine-Grained MoE`, `Depth-Loop`, and `Multi-Token Prediction (MTP)` |
| [`nanolm/tokenizer/`](nanolm/tokenizer/__init__.py) | Byte-Level BPE trainer with individual digit splitting, 32 special tokens, and 6-domain fertility auditing |
| [`nanolm/data/`](nanolm/data/__init__.py) | Intra-document causal sequence packing (`doc_ids` + per-document RoPE `position_ids` reset) and 13-gram benchmark decontamination |
| [`nanolm/train/`](nanolm/train/__init__.py) | Reproducible `WSD` / Cosine trainer with `bfloat16` MPS/CUDA autocast, tokenizer-invariant `Bits-Per-Byte (BPB)` validation, and atomic checkpoint resume |
| [`nanolm/eval/`](nanolm/eval/__init__.py) | Independent 6-suite programmatic evaluation: subprocess Python code execution, math/logic, exception-preserving summarization, JSON schema validation, Mermaid/chart validation, and long-context reasoning |
| [`nanolm/infra/`](nanolm/infra/__init__.py) | Live hardware preflight benchmark (`bfloat16` MPS/CUDA) and 8xH100 / 4xB200 compute-cost scaling estimator |

## Canonical Documentation

- [Official NanoLM Charter](docs/nanolm/CHARTER.md)
- [Phase 1 Technical Audit](docs/nanolm/PHASE1_AUDIT.md)
- [Architecture, Data & Training Specification](docs/nanolm/ARCHITECTURE_AND_TRAINING.md)
- [Empirical Baseline, Decisions & Compute Roadmap](docs/nanolm/ROADMAP_AND_COMPUTE.md)
- [Verified Empirical Baseline JSON](docs/nanolm/EMPIRICAL_BASELINE_V1.json)
- [Active Program Status](docs/ACTIVE_NOW.md) · [RunPod Compute Policy](docs/infrastructure/RUNPOD.md)

## Quick Verification

```bash
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
python3 -m pytest -q nanolm/tests fabric trajectory
python3 scripts/nanolm/pretrain_cli.py presets
```

## License

MIT — [`LICENSE`](LICENSE)
