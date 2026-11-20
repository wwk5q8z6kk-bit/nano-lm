# NanoLM (2B–4B) — Multi-Domain Post-Training & Deterministic Verification (`v1`)

**Date:** 2026-11-16 · **Schema:** `nanolm.foundation_posttrain_eval.v1` · **Paid compute:** $0 (local Apple M3 Max 40-core GPU, 64 GB UMA)

This report documents the independent **NanoLM 2B–4B Foundation Model** architecture, deterministic 6-domain evaluation suite, and pure-PyTorch LoRA post-training run implemented in [`nanolm_posttrain/`](../../nanolm_posttrain/). Machine-readable outputs are recorded in [`results.json`](results.json).

## 1. Model Identity & Strict Parameter Ceiling

- **Program Identity:** NanoLM is an independent general-purpose small foundation language model (`~2B` preferred target, `4B` absolute hard upper limit). Historical Scribe experiments remain preserved provenance only; external projects (`DevX`, etc.) do not govern NanoLM.
- **Architectural Configurations (`nanolm_posttrain/architecture.py`):**
  - `nanolm-2b`: **`1,978,345,602` parameters** (`< 2.0B` preferred target; 36 physical layers, 54 effective depth layers via 2-pass recurrent thought loops, GQA `16:4`, per-head QK-Norm, tied embeddings, 64k RoPE).
  - `nanolm-4b`: **`3,864,791,042` parameters** (`< 4.0B` hard ceiling; 48 physical layers, 72 effective depth layers). Any config `> 4,000,000,000` raises `ParameterBudgetExceededError`.

## 2. Live 1.54B Multi-Domain Post-Training Experiment (`mps` FP16 + FP32 LoRA)

To test high-density multi-domain post-training within the `<= 4B` budget, we evaluated pinned `1.54B` (`1,543,714,816` base parameters; checkpoint `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`) before and after 6 epochs (`81.81 s`) of prompt-masked causal SFT using pure-PyTorch `LoRALinear` (`rank=16`, `alpha=32`, `196` wrapped projections, `18,464,768` trainable parameters = `1.18%`) on an 11-example multi-domain curriculum **strictly disjoint** from the 12-task held-out evaluation suite:

- **Training loss trajectory (6 epochs):** `0.3975 → 0.2034 → 0.0841 → 0.0286 → 0.0115 → 0.0019`

### Held-Out 6-Domain Evaluation Results (`12` Unseen Tasks)

| Capability Domain | Baseline Pass Rate | Adapted Pass Rate | Baseline Score | Adapted Score | Key Observed Shift |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Coding (`coding`)** | `0.0%` (`0/2`) | **`50.0%` (`1/2`)** | `0.000` | **`0.500`** | Solved unseen Kahn's topological sort + cycle `ValueError` (`eval-code-02`) |
| **Math & Logic (`reasoning`)** | `0.0%` (`0/2`) | **`50.0%` (`1/2`)** | `0.000` | **`0.500`** | Fixed per-worker vs cluster-total rate error & token-truncation (`eval-reason-01`: `900`) |
| **Summarization (`summarization`)** | `50.0%` (`1/2`) | `50.0%` (`1/2`) | `0.750` | `0.625` | Zero hallucinated numbers across all summaries; preserved all 4 metrics on `eval-sum-01` |
| **Structured JSON (`structured_extraction`)** | `100.0%` (`2/2`) | **`100.0%` (`2/2`)** | `1.000` | **`1.000`** | Exact schema keys, types, booleans, and source-value conservation |
| **Diagrams & Charts (`diagrams_and_charts`)** | `100.0%` (`2/2`) | **`100.0%` (`2/2`)** | `1.000` | **`1.000`** | Valid `flowchart LR` quoted-label topology + exact bar-chart series & total conservation |
| **Long Context (`long_context`)** | `50.0%` (`1/2`) | `50.0%` (`1/2`) | `0.500` | `0.500` | Exact 30-section multi-needle retrieval & arithmetic synthesis (`eval-long-01`) |
| **Overall (All 6 Pillars)** | **`50.0%` (`6/12`)** | **`66.7%` (`8/12`)** | **`0.5417`** | **`0.6875`** | **`+16.7 pp` pass rate, `+0.1458` mean score** |

The table records the original verifier. Its topological-order checks missed lexicographic ties. Current coding evaluations use the Monty Python subset and enforce those ties. Standard-library availability differs from CPython. Fresh scores require rescoring.

## 3. Reproduce

```bash
python3 -m pip install pydantic-monty==1.6.0
python3 -m pytest -q nanolm_posttrain/
python3 -m nanolm_posttrain.runner --checkpoint /path/to/local/model --adapt-epochs 6 --lora-rank 16 --lr 0.00015
```
