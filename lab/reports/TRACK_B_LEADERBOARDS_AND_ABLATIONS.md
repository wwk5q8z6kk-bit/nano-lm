# Track B Competitive Foundation Evaluation: 3 Leaderboards & Distillation Ablations

All models evaluated under identical zero-temperature vLLM (`v0.34.0`, `bfloat16`) and `lm_eval` (`v0.4.13`) harnesses on NVIDIA A40 48GB (`seed=1234`), plus our deterministic 132-item 8-pillar capability, tool-calling, RULER long-context, and epistemic reliability suite (zero LLM-judge subjectivity).

## 1. Leaderboard A — Intelligence (Reasoning, Math, Code, Tool Use, Long-Context & Knowledge)

| Model | Stored / Active Params | GSM8K (CoT) | MATH-500 | HumanEval | IFEval | ARC-C | 8-Pillar Pass (`think=0` / `think=1`) | LB-A Intelligence (`think=0` / `think=1`) | Tool Calling (`think=0` / `think=1`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen2.5-1.5B-Instruct (historical baseline)** | 1.54B / 1.54B | 67.5% | 48.6% | 52.4% | 40.5% | 53.9% | 65.1% / — | 72.3% / — | 84.4% / — |
| **Qwen3-1.7B (unmodified Track B base)** | 1.72B / 1.72B | 75.0% | 70.0% | 66.5% | 66.9% | 52.1% | 70.5% / 83.3% | 75.6% / 75.0% | 71.9% / 93.8% |
| **Qwen3.5-2B** | 2.27B / 1.88B | 70.1% | 63.6% | 53.7% | 66.9% | 52.5% | 76.5% / 37.9% | 80.6% / 40.8% | 78.1% / 87.5% |
| **SmolLM3-3B** | 3.08B / 3.08B | 82.5% | 63.2% | 72.0% | 67.5% | 59.8% | 77.3% / 87.1% | 86.2% / 84.0% | 81.2% / 96.9% |
| **Granite-4.0-H-Micro (3.2B)** | 3.19B / 3.19B | 80.7% | 49.2% | 77.4% | 79.7% | 65.5% | 77.3% / — | 82.7% / — | 40.6% / — |
| **Phi-4-mini-instruct (3.84B)** | 3.84B / 3.84B | 82.3% | 34.0% | 70.7% | 65.2% | 57.6% | 74.2% / — | 79.4% / — | 96.9% / — |
| **NanoLM-B1-SFT (1.72B / 2.03B untied)** | 1.72B / 1.72B | 55.6% | 41.0% | 50.6% | 60.4% | 55.0% | 73.5% / 32.6% | 69.4% / 18.1% | 65.6% / 28.1% |
| **NanoLM-B2-KD8B (1.72B / 2.03B untied)** | 1.72B / 1.72B | 68.0% | 56.0% | 59.1% | 56.6% | 52.6% | 83.3% / 57.6% | 79.6% / 42.7% | 81.2% / 28.1% |
| **NanoLM-B3-OnPolicy-KD8B (1.72B / 2.03B untied)** | 1.72B / 1.72B | 73.2% | 62.6% | 63.4% | 65.1% | 52.6% | 78.0% / 75.0% | 76.0% / 67.3% | 71.9% / 96.9% |
| *Aspirational Frontier Reference: `openai/gpt-oss-120b` (publisher / open-eval reference, out-of-class MoE)* | *117B / 5.1B* | *~95%* | *~90%+* | *~88%+* | *~86%+* | *—* | *—* | *—* | *Reference ceiling (~58× stored params)* |

## 2. Leaderboard B — Reliability, Groundedness & Schema/Abstention Calibration

| Model | LB-B Reliability (`think=0` / `think=1`) | When2Call Abstention (`t=0` / `t=1`) | Tool Error Recovery (`t=0` / `t=1`) | Strict JSON & Arith (`t=0` / `t=1`) | Epistemic Reliability (`t=0` / `t=1`) | Faithful Summarization (`t=0` / `t=1`) | TruthfulQA MC2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen2.5-1.5B-Instruct (historical baseline)** | 60.9% / — | 87.5% / — | 50.0% / — | 31.2% / — | 100.0% / — | 25.0% / — | 46.7% |
| **Qwen3-1.7B (unmodified Track B base)** | 76.0% / 97.9% | 100.0% / 100.0% | 37.5% / 87.5% | 25.0% / 100.0% | 91.7% / 91.7% | 87.5% / 100.0% | 45.9% |
| **Qwen3.5-2B** | 64.1% / 25.0% | 37.5% / 62.5% | 100.0% / 87.5% | 18.8% / 37.5% | 100.0% / 0.0% | 100.0% / 0.0% | 43.4% |
| **SmolLM3-3B** | 74.5% / 91.7% | 100.0% / 100.0% | 37.5% / 87.5% | 12.5% / 81.2% | 91.7% / 91.7% | 93.8% / 93.8% | 53.8% |
| **Granite-4.0-H-Micro (3.2B)** | 88.5% / — | 100.0% / — | 12.5% / — | 68.8% / — | 91.7% / — | 93.8% / — | 58.1% |
| **Phi-4-mini-instruct (3.84B)** | 69.8% / — | 100.0% / — | 87.5% / — | 18.8% / — | 66.7% / — | 93.8% / — | 48.3% |
| **NanoLM-B1-SFT (1.72B / 2.03B untied)** | 91.7% / 62.5% | 100.0% / 100.0% | 12.5% / 12.5% | 81.2% / 81.2% | 91.7% / 0.0% | 93.8% / 68.8% | 46.8% |
| **NanoLM-B2-KD8B (1.72B / 2.03B untied)** | 94.8% / 96.4% | 100.0% / 100.0% | 50.0% / 12.5% | 87.5% / 93.8% | 91.7% / 91.7% | 100.0% / 100.0% | 48.5% |
| **NanoLM-B3-OnPolicy-KD8B (1.72B / 2.03B untied)** | 90.1% / 88.5% | 100.0% / 100.0% | 37.5% / 87.5% | 68.8% / 100.0% | 91.7% / 91.7% | 100.0% / 62.5% | 47.5% |

## 3. Leaderboard C — Efficiency, Throughput & Strict Parameter Accounting

| Model | License | Architecture | Stored Params | Untied Equiv | Active Params | Fast Output `tok/s` (A40) | Fast Eval Wall (s) | Intelligence / 1B Active (`think=0`) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen2.5-1.5B-Instruct (historical baseline)** | `Apache-2.0` | Dense GQA Transformer | 1.54B | 1.54B | 1.54B | 1,498.1 | 18.3s | 0.4694 |
| **Qwen3-1.7B (unmodified Track B base)** | `Apache-2.0` | Dense GQA + QK-Norm (dual think) | 1.72B | 2.03B | 1.72B | 1,406.0 | 12.2s | 0.4397 |
| **Qwen3.5-2B** | `Apache-2.0` | 3:1 Gated-DeltaNet + GQA Hybrid (VLM) | 2.27B | 2.27B | 1.88B | 1,472.1 | 18.8s | 0.4288 |
| **SmolLM3-3B** | `Apache-2.0` | Dense GQA + NoPE/RoPE (dual think) | 3.08B | 3.08B | 3.08B | 915.4 | 17.6s | 0.2800 |
| **Granite-4.0-H-Micro (3.2B)** | `Apache-2.0` | Mamba-2 + GQA Hybrid | 3.19B | 3.19B | 3.19B | 854.7 | 15.9s | 0.2593 |
| **Phi-4-mini-instruct (3.84B)** | `MIT` | Dense GQA Transformer | 3.84B | 3.84B | 3.84B | 607.5 | 25.4s | 0.2067 |
| **NanoLM-B1-SFT (1.72B / 2.03B untied)** | `Apache-2.0` | Qwen3-1.7B + Verified SFT (alpha_kd=0.0) | 1.72B | 2.03B | 1.72B | 533.9 | 16.8s | 0.4033 |
| **NanoLM-B2-KD8B (1.72B / 2.03B untied)** | `Apache-2.0` | Qwen3-1.7B + Top-64+Tail KD from Qwen3-8B (off-policy) | 1.72B | 2.03B | 1.72B | 1,305.5 | 10.4s | 0.4627 |
| **NanoLM-B3-OnPolicy-KD8B (1.72B / 2.03B untied)** | `Apache-2.0` | Qwen3-1.7B + Verified Dual-Mode On-Policy Top-64+Tail KD (Qwen3-8B) | 1.72B | 2.03B | 1.72B | 1,297.8 | 13.8s | 0.4421 |

## 4. Controlled Post-Training & Teacher-Distillation Ablation (`Qwen3-1.7B` Foundation)

| Metric | Base `Qwen3-1.7B` | `B1` Verified SFT (`alpha=0`) | `B2` Off-Policy KD-8B (`alpha=0.6`) | `B3` Verified Dual-Mode On-Policy KD-8B (`alpha=0.85`) | Mechanism Finding |
|---|---:|---:|---:|---:|---|
| 8-Pillar Overall (`think=0`) | 70.45% | 73.48% | 83.33% | **78.03%** | `B2` (+12.9pp) & `B3` (+7.6pp) beat Base & `B1` |
| 8-Pillar Overall (`think=1`) | 83.33% | 32.58% | 57.58% | **75.00%** | `B3` Dual-Mode rescues `think=1` (+42.4pp vs `B1`, +17.4pp vs `B2`) |
| LB-A Intelligence (`think=0`) | 75.63% | 69.37% | 79.58% | **76.04%** | Both KD arms (`B2`, `B3`) beat Base; pure SFT (`B1`) regresses |
| LB-B Reliability (`think=0`) | 76.04% | 91.67% | 94.79% | **90.11%** | +14.1pp (`B3`) to +18.8pp (`B2`) Reliability jump over Base |
| Coding Execution (`think=0`) | 56.25% | 25.00% | 43.75% | **68.75%** | `B3` on-policy MBPP-train KD lifts coding +12.6pp over Base (+43.8pp vs `B1`) |
| Tool Calling (`think=1`) | 93.75% | 28.12% | 28.12% | **96.88%** | `B3` reaches **96.9%** (31/32), matching 3.84B Phi-4-mini |
| Strict JSON & Arith (`think=0`) | 25.00% | 81.25% | 87.50% | **68.75%** | Fixes single-pass JSON schema/sum failures (`25% -> 68.8%–87.5%`) |
| Strict JSON & Arith (`think=1`) | 100.00% | 81.25% | 93.75% | **100.00%** | `B3` preserves **100.0%** structured JSON in thinking mode |
| Public `GSM8K` (CoT) | 74.98% | 55.57% | 68.01% | **73.16%** | On-policy `\boxed{}` KD (`B3`) eliminates 90% of `B1` forgetting (+17.6pp vs `B1`) |
| Public `MATH-500` | 70.00% | 41.00% | 56.00% | **62.60%** | `B1` (-29pp) -> `B2` (+15pp rescue) -> `B3` (+21.6pp rescue to 62.6%) |
| Public `HumanEval` | 66.46% | 50.61% | 59.15% | **63.41%** | `B3` recovers +12.8pp over `B1` (+4.3pp over `B2`) |
| Public `IFEval` (Strict) | 66.91% | 60.44% | 56.56% | **65.06%** | `B3` constraint verification recovers +8.5pp over `B2` (within 1.85pp of Base) |
| Public `TruthfulQA MC2` | 45.88% | 46.77% | 48.49% | **47.47%** | Both KD models (`B2` +2.6pp, `B3` +1.6pp) improve truthfulness over Base |
| Public `ARC-Challenge` | 52.13% | 55.03% | 52.56% | **52.65%** | Zero factual/science degradation (`52.13% -> 52.65%`) |

## 5. Diagnostic Failure-Mode Taxonomy Across Candidate Families

- **Qwen2.5-1.5B-Instruct (historical baseline)**
  - `think=0` top failures: `summarization:incomplete_facts` (12), `structured_json:semantic_check_failed` (10), `coding:AssertionError` (7), `long_context:wrong_answer_long_context` (5)
- **Qwen3-1.7B (unmodified Track B base)**
  - `think=0` top failures: `structured_json:semantic_check_failed` (11), `math_logic:wrong_answer_math_logic` (8), `coding:AssertionError` (6), `tool_calling:error_recovery_failed` (5)
  - `think=1` top failures: `math_logic:wrong_answer_math_logic` (11), `coding:no_function_definition` (7), `tool_calling:argument_value_error` (1), `tool_calling:error_recovery_failed` (1)
- **Qwen3.5-2B**
  - `think=0` top failures: `structured_json:semantic_check_failed` (13), `tool_calling:false_positive_tool_call` (4), `coding:AssertionError` (3), `tool_calling:invalid_tool_json` (2)
  - `think=1` top failures: `summarization:word_limit_exceeded` (16), `math_logic:wrong_answer_math_logic` (14), `long_context:wrong_answer_long_context` (12), `epistemic_reliability:wrong_answer_epistemic_reliability` (12)
- **SmolLM3-3B**
  - `think=0` top failures: `structured_json:invalid_json` (10), `tool_calling:invalid_tool_json` (5), `coding:AssertionError` (5), `structured_json:semantic_check_failed` (4)
  - `think=1` top failures: `coding:no_function_definition` (4), `math_logic:wrong_answer_math_logic` (3), `structured_json:semantic_check_failed` (3), `diagrams_charts:no_valid_edges` (2)
- **Granite-4.0-H-Micro (3.2B)**
  - `think=0` top failures: `tool_calling:invalid_tool_json` (19), `structured_json:semantic_check_failed` (5), `coding:timeout` (1), `coding:AssertionError` (1)
- **Phi-4-mini-instruct (3.84B)**
  - `think=0` top failures: `structured_json:semantic_check_failed` (13), `math_logic:wrong_answer_math_logic` (6), `epistemic_reliability:wrong_answer_epistemic_reliability` (4), `coding:AssertionError` (2)
- **NanoLM-B1-SFT (1.72B / 2.03B untied)**
  - `think=0` top failures: `tool_calling:error_recovery_failed` (7), `math_logic:wrong_answer_math_logic` (7), `coding:AssertionError` (6), `structured_json:semantic_check_failed` (3)
  - `think=1` top failures: `tool_calling:missed_tool_call` (15), `math_logic:wrong_answer_math_logic` (14), `long_context:wrong_answer_long_context` (12), `epistemic_reliability:wrong_answer_epistemic_reliability` (12)
- **NanoLM-B2-KD8B (1.72B / 2.03B untied)**
  - `think=0` top failures: `coding:AssertionError` (7), `tool_calling:error_recovery_failed` (4), `math_logic:wrong_answer_math_logic` (3), `tool_calling:missed_tool_call` (2)
  - `think=1` top failures: `tool_calling:missed_tool_call` (16), `math_logic:wrong_answer_math_logic` (11), `coding:no_function_definition` (9), `tool_calling:error_recovery_failed` (7)
- **NanoLM-B3-OnPolicy-KD8B (1.72B / 2.03B untied)**
  - `think=0` top failures: `math_logic:wrong_answer_math_logic` (7), `tool_calling:error_recovery_failed` (5), `tool_calling:missed_tool_call` (4), `structured_json:semantic_check_failed` (4)
  - `think=1` top failures: `math_logic:wrong_answer_math_logic` (12), `coding:no_function_definition` (11), `summarization:missing_qualification` (6), `tool_calling:error_recovery_failed` (1)
