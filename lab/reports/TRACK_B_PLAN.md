# Track B plan v1: competitive NanoLM from a pretrained foundation

*2026-11-17. Gate: the base model is chosen only after LAB-BASE-03 and LAB-BASE-04 results. Qwen3.5-2B is the provisional lead.*

## B0. Principles (from the owner's decision)
- The original checkpoint stays untouched. Each intervention is a new versioned artifact (adapter or full weights) with before and after evaluation.
- No architecture surgery until it is shown to preserve capability. Qwen3.5-2B is already a 3:1 Gated-DeltaNet hybrid, so no conversion is needed to get delta-rule efficiency.
- Teachers are compared. Teacher outputs become training targets **only after verification** (unit tests, exact-answer match, JSON-schema / tool-execution checks).
- Every training source is 13-gram decontaminated against every evaluation set (GSM8K/MATH test, HumanEval/MBPP, IFEval, BFCL, MMLU(-Pro), GPQA, RULER), and the removals are logged.

## B1. Base selection (LAB-BASE-03 + 04)
Candidates and licenses: see the plan artifact §10.2. BFCL (tool calling: simple, multiple, parallel, irrelevance, multi-turn) and RULER (4k–32k) are added before the final choice.

## B2. Teachers (licenses verified)
| Teacher | License | Tokenizer shared with Qwen3.5-2B? | Use |
|---|---|---|---|
| Qwen3.5-35B-A3B | Apache-2.0 | **Yes** | Top-k **logit KD** (off-policy, then on-policy), reasoning traces. MoE with ~3B active, so it is cheap to serve on one 80 GB GPU. |
| Qwen3.5-27B | Apache-2.0 | **Yes** | Second logit teacher for the teacher comparison |
| gpt-oss-120b | Apache-2.0 | No (o200k) | Sequence-level traces only, plus the ~120B **aspirational comparison target** |

## B3. Data sources (licenses verified; non-commercial sources excluded)
- Tool calling: glaive-function-calling-v2 (Apache), hermes-function-calling-v1 (Apache), ToolACE (Apache), When2Call (CC-BY-4.0, teaches when *not* to call a tool), xlam-60k (CC-BY-4.0, gated: needs acceptance).
- Reasoning: OpenMathReasoning and OpenCodeReasoning (CC-BY-4.0), filtered to **verified-correct** samples, plus teacher traces re-verified by execution.
- General SFT replay (forgetting control): tulu-3-sft-mixture (ODC-BY).
- **Excluded:** KodCode (CC-BY-NC).

## B4. Experiments, in order
| ID | Hypothesis | Intervention | Primary metric | Guardrail |
|---|---|---|---|---|
| B-TOOL-01 | Verified tool-use SFT raises BFCL without hurting general ability | LoRA r=64 on the base, then full FT if LoRA saturates; synthetic tool trajectories with execution feedback, error-recovery turns, and "no tool needed" negatives | BFCL overall and multi-turn | MMLU-Pro, GSM8K, IFEval within 1 SE of base |
| B-KD-01 | Same-tokenizer logit KD beats SFT on teacher text at equal tokens | top-k=64 KL from Qwen3.5-35B-A3B vs cross-entropy on the same samples | GSM8K, MATH-500, HumanEval | Same guardrails |
| B-TEACH-01 | Teacher choice matters | 35B-A3B vs 27B vs gpt-oss-120b traces (verified only) | Same | |
| B-RLVR-01 | Verifiable-reward RL (GRPO) adds reasoning and tool reliability beyond SFT/KD | Rewards: unit tests, exact math answers, schema-valid and executed tool calls | Held-out math, code, BFCL | Length and verbosity tracked |
| B-VOCAB-01 | The 248k vocabulary (~0.5B params) can be pruned to frequently used tokens with no measurable loss, freeing budget | Frequency-based prune and remap, then short healing | Perplexity and full suite | Multilingual (MGSM) |
| B-XFER-01 | Track A winners transfer | e.g. recurrent-depth core added to the base via progressive adaptation and distillation | Full suite at matched inference FLOPs | Original-capability retention |

## B5. Reporting
Three leaderboards (Intelligence / Reliability / Efficiency). Every entry records model and revision, total, active, and embedding parameters, think mode, tool access, context length, precision, and the harness version. NanoLM-B variants are compared against their own base model and all reproduced competitors.
