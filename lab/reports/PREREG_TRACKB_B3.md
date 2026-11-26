# PREREG TRACKB-B3 — Verified Dual-Mode On-Policy Teacher Distillation (`Qwen3-8B -> Qwen3-1.7B`)

*Registered 2026-11-17 after diagnosing B1/B2 format-shift vs dark-knowledge trade-off.*

## 1. Empirical Diagnosis from B1 & B2
In `PREREG_TRACKB_B1_B2`:
- **`B2_teacher_kd_8b` (`alpha_kd=0.6`) decisively beat `B1_verified_sft` (`alpha_kd=0.0`) on almost every metric:**
  - 8-pillar `think=0` Intelligence: `0.6937 (B1) -> 0.7958 (B2)` (vs `0.7563` Base `Qwen3-1.7B`)
  - 8-pillar `think=0` Reliability: `0.9167 (B1) -> 0.9479 (B2)` (vs `0.7604` Base `Qwen3-1.7B`, lifting `structured_json` `25.0% -> 87.5%` and `tool_calling` `71.9% -> 81.2%`)
  - `MATH-500`: `41.0% (B1) -> 56.0% (B2)` (`+15.0 pp` KD rescue)
  - `GSM8K`: `55.6% (B1) -> 68.0% (B2)` (`+12.4 pp` KD rescue)
  - `HumanEval`: `50.6% (B1) -> 59.2% (B2)` (`+8.6 pp` KD rescue)
- **Root Cause of Remaining Gap vs Base on Public CoT/Code/IFEval (`MATH-500` `70.0%`, `HumanEval` `66.5%`, `IFEval` `66.9%`) and `think=1` (`0.750 -> 0.427`):**
  1. `B1`/`B2` trained on **synthetic off-policy target strings** (`FINAL_ANSWER:` instead of natural `\boxed{}`, repeated 6-function Python snippets, zero instruction-constraint prompts).
  2. `B1`/`B2` encoded 100% of examples with `enable_thinking=False`, degrading `</think>` termination when evaluated in `think=1` mode.
  3. Two epochs over repeated templates at `lr=1.5e-4, rank=32` over-adapted output formatting.

## 2. Registered Design for `B3` (`nanolm_b3_onpolicy_kd8b_2b`)
1. **On-Policy Teacher Trajectory Generation (`Qwen/Qwen3-8B`, Apache-2.0):**
   - Generate natural `Qwen/Qwen3-8B` completions across ~620 **unique, non-repeating** prompts spanning:
     - Disjoint single/parallel tool calling, `When2Call` abstention, and multi-turn API error recovery (6 disjoint domains),
     - Strict arithmetic-in-JSON normalization,
     - Unique algorithmic Python problems (`mbpp` `train` split [CC-BY-4.0] + disjoint algorithmic specs, unit-test verified),
     - `openai/gsm8k` `train` split [MIT] keeping natural `\boxed{}` LaTeX step-by-step derivations,
     - Deterministic instruction-constraint prompts (bullet counts, forbidden words, JSON-only, `<<title>>`, postscripts, Mermaid syntax),
     - Grounded epistemic abstention & faithful numeric summarization.
   - **Dual-Mode Split (`70%` `think=0` + `30%` bounded `think=1` `<= 896` tokens):** explicitly preserves and sharpens both fast non-thinking inference AND concise, bounded `<think>...</think>` reasoning that terminates cleanly before token limits.
2. **Deterministic Verifier + 13-gram Decontamination Gate:**
   - Every `Qwen3-8B` completion must pass its domain verifier (tool JSON/schema/abstention check, isolated Python `assert` execution, exact math `\boxed{}` check, exact JSON arithmetic/schema check, or deterministic IFEval-style constraint check) AND have zero 13-gram overlap with our evaluation suite. Teacher mistakes are discarded before distillation.
3. **Gentle High-Weight Proper Top-64+Tail Logit KD:**
   - `rsLoRA (rank=16, alpha=16)`, `alpha_kd = 0.85`, `temperature = 1.5`, `topk = 64` + exact remainder tail bucket, `lr = 5e-5`, `epochs = 1`.

## 3. Preregistered Success & Falsification Criteria
- **H-B3a (Fast-Mode Gain Retained):** `nanolm_b3_onpolicy_kd8b_2b` (`think=0`) improves over Base `Qwen3-1.7B` (`think=0`) on BOTH Leaderboard A (`> 0.7563`) and Leaderboard B (`> 0.85`, vs `0.7604` Base).
- **H-B3b (Capability & Dual-Mode Preservation):** Eliminates the catastrophic forgetting seen in `B1`/`B2`:
  - `pillars8_think1` Intelligence recovers from `0.4271 (B2)` to `>= 0.750` (matching or beating Base `0.750`),
  - Public `MATH-500`, `GSM8K`, `HumanEval`, and `IFEval` recover substantially over `B2` (`56.0% / 68.0% / 59.2% / 56.6%`) and stay within `2.0 pp` (or higher) of unmodified Base `Qwen3-1.7B`.
- **Compute Budget:** Warm A40 pod `2wjeaobs3gnszi` (`$0.59/h`), ~28 minutes (~`$0.28`), followed by immediate pod teardown upon result sync.
