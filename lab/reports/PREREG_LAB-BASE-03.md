# PREREG LAB-BASE-03 — Track B candidate baselines (registered 2026-11-17 before spend)

**Question.** Which license-eligible ≤4B checkpoint is the strongest starting point for NanoLM Track B,
measured in one harness under identical conditions?

**Candidates.** Qwen3.5-2B, Qwen3-1.7B, SmolLM3-3B, granite-4.0-h-micro, granite-4.0-micro,
Phi-4-mini-instruct, Qwen2.5-1.5B-Instruct (reference). Instruct/chat variants, thinking disabled
(recorded as `think=0`); a thinking-on pass is a separate, labelled condition.

**Protocol.** `lab/trackb/eval_pod.sh` — vLLM bf16, greedy, max_model_len 8192, full test sets, pinned
model revision SHA, lm_eval version recorded. Scores are reported with bootstrap standard errors from
lm_eval. No candidate's published numbers are used in rankings unless reproduced here.

**Selection rule.** Rank on the unweighted mean of normalized scores across: knowledge (MMLU), hard
reasoning (MMLU-Pro, GPQA-D, BBH), math (GSM8K, MATH-500), code (HumanEval, MBPP+), instruction (IFEval),
multilingual (MGSM). Tool-calling (BFCL) and long-context (RULER) are added in LAB-BASE-04 before final
selection. Ties within 1 SE → prefer fewer total params and the more permissive data provenance.

**Compute & cost.** One A40 48GB (secure, \$0.59/h). Pilot = Qwen3.5-2B alone to measure wall time;
projected total must stay < \$15 or the suite is trimmed. Pod removed when the queue empties.

**Stopping.** Any task failing twice is recorded as FAILED with its log, not silently dropped.
