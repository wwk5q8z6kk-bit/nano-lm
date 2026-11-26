# PREREG TRACKB-B1-B2 — Verified Post-Training vs Same-Tokenizer Teacher Logit KD on Qwen3-1.7B (2.03B)

*Registered 2026-11-17 before execution.*

## 1. Scientific Question & Hypotheses
Can a `2.03B` Apache-2.0 foundation model (`Qwen/Qwen3-1.7B`) be substantially improved on fast (`think=0`)
tool calling, multi-turn API error recovery, strict arithmetic JSON extraction, and mathematical/coding
reliability **without degrading its general capabilities** (`MATH-500`, `GSM8K`, `HumanEval`, `IFEval`),
and does **Proper Top-K + Tail-Bucket Teacher Logit Distillation** from `Qwen/Qwen3-8B` (`8.19B`, Apache-2.0)
outperform standard verified SFT on the exact same training examples?

- **H-B1 (Verified Curriculum Effect):** `B1_verified_sft` (`alpha_kd = 0.0`, `rsLoRA r=32`) lifts 8-pillar
  `think=0` overall score over unmodified `Qwen/Qwen3-1.7B__think0` (`70.45%`) by `>= +8.0 pp` while keeping
  `humaneval_instruct`, `ifeval`, and `minerva_math500` within `1.5 SE` (or higher) of the unmodified base.
- **H-B2 (Proper Teacher Logit KD Effect):** `B2_teacher_kd_8b` (`alpha_kd = 0.6, T = 1.5, K = 64` + exact
  tail mass bucket from `Qwen/Qwen3-8B` on the identical 566 verified examples, seed, LR, and steps) beats
  `B1_verified_sft` on held-out public benchmarks (`GSM8K` / `MATH-500` / `HumanEval` / `IFEval` mean) or
  8-pillar composite by preserving richer dark-knowledge token distributions than hard 1-hot SFT targets.

## 2. Safeguards
- Base `Qwen/Qwen3-1.7B` weights are never mutated; merged checkpoints are written to isolated paths
  `/workspace/models/nanolm_b1_sft_2b` and `/workspace/models/nanolm_b2_kd8b_2b`.
- All 566 training trajectories pass deterministic execution/schema/math verifiers (`verification_failed = 0`)
  and 13-gram benchmark shield (`contaminated_dropped = 0`).

## 3. Compute & Cost
- Executed on existing warm A40 48GB pod (`2wjeaobs3gnszi`, `$0.59/h`).
- Estimated runtime: ~8 min (`B1` train) + ~14 min (`B2` teacher Top-64 logit cache + train) + ~20 min
  (full 8-pillar `think=0` & `think=1` + full `GSM8K`, `MATH-500`, `HumanEval`, `IFEval` evaluation on both
  `B1` and `B2`) = **~42 minutes (`~$0.42` total cost)**.
- Stopping rule: abort if training loss becomes non-finite or if `B1` degrades `IFEval` > 5 pp (indicating LR too high).
