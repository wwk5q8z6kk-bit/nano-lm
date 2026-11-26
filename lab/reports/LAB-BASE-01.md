# LAB-BASE-01 — Qwen2.5-1.5B-Instruct ("the 1.54B") log-likelihood baseline  [MEASURED]

- Date: 2026-11-17 · Harness: lm_eval 0.4.12, HF backend, bf16, Apple M3 Max MPS, batch 4, seed 1234
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (1,543,714,304 params, Apache-2.0), no chat template, 0-shot
- **Subsampled: `--limit 400` per task/subtask** (MMLU = 57 subtasks × ≤400 → 13.3k items ≈ full; others 400)
- Raw outputs + per-sample logs: `lab/baselines/results/qwen25_1p5b_instruct_loglik/`

| Task | Metric | Score | ± SE |
|---|---|---:|---:|
| MMLU (all) | acc | **62.00** | 0.43 |
| — humanities / other / social / STEM | acc | 61.5 / 63.1 / 70.7 / 54.0 | ~0.9 |
| ARC-Challenge | acc_norm | 43.75 | 2.48 |
| ARC-Easy | acc_norm | 76.25 | 2.13 |
| HellaSwag | acc_norm | 60.00 | 2.45 |
| PIQA | acc_norm | 77.75 | 2.08 |
| WinoGrande | acc | 62.50 | 2.42 |

Notes: scores sit in the publicly reported range for this model family, which is evidence the harness
is functioning, not a reproduction claim (publisher numbers use different shots/prompts). The full-set,
GPU, multi-model comparison is LAB-BASE-03; generative tasks (GSM8K, IFEval, HumanEval, MBPP) for this
model are running locally as LAB-BASE-02.
