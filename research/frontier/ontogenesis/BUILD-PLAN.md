# ONTOGENESIS build plan, 2026-11-20

Three workstreams: Build (a strong conventional foundation), Improve (changes that must beat it), Discover (open research). This file records engineering decisions and what they rest on. Verified = checked in this session; Unverified = from memory or a search summary.

## Decisions

| Question | Decision | Basis |
|---|---|---|
| Baseline architecture | Decoder-only Transformer: RMSNorm pre-norm, rotary positions, query-key normalisation, grouped-query attention, SwiGLU, tied embeddings | Already implemented and tested on master: `research/frontier/m21_dense_code/dense.py` (Verified) |
| Model B (recurrent) | The same file's loop option: prelude, core blocks applied r times, coda | Same code path, so the comparison isolates looping (Verified) |
| Model C (experimental) | Not chosen. The slot stays empty until a mechanism beats A and B at small scale | No candidate has earned it |
| Tokenizer | Existing 32k tokenizer | `/Volumes/Express1M2/nanolm-lab/data/v1/tokenizer_32k.json` (Verified present) |
| Corpus for stages 0–1 | Existing v1 corpus: 800M tokens (web 62%, textbook 13%, mathematics 13%, code 12%), with 1M-token validation sets per domain | Same path, `manifest.json` (Verified present) |
| Corpus for 125M and above | Needs building: 800M tokens is too little | See data section |
| Frameworks | PyTorch for laptop and Kaggle experiments; JAX for TPU runs. Port only what wins | TPU benchmark ran in JAX (Verified); MPS does not run JAX reliably |
| Comparison rule | Step-matched or time-matched on one device, same optimiser, three seeds, protocol committed first | Standing practice in this repository |
| Metric | Bits per byte per domain and on the mix; then grammar contrasts and the public-text set already on master | `experiments/grammar_contrast`, `experiments/native_public_language` |

## Compute, measured and estimated

- Laptop (Apple GPU, 64 GB): all experiments so far.
- TPU v6e, one chip, JAX, dense model, sequence 1,024: 110M parameters 176,662 tokens/s; 336M parameters 64,842 tokens/s (Verified, `tpu-probe/RESULT.md`). 16 chips per region available on demand.
- GPUs on Google: blocked, limit 0 (Verified again this evening).
- TPU hourly price: Unverified. The public price catalogue refused my request (HTTP 403). The figure below assumes about 2.70 USD per chip-hour from a search summary.

| Run | Tokens | Chip-hours | Approx. cost |
|---|---|---|---|
| 125M, one run | 2B | 3.1 | 8 USD |
| 125M, three arms × three seeds | 18B | 28 | 76 USD |
| 350M, one run | 5B | 21 | 58 USD |
| 2B, one run at a modest budget | 100B | about 2,500 (estimated by linear scaling) | about 6,800 USD |
| 2B at published budgets | 1T and more | about 25,000 and more | about 68,000 USD and more |

So the credits cover stage 2 (125M) fully and one or two 350M runs. A 2B model is a separate funding decision.

## Data

- Have: 800M tokens, four domains, deduplicated against validation (`skipped_overlap` recorded in the manifest).
- Stage 2 (125M, 2B tokens per run) can cycle the 800M corpus about 2.5 times; acceptable for an architecture comparison, not for a final model.
- Stage 3 and beyond need a corpus of at least 5–10B tokens with licence filtering, deduplication and contamination checks against every evaluation set. Not started.
- Long documents for a long-context measurement: not available locally.

## Stages and gates

| Stage | What | Where | Gate to the next |
|---|---|---|---|
| 0 | Synthetic probes: chains, retention, plasticity | Laptop | Used to choose candidates, not to claim results |
| 1 | 15M language models, A against B, three seeds | Laptop | B within 0.01 bits/byte of A or better at equal steps, and better on the chain probe |
| 2 | 125M, A, B and at most one C, three seeds | TPU, needs approval | A repeatable gain for B or C |
| 3 | 350M pilot of the winner against A | TPU | Gain holds |
| 4 | 1B–2B | Separate funding | — |

## Seven-day schedule against the plan

| Day | Item | State |
|---|---|---|
| 1 | Recommended baseline | Done: the existing dense model |
| 2 | Hardware and throughput | Done for TPU and laptop; price unverified |
| 3 | Data audit | Done for stages 0–1; larger corpus not started |
| 4 | Working baseline training run | Started: `lm_compare.py`, pilot running |
| 5 | Evaluation | Bits per byte in place; grammar and public-text harnesses exist on master and need wiring in |
| 6 | Recurrent comparison | Synthetic replication running under `PROTOCOL-LOOP.md`; language-model pilot running |
| 7 | Review | — |

## Rules adopted

The nine rules in the directive are adopted as written. Two additions from this session's mistakes: every arm gets its own learning-rate sweep before any comparison is reported, and a result is not reported as a difference between arms until it has three seeds.
