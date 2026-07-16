# nano-lm — a language model pretrained and instruction-tuned from scratch on a MacBook

A 3.15M-parameter decoder-only transformer built end-to-end: data pipeline, tokenizer,
pretraining, and supervised fine-tuning (SFT) — all trained locally on Apple Silicon (MPS),
no cloud GPUs. Total pretraining wall-clock: **20 minutes**.

The point is not the model's size — it's exercising the *entire* modern LLM build stack
at a scale where every decision is legible and auditable.

## Purpose

nano-lm is a **testbed for evaluation-gated model development**: every capability the
model gains must pass a pre-registered behavioral gate (held-out prompts, base-model
control, honest failure reporting) before the next stage begins. At this scale a full
train-evaluate-gate-iterate cycle takes minutes, so evaluation *design* — what to
measure, how to prevent contamination, when a pass is real — can be prototyped and
stress-tested far faster than at production scale. The end goal is a miniature
**ambient-scribe** task: converting short dialogue transcripts into structured summaries,
gated on *faithfulness* (no content in the summary that isn't grounded in the dialogue) —
the same evaluation problem that production clinical documentation AI faces.

## Progression

| Stage | Status | Gate result |
|---|---|---|
| Pretrain (FineWeb, Chinchilla-scaled) | ✅ done | val loss 3.96, zero spikes |
| SFT v1 (SmolTalk, ChatML) | ✅ done | ❌ honest FAIL (72% stop, 33% refusal) |
| SFT v2 (diversified refusal slice, +format) | ✅ done | ✅ PASS (98% stop, 92% held-out refusal) |
| Preference pairs (best-of-n + rubric judge) | ✅ done | 163 pairs, margin ≥ 0.6 |
| DPO (β=0.1) | ✅ done | ✅ PASS (win-rate 80.6% vs SFT, 95% CI [75.6, 85.6]) |
| RLVR/GRPO (verifiable-reward slice) | ✅ done | ✅ PASS (pass@1 12.5% → 85.9%, Δ CI [+61.7, +85.2]) |
| Scribe v1 (faithfulness gate, pre-registered) | ✅ done | ❌ honest FAIL (recall 74%, halluc 14%) — diagnosis in audit |
| Scribe v2 (pre-specified diversity sweep) | ✅ done | ❌ FAIL by 1.5 pts (recall 81% ✅, halluc 11.5% vs ≤10%) — **stage closed per protocol** |
| Stage G: grounding-verifier guardrail | ✅ done | ❌ FAIL by 0.8 pts — but **0.0% residual hallucination** (23/23 caught, 14% review load); miss = unverifiable absence claims |
| Absence-verifier axis / copy-curriculum / scale test | 🔜 planned | each requires fresh pre-registration |

The scribe stages closing at FAIL is deliberate: each protocol allowed one measurement
(plus, for v1→v2, one pre-specified sweep) — running "one more try" against seen results
would be bar-chasing. The cumulative arc is the finding: model hallucination 14.0% (v1)
→ 11.5% (v2, training-side diversity) → **0.0% as presented** (Stage G, verification-side
guardrail, 100% catch at 14% review load). Hallucination in high-stakes drafting is a
*systems* problem — training reduces it; verification architecture removes it from the
output at a measured human-review cost. Stage G's residual risk moved from fabrication
to omission (unverifiable `none` claims), a quieter failure mode needing its own gate
axis. Full trail: `scribe/AUDIT.md`.
| Over-refusal gate axis (XSTest-style) | 🔜 planned | known gap, documented in audit |

## Results

| Stage | Metric | Value |
|---|---|---|
| Pretrain | train loss | 8.35 (= ln V, init sanity) → 3.70 |
| Pretrain | val loss (held-out stream) | 3.96 |
| Pretrain | throughput | ~25k tok/s on MPS, zero loss spikes |
| SFT | masked loss (assistant tokens only) | 3.97 → ~1.5 (7.6 min) |
| SFT | throughput | ~53k tok/s |
| Behavioral gate (v2) | clean ChatML stop, sampled decoding | base 2% → **98%** |
| Behavioral gate (v2) | refusal on held-out unsafe phrasings | base 8% → **92%** |

## Architecture (`pretrain/train.py`)

- Decoder-only transformer, **3.15M params**, d=192, seq len 512
- RMSNorm (pre-norm), **SwiGLU** MLP (ff = 8/3·d), **RoPE** (base 1e4)
- **Grouped-query attention** 6q:2kv, tied input/output embeddings
- No dropout; init 0.02 depth-scaled; z-loss 1e-4

## Data & tokenizer

- Corpus: `HuggingFaceFW/fineweb` (sample-10BT), 12,000 docs streamed
- Filtering: Gopher/C4 heuristics; dedup via exact SHA-1 + MinHash (5-gram, 112 hashes, 14×8 bands)
- Tokenizer: byte-level BPE (HF `tokenizers`) + digit pre-tokenization, **V=4096**
  (sized by the 2·V·d embedding-budget rule); measured fertility 1.84 tok/word
- Tokenize-once → uint16 shards: **10.96M unique tokens**

## Compute budgeting

Chinchilla-style arithmetic drove the model size, not vibes:
10.96M unique tokens × ≤4-epoch cap → D≈33M tokens → **N≈3M params** (D≈20N).
Trained 4,000 steps / 32.8M tokens (~3.1 epochs). Optimizer: AdamW(0.9, 0.95),
peak LR 3e-3, linear warmup → cosine decay to 10% floor, grad clip 1.0, wd 0.1
on ≥2D params only.

## SFT (`sft/`)

- Chat format: ChatML (`<|im_start|>` / `<|im_end|>`); the pretrain tokenizer lacked
  role tokens, so embeddings were **resized 4096 → 4098** with new rows initialized
  from the `<|endoftext|>` embedding
- **Loss masking**: cross-entropy computed only on assistant completion tokens
- Data: `HuggingFaceTB/smoltalk`, capacity-adapted mixture (heavy code/LaTeX slices
  dropped at nano scale; refusal + format slices kept), 23,685 examples, 2 epochs
- `gate_sft.py`: post-SFT behavioral gate with **pre-registered pass/fail bars**, held-out
  refusal prompts (no train/test phrase overlap), and a base-model control — PASS requires
  the SFT model to clear the bars AND the base to fail them
- **v1 failed its bars honestly** (72% stop < 80%, 33% refusal < 66%); diagnosis traced the
  refusal miss to a low-diversity refusal slice. One pre-specified improvement sweep
  (refusal diversity 7→160 templates, +format slice, 3 epochs) → **v2 passed: 98% / 92%**.
  Full trail in `sft/AUDIT.md`, including a known limitation (mild over-refusal on benign
  prompts — the gate lacks an XSTest-style over-refusal axis)

## Methodology: guided-build audits

Each stage was executed strictly from my own research-vault recipes, with an audit log
(`pretrain/AUDIT.md`, `sft/AUDIT.md`) recording every decision's source — and every
**STALL**: a point where the documentation was insufficient and outside knowledge was
required. Each stall becomes a documentation fix. The build validates the knowledge
base as much as the knowledge base drives the build.

## Reproduce

```bash
pip install torch tokenizers datasets
cd pretrain && python train.py     # ~20 min on Apple Silicon
python generate.py                 # sample from the checkpoint
cd ../sft && python build_sft_data.py && python sft.py
```

Checkpoints and tokenized shards are excluded from the repo (see `.gitignore`);
the trained checkpoints are attached as release assets.
