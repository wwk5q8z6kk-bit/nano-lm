# Stage S — ~10M-parameter scale test (2026-07-17, fresh pre-registration)

## Hypothesis under test

The scribe track's surviving explanation for out-of-distribution hallucination is
CAPACITY: at 3.15M params the model cannot do reliable content-addressed copying
(Stage C refuted the curriculum explanation — the 22-pt held-out-value gap was
unmoved by an unmemorizable-value training slice). Stage S tests capacity directly:
same tokenizer, same scribe data recipe, same eval set, same bars — ~3.2× the
parameters, Chinchilla-rescaled pretraining.

## Build (fixed before training)

- **Model**: d=320, L=8, H=8, KV=2 (GQA 8q:2kv, hd=40), SwiGLU ff=864, RoPE,
  RMSNorm pre-norm, tied embeddings, V=4098, S=512 → **≈10M params**.
- **Pretrain**: FineWeb sample-10BT streamed (~240k docs), same 4096-BPE tokenizer
  (downloaded from the repo — no retraining), D≈20N → ~200M tokens, one epoch-ish.
  AdamW(0.9,0.95), peak LR 1.8e-3 (width-scaled from nano's 3e-3 × 192/320),
  warmup 3%, cosine → 10% floor, wd 0.1 on ≥2D params, clip 1.0, z-loss 1e-4,
  batch 32×512. fp16 autocast + GradScaler on CUDA (T4 has no bf16 tensor cores).
  Checkpoint every 1000 steps, resume-able (free-tier sessions can die).
- **Scribe finetune**: the v2 data recipe verbatim (diverse templates + compositional
  values, NO gibberish slice — Stage C showed it converts omissions to fabrications),
  12000 examples, 3 epochs, LR 1e-4.
- **Venue**: Kaggle free T4/P100 (background execution); Colab free as fallback.

## Acknowledged confound (stated in advance)

The nano scribe lineage was pretrain→SFT→DPO→scribe; Stage S is pretrain→scribe
(no chat SFT, no DPO). Rationale: chat/refusal shaping is orthogonal to the
field-extraction skill the gate measures, and the scribe finetune re-formats output
behavior entirely. Risk accepted and logged: if Stage S fails in a way that smells
like formatting (parse rate), the missing SFT stage is the first suspect.

## Pre-registered evaluation (ONE run, no post-hoc tuning)

Same 40-dialogue eval set, byte-identical (downloaded from the repo), greedy primary.

Primary bars (the original scribe faithfulness bars — never yet passed by a model):
1. parse ≥ 90%   2. recall ≥ 80%   3. hallucination ≤ 10%
4. base control (the 10M pretrain ckpt, pre-scribe) must fail the bars

Hypothesis decision rule (report + interpretation, fixed now):
- held-out-value recall gap **< 10 pts** (nano-v2 baseline: 22 pts) → capacity
  explanation CONFIRMED — scale bought copying.
- gap **≥ 15 pts** → capacity explanation WEAKENED at this scale; next suspect is
  the objective/data (e.g., needs longer training or an extraction-specific head).
- 10–15 pts → inconclusive; report as such.

## Result
- (to be filled after the single Kaggle run)
