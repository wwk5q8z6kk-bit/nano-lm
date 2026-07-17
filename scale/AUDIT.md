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

## Result — measured once on Kaggle T4, STAGE S GATE PASS; hypothesis verdict: WEAKENED

Run notes: first attempt on P100 failed fast (sm_60 unsupported by Kaggle's PyTorch —
capability guard added to the script from the live lesson). T4 run: 81k tok/s, 200M
tokens / 12,207 steps in ~75 min, val loss 8.49 → **3.284** (nano: 3.96). Scribe
finetune 3.3 min.

**Gate (greedy, same 40-dialogue eval set):**
- parse 40/40 = **100%** ✅  recall 177/200 = **88%** ✅ (bar 80)  halluc 15/200 = **7.5%** ✅ (bar ≤10)
- omissions 8; base control (10M pretrain): parse 0% — discrimination clean
- **First MODEL to pass the pre-registered faithfulness bars** (nano v1 72/14.0, v2 81/11.5)

**Hypothesis decision rule, applied as pre-registered:**
- held-out-value recall 77% vs seen **100%** → **GAP 23 pts** (nano-v2 baseline: 22)
- Rule said: <10 pts = capacity CONFIRMED; ≥15 pts = capacity WEAKENED → **WEAKENED.**

**Interpretation (the finding of the whole track):** 3.2× parameters bought complete
on-distribution mastery (seen-value recall 94% → 100%) and enough overall hallucination
reduction to pass the bars — but did NOT move the out-of-distribution gap at all
(22 → 23 pts). Combined with Stage C (copy-curriculum refuted at 3M), the OOD copying
failure is now robust to BOTH interventions tested: curriculum and modest scale. Next
suspects are much larger scale, an architectural copy mechanism (pointer/induction
capacity), or the objective itself.

**Systems conclusion, strengthened:** a model can PASS a well-designed average-case
gate while retaining its tail failure mode — the 10M scribe hallucinates on exactly
the inputs that leave its training distribution, same as the 3M one, just less often.
This is why the Stage G/A verification layer is not retired by scale: at any scale
tested here, trust in the presented output still comes from verification architecture.
The gates passed; the guardrail stays.

## Artifacts

Checkpoints published as v0.1 release assets
(https://github.com/wwk5q8z6kk-bit/nano-lm/releases/tag/v0.1):

- `scale10m_pretrain.pt` (120 MB) — 10M pretrain, val loss 3.284
  sha256 `892180f02d09cacd2d129ba041dcbcca7635594bc98d8402203af64afc2fc88d`
- `scale10m_scribe.pt` (40 MB) — scribe finetune, the gate-passing model
  sha256 `f5aca5f04bd1045cc158d46a27b84024bb94baa349ed330933631c8b8d5acf0d`
