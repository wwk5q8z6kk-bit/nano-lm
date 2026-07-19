# Scale doesn't buy copying: a within-stack control isolates the training stack in a held-out value-copying failure

*Working draft — Paper 2 (causality). Companion and sequel to Paper 1 ("Held-out value
copying in small language models"). All numbers trace to immutable JSONs under
`trajectory/`; instrument identical to Paper 1 (5×(100 held + 100 seen), gap = seen −
held recall, mean ± across-instance SD). Status: skeleton with the full-FT arm landed;
LoRA arm running — its cell is marked ⏳.*

## Abstract (draft)

Paper 1 measured a severe held-out value-copying failure in small from-scratch language
models (~18 points diluted, ~80–87 points on the value-level metric at 3–10M) that is
far smaller under the Pythia pipeline (3.5–4.2 points at 160M–410M) — but the comparison
confounded parameter count with the entire training stack. Here we run the pre-registered
within-stack control: a **160M-parameter model in the same architecture family, tokenizer,
pretraining recipe (~200M FineWeb tokens), and finetuning method** as the small anchors,
measured on the identical instrument. The gap **does not close: 16.9 ± 1.7** diluted
(66.6 ± 5.0 value-level) — statistically indistinguishable from the 3.15M anchor. Across
**50× of within-stack scale** (3.15M → 10M → 160M) the copying gap is flat
(18.3 → 18.7 → 16.9), while Pythia at the *same parameter count* reads 3.5. The
pre-registered decision rule fires **stack-dominant**: parameter count alone does not
produce the collapse; properties of the training stack do. A second arm holds the
checkpoint fixed and swaps only the finetuning method (full FT → LoRA r=16):
⏳ *LoRA-arm result pending — completes the 2×2*. Per-slot structure sharpens the
attribution: at every scale, in both stacks, the failure concentrates in low-diversity
slots (the 5-value allergy slot fails totally even in Pythia-410M), consistent with a
per-slot competition between memorization and copying that capacity alone does not settle.

## 1. The question Paper 1 left open

Paper 1's primary limitation was explicit: the nano→Pythia comparison changes parameter
count *and* pretraining data (~1500×), tokenizer (4098 vs ~50k vocab), architecture
family, and finetuning method (full FT vs LoRA) simultaneously. Its one controlled
within-stack step (3.15M → 10M, 3.2×) moved nothing. The open question: **is the collapse
a scale effect or a stack effect?**

## 2. Design (pre-registered)

`trajectory/PREREG_ownstack_160m.md`, fixed before any run, including the decision rule:
diluted gap ≥ 14 → STACK-dominant; ≤ 6 → scale-plausible within family; 6–14 → extend the
ladder (40M/80M).

- **Model:** own-stack GPT family (RoPE, GQA 4:1, SwiGLU, RMSNorm, tied embeddings,
  4098-vocab BPE, 512 ctx) at d=1024, L=14, H=16, KV=4, hd=64, ff=2752 → **159.3M**.
- **Held identical to the anchors:** pretraining recipe (~200M FineWeb tokens, D≈20N for
  the anchors; deliberately *not* rescaled — the "identical recipe" comparison), scribe
  finetune (v2 data, byte-identical generator, full FT, 3 epochs, LR 1e-4), scorer,
  eval instances (m0–m4 + inst0), both metrics (diluted + clean).
- **Deviation (pre-authorized):** effective batch 32 realized as micro-8 × accum-4
  (T4 memory); optimizer trajectory unchanged.
- **Method arm (2×2):** the same pretrained 160M checkpoint finetuned with LoRA r=16
  α=32 (98 wrapped modules, 4.03M trainables) instead of full FT — isolating the
  finetuning-method member of the stack bundle. ⏳ running.

## 3. Results

### 3.1 The within-stack curve is flat across 50×

| model | params | diluted gap | clean (value-level) gap |
|---|---|---|---|
| nano | 3.15M | 18.3 ± 1.3 | 87.3 ± 2.7 |
| scale | 10M | 18.7 ± 1.5 | 79.5 ± 2.1 |
| **own-160M (full FT)** | **159.3M** | **16.9 ± 1.7** | **66.6 ± 5.0** |
| pythia-160m (reference) | 162M | 3.5 ± 0.7 | 14.7 ± 2.1 |

Model quality scales normally — pretrain val loss 2.86 (vs 3.28 at 10M, 3.96 at 3.15M),
scribe parse 100%, base control parse 0% — yet the copying gap barely moves. The
pre-registered rule fires **stack-dominant**. The clean metric declines modestly with
scale (87.3 → 79.5 → 66.6) — capacity helps at the margin — but at 160M own-stack the
pure held-out-value failure is still **4.5×** the Pythia-160M value; per-field, the
allergy slot remains at **100.0** (clean) at all three own-stack scales.

### 3.2 The 2×2 method control ⏳

| | full FT | LoRA r=16 |
|---|---|---|
| own-stack 160M | 16.9 ± 1.7 | ⏳ |
| pythia-160m | (not run) | 3.5 ± 0.7 |

Reading (pre-specified): LoRA ≈ full-FT → the finetuning method is innocent and the
remaining stack bundle (pretraining breadth/content, tokenizer) carries the effect;
LoRA ≪ full-FT → parameter-efficient adaptation itself suppresses the memorization
pathway and is part of the explanation.

### 3.3 Familiar structure at the new rung

The public instance is again the hard draw (inst0 28.0 vs fresh 16.9 — now 6/6 rungs),
and the per-slot pattern replicates (clean: cc 73.1, med 51.7, alg 100.0 at m4) — the
slot-diversity gradient (~190/18/5 training values) shapes the failure identically at
159M as at 3M.

## 4. What this establishes, and what it doesn't

**Established:** within this stack and recipe, parameter count from 3.15M to 159M does
not produce the low-gap regime that the Pythia pipeline exhibits at the same size. The
collapse Paper 1 measured is a **stack effect**, not a scale effect, under the tested
conditions. Combined with the slot structure, the natural reading is that copying is
induced by *pressure* (slot diversity) and *pretraining breadth*, not by capacity alone.

**Not established:** *which* stack member carries the effect. The bundle still contains
pretraining data quantity/breadth (~200M vs ~300B tokens — and our 160M is ~16× under
compute-optimal; a Chinchilla-scaled control run is the designed follow-up), tokenizer
granularity, and architecture-family details; the LoRA arm ⏳ isolates the finetuning
method. Single training run at 160M (no duplicate-finetune variance probe yet).

## 5. Next decompositions (designed, not run)

1. **Chinchilla control** — same 160M, TARGET_TOKENS ≈ 3.2B: is it data *quantity*?
2. **Tokenizer swap** — own-stack with a ~50k vocab: is it value fragmentation?
3. **Slot-diversity intervention** — vary the allergy slot's training diversity at fixed
   scale: the direct test of the per-slot hypothesis (also Paper 1 §8's prediction).
4. **Duplicate 160M finetune** — training-run variance at the new rung.

---
*All artifacts: `results_ownstack_v2_160m_fullft.json` (immutable), kernels
`kaggle_ownstack_160m{,_lora}.py`, decision rule in `PREREG_ownstack_160m.md`.*
