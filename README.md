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
| Stage A: absence-verifier axis | ✅ done | ✅ **PASS — presented precision 100%** (0.0% residual halluc AND omissions, 33/33 errors caught, 19% review load) |
| Stage C: copy-curriculum hypothesis test | ✅ done | ❌ FAIL — **hypothesis decided**: held-out gap unchanged (22 pts) → capacity, not curriculum; omissions (10→0) converted into fabrications (11.5→17.5%) |
| Stage S: ~10M scale test (Kaggle T4) | ✅ done | ✅ **GATE PASS — first model to clear the bars** (parse 100%, recall 88%, halluc 7.5%) — but held-out GAP 23 pts, unchanged → capacity hypothesis **weakened** |
| Over-refusal gate axis (XSTest-style) | 🔜 planned | known gap, documented in audit |

The scribe track's arc is the finding. Three stages closed at honest FAIL — each protocol
allowed one measurement (plus, for v1→v2, one pre-specified sweep), and "one more try"
against seen results would be bar-chasing. Each FAIL located the next lever: v1's
position-anchored extraction → template diversity; v2's out-of-distribution hallucination
→ verification architecture; Stage G's unverifiable absence claims → the lexicon axis.
Stage A then passed everything: with a drafting model whose intrinsic hallucination rate
is 11.5%, the two-axis verification layer yields an output channel where **every presented
field is correct (100% precision), every model error is routed to human review, at 19%
review cost**. Trust came from verification architecture, not model scale — and the git
history proves every bar preceded every result. Full trail: `scribe/AUDIT.md`.

Stage S (10M params, Kaggle T4) sharpened the conclusion: the larger model became the
first to PASS the model-side bars (halluc 7.5%) — yet its out-of-distribution gap didn't
move (23 pts vs 22 at 3M). A model can pass a well-designed average-case gate while
keeping its tail failure mode, which is exactly why the verification guardrail is not
retired by scale. Full trail: `scale/AUDIT.md`.

## Research track: why does a gated model still fail held-out copying? (`trajectory/`, `papers/`)

Stage S ended on a puzzle: a model can pass every model-side gate and keep a ~23-point
held-out gap. The research track turned that puzzle into a measured, pre-registered
program. Every number below traces to immutable per-run JSONs under `trajectory/`,
with pre-registrations frozen before results.

**Paper 1 — the failure mode and the instrument** (`papers/latex/paper1.pdf`, 13 pp,
manuscript — not yet submitted): the gap is **held-out value copying** — the model
copies field values it saw during finetuning but errs on held-out values under held-out
phrasings, even though both are verbatim in the input. Measured on a 5-instance
instrument (100 held + 100 seen prompts each): **18.3±1.3 pts** at 3.15M and
**18.7±1.5** at 10M. The failure localizes *entirely* to the three open-vocabulary
fields; the two closed-value fields sit at exactly zero — a built-in control showing
this is held-out-*value* copying, not generic degradation. Pythia (160M/410M/1B)
finetuned on the identical task reads single digits (3.5±0.7 at 160M); at 1B a
determinism cross-check revealed training-run nondeterminism dominates the residual,
so that rung is honestly reported as the interval [0,5] rather than a point.

**Paper 2 — the cause** (`papers/paper2_draft.md`, draft): the pre-registered
within-stack control — 160M params, same architecture family, tokenizer, and recipe —
reads **16.9±1.7**: flat across 50× of scale, while Pythia at the same parameter count
reads 3.5. The frozen decision rule fires **stack-dominant**: parameter count alone
does not close the gap. Two single-factor arms then each cut it by more than half —
LoRA instead of full finetuning (**7.1±1.2**) and Chinchilla-scale pretraining data,
200M→3.2B tokens (**7.0±1.0**) — near-identical effects, i.e. substitutes, not
independent factors. The factorial corner (3.2B tokens + LoRA) lands at **4.2±0.9**,
≈ Pythia level, and is behaviorally deterministic across training seeds (|Δ|=0.00).

**Mechanism probes — both closed at pre-registered verdicts:** C-1b (lexical
interference) — **REFUTED** (−4 pts against a ≤15 rule; 0/77 predicted substitutions).
C-3 (transition × boundary × length factorial, 93 held types) — H-transition
**REFUTED**, H-boundary **REFUTED**, H-length UNRESOLVED (noise-dominated at per-cell
n). The error census surfaced a genuinely new dominant failure mode: **morphological
re-inflection** (chiefly singular/plural suffix flips, ~44% of cell-type misses), not
truncation. Results are triple-cross-checked (kernel, from-scratch recompute,
independent harness — which surfaced and fixed a real bug in the harness itself) and
**replicated on a second GPU venue** with verdicts reproducing
(`trajectory/replications/`).

**Fabric — the verification layer** (`fabric/`): typed Claim / EvidenceSpan /
VerificationResult / Decision packets with content-addressed IDs; grounding,
provenance, and absence-never-from-silence gates enforced in code, not prompts.
Measured across all 24 model × verifier × instrument cells: presented-error rate
**18.4% → 0.0%** (3.15M) and **11.5% → 0.0%** (10M), zero correct claims lost, 100%
span provenance. (The v2 verifier is a rules-perfect reference extractor on this
synthetic task — caveat documented in `fabric/README.md`.)

Compute venues: local Apple Silicon (MPS), Kaggle T4, RunPod (A6000 / RTX 4090 /
H100 NVL); largest single run ≈ $37 (the 3.2B-token Chinchilla control). Program map:
`papers/RESEARCH_PROGRAM.md` · writing audit: `papers/writing_audit.md`.

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
