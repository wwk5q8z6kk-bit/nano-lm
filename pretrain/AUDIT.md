# Guided-build execution audit — Build Path → nano pretrain (2026-07-15)
Rule: every step follows ONLY vault instructions; every outside-the-vault reach = STALL logged.

## Stage 0 · Orientation (Build Path)
- Vault: "set up Python + PyTorch; run a notebook on free GPU"
- Did: local env probe — py3.14.5, torch 2.11 (MPS), tokenizers, datasets. CLEAN.

## Stage 5 · Data (Build Path → Recipe ⑦ §1-5, pretraining-corpora)
- Acquisition: `HuggingFaceFW/fineweb` sample-10BT streamed by the exact HF ID the vault now lists — 12,000 docs / 36.9M chars in 8s. CLEAN (iteration-12 fix load-bearing).
- Heuristics (Gopher/C4 per §3): 96% kept. Exact SHA-1 + MinHash 5-gram/112-hash/14×8 (§4 constants): 0 dups.
  → Both near-no-ops BECAUSE FineWeb is pre-filtered/pre-deduped — validates §11's hobby-tier advice ("mostly reuse FineWeb/DCLM"). CLEAN.
- Decontamination (§5): method fully specified (13-gram + canaries); no eval suite exists at nano scale → executed as documented no-op. CLEAN.
- OBSERVATION (not a stall): attrition funnel percentages in §0 describe RAW CommonCrawl; a FineWeb-start builder sees ~4% attrition. §11 implies this; §0 could say it once.

## Stage 4 · Tokenizer (Recipe ⑦ §9, tokenizer-training)
- HF `tokenizers` byte-level BPE + Digits pre-tokenizer, V=4096 (sized by the vault's own 2·V·d embedding-budget formula — frontier 100k-256k table N/A at nano; formula carried the decision). Fertility measured per instruction: 1.84 tok/word. CLEAN.
- Shard: tokenize-once → uint16 binary (Recipe ⑦ §10 / Recipe ② §1.4). 10.96M tokens. CLEAN.

## Stage 6 · Model + run sizing (Recipe ① components, Recipe ② §3/§6)
- Budget arithmetic drove everything: U=10.96M unique tokens × ≤4-epoch cap (§1.4) → D≈33M → N≈3M params (Chinchilla D≈20N loose at nano; data-constrained accepted per vault).
- Spec per Recipe ①: RMSNorm pre-norm, SwiGLU (ff=8/3·d), RoPE base 1e4, GQA 6q:2kv, tied embeddings, no dropout, init 0.02 depth-scaled. 3.15M params.
- Config per Recipe ② §3 scaled: AdamW(0.9,0.95) eps 1e-8, wd 0.1 on ≥2D params only, peak LR 3e-3, linear warmup → cosine to 10% floor, grad clip 1.0, z-loss 1e-4, seq 512, no batch ramp (nano).
- Sanity: step-1 loss 8.35 ≈ ln(4096) — uniform-init check passes.
- STALL-lite #1: bf16/fp8 precision guidance (§5) is CUDA-centric; nothing covers Apple-Silicon/MPS local training (Build Path routes to Colab instead). fp32 chosen by outside knowledge.
- STALL-lite #2: "deterministic, resumable data order" specified as a requirement everywhere, but no vault page shows the seeded-sampler/skip-to-step mechanic a from-scratch builder must write. Seeded RNG used from outside knowledge.
- Harness note: first launch died at step 1 (detached child reaped) — an operational lesson §7 covers conceptually ("assume failures; checkpoint; restart") and the restart worked from script, not checkpoint (step 1). CLEAN-ish.

## Run result (Stage 6 executed)
- 4000 steps / 32.8M tokens (~3.1 epochs) / 20.3 min on MPS @ ~25k tok/s
- loss 8.35 (=ln V, init sanity) -> 3.70 train / 3.95 val; grad-norm 0.36-0.48 flat; ZERO spikes; checkpoints at 1k/2k/3k/4k
- Generation test PASSES: fluent local English in web register, incoherent globally — expected at 3M params.

## VERDICT
Empty folder -> working LM in ~35 min wall-clock using ONLY vault instructions.
Stalls: 2 doc gaps (Apple-Silicon/local precision unaddressed; deterministic-resumable
data-order mechanic never shown) + 1 caveat (§0 funnel attrition assumes raw CC).
All three patched into the vault (iteration 16). The Build Path executes.
