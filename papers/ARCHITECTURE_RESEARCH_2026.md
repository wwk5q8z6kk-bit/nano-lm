# Architecture Research 2026

*Generated 2026-08-05 by a 9-agent research swarm (4 external literature lenses, each mapped onto nano_ai code, then synthesized). Literature grades: [REP] replicated, [1P] single paper, [LM-ONLY] large-model-only, [UNVERIFIED] post-cutoff preprint — no gate is built on an [UNVERIFIED] claim.*


Synthesis of four external research lenses (attention, memory/long-context, reasoning, small-scale) mapped onto Nano's measured record. Every code claim carries a file path; every literature claim carries a source and a replication grade.

**Status of this document.** It is a research synthesis, not a preregistration. Nothing here authorizes a run. Items that require a run are written as preregistration *candidates* with frozen thresholds so they can be lifted into `papers/SCALE_PROGRAM_PREREG.md` or a new prereg without redesign.

---

## 0. Three corrections that reframe the brief

These came out of reading the code, and they change what the rest of the document can honestly say. They are stated first because every downstream section depends on them.

### 0.1 The trunk was never frozen. H2–H6 are full-parameter finetunes.

The framing that Nano's pointer/evidence-query heads were bolted onto a frozen trunk is **false**, and one of the four lenses asserted it. Verified:

- `nano_ai/training/train_evidence_query.py:298` — `_optimizer` partitions `model.parameters()` by dimensionality and hands **all** of them to AdamW at `PEAK_LEARNING_RATE` / `WEIGHT_DECAY`. No parameter group is excluded.
- `grep -rn requires_grad nano_ai/` returns hits only in `nano_ai/tests/test_train_pointer.py` (test fixtures). The only `requires_grad_(False)` calls in the repo are the frozen *reference* policies in `sft/grpo.py:39` and `sft/dpo.py:17`.
- Seven training/eval scripts record `"full_trunk_trainable": True` explicitly — `train_evidence_query.py:807`, `train_evidence_query_h4.py:970`, `train_evidence_query_h5.py:558`, `train_evidence_query_h6.py:610`, `train_pointer.py:633`, `evaluate_evidence_query.py:339`, `evaluate_pointer.py:426`.

So H2, H3, H4, H5 and H6 all initialized from the anchor and then **fully rewrote it**. This matters because `papers/paper2_draft.md:122-123` measures that exact regime as the worst cell of a 2×2, and every one of the five interventions sat inside it.

### 0.2 Held-out value copying is not at the floor, and the residual is one slot.

The brief states held-out value copying "fails." On the H-line instrument it does not. `papers/DECISION_GATES.md:109` records H5 held value **2,220/2,987 = 74.3%**, *passing* its frozen 2,167 requirement. H5 was rejected on absence (280/413 vs 383), conflict (149/250 vs 236) and uncertainty (162/250 vs 228) — `papers/DECISION_GATES.md:100-118`. On that instrument the binding constraint is **abstention calibration, not copying**.

On the paper2 instrument the failure is real but *localized*. `papers/paper2_draft.md:145-155`: at the best measured corner (160M / 3.2B tokens / LoRA r=16, diluted 4.2 ± 0.9, clean 17.7 ± 3.2) the per-slot fingerprint is **cc 0.0, med 0.0, alg 100.0** — identical to Pythia's. Under Chinchilla pretraining with full FT: cc 9.2, med 25.5, **alg 100.0**. The allergy slot reads clean 100.0 ± 0.0 in **all five own-stack configurations ever run**.

The honest restatement of the central open problem is therefore not "Nano cannot copy." It is: **cc and med copying was solved by data quantity and adaptation method; alg was never moved by either, at any scale, by any head.**

### 0.3 One intervention has moved the allergy slot, and it was never combined with the levers that solved cc and med.

`trajectory/PREREG_slot_diversity.md:98-103` — allergy-slot held-**type** recall as a function of finetune value-pool size, at fixed 10M scale, full-FT, weak base, with the position control innocent:

| arm | held-type recall |
|---|---:|
| D5 | 0.00 |
| D20 | 24.53 |
| D80 | **66.67** |

Diversity effect D80 − D5 = **66.7 pts**, decision **H-slot SUPPORTED** (≥30 pt threshold). Monotonic, position-controlled.

The cause of D5 is a single line: `scribe/build_scribe_data_v2.py:30`, `ALG_TRAIN = ["penicillin","peanuts","pollen","latex","shellfish"]`, held-out `sulfa drugs`; and line 86 puts `alg` in only ~50% of examples. That recipe is the finetune spine of **every cell** of the paper2 grid — including the 4.2 corner. The one lever that moves the residual was never crossed with the two levers that were bought at $37/run.

---

## 1. State of the art that matters to Nano

Grades used throughout: **[REP]** independently replicated · **[1P]** one strong paper, no independent replication found · **[CONTESTED]** primary sources disagree · **[LM-ONLY]** demonstrated only at ≫Nano scale · **[UNVERIFIED]** post-cutoff preprint, cited but not opened.

> **A methodological conflict between the lenses, stated openly.** The memory and small-scale lenses deliberately omitted every arXiv ID in the 2512/2601/2602/2604/2605/2606/2607 ranges as post-cutoff and unverifiable. The attention and reasoning lenses cited several. This document keeps those citations but marks them **[UNVERIFIED]** and **builds no gate on any of them**. Where an unverified 2026 preprint is the only support for a claim, the claim is demoted to "hypothesis, not evidence."

### 1.1 Attention / retrieval

**Load-bearing and replicated.**

- **Zoology** (Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré; arXiv 2312.04927, Dec 2023) **[REP]** — softmax attention solves multi-query associative recall at model dimension **64**, independent of sequence length; a 70M attention model beats a 1.4B Hyena on associative recall. Nano runs full softmax attention at **d=192** (`nano_ai/training/model.py`), 3× the width Zoology requires. *Scope limit, and both lenses that raised it agree:* MQAR is **single-token** key→value recall over a small synthetic vocab. It does **not** license a claim about **multi-token BPE span** copy at V=4098. It does license "capacity is not the binding constraint for in-context recall at this geometry."
- **Retrieval heads** (Wu, Wang, Xiao, Peng, Fu; arXiv 2404.15574, ICLR 2025) **[REP]** — retrieval heads are universal, **sparse (<5% of heads)**, **intrinsic to pretraining**, dynamically activated, and **causal**: masking them produces hallucination instead of copying. Nano has 36 heads (6×6); 5% is 1–2 heads — a thin detection margin.
- **Induction heads** (Olsson et al., Transformer Circuits, Mar 2022) **[REP]** — abrupt phase change, requires ≥2 layers via cross-layer head composition. Nano has 6. *The commonly-quoted "2.5–5B token" window for the bump is secondary-source only; the primary page says "early in training" without numbers. Do not cite the figure.*
- **Repeat After Me** (Jelassi, Brandfonbrener, Kakade, Malach; arXiv 2402.01032, ICML 2024) **[REP]** — constructive: layer 1 builds n-gram hashes, layer 2 matches and emits the successor; a **two-layer** transformer copies strings of exponential length, while state-space models are bounded by fixed latent state. **Depth is provably not Nano's obstacle.**
- **Attention sinks** (Barbero et al., arXiv 2504.02732, COLM 2025; "When Attention Sink Emerges," ICLR 2025) **[1P each, converging]** — sinks prevent over-mixing and **preserve token identity through depth**; a dedicated sink token in every pretraining sample improved convergence, validated at 160M. Token-identity preservation is the precondition both for span copy and for Nano's readout (`nano_ai/training/evidence_query_model.py:110` — `boundary_key = nn.Linear(width, 64, bias=False)`, a 64-d projection of the **final-layer** state at line 267).

**Nano-specific kills, by arithmetic on the code.**

- **p-RoPE** (Barbero et al., arXiv 2410.06205, DeepMind) — proposes truncating the lowest RoPE frequencies to create content-only channels. **No-op at Nano's geometry.** `nano_ai/training/model.py:191-196` uses base 10000 with head_dim=32, so `arange(0,32,2)/32` gives max exponent 0.9375, lowest inv_freq = 10000^-0.9375 = 1.778e-4 rad/token; over 511 positions the maximum accumulated rotation is **0.0909 rad ≈ 5.2°**. Five of 16 rotary pairs rotate under 90° across the entire context. Nano's bottom pairs are **already** effectively NoPE content channels. Two lenses derived this independently.
- **MLA** (DeepSeek-V2/V3) — the only small-scale study (Mehta et al., arXiv 2506.09342, 17.5M–202.7M) finds it **loss-neutral at best**: 9L-512d MHA 2.147 vs MLA+RoPE r=d/2 2.154, r=d/4 2.241, collapse near r=d/16, and reports **no copying or retrieval evaluation**. Nano's KV cache at ctx=512 / 3.15M params is trivial — nothing to buy. Compressing K/V is directionally wrong for retrieving exact novel token identities.
- **NSA / MoBA** (arXiv 2502.11089) **[LM-ONLY]** — all gains are long-context *compute* wins at 64k (11.6× decode). At ctx=512 on a laptop there is no compute problem and no recall gain on offer.
- **Sliding-window hybrids** (Gemma 3, gpt-oss) — Gemma 3's *local* window (1024) is **twice Nano's entire context**.
- **YaRN / NTK / positional interpolation / LongRoPE** — extend a model beyond its trained context. Nano trains at its full 512.

**Real but not yet applicable.**

- **DiffAttn** (Ye et al., arXiv 2410.05258, ICLR 2025 oral) **[LM-ONLY]** — verified claim is "~65% of model size or training tokens for comparable LM performance"; 6.8B DIFF ≈ 11B Transformer; needle tested at 4K and 64K only; **smallest from-scratch model is 830M**. *Two fabrications were caught and discarded during research and must not propagate: "70M matches 1B" and "4M-token needle" are both FALSE.* DiffAttn **halves head count** (6×32 → 3×64); with retrieval heads at <5% of heads, cutting a 6-head model to 3 is a live risk to the exact capability under repair. Follow-up (arXiv 2505.16333, NeurIPS 2025) attributes gains to negative-attention expressivity and reduced head redundancy, retrofittable post-hoc (DEX).
- **Gated attention** (Qwen, arXiv 2505.06708, NeurIPS 2025 Best Paper) **[LM-ONLY]** — head-specific sigmoid gate after SDPA; 30 variants, 1.7B dense + 15B MoE, 3.5T tokens; shipped in Qwen3-Next. **Zero evidence below 1.7B.** Unresolved tension with Barbero: that paper says sinks are load-bearing; Qwen removes them. At L=6 over-mixing is less acute, so the expected effect is smaller than the headline.
- **KV-shifting attention** (Xu, Cheng, Wang, Chen; arXiv 2411.19574) **[1P]** — shifts K and V by one position with learned coefficients, *provably reducing the depth and width required for the induction-head mechanism*. Evaluated from toy models to >10B, unusually broad scale coverage. **The single most mechanism-targeted architectural lever available**, at ~two learned scalars per head. No independent replication found.

**Attribution correction.** The widely-repeated "MQA degrades retrieval, GQA with 4–8 groups recovers it" is **blog folklore, not a result in Ainslie et al. (arXiv 2305.13245)**, which reports summarization and translation and predates needle-in-a-haystack. No rigorous ablation isolating KV-group count against retrieval was found at any scale.

**Unverified 2026 items, recorded but not built on:** Kawata et al. "From Shortcut to Induction Head" (arXiv 2512.18634) **[UNVERIFIED]**, whose max-sum-ratio prescription `q_ℓ ∝ ℓ` over inter-occurrence distances is the sharpest data-side mechanism proposed by any lens; Lavie et al. first-order copy-head transition (arXiv 2606.12058) **[UNVERIFIED]**; Lin et al. dynamic retrieval heads (arXiv 2602.11162) **[UNVERIFIED]**; Bayram RoPE/retrieval-head analysis (arXiv 2606.21249) **[UNVERIFIED, single-author]**.

### 1.2 Memory / long context

**The reframe that matters more than any technique.** Nano's copying failure occurs at **~130 tokens** inside a **512-token** window. It is not a long-context problem. Two thirds of the memory toolkit answers a question Nano does not have, and two of its techniques would make the central problem worse.

- **Jelassi et al. (2402.01032)** + **Bick et al. Gather-and-Aggregate** (arXiv 2504.18574, ICML 2025) **[REP]** — SSMs are fundamentally bounded by fixed latent state on copying; the gap localizes to a few G&A heads producing smooth attention where sharp transitions are required. **This is the single most valuable rejection in the survey**, because "local-first, small, long messy transcripts" makes Mamba/SSM look like the obvious answer precisely where the evidence says it makes the central problem worse.
- **Recurrent Memory Transformer** (Bulatov et al., arXiv 2207.06881) **[1P]** — removes the length limit but reintroduces the **fixed-size inter-segment state** Jelassi proves fatal for copying.
- **KV-cache compression/eviction** (StreamingLLM arXiv 2309.17453; H2O; SnapKV; benchmark at Findings of EMNLP 2024, 2024.findings-emnlp.266) **[REP]** — Wu et al. show compression breaks **precisely the retrieval heads that perform copy-paste**; eviction is irreversible. An active hazard for L5 streaming, not a lever now.
- **Repeated data** (Hernandez et al., arXiv 2205.10487, Anthropic) **[1P, strong]** — repeating 0.1% of data 100× degrades an 800M model to 400M-equivalent, with the disproportionate hit falling on **copying**, coinciding with disproportionate degradation of **induction heads**. Nano's anchor ran **~3.1 epochs over a 10.96M-token shard** (`papers/PRETRAINING_PROVENANCE.md:20-21`).
- **Context-extension recipes** **[LM-ONLY]** — ProLong (arXiv 2410.02660, ACL 2025) spends ~40B continued-training tokens from Llama-3-8B and calls that 5% of competitors' budgets; SmolLM2 (arXiv 2502.02737) needed a dedicated 2k→8k stage with 40% long documents; SmolLM3 needed two 50B-token stages. **The token budgets do not transfer.** What does transfer at near-zero cost: ProLong reports **document masking during attention improves both long- and short-context performance**.
- **Evaluation** — RULER (arXiv 2404.06654, COLM 2024) **[REP]**: single-needle NIAH tests "only a superficial form of long-context understanding." NoLiMa (arXiv 2502.05167, ICML 2025) **[1P]**: removing lexical overlap halves 12 models' scores by 32k; GPT-4o 99.3 → 69.7. Lost-in-the-middle (Liu et al., TACL 2023): U-shaped position bias, **worse at smaller scale**. **Nano's held-out-value instrument is already structurally closer to NoLiMa than to NIAH.** That is a methodological strength the papers should claim, not a gap to fill.
- **Just Read Twice** (arXiv 2407.05483) **[1P]** — a second pass closes part of the recall gap and presentation **order** changes selection difficulty. Developed for *recurrent* models whose deficit Nano does not have, so expect a smaller effect; but it is free to test.

**Tokens-per-parameter, the number the memory lens contributes.** Nano anchor **D/N ≈ 10** (32.8M tokens / 3.15M params, and only ~3.5 *unique* tokens per parameter); 10M variant D/N = 20; 160M at 3.2B tokens **D/N ≈ 20**. Compare Pythia-160m ≈ 1,875 · SmolLM2-135M ≈ 14,800 · Llama-3.2-1B ≈ 7,300 · Sardana & Frankle (arXiv 2401.00448, ICML 2024, 47 models) **[REP]** find quality still improving out to **10,000 tokens/param**. Nano's entire own stack sits **two to three orders of magnitude** below the modern small-model norm.

### 1.3 Reasoning / think-arm

**The settled shape** (DeepSeek-R1 arXiv 2501.12948; Qwen3 arXiv 2505.09388; Phi-4-Mini-Reasoning arXiv 2504.21233) **[REP]**: reasoning-heavy mid-training → long-CoT cold start → RLVR → optional preference stage → test-time controls.

**What breaks below ~1.5–3B — this is the decisive cluster.**

- **Distillation beats RL below ~7B** (DeepSeek-R1 §4.1, Table 6) **[REP]** — RL on Qwen-32B-Base >10K steps: AIME24 47.0 / MATH500 91.6; distilling R1 into the same base: 72.6 / 94.3. Verbatim: *"smaller models relying on the large-scale RL mentioned in this paper require enormous computational power and may not even achieve the performance of distillation."*
- **RL emergence floor** — TinyZero (public repo, widely reproduced, not peer-reviewed): Qwen2.5-0.5B **"fails to learn reasoning"** on Countdown; 3B+ develops it.
- **Long-CoT distillation actively hurts small students** — "Small Models Struggle to Learn from Strong Reasoners" (arXiv 2502.12143, Findings ACL 2025) names the **Small Model Learnability Gap**. Phi-4-Mini-Reasoning quantifies it: applying LIMO and s1K directly to 3.8B dropped MATH-500 **71.8% → 57.8% (LIMO) and → 47.0% (s1K)**. **The "1000 examples is all you need" result does not survive scale-down** — s1 (arXiv 2501.19393) was Qwen2.5-**32B**.
- **Most small-model RLVR wins are Qwen artifacts** — Spurious Rewards (arXiv 2506.10947) **[REP]**: **random** rewards give +21.4% on MATH-500 for Qwen2.5-Math-7B, *incorrect* labels +24.1%, vs +29.1% for ground truth — and **none transfers to Llama3 or OLMo2**. A from-scratch model on licensed non-math data has none of the latent behavior these results amplify.
- **Extend vs sharpen is [CONTESTED]** — Yue et al. (arXiv 2504.13837, NeurIPS 2025) show RLVR raises pass@1 but *lowers* pass@k at large k; NVIDIA ProRL (arXiv 2505.24864) counters at 1.5B with >2K RL steps + KL control. Do not build on either side.
- **The nearest published regime to Nano** — Zhang, Neubig & Yue (arXiv 2512.07783, ICML 2026 Spotlight) **[UNVERIFIED, post-cutoff]** run the full pipeline on **from-scratch 100M decoder-only models** on synthetic tasks with parseable traces. Reported: RL yields true pass@128 gains only when pretraining leaves headroom; RL enables contextual generalization only when primitives already exist, with **≥1% pretraining exposure sufficient**; mid-training beats RL alone at fixed compute; process-aware rewards mitigate reward hacking. *Because it is unverified, it motivates a hypothesis here and gates nothing.*

**Does CoT help *this* task? The strongest evidence says no.** "To CoT or not to CoT" (Sprague et al., arXiv 2409.12183, ICLR 2025) **[REP]** — 100+ papers, 20 datasets × 14 models: gains concentrate almost entirely on math/symbolic/logic; on MMLU CoT ≈ direct answering unless an "=" appears. Evidence-bound extraction sits squarely in the "CoT doesn't help" bucket. *Scope limit:* it measures **prompted** CoT on non-reasoning-trained models and predates RL-trained reasoners.

**The counterweight cuts toward Nano.** Li, Liu, Zhou & Ma (arXiv 2402.12875, ICLR 2024) **[1P, theoretical]** prove constant-depth finite-precision transformers are confined to TC⁰ without CoT, while T CoT steps let a constant-depth model simulate a size-T boolean circuit — **and empirical gains are largest for low-depth transformers**. Nano is L=6/L=8. Merrill & Sabharwal (arXiv 2310.07923, ICLR 2024) give the same result from the expressivity side. This argues for a scratchpad doing **serial work**, not imitated deliberation.

**On the copying problem specifically the answer is null.** No paper was found showing a think-arm or scratchpad fixes verbatim novel-span copying. The closest analogue is LLMQuoter (arXiv 2501.05554, LLaMA-3B + LoRA, quote-then-answer, >20 pts over RAFT) **[1P, 3B, one task]**.

**Risks a think-arm imports into an abstention-first product.**

- **Reasoning models hallucinate more.** OpenAI o3/o4-mini system card (2025-04-16): PersonQA hallucination **33% (o3) and 48% (o4-mini) vs 16% (o1)**. OpenAI's explanation: the models "make more claims overall." Direct threat to fabric v2's 0.0% presented-error identity.
- **Hybrid think/no-think in one checkpoint is known-failed design.** Qwen3 shipped thinking-mode fusion (arXiv 2505.09388) and the July-2025 "2507" refresh **abandoned it for separate Instruct and Thinking checkpoints**. At 160M capacity contention is worse — a fused ablation would measure contention and mislabel it "think-arm doesn't help."
- **Do not reward the trace.** Baker et al. (arXiv 2503.11926) — optimization pressure on the CoT teaches **obfuscated** reward hacking. Anthropic (arXiv 2505.05410) — CoT often omits the true causes of an answer. Both labs converge on: **reward the artifact, leave the trace unsupervised.** The `<think>` block is an engineering scratchpad, never an evidence trail.

**The constraint collision.** The most-replicated small-model reasoning win — distill verified long-CoT from a stronger reasoner — is the one Nano cannot straightforwardly take. Open-licensed R1-derived corpora exist and are usable (OpenR1-Math-220k Apache-2.0; DeepScaleR MIT; OpenMathReasoning CC-BY-4.0; R1 weights are MIT so its outputs are redistributable) but they are **math**, off-mission for a scribe. Generating *scribing* traces needs a frontier API, colliding with competing-model ToS. Three licensed paths remain: self-distillation / rejection sampling; distillation from an openly-licensed mid-size model run locally; or **deterministic programmatic trace construction** from the generator that already owns the ground-truth field/span assignments. The third is the best fit and costs no license exposure.

### 1.4 Small-scale (3M–1B)

- **Over-training past Chinchilla** **[REP]** — see §1.2. Nano's own 159M row replicates it internally: `papers/paper2_draft.md:122` — 16× more pretraining tokens at fixed params moved diluted **16.9 → 7.0**.
- **Data quality** **[REP with a caveat]** — FineWeb/FineWeb-Edu (arXiv 2406.17557, NeurIPS 2024 D&B): the educational classifier reached 33.6% MMLU at **38B tokens vs ~300B** for the next-best corpus. TinyStories (arXiv 2305.07759) is the direct sub-10M evidence that curation dominates as models shrink. **Contested branch:** the phi-style "textbooks" line (arXiv 2306.11644, 2309.05463) has documented contamination and HuggingFace's Cosmo-1B reproduction did not match — **do not adopt**. *Nano-specific counter-consideration:* educational filtering strips exactly the messy disfluent transcripts L4 targets, and may reduce the entity burstiness the retrieval hypothesis depends on. Mixture, not swap.
- **Curriculum learning** — **[REP NEGATIVE]**. BabyLM has run this at Nano's exact data scale three times (10M-word strict-small, 100M-word strict); Warstadt et al. (arXiv 2504.08165) and the 2024/2025 findings report curriculum submissions were the largest category and **largely unsuccessful**. Nano's own negative curriculum result is the expected outcome. **Do not re-litigate.**
- **Data-distributional drivers of ICL** — Chan et al. (arXiv 2205.05055, NeurIPS 2022) **[REP]**: ICL emerges only under **burstiness**, **many rare classes**, and dynamic item meanings; ICL and in-weights learning **trade off** unless the distribution is Zipfian-skewed. D5 is the exact opposite of "many rare classes."
- **ICL transience** — Singh, Chan, Moskovitz, Grant, Saxe, Hill (arXiv 2311.08360, NeurIPS 2023) **[1P, strong]**: ICL emerges and then **disappears** in favour of in-weights learning **while training loss keeps decreasing**. Follow-up (arXiv 2404.07129) decomposes induction-head formation into three interacting subcircuits with abrupt, data-dependent timing. Reddy (arXiv 2312.03002, ICLR 2024) shows formation is diversity-gated and abrupt.
- **Knowledge extractability is set at pretraining** — Allen-Zhu & Li (arXiv 2309.14316, ICML 2024, *Physics 3.1*) **[REP]**: without pretraining-time augmentation, knowledge is memorized but **not extractable — 0% accuracy regardless of subsequent instruction finetuning**. This is why a finetune-stage curriculum could not have worked.
- **LoRA learns less and forgets less** — Biderman et al. (arXiv 2405.09673, TMLR 2024) **[REP]**: LoRA underperforms full FT in-domain but preserves base capability better than weight decay or dropout. Nano's 159M cells (7.1 vs 16.9; 4.2 vs 7.0) are exactly that signature.
- **Error-correction data in pretraining** — Ye, Xu, Li, Allen-Zhu (arXiv 2408.16293, ICLR 2025, *Physics 2.2*) **[1P, methodologically closest]**: mistake→immediate-correction data placed in **pretraining** raises reasoning accuracy over the same volume of error-free data, by plain autoregression, no multi-round prompting. Ablates masking policy and required error rate. Run on GPT-2-class synthetic models trained from scratch.
- **Smallest demonstrated reliable structured extraction.** 304M — GLiNER-L (arXiv 2311.08526, NAACL 2024), a DeBERTa **encoder** doing span-extractive zero-shot NER that beats ChatGPT. 0.5B — NuExtract-tiny (Qwen1.5-0.5B). 0.36–0.5B — Christou & Tsoumakas (arXiv 2606.22606) **[UNVERIFIED]** report Qwen2.5-0.5B SFT at micro-F1 0.83 vs 0.69 for a zero-shot frontier model, with the explicit finding that gains came from **task-specific training, not scale**. Counter-anchor: in-context *retrieval* is weak even at 0.5B — needle accuracy 22.1% (Qwen1.5-0.5B) / 43.8% (Qwen2-0.5B) per the SLM survey (arXiv 2409.15790). **No published result was found for reliable structured extraction with citation at or below ~100M, and none near 3–10M. That gap is itself a finding**, and it means every architectural candidate below is an extrapolation into unmeasured territory — which is what preregistration is for.
- **Depth over width** — MobileLLM (arXiv 2402.14905, ICML 2024) **[1P]**: +2.7%/+4.3% at 125M/350M. Nano already has SwiGLU, GQA and tied embeddings. Worth single-digit percent versus ~10× from the token budget. Second-order.
- **Vocabulary scaling** — Tao et al. (arXiv 2407.13623, NeurIPS 2024): optimal vocab grows with compute, **smallest model 33M**. A 3.15M extrapolation is unlicensed. (For reference Nano's embedding is 4098×192 = 787K ≈ **25% of all parameters**.)
- **Canon layers** — Allen-Zhu (arXiv 2512.17351) **[UNVERIFIED, single-lab, 1.3B/100B tokens]**. No independent replication. Do not act.

---

## 2. The copying-gap verdict

**This is the section that matters.** It answers: given H2 (pointer head, REJECTED), H3 (evidence-query head, REJECTED), H6 (state-conditioned boundaries, un-evaluated), and the external literature — what is actually going on, and what is the most promising remaining intervention?

### 2.1 What the evidence rules out

**Capacity is not the constraint.** Zoology solves MQAR at d=64; Nano is d=192. Jelassi's constructive copy circuit needs 2 layers; Nano has 6. Nano's copy failure additionally **survived a 3.2× scale-up** and a 50× scale-up to 160M. A capacity story predicts scale fixes it. Scale did not. `[REP + in-project]`

**Depth is not the constraint.** Same source. Two layers suffice.

**The readout head is not the constraint — and this is now refuted by family, not by one experiment.** H2 (direct pointer), H3 (global evidence-query bilinear readout), and H6 (state-conditioned 640-param query residual) are **three variants of a single move**: improve the boundary-query readout over a projection of the final-layer hidden state (`nano_ai/training/evidence_query_model.py:110,267`). Add C_POINTER_P1 (VOID) and C_POINTER_P2 (REFUTED) and the record is **five model-side interventions, zero movement on the residual slot**. A sixth head design is refuted-by-family until the confound in §2.2 is removed.

**Finetune-stage curricula could not have worked.** Allen-Zhu & Li: without pretraining-time augmentation, extraction is 0% *regardless of any subsequent instruction finetuning*. H4 tested the finetune-side version and made things worse — held-value fell 72.55% → 38.03%, and `papers/DECISION_GATES.md` accordingly bans repeating generic surface expansion.

**Long context is not the constraint.** The failure occurs at ~130 tokens in a 512-token window. Every long-context technique buys more of a resource that is ~74% unused.

### 2.2 What the evidence points at

**Three accounts survive. They are separable, and no in-project instrument currently separates them.**

**Account A — adaptation-regime destruction.** `papers/paper2_draft.md:122-123` is the sharpest experiment in the repo:

| | 200M pretrain tokens | 3.2B pretrain tokens |
|---|---|---|
| **full FT** | 16.9 ± 1.7 (clean 66.6 ± 5.0) | 7.0 ± 1.0 (29.4 ± 4.0) |
| **LoRA r=16** | 7.1 ± 1.2 (29.6 ± 3.7) | **4.2 ± 0.9 (17.7 ± 3.2)** |

Same architecture, same tokenizer, same checkpoint at each column. Changing **only the adaptation regime** moved the diluted gap 16.9 → 7.1 and moved per-field **cc clean 64.4 → 0.0**. Copying is demonstrably reachable in this exact architecture family; full-parameter adaptation on an under-trained base removes it. Supported externally by Biderman et al. (LoRA forgets less) and Singh et al. (ICL/IWL competition). **Every H2–H6 run sat in the full-FT cell** (§0.1) — so none of them ever tested this, and all five carried the confound.

**Account B — the circuit never formed, because pretraining never asked for it.** Retrieval heads are intrinsic to pretraining (Wu et al.); extractability is gated at pretraining (Allen-Zhu & Li); ICL emerges only under burstiness and many rare classes (Chan et al.); induction-head formation is diversity-gated and abrupt (Reddy). Nano's anchor saw **32.8M tokens over a 10.96M-token shard, ~3.1 epochs** — and Hernandez et al. show repeated data damages **copying and induction heads specifically**. `nano_ai/pretraining/dataset.py:39-50` draws a **uniformly random window from a flat concatenated token stream**, and `nano_ai/training/model.py` calls SDPA with `is_causal=True` and no mask — so a large fraction of every training window's attention signal is **cross-document**. No pretraining-side variable has ever been manipulated in this project.

**Account C — the training data never made pointing necessary, per slot.** `scribe/build_scribe_data_v2.py:30` — 5 allergy values, one held out. The allergy slot reads clean 100.0 ± 0.0 in **all five own-stack configurations**; tokens moved cc and med and never touched it. The one intervention that moved it is D5→D80 (§0.3, +66.7 pts, monotonic, position-controlled).

**These are not competing — they compose.** A and C are both *finetune-side* and are both **cheap, executable now, and never crossed with each other.** B is *pretraining-side*, is the most-supported by external literature, and is the only one that costs a run.

### 2.3 The measurement defect underneath all of it

Two instrument problems mean several H-cycle rejections are less informative than they look.

**Terminal-only evaluation cannot see transience.** Every H-cycle gate is scored at epoch granularity — three observation points per seed across a 1,050-step schedule. Under Singh et al.'s ICL-transience dynamic, three points cannot distinguish "never learned to copy" from "learned it at step 300 and lost it by step 1050." **Every H1–H6 rejection was scored under that ambiguity.** A free partial check exists today: `artifacts/nano_h5/` and `artifacts/nano_h6/` already hold per-epoch checkpoints; if epoch-2 held-value exceeds epoch-3 in any seed, the effect is confirmed at coarse resolution before a single new step is trained.

**Aggregate metrics cannot carry the decision.** `papers/paper2_draft.md:149-152` states it outright: three models across two stacks with different seeds were **metrically indistinguishable because they occupy the same categorical flip state** (cc ✓, med ✓, alg ✗), and aggregate metrics are composition arithmetic over that state. **The flip matrix, not the scalar gap, is the fundamental object.** Any gate written on the aggregate inherits this.

**A third defect, in the RL gate.** `sft/gate_grpo.py:26` computes `se = ((p_sft*(1-p_sft) + p_grp*(1-p_grp)) / n)**0.5` with `n` = prompts × K — treating clustered samples as independent Bernoulli draws. The design effect is `1 + (K−1)ρ`; at K=8, ρ=0.5 the SE is understated ~2.1×. Any historical PASS from that gate is not reproducible at the stated confidence.

### 2.4 The verdict

> **The copying gap is not one failure. It is a solved part and an unsolved part, and the project has been measuring their sum.**
>
> **The solved part** — chief-complaint and medication copying — was closed by pretraining tokens and adaptation method, independently and compoundingly (16.9 → 7.0/7.1 → 4.2). Architecture and tokenizer are ~innocent.
>
> **The unsolved part** is a single slot, and it is pinned by a five-value finetune pool, not by the model. Nothing model-side has ever moved it — five attempts, zero movement — and the one data-side intervention that did move it (+66.7 pts) was run at 10M with full-FT on a weak base and **has never been combined with the two levers that solved the other slots**.
>
> Therefore the most promising remaining intervention is **not another head, not a longer context, not an SSM, and not more parameters**. In order:

**1. Run Stage M. It is built, preregistered, locally validated, and has never been executed.** `stage_m/stage_m_kernel.py` (17.5KB) and `stage_m/PREREG_induction_curriculum.md` exist; `ls stage_m/` shows **no result file**, `artifacts/` contains **no `stage_m/` directory**, and there is no ledger row. Its decision tree is frozen in code at `stage_m_kernel.py:288-294` with four verdicts already written, including `"FULL-FT-DESTROYS — induced at pretrain but collapsed post-SFT; NOT a refute -> LoRA/frozen-layer follow-up (vNext priority A)"`. **The project pre-committed to the Account-A mechanism and then never fired the test.** It is the only instrument in the repo that measures whether a copy circuit *survives* finetuning; it runs at Nano's exact geometry on hardware in hand, in minutes-to-hours, and it discriminates Account A from Account B directly.

**2. Remove the adaptation-regime confound.** Add a `trunk_adaptation` argument to `nano_ai/training/train_evidence_query.py::_optimizer` with three preregistered arms — `full` (control), `attn_frozen` (`requires_grad=False` on `blocks[*].{q,k,v,o}` only), `lora_r8`. The `attn_frozen` arm is mechanistically sharper than blanket LoRA: paper2's LoRA arm wrapped 98 modules indiscriminately and cannot say the capability lives in attention; freezing exactly the attention projections tests that claim. No geometry change, so `_FROZEN_GEOMETRY`, the exact-parameter-count assertion (`evidence_query_model.py:41,43,77`) and the SHA-256 gate (`model.py:232,269-270`) never fire.

**3. Cross D80 with the good corner.** The two solved-slot levers (3.2B tokens, LoRA) and the one residual-slot lever (D80) have never met. `trajectory/slot_diversity_pools.py` already holds the frozen 80-value pool with held types excluded by construction — reuse it rather than minting a new one, which keeps the arm preregistration-clean.

**4. Build the within-checkpoint retrieval-head contrast.** A probe-only subclass of `_NanoBlock` replacing SDPA with the explicit `softmax(QKᵀ/√32 + causal_mask)V` path (asserted to match SDPA to <1e-5 on the loaded weights, which is what makes it provably the real model), scoring the Wu et al. copy-paste score **within one checkpoint** on a field that copies (cc, clean 9.2) versus a field that never does (alg, clean 100.0), same forward pass, zero cross-model confound. This is what makes a null at 36 heads interpretable rather than a detection-threshold artifact.

**Deferred, not rejected: KV-shifting attention.** It is the single most mechanism-targeted architectural lever in the survey — it *provably lowers the depth and width required for the induction-head mechanism*, and Nano is precisely depth- and width-poor — at ~two learned scalars per head. But it changes trunk geometry, so it is a **new pretrain rung**, and putting an architecture change into rung 1 would confound the rung's stated capability question. It unlocks only if items 1–4 show the failure is head-level.

---

## 3. Ranked adoption plan

### 3.1 ADOPT_NOW

Ordered by information-per-dollar. Items 1–3 require **zero training**.

| # | Item | Files | Effort |
|---|---|---|---|
| A1 | **Run Stage M** exactly as preregistered; add the RESULT section its own honest-reporting rule requires; open a claim ID | `stage_m/stage_m_kernel.py`, `stage_m/PREREG_induction_curriculum.md`, `papers/EVIDENCE_LEDGER.md`, `papers/EXECUTION_QUEUE.md` | days |
| A2 | **Re-anchor G-scale-1** — it is already passed by paid-for runs (see §4.1) | `papers/SCALE_PROGRAM_PREREG.md`, `papers/EVIDENCE_LEDGER.md` | hours |
| A3 | **Re-score existing per-epoch checkpoints** for non-monotonic held-value (free transience check) | `artifacts/nano_h5/`, `artifacts/nano_h6/`, `nano_ai/training/evaluate_evidence_query_h6.py` | hours |
| A4 | **Per-head retrieval/copy-score probe**, cc-vs-alg within one checkpoint, plus span-length sweep (1→N BPE tokens) | new `nano_ai/training/retrieval_head_probe.py`, `nano_ai/training/model.py`, `checkpoints/chinchilla-160m/*`, `checkpoints/anchors/nano_v01_scribe.pt` | days |
| A5 | **`trunk_adaptation` arms** (`full` / `attn_frozen` / `lora_r8`) — removes the confound carried by all five H-cycle runs | `nano_ai/training/train_evidence_query.py`, `nano_ai/training/pointer_model.py`, `scribe/scribe_sft.py` | days |
| A6 | **Step-resolved copy probe** every 50 steps (21 points/seed vs 3); make peak-step, peak-value, final-value, peak−final reported fields | `nano_ai/training/train_evidence_query*.py`, `nano_ai/training/train_state_span.py` | days |
| A7 | **Cluster-robust gate estimator** — replace the iid SE in `gate_grpo.py:26` with a paired prompt-level bootstrap | `sft/gate_grpo.py` | hours |
| A8 | **Freeze the rung-1 special-token block** (24 reserved IDs: `<think>`, `</think>`, `<field>`, `</field>`, `<quote>`, `</quote>`, `<state>`, `<no_evidence>`, `<retry>`, + 15 reserved). Prereg gate 4 is still open; the window is irreversible | `papers/SCALE_PROGRAM_PREREG.md`, `pretrain/tokenizer.json`, `nano_ai/pretraining/sources.py` | hours |
| A9 | **Pretraining pipeline hygiene**: document-offset index in the manifest, `unique_tokens_written` as a first-class stat, optional document-masked attention, `rope_base` as a recorded config field (currently hardcoded at `model.py:196` and `anchor_checkpoint.py:409`) | `nano_ai/pretraining/prepare.py`, `nano_ai/pretraining/dataset.py`, `nano_ai/training/model.py`, `papers/PRETRAINING_PROVENANCE.md` | days |

**Note on A8/A9 provenance.** `architecture_identity` feeds `solver_id`, which is baked into recorded benchmark results. Append `-rope{base}` **only** when base ≠ 10000.0 so every existing `solver_id` stays byte-stable. A9's document-masking path loses the fused causal SDPA kernel — measure and report the slowdown, do not assume it is minor.

**A4 preregistered read.** On `ownstack160m_scribe.pt`, N=200 instances/field, metric = max-over-heads per-instance copy-paste score on gold evidence spans. Predict: cc held-out shows max ≥ 0.30 in ≥50% of instances; alg held-out shows max < 0.10 in ≥80%. **If cc and alg score within 0.05 of each other — the heads fire identically on the field that copies and the field that never does — the attention hypothesis is FALSIFIED for Nano and every architectural attention lever should be struck from the rung-1/1b list.**

**A5 preregistered read.** Frozen paper-alpha diluted instrument, recorded full-FT baseline for the 3.15M anchor **18.3 ± 1.3** (clean 87.3 ± 2.7). Threshold: `attn_frozen` or `lora_r8` reads diluted ≤ **12.0** (≥6.0 pts improvement — ~60% of the 9.8-pt effect measured at 160M, discounted because the anchor is ~2× under-Chinchilla where the 160M cell was ~16×). >12.0 REJECTS the adaptation-regime account at 3.15M and is a retained negative. Guardrail: scribe parse rate stays 100%. **Named second outcome, recorded before the run:** with attention frozen the trunk may underfit the pointer head — H3 hit 4,000/4,000 training-only calibration under full-FT, so if the frozen arm's training-only calibration falls below 3,800/4,000 the arm is **capacity-blocked, not regime-refuted**, and that distinction must be logged in advance.

### 3.2 PREREGISTER_AS_HYPOTHESIS

**P1 — D80 × LoRA at the good corner (highest-expected-value hypothesis).**

- *Design:* 2×1 factorial at fixed 160M / 3.2B tokens / LoRA r=16. Arm A = D5 (preserves byte-comparability to the whole existing grid — non-negotiable). Arm B = the already-frozen 80-value pool in `trajectory/slot_diversity_pools.py`.
- *Prediction:* arm B drives clean per-field **allergy** gap from 100.0 ± 0.0 to **≤ 50.0**, and clean aggregate value-level gap from 17.7 ± 3.2 to **≤ 10.0**. (Justification: the 10M D80 arm reached 66.67% held-type recall, ~33% miss, so ≤50 is conservative-but-discriminating; and with cc/med already at 0.0 the aggregate is ≈ alg/5, so alg=33 implies aggregate ≈6.6.)
- *Hard falsifier:* alg clean stays **≥ 90.0** → the D5→D80 result is scale- or method-local, does not transfer from 10M/full-FT to 160M/LoRA, and data composition is demoted below adaptation method.
- *Instrument caveat, stated in advance:* the 10M result is held-**type** recall; the rung-1 metric is clean **value-level** gap. **These are not the same estimand.** The arm must report both.

**P2 — Pretraining-side retrieval-shaped data (Account B).**

- *Design:* ≥1–5% of rung-1 pretraining tokens are self-generated verbatim-span-copy / bursty-repeated-entity exemplars (license-clean by construction), implemented in `nano_ai/pretraining/prepare.py` as an additional manifest-verified shard family with its own sha256 and contamination digest against `fresh_v1` and all H-cycle values. The inter-occurrence-distance prescription (`q_ℓ ∝ ℓ`) from arXiv 2512.18634 is **[UNVERIFIED]** and is used as a *design heuristic only* — the gate does not depend on it.
- *Must be pretraining, not SFT.* The project's own record is why: H4 put generic surface expansion into the finetune mixture and held-value fell 72.55% → 38.03%; `papers/DECISION_GATES.md` bans repeating it. What H4's limitations record explicitly did **not** vary is evidence order, distractors and long-context distance.
- *Primary gate (categorical, far sharper than any aggregate):* **allergy clean gap < 100.0.** It has read exactly 100.0 ± 0.0 in five own-stack configurations and is the most robust null in the project. Any movement off the ceiling is real signal; staying at exactly 100.0 rejects the intervention.
- *Secondary:* held-out-value clean aggregate ≤ 40.0 against the re-anchored control (§4.1).
- *Guardrail, non-negotiable:* G-scale-2 holds — fabric grounding.v2 presented-error 0.0% with zero lost-correct, and abstention/absence must not regress. **H5 is the standing proof that a held-value gain can be bought by wrecking abstention** (held-value 74.3% PASS while absence, conflict and uncertainty all FAILED).

**P3 — Serial extractive scratchpad, judged against a transcript-echo placebo.**

- *Design:* SFT (not RL) on traces constructed **deterministically** from data the generator already owns — `nano_ai/training/state_span_data.py::_extract_fields` is deterministic and `encode_state_span` already carries the gold span text. Form: `<think><field>medication</field><quote>…</quote><state>supported</state>…</think>` then the **unchanged** five-slot target. Quote precedes state deliberately: H4 measured **832/1000 correct states against 194/1000 exact spans** — the model knows *which* field, not *where* — so the copy must not compete for token positions with the state decision.
- *Three hard constraints, each from a named failure:* (i) **separate checkpoints, never mode-fusion** (Qwen3 shipped fusion and abandoned it); (ii) **the trace is never rewarded, verified, displayed, or counted as evidence** (Baker et al.; Anthropic 2505.05410) — enforced by a `strip_trace()` in every scoring path plus a test asserting the score is bit-identical under arbitrary perturbation of the `<think>` block; (iii) **mandatory transcript-echo placebo** — a matched-length scratchpad that merely re-emits the transcript. The echo grants the short-range-copy benefit without serialization.
- *Gate:* held-out-value exact match beats the equal-budget no-think control by ≥5.0 pts **and** the echo placebo by ≥3.0 pts, McNemar paired 95% CI lower bound >0 on both. (n=2,987 held-value fields, paired SE ≈0.8 pts, so these are ~6σ and ~3.7σ — decision-sized, not significance-sized.) Must also beat verifier-guided best-of-N at matched inference token budget by ≥2.0 pts, or the honest conclusion is that **test-time compute, not training, is the lever**.
- *Guardrails (this is the o3/o4-mini hallucination result made into a gate):* verifier-final wrong-presented **== 0**; absence/conflict/uncertain do not fall below H5's 280/413, 149/250, 162/250.
- *Falsifier for the hypothesis itself:* if the echo matches the scratchpad within 3.0 pts, the serialization claim is dead and the retained finding is the much cheaper "short-range copy is easier than long-range copy."

**P4 — Error-correction ("retry") traces in the rung-1 mixture, ≥1% exposure.**

- Targets **over-abstention**, the other measured product failure (`papers/PRODUCT_THESIS.md:109-122`: 3/10 useful, 3/10 OVER_ABSTENTION on the only non-fixture corpus, and the doc's own note that "even 3/10 is generous"). Today the only recovery from low confidence is silence; a retry affordance gives a second option.
- Two choices fixed, **deliberately not swept** (DECISION_GATES forbids ratio-sweeping after H5): error rate 20%; erroneous tokens **not** masked from the loss (arXiv 2408.16293 ablates masking and finds it unnecessary). Distractor generator already exists — `state_span_data.py::_alternate_value`.
- *Prediction:* absence accuracy +≥15 pts absolute over the equal-budget control (H5 measured 280/413 = 67.8%; predict ≥345/413), n=413, paired SE ≈2.7 pts → ~5.6σ. **Admission remains the pre-existing frozen 383/413 that H5 failed** — clearing the effect prediction without clearing admission is a retained negative, not a pass.
- *Honest cost, stated not hidden:* keeping the comparison clean requires rung 1 to run **both** mixtures — two 160M pretrains at the measured $37 = $74 against a $150 cap that also covers finetune, eval and margin. Preregistering costs nothing; the second arm needs **explicit owner budget re-approval**.

**P5 — Distance-resolved copy instrument (kill gate for the L4/L5 branch).**

- Insert k neutral filler turns between the evidence turn and transcript end, holding target text byte-identical and shifting offsets deterministically (contract validation re-checks offsets, so correctness is machine-verified). Arms at evidence-to-end distance ~130 / ~256 / ~384 / ~480. Development partition only; `fresh_v1` stays sealed. Zero training, CPU-only.
- *Gate:* if acc(d~130) − acc(d~480) ≥ 15.0 pts, the failure has a distance component and rung-2 context extension is warranted. If ≤ 5.0 pts, **all context-extension spend is deferred** and the NANO_V2 L4/L5 long-context items are reclassified data-gated behind copying working at d~130. 5–15 pts is inconclusive and needs a powered rerun. *Prior, given a ~130-token failure inside a 512-token window: the ≤5.0 outcome.*

**P6 — Contract v1: document-scoped evidence + deterministic multi-doc reconciler.** `nano_ai/contract.py:202,210-211,228` pins `speaker` to exactly `"patient"` and resolves spans via `_patient_content_ranges`, which only requires the span to sit inside *some* `Patient:` line — so concatenating two transcripts lets a span from the **wrong document** validate cleanly. That is a live correctness hole and it makes L5 unmeasurable. Bump `CONTRACT_VERSION` to `nano.scribe.v1` (additive; v0 objects keep validating), add `SourceDocument` + `EvidenceSpan.document_id`, relax the speaker pin to a per-document configurable role (L4 needs multi-speaker), and add a per-document-inference + field-reconciliation adapter. *Gate:* the reconciler achieves cross-document misattribution ≤2.0% and cross-document conflict recall ≥95% at k=8, where the frozen anchor on the concatenated transcript is **structurally incapable of k≥4** (4 × ~130 tokens > 512). If it clears both, **L5 multi-document is solved by the smallest sufficient solver** and every future long-context proposal must beat it rather than being funded on ambition.

### 3.3 DEFER (with named, mechanically checkable unblock conditions)

| Item | Unblock condition |
|---|---|
| **KV-shifting attention** (arXiv 2411.19574) | Rung 1 passes G-scale-1 (re-anchored) **and** G-scale-2 **and** A4 shows the failure is head-level. New pretrain rung (rung 1b) — it changes trunk geometry. |
| **Gated attention** (arXiv 2505.06708) | Same gate, second in queue behind KV-shifting. Zero evidence below 1.7B. |
| **Sink token in every pretraining window** | Only if the layer-6 token-identity linear probe reads **<50%** top-1 on open-vocabulary value tokens. **≥80% ⇒ struck from the prereg, a preregistered decision to spend nothing.** 50–80% ⇒ unresolved, record the band, do not add the variable. |
| **RLVR / GRPO** | **Both** required: (i) cluster-bootstrap (pass@32 − pass@1) ≥ 10.0 pts on held-out-value exact match — below that there is nothing latent to sharpen; (ii) one full GRPO config (200 × 6 × G=16 = 19,200 rollouts) completes in ≤4h locally. Currently impossible: `sft/model_nano.py:49-56` re-runs a **full 512-padded forward per generated token**, so ~250 tokens/rollout = 4.8M forwards, projecting ≥24h at 160M for one configuration. |
| **D/N ≥ 200 over-training** | See §5 — the in-project evidence neither confirms nor refutes it for the residual slot, and the budget consequence is 3–25× the $150 cap. Gate on P1 first: it is nearly free and targets the same residual. |

**RLVR design constraints to record now, so a future session cannot improvise them:** rewards target the **output artifact only**, never the `<think>` trace; the reward **is** the already-frozen decision utility (E1's U / E4's U_R\*), not a fresh hand-set weighting, which also means an RLVR arm consumes the ≤1 R\* revision budget in C_RSTAR_VALUE and needs its own claim ID; the abstention/assertion asymmetry (λ, μ) comes from that same frozen U, because μ/λ **is** the product's operating point and tuning it is exactly how an over-abstaining model becomes a confident-and-wrong one. Also: mermaid-cli in the reward loop is a subprocess per rollout and will dominate wall-clock — use an in-process parser and reserve compile-checking for a periodic audit sample.

### 3.4 REJECT (with reason)

| Rejected | Reason |
|---|---|
| **MLA / latent attention** | Loss-neutral at best at small scale (arXiv 2506.09342: MHA 2.147 vs MLA r=d/2 2.154, r=d/4 2.241), **no retrieval evaluation exists**, and Nano's KV cache at ctx=512/3.15M is already trivial. Compressing K/V is the wrong direction for retrieving exact novel token identities. |
| **NSA / MoBA / sparse attention** | All gains are long-context *compute* wins at 64k. At ctx=512 on a laptop there is no compute problem and no recall gain on offer. |
| **Sliding-window + global hybrids** | Gemma 3's **local** window (1024) is twice Nano's **entire** context. |
| **YaRN / NTK / PI / LongRoPE** | Extend a model past its trained context. Nano trains at its full 512. |
| **p-RoPE** | Killed by arithmetic on `model.py:191-196`: max accumulated rotation over 511 positions is **5.2°**; 5 of 16 pairs rotate <90° across the whole context. Nano's bottom pairs are already content-only. Truncating them is a no-op. |
| **DiffAttn** | No from-scratch validation below **830M** anywhere in the paper, and it **halves head count** (6×32 → 3×64) while retrieval heads are <5% of heads — at 6 heads that is a live risk to the exact capability under repair. |
| **GQA 2→6** | The supporting claim is blog folklore, not a result in Ainslie et al. (arXiv 2305.13245, which tests summarization/translation and predates NIAH). Additionally **pre-killed by A4**: if any head already scores high copy-paste on cc, the KV subspace is demonstrably sufficient and this dies without a run. |
| **SSM / Mamba / gated-conv blocks** | Jelassi et al. prove GSSMs are bounded by fixed latent state on copying; Bick et al. localize the gap to Gather-and-Aggregate heads. **The most valuable rejection in the survey** — "local-first, small, long transcripts" makes SSMs look like the obvious answer exactly where they make the central problem worse. |
| **Recurrent Memory Transformer / segment memory** | Reintroduces the fixed-size inter-segment state Jelassi proves fatal. Revisit only at L5 streaming, and only after retrieval works. |
| **KV-cache eviction / compression** | Wu et al.: compression breaks precisely the retrieval heads that do copy-paste; eviction is irreversible. |
| **A sixth readout-head redesign** | **Project-specific kill.** H2, H3 and H6 are three variants of one move (improve the boundary-query readout over a final-layer projection), plus P1/P2 — five model-side interventions, zero movement on the residual slot, all five carrying the full-FT confound. Refuted **by family** until A5 removes the confound and A4 localizes the failure. |
| **MobileLLM depth-over-width reallocation** | Worth single-digit percent; Nano already has SwiGLU/GQA/tied embeddings; and refuted in spirit by the 0-for-5 model-side record. |
| **Canon layers** | Single-lab, unreplicated, validated at 1.3B/100B tokens **[UNVERIFIED]**. |
| **GPT-BERT hybrid-objective fork** | BabyLM winners are ~100M, untested at 3M; bidirectionality **breaks the frozen-anchor lineage** and conflicts with the L5 streaming ambition. |
| **Logit distillation from an open teacher** | V=4098 byte-BPE cannot be token-aligned to a 49k/151k teacher, and teacher forward passes over 3.2B tokens break local-first. (The regime boundary itself is favourable — arXiv 2502.08606 — so this is a *tokenizer and compute* rejection, not a technique rejection.) |
| **Vocabulary resizing** | Tao et al.'s smallest model is 33M; a 3.15M extrapolation is unlicensed. |
| **phi-style synthetic textbooks** | Documented contamination; HuggingFace's Cosmo-1B reproduction did not match. |
| **Curriculum learning (re-litigating)** | BabyLM's replicated negative at exactly this data scale, matching the project's own Stage C failure. |
| **Single-needle NIAH as an instrument** | RULER: superficial. Nano's existing held-out-value instrument is already closer to NoLiMa. "Upgrading" to NIAH would flatter the model. |

**Reopen conditions for the architecture branch** (so this is falsifiable rather than permanent): run a direct MQAR probe at Nano's exact frozen geometry (V=4098, d=192, L=6, H=6, KV=2, head_dim=32, ctx=512), training the architecture on the task itself. **≥90% 2-key recall within 2,000 steps CONFIRMS** the rejection — no further architecture work is licensed on the copying gap. **<90% FALSIFIES it** and reopens the branch under a new claim ID. Zoology predicts the former at 3× the width it required. `stage_m/stage_m_kernel.py::induction_probe()` already implements a close variant, so this is mostly assembly.

---

## 4. What rung-1 should be

Data and method are **known substitutes in this project** — `paper2_draft.md` establishes it as a measured 2×2, not a hypothesis. And slot diversity gave the single biggest measured win (+66.7 pts) of any intervention in the project's history. Rung 1 should be designed around both facts.

### 4.1 Re-anchor G-scale-1 — it is already passed, by runs that are paid for

`papers/SCALE_PROGRAM_PREREG.md:62-65` requires held-out-value clean aggregate to improve **≥10 points over the existing within-stack 160M control (66.6 ± 5.0)**. But `papers/paper2_draft.md:105,122-123` shows 66.6 is the **200M-token / full-FT** cell — **the worst of the four measured**. Scoring the completed runs against the gate as written:

| completed cell | clean | vs. 66.6 baseline | verdict under current G-scale-1 |
|---|---:|---:|---|
| 3.2B + full FT | 29.4 ± 4.0 | **+37.2** | passes by 27 pts of margin |
| 3.2B + LoRA r=16 | 17.7 ± 3.2 | **+48.9** | passes by 39 pts of margin |

**G-scale-1 has at least two known passing configurations that add no information, and it contradicts its own §1 question** ("does capability breadth help *beyond* the cheap fixes"). Re-anchor to the **best** measured cell (3.2B + LoRA, clean 17.7 ± 3.2) and require ≥5 points of improvement on it (i.e. clean ≤ 12.7). Under the re-anchored gate, **zero of the four measured cells passes** — which is what a discriminating gate looks like.

Add two clauses:

- **G-scale-0 (comparability):** any arm that changes the finetune data is scored against its own D-matched control, never against the D5 grid.
- **Per-slot reporting is mandatory** — cc/dur/sev/med/alg reported separately, **alg primary**. `paper2_draft.md:149-152` proves the aggregate cannot carry the decision: three models across two stacks are metrically identical because they occupy the same categorical flip state.

### 4.2 The finetune spine must not be reused verbatim

`SCALE_PROGRAM_PREREG.md:§2` says the scribe finetune protocol is "reused verbatim as the comparison spine." Verbatim = `build_scribe_data_v2.py:30` = D5 = **alg pinned at 100.0 by construction**. The prereg as written guarantees that the residual it needs to move stays frozen.

Replace the single verbatim arm with the 2×1 of P1 (D5 control + D80 arm at fixed 160M/3.2B/LoRA r=16), reusing the already-frozen `trajectory/slot_diversity_pools.py`. Adaptation is **LoRA r=16 in both arms, not full-FT** — the project's own grid measures 7.1 vs 16.9 on the same weak base and 4.2 vs 7.0 on the strong base, a ~2.4× reduction from simply not rewriting the trunk, at zero extra pretraining cost. Do **not** extend to D320 yet; gate that on the D80 result.

### 4.3 Sequence length by computed evidence, not ambition

Define **LRSF(seq)** = fraction of training token positions whose *in-document* causal prefix ≥ seq/2 — computable in one pass over the document-offset array from A9, **before spending a dollar**, requiring no token data. Rule: rung 1 trains at the largest seq in {512, 1024, 2048, 4096, 8192} with LRSF ≥ 0.25. Below that, more than three quarters of positions never exercise attention past half the window and the extra L² cost buys distance the corpus cannot supply.

This is why **seq-8192 is not merely expensive but wasteful**: with a dense block-diagonal mask, SDPA computes the full 8192² matrix and discards ~90% of it while no single document spans the window, so reachable distance does not increase. Rough FLOP accounting for the record (to be replaced by a measured short run): attention/token ≈ 12·n_layer·d·seq versus dense 6N ≈ 9.6e8 at 160M → ~+6% at 512, ~+12% at 1024, ~+25% at 2048, ~+95% at 8192; i.e. $37 → ~$46 at 2048, ~$68 at 8192. **All headroom under the cap goes to D/N and to the second mixture arm, not to range.**

This contradicts `NANO_V2_AMBITION.md`'s line that long-context discipline is "not a rung-1 concern." The *range* is not a rung-1 concern. The **knobs** (recorded RoPE theta, document offsets, document masking, unique-token accounting, reserved special tokens) are rung-1 concerns, are nearly free, and are **irreversible if skipped**.

### 4.4 Rung-1 checklist

1. **Run Stage M first.** It is the decide-then-spend gate for the central open problem, at minutes-to-hours on hardware in hand versus $150 of pod time, and its four verdicts are already frozen in code. Its blocking Arm-C feasibility pre-gate (parse ≥90%, recall ≥80%) also protects against drawing any conclusion from a broken lineage.
2. **Re-anchor G-scale-1** (§4.1) and add G-scale-0 + mandatory per-slot reporting.
3. **Finetune = D80 × LoRA r=16, with a D5 × LoRA control** (§4.2).
4. **Freeze the 24-ID special-token block** (A8). Cost is bounded and checkable before the freeze: at a 160M geometry with tied embeddings, 24 reserved rows = 24×768 = **18,432 params = 0.0115% of 160M and 0 training FLOPs**. A unit test asserts the block size and that every ID round-trips without splitting. **This decision cannot be taken later without a full retrain** — and it is what makes P3 and P4 runnable at rung-2 prices instead of re-costed at $500–700.
5. **Land A9's pipeline hygiene** (document offsets, unique-token stats, optional document masking, recorded `rope_base`). Zero compute, no license gate. The license path is clear: `nano_ai/pretraining/sources.py:77-119` records ODC-By-1.0 **CLEARED 2026-08-05** for research + commercial with attribution, the split-isolation policy **DEFINED**, 14 parquet files pinned with per-file sha256, and the smoke-1m prep run **SATISFIED**.
6. **Instrument before training** (A6): held-out-copy probe every 50 steps with peak/final both reported. Given ICL transience, a rung-1 null measured only at the end would be as uninterpretable as H1–H6's.
7. **Guardrail unchanged:** G-scale-2 must hold — fabric grounding.v2 presented-error 0.0% with zero lost-correct, and abstention must not regress. H5 is the standing proof that held-value can be bought by wrecking abstention.

**The one-line version.** Rung 1 as currently written spends $150 to re-measure a gate that two completed runs already clear, using a finetune recipe that pins the only slot in question. Re-anchoring the gate and swapping D5 for D80 costs one extra finetune (minutes) and is the difference between a rung that can answer its own question and one that cannot.

---

## 5. Honest uncertainty

**What the literature does not settle.**

- **No published paper trains a sub-20M-parameter model and measures novel-span copy directly.** Every architectural candidate here is an extrapolation into unmeasured territory. This is the single largest uncertainty in the document.
- **MQAR does not license a multi-token claim.** Zoology proves single-token in-context recall is easy at d=64. Nano must copy a **multi-token BPE span** at V=4098 out of a messy transcript. The discriminating measurement is A4's span-length sweep: if the gap opens only at span >1, a circuit/capacity story returns; if it is flat-bad at span=1, the circuit-selection diagnosis is confirmed hard.
- **Whether RL extends or only sharpens is genuinely contested** — Yue et al. (pass@k collapse) versus ProRL at 1.5B. Do not build on either side.
- **The sub-1B RLVR literature is largely a Qwen artifact** and its transferability to a from-scratch licensed-data base is unknown, not merely unmeasured.
- **Sinks are load-bearing (Barbero) versus removable (Qwen gated attention) is unresolved.** The reconciliation offered — that the gate supplies the same attend-to-nothing escape valve — is plausible and untested at L=6.
- **The nearest-regime reasoning result (from-scratch 100M, ≥1% pretraining exposure sufficient) is an unverified post-cutoff preprint.** It is the strongest evidence a nano-scale think-arm is not hopeless, and it is exactly the kind of citation that should not become load-bearing without being opened.
- **`q_ℓ ∝ ℓ` inter-occurrence-distance shaping is theory on 1–3 layer models with single-token targets** and is unverified. It is used here as a design heuristic, never as a gate.

**What only an experiment can decide.**

| Question | Deciding experiment | Cost |
|---|---|---|
| Is the copy circuit absent, or present-then-destroyed? | **Stage M** (built, never run) | hours, local |
| Does the failure live in attention or downstream of it? | **A4** cc-vs-alg within-checkpoint head contrast | days, no training |
| Did ICL transience make H1–H6's rejections uninterpretable? | **A3** re-score existing per-epoch checkpoints | hours, free |
| Does D5→D80 transfer from 10M/full-FT to 160M/LoRA? | **P1** | one extra finetune |
| Does the failure have a *distance* component at all? | **P5** | days, CPU-only |
| Does pretraining-side retrieval-shaped data move the residual slot? | **P2** | one rung-1 arm |
| Is D/N ≥ 200 the lever for the residual slot? | **Untested at any scale in-stack.** The 16× token increase (200M→3.2B, still D/N ≈ 20) moved cc and med and did **not** move alg. Whether a further 10× would is unknown, and the budget consequence is 3–25× the $150 cap. **Run the free lever (P1) first.** | $400–3,700 |

**Two standing facts that must not be lost.**

- **H6 is un-evaluated, not refuted.** `artifacts/nano_h6/TERMINAL_INFRASTRUCTURE_CLOSEOUT.json` records `status: INCONCLUSIVE_INFRASTRUCTURE_NO_DEVELOPMENT_ACCESS`, `scientific_result: null`, training **completed** with both seed checkpoints recovered and backed up, `one_shot_consumed: false`, terminated at `pre_development_runtime_admission` on a host-kernel platform-string mismatch (required `6.8.0-90`, observed `6.8.0-134`) that the provider interface cannot select. **Its one-shot is unconsumed.** Any statement that "H6 failed" is false.
- **Every allergy number is a type-level n=1** — `paper2_draft.md` says so itself, and across four Pythia-1B training draws the allergy slot spans ~0–25 while cc/med sit at 0.0 in all of them. **Training-run variance concentrates in the hardest slot.** The D80 prediction in P1 must therefore be read against that variance, and a single passing draw is not a solved slot.

---

## Appendix: verification log

Claims in this document were checked against source rather than accepted from the lens reports. Notable corrections made during verification:

| Asserted by a lens | Verified status |
|---|---|
| H3–H6 train heads on a **frozen** trunk | **FALSE.** `train_evidence_query.py:298` optimizes all `model.parameters()`; seven scripts record `"full_trunk_trainable": True`; zero `requires_grad=False` in the training path. |
| Held-out value copying is at the floor | **FALSE on the H-line.** H5 held value 2,220/2,987 = 74.3%, **passing** its 2,167 gate (`DECISION_GATES.md:109`). H5 failed on absence/conflict/uncertainty. |
| The corpus pipeline never existed / fineweb-edu is license-blocked | **STALE.** `sources.py:77-119` — ODC-By CLEARED 2026-08-05, 14 files sha256-pinned, smoke-1m prep SATISFIED. |
| DiffAttn: "70M matches 1B", "4M-token needle" | **BOTH FABRICATIONS**, caught during research and excluded. Verified claims are ~65% of size/tokens, 6.8B ≈ 11B, needle at 4K/64K only, smallest from-scratch 830M. |
| "MQA hurts retrieval, GQA recovers it" from Ainslie et al. | **MISATTRIBUTED.** That paper reports summarization/translation and predates NIAH. |
| G-scale-1 is a discriminating gate | **FALSE.** Baseline 66.6 is the worst measured cell; two completed runs clear +10 by 27 and 39 points. |
| Stage M has been run | **FALSE.** `ls stage_m/` shows only the prereg and the kernel; no result file, no `artifacts/stage_m/`, no ledger row. |
| p-RoPE kill arithmetic | **CONFIRMED independently.** head_dim=32, base 10000 (`model.py:196`) → lowest inv_freq 1.778e-4 rad/token → 0.0909 rad = 5.21° over 511 positions. |
| Pretraining has no BOS/sink at position 0 | **CONFIRMED.** `pretrain/train.py:14` and `dataset.py:39-50` draw uniformly random windows from a flat stream; `prepare.py:23` appends EOD **between documents**, which is not a per-window sink. |
| `gate_grpo.py` treats clustered samples as iid | **CONFIRMED**, `sft/gate_grpo.py:26`. |
| Contract lets a wrong-document span validate | **CONFIRMED**, `contract.py:202,210-211,226-228`. |