# Nano LM: integrated research plan

Version 1, 2026-11-23. This file is the entry point for the project's direction. Historical results stay where they are (`INVESTIGATION.md`, `LEDGER.md`, `results/`); this plan links to them and corrects how some of them should be read. Nothing here has been implemented, trained or bought.

**Objective (locked by the owner).** Discover and validate architectures, learning methods and engineering techniques that deliver stronger capability with less computation, memory and energy. The flagship is about 3–4B parameters, explored from 2B to 4.5B, never above 7B, and it must outperform substantially larger models across several meaningful capabilities. The 386M model is the experimental foundation, not the destination. Novelty counts when it produces a measured advantage.

**How this plan was made.** Six independent read-only investigations, each checked against primary sources or our own files:
- a reference-model matrix (model cards, config files, technical reports);
- a methodology audit of our trainer, tokenizer, data and harness;
- the distillation and knowledge-transfer literature;
- the architecture literature at 1–8B;
- an inventory of every branch;
- an assessment of the V5 architecture package.

Claims below carry their source. "From memory" means the claim is not yet verified.

---

## Contents

1. Summary and recommendation
2. What we have: reconciled inventory
3. Corrected conclusions from completed work
4. The compute comparison, scoped properly
5. Where capability is lost: bottlenecks from first principles
6. Research landscape: what is established, what is open
7. Knowledge inheritance as a research programme
8. Architecture candidates
9. Recommended direction and why
10. Evaluation framework
11. Compute economics
12. Experiment queue
13. Staged path to the flagship, with gates
14. Risks, unknowns and falsifiable hypotheses
15. What would make Nano LM genuinely different
16. Approval package
17. Sources

---

## 1. Summary and recommendation

**Plain summary.** Our best model (386M, a standard dense Transformer) is a sound base for experiments, but it is not evidence that we have a better architecture. On a fair reading it is behind modern small models, mainly because of how little it was trained. Six investigations point to one direction with real evidence behind it: inherit the knowledge of large open models, then redesign how the model computes, remembers and reasons, and measure every gain against cost. The generic "efficient hybrid" is already used by strong models at our target size (Qwen3.5 at 0.8–9B, March 2026), so our novelty has to come from specific mechanisms that fix measured weaknesses. The plan tests those mechanisms cheaply at about 0.5B before anything larger.

**Recommendation.** Investigate one integrated architecture with three parts:
1. **Inherit, then convert.** Start from a larger open teacher with a permissive licence. Prune it to the target size, convert most attention layers to recurrent layers initialised from the teacher, and distil from the full teacher. This is established engineering; the specific combination and the data-driven placement of the remaining attention layers are our synthesis.
2. **Budgeted exact memory.** Recurrent layers compress context and lose exact recall. Add a small, fixed-size exact memory. A write policy, trained on how much later attention the *teacher* pays each token, decides which tokens to keep exactly. This is a new hypothesis.
3. **Adaptive recurrent depth with verified halting.** A shared reasoning block runs a variable number of times per token over the recurrent state. It stops when its update stops changing the state, as an iterative solver stops at convergence. This is partly new: no prior work was found for this combination with a recurrent state.

The first experiments run at about 0.5B with a 1.5B teacher, for roughly $150–250 in total. Each part has a falsification test. The flagship (3–4B) costs roughly $3,000–7,000 and needs funding beyond the current credit; it is gated on the small-scale evidence (§11, §13).

---

## 2. What we have: reconciled inventory

**Plain summary.** The project trained many models across five lines, but most weights from paid RunPod work were lost on temporary machines. Only the Google Cloud line kept everything, with checkpoints in one bucket. The lines don't share tokenizers or goals, so they can't be merged. What can be reused is our evaluation tooling, our training stack, the lessons from failed ideas, and our trained models as controlled references.

### 2.1 Model lines

| Line | Location / recoverability | What it trained | Tokenizer, context | Data, tokens | Key result | Cost (method) | Use next |
|---|---|---|---|---|---|---|---|
| TPU dense ladder (this branch) | `gs://nanolm-lab-eu-0478806098/runs/`, 79 checkpoints, CRC32C recorded in `results/checkpoint-index.tsv`; code hash in each result JSON | 14M, 40M, 101M, 211M, 386M dense GQA/RoPE/SwiGLU; AdamW and Muon | own 32k BPE; 512 | corpus v1 (800M) and v2 (10B); crops sampled with replacement, so 7.7B tokens ≈ 5.3B unique (audit) | 386M: 0.8049 bits/byte; benchmarks in §3 | ≈ $165 (22 Nov) + ≤ $110 (Q13) + ≤ $22 (Q14), list-price upper bounds, billing unread | Controlled from-scratch reference; harness; trainer |
| RunPod native (Track A, arch-evolution) | 3 of 41 checkpoints kept on Express1M2; the rest lost (`LEDGER:80`) | 46M–350M hybrids, 400–600M tokens | same 32k | LAB-DATA-01 | No hybrid beat dense on public text; Pythia-410M beat all three kept models (`native_public_language/README:7,21`) | RunPod ledger (local branch, unverified total) | Lessons only |
| RunPod post-training (Track B) | lost (ephemeral pods) | Qwen3-1.7B and SmolLM3-3B: SFT, distillation, GRPO | model's own | KD traces | Qwen3-1.7B after 8B distillation fell below its base (GSM8K 73.2 vs 75.0; MATH-500 62.6 vs 70.0) | unverified | Lesson: off-policy SFT distillation damaged an already-distilled model |
| Scribing (Nano v0.1, H1–H7, P1) | release assets, Kaggle outputs | 3M trunk with heads; Qwen 1.5B/7B LoRA | own 3M; Qwen | synthetic | H6 rejected; H7 invalid labels; native30/100 had a non-causal decoder (false null) | mixed | Domain task for evaluation (§10), not a model |
| Toy mechanism studies (ontogenesis, V5, G04/G06) | in Git | tiny bilinear and toy models | synthetic | synthetic | Routing, correction and memory results in toy worlds; V5 largest executed model 840 parameters | ≈ $0 | Ideas and evaluation contracts, not capability |
| Kaggle tournaments (gate1, H13/H10) | Kaggle outputs | 14M variants, 98M tokens per arm | 32k | v1 | value-residual and attention-residual arms ≈ a 10% larger dense model (one seed); owned by another workstream | free | Its owner's scorer decides |

Full per-run detail: the inventory report (§17) and `LEDGER.md`. Unverified fields: RunPod totals, several weight locations, and Kaggle output retention.

### 2.2 Shared assets

- **Evaluation harness** (`tpu/eval_*.py`): seven zero-shot tasks with lm-eval prompt formats, a no-context control and a 13-gram contamination check. The audit's limits apply (§3); it needs extending (§10).
- **JAX trainer** (`tpu/train_jax.py`): data parallelism, Muon, resume, per-run code hash. Fixes needed: sampling without replacement, final optimiser state, weight hashes in results.
- **Corpus v2:** 10B tokens of education-filtered web, textbook, maths and code. The held-out decontamination is weaker than described (§3).
- **Machine control:** layered deletion deadlines, a spot supervisor and a job queue.
- **Evaluation discipline from V5:** score a token before observing it, equal-information memory comparisons, and three separate scoreboards (native, adapted, augmented).

### 2.3 Integration strategy

- **Merging weights across our lines:** not possible. Tokenizers, shapes and training differ, and most weights are gone. Merging methods need identical shapes.
- **Initialisation and transplantation:** possible from *open* models with permissive licences into a new architecture (§7). Our own small models are too weak to be useful teachers.
- **Our models as references:** the 386M and the ladder are the from-scratch control for every inheritance experiment, at matched student size.
- **Lessons carried forward:**
  - store/cache augmentation does not pay at 40–101M;
  - architecture variants under our setup were within noise of dense;
  - whole-block reuse learned two-step composition where untied layers did not (toy);
  - off-policy distillation can damage a distilled model;
  - every spot machine needs a supervisor;
  - every result needs weight and data hashes.

---

## 3. Corrected conclusions from completed work

**Plain summary.** Several of our recent statements were stronger than the evidence. The table restates each one at the strength the evidence supports, and names the test that would settle it.

| Claim as made | Supporting evidence | Confounders | Justified conclusion | Next discriminating test |
|---|---|---|---|---|
| "The recipe is not broken; it beats Pythia-410M" | 386M beats Pythia-410M's 6.3B and 8.4B checkpoints on every task (ARC-E 0.596 vs 0.428) | Pythia checkpoints are at full learning rate (not annealed); data differs (Pile vs education-filtered); tokenizer differs | No severe defect in the pipeline at this compute. Nothing about relative recipe efficiency | Same data and tokenizer, our recipe vs a standard recipe at matched compute |
| "Beats fully trained Pythia-410M on PIQA" | 0.678 vs 0.668 | 0.7 SE apart; Pythia figure is from official lm-eval, not our scorer | Not distinguishable. Pythia-410M is clearly better on LAMBADA (0.516 vs 0.354; perplexity 10.8 vs 32) | Score Pythia-410M final through our scorer |
| "Trails SmolLM2 because of training compute" | 386M used ≈ 1/90 of SmolLM2-135M's compute | Data, tokenizer, context, objective and schedule all differ; no overlap in compute (§4) | A plausible hypothesis, not a tested cause | Our own scaling curve on fixed data, compared at matched compute (§4) |
| "Without the question, our model equals SmolLM2" | Within 0.004 on every task with the question removed | WinoGrande's control is degenerate; the decomposition is arithmetic | Describes behaviour: answer priors are similar. It does not identify why we gain less from the question | Controlled probes of context use (shuffled or irrelevant contexts, retrieval tasks) at matched compute |
| "Seed noise is 0.0008 bits/byte, so 0.003 effects are detectable" | 3 configurations at 14–40M, 3 seeds each | Small models, one metric, one data order | Preliminary for those settings only | Measure seed spread at 101M and on benchmarks |
| "Novel architectures never beat dense" | Nested loops, store, cache, depth routing, wider heads: all within noise or worse | Our implementations, small budgets (≤ 0.8B tokens), single learning rates in places, toy tasks for several | Retired *as implemented at these budgets*. Each mechanism addresses a different question (below) | Re-test a mechanism only if a new design targets a measured failure (§5) |
| "Recurrent depth helps reasoning" | Whole-block reuse 4/5 vs untied 0/5 at every learning rate (toy two-step chains) | Toy task; small model; fixed step budget | A real effect in that setting. No bridge to language capability yet | Recurrent-depth test on language and held-out composition (E3) |
| "Memory store or cache helps" | Leave-document-out store gains 0.0096 / 0.0037 / 0.0017 bits/byte at 14M / 40M / 101M; 40M cache below 0.01 | n-gram stores, not learned memory | n-gram augmentation does not pay at this scale. Says nothing about learned exact memory | Budgeted exact memory on a recurrent model (E2) |
| "Fused attention is not a speed win" | 1.05× at 101M with heads of 128 | One size and sequence length; profiling not done; spot machine lost after 2,000 steps | Not worth adopting *here*. Where time goes is unmeasured | Profile the trainer (E0b) before any kernel work |
| "Hardware efficiency 36% of peak" | 6 × N × tokens/s against chip peak | Theoretical peak, not measured bottlenecks | An estimate only | Profiler trace: matmul, attention, optimiser, input pipeline, communication |
| "Distillation into our model gives +0.02 bits/byte" | 101M teacher into 14M/40M students | Weak teacher, same tokenizer, small budget | A small gain in that setting | Same-tokenizer distillation from a strong teacher (E1) |
| "Q14: wider heads not adopted" | W1 failed (1.05×) | Run interrupted; W2/W3 never run | Infrastructure-interrupted; the speed result stands as measured | None needed unless profiling points at attention |
| "Cerebras-GPT comparison running" | none | Official repos return 401 | No result exists; the comparison is unavailable | None |

**Data and evaluation issues the audit found** (`INVESTIGATION.md` will carry the corrections):
- **Held-out leakage.** Code and maths validation overlap training text verbatim (13-token windows: maths 6.9%, code 7.3% in v1). The builder's 13-word check only compares windows at multiples of 7 words. Code and maths bits/byte are optimistic.
- **Benchmark contamination.** Benchmark overlap was measured (HellaSwag 2.1% of items, PIQA 1.25%, ARC 1.5–1.8%) but never removed or reported as score-with-and-without.
- **Unchecked scoring path.** Our models are scored by `eval_score.py` (bf16, EOS prefix). The calibration against Pythia only checked `eval_score_hf.py`. The two scorers have never been run on the same model.
- **Repeated tokens.** The sampler draws with replacement: 7.7B training tokens covered about 54% of the corpus at least once and 18% at least twice.
- **Unrecoverable final state.** Final weights were not hashed in result files, and final optimiser state is not saved, so the 386M run cannot be resumed as-is.
- **Untested tuning.** The 386M learning rate and Muon settings were chosen at 101M and never checked at 386M.

---

## 4. The compute comparison, scoped properly

**Plain summary.** Our model and SmolLM2's published checkpoints never meet on the compute axis. SmolLM2's earliest checkpoint already used 11× our compute, so any "compute multiplier" against SmolLM2 needs extrapolation. Pythia's checkpoints *do* overlap ours, but on different data and with an unfinished learning-rate schedule. Neither comparison can attribute a difference to our architecture or recipe; only controlled experiments can.

Training FLOPs, approximated as 6 × parameters × tokens:

| Point | Tokens | FLOPs | × our 386M |
|---|---|---|---|
| Ours 101M (0.8B) | 0.8B | 4.8e17 | 0.03 |
| Ours 211M (4.2B) | 4.2B | 5.3e18 | 0.30 |
| **Ours 386M (7.7B)** | 7.7B | **1.8e19** | 1.0 |
| Pythia-410M step 3000 | 6.3B | 1.5e19 | 0.86 |
| Pythia-410M step 4000 | 8.4B | 2.0e19 | 1.14 |
| Pythia-410M final | 300B | 7.3e20 | 41 |
| SmolLM2-135M first checkpoint (step 240k × 1,048,576) | 252B | 2.0e20 | 11 |
| SmolLM2-135M final | 2T | 1.6e21 | 90 |
| SmolLM2-360M first checkpoint (step 160k × 1,572,864) | 252B | 5.5e20 | 31 |

**What each comparison can and cannot establish:**
- **Pythia (observed overlap).** At matched compute our model scores higher on every task. Data, tokenizer and schedule differ, so this shows only that our pipeline has no severe defect at this compute.
- **SmolLM2 (no overlap).** Comparing our curve with theirs needs extrapolation across at least an 11× compute gap. That requires assuming both curves keep their shape (a power law in compute), with uncertainty that grows with the gap. Any such figure must be labelled an extrapolation, with its fit and residuals shown.
- **Attribution.** Neither comparison can be attributed to architecture or recipe. That needs runs on the same data and tokenizer, matched compute, several seeds, and a tuning budget for each arm.
- **Transfer to new architectures.** Results from the dense model do not transfer automatically. What is likely to carry over is optimiser behaviour, data quality and tokenizer effects. What needs re-testing for any new architecture is learning-rate scaling, stability, compute per token, and how capability scales with size.

---

## 5. Where capability is lost: bottlenecks from first principles

**Plain summary.** We trace each observed weakness to what the task needs, where the current system loses it, and which mechanism could change that. Each mechanism adds its own costs and needs a test against simpler explanations.

| Observed failure | What the task needs | Where it is lost | Candidate mechanism | New costs / risks | Distinguishing test |
|---|---|---|---|---|---|
| Weak knowledge per parameter (MMLU-like; behind SmolLM2) | Facts stored in weights, reliably retrievable | Too few training tokens; knowledge is learned slowly by next-token prediction | Inherit knowledge from a model trained on trillions of tokens (§7) | Teacher licence and quality ceiling; converted students lose 7–14 MMLU points | Converted student against from-scratch at equal student compute, on knowledge probes |
| Exact in-context recall (names, numbers, quotes) | Copy precise tokens from context at any distance | Attention does this; recurrent states compress and lose it (Based; Gated DeltaNet recall 30.6 vs 37.0) | Budgeted exact memory with a learned write policy (Candidate 2) | Memory traffic; write-policy errors; training instability | Fixed memory budget: learned policy vs recency, heuristic eviction and full attention, on recall tasks |
| Multi-step composition | Apply a learned operation repeatedly | Fixed depth limits sequential steps; untied layers don't share the operation | Adaptive recurrent depth (Candidate 3) | More inference FLOPs; training instability (Huginn had collapsed runs) | Matched-FLOP comparison with an untied deeper model and with fixed-depth recursion, on held-out compositions |
| Use of context (gain from the question) | Read and condition on the prompt | Unclear: could be undertraining, tokenizer or capacity | First diagnose; then choose a mechanism | — | Context-use probes at matched compute; inherited vs scratch models |
| Inference cost at long context | Memory and bandwidth that don't grow with every token | Attention's KV cache grows linearly | Recurrent layers with few attention layers (established) | Recall loss (above) | Quality–memory–latency curves at 4K–64K |
| Training efficiency | High hardware utilisation | Unmeasured: estimated 36% of peak, 86% scaling over 4 chips | Profile first, then targeted fixes | — | Profiler trace |

**Fields that inform the design:**
- **Information theory.** A rate–distortion view of memory: keep exactly the tokens with the highest future retrieval value, and compress the rest. That is Candidate 2's write rule.
- **Numerical methods.** Treat a reasoning block as a fixed-point iteration, and stop when the residual (the change in state per iteration) is small. That gives Candidate 3 a halting rule with a measurable meaning rather than an opaque learned gate.
- **Control.** Use a feedback signal (verifier confidence) to allocate compute, with a budget constraint.
- **Compilers and systems.** Choose mechanisms that map onto dense matrix operations and chunked scans, since TPUs and GPUs punish irregular memory access.

---

## 6. Research landscape: what is established, what is open

**Plain summary.** Mixing recurrent and attention layers is now standard practice at our target size. Several other ideas are well established. A few questions remain genuinely open, and a small team could answer them.

**Established, so not novel:**
- **Recurrent and attention layers mixed 3:1.** Qwen3.5 (0.8–9B, March 2026), Olmo Hybrid 7B (reached Olmo 3's MMLU with 49% fewer tokens), Kimi Linear, NVIDIA's 8B hybrid (+2.65 average over 12 tasks).
- **Fine-grained mixture-of-experts:** DeepSeekMoE, OLMoE.
- **Muon:** about 2× compute efficiency in Moonlight. We already use it.
- **Attention variants:** latent attention (MLA), gated attention, sliding-window plus global attention.
- **Multi-token prediction.**
- **Looped models combined with Gated DeltaNet** (LT2).
- **Prune-then-distil** (Minitron) and **attention-to-recurrent conversion** (MOHAWK, Llamba, Mamba-in-the-Llama, LoLCATs).

**Weak or contested:**
- **Recurrent depth (Huginn-3.5B):** no matched baseline.
- **Titans and Atlas:** not replicated.
- **nGPT:** 1.29× in an independent check against its claimed 4–20×.
- **Differential Transformer:** one lab only.

**Open questions identified by the literature review (no prior work found):**
1. Per-token adaptive recursion over a *recurrent* state inside a hybrid.
2. Why the attention layers in hybrids work: placement, position encoding (Kimi Linear's gain disappears with RoPE), and state size, measured on retrieval.
3. Learned dynamic chunking (H-Net) with a hybrid backbone at 1–3B.
4. Matched baselines for recurrent depth at 1B and above.
5. Shared versus no shared expert in small mixture-of-experts.

**Gaps this plan adds, from the inheritance evidence:**
6. Recovering the knowledge and recall that converted students lose, with mechanisms designed for that purpose.
7. Prune-from-larger, then convert, then distil, as one pipeline. Each step is published separately; the combination is not, as far as we found.
8. Training a memory policy from the teacher's attention.

---

## 7. Knowledge inheritance as a research programme

**Plain summary.** Large open models already hold what trillions of tokens taught them. The published evidence says the cheapest way to pass that knowledge to a new architecture is to inherit weights and the tokenizer, then train only what changed. Llamba-3B needed 10B tokens, about 0.1% of its teacher's. Learning only from a teacher's outputs, with a different tokenizer and no weight transfer, has no evidence of savings on that scale. A student usually does not beat its own teacher, so beating much larger models means inheriting from a larger teacher.

| Route | What transfers | Access needed | Alignment needed | What still has to be learned | Our cost (incremental) | How we know it worked |
|---|---|---|---|---|---|---|
| Train from scratch (control) | nothing | data | none | everything | 3.5B at 70B tokens ≈ 1.5e21 FLOPs (§11) | Reference curve |
| Continue training an open model | everything in the weights | weights | none | domain or format adaptation | small | Not novel. Reference only |
| Learn from teacher-written examples (sequence-level) | problem-solving traces, style | API or weights | none (works across tokenizers) | how to generalise beyond the traces | $40–320 per 1B generated tokens (API prices, §17) | Held-out and changed-composition problems; known risk: gains stay in maths and code (2507.00432) |
| Distil output probabilities | the full predictive distribution | weights (logits) | same tokenizer; cross-tokenizer methods lose much of the gain at small scale | everything the student can't fit | teacher forward ≈ $30 / $130 / $280 per 1B tokens for 7B / 32B / 70B on H100; top-256 logits ≈ 1.5 TB per 1B tokens | Token multiplier against a no-distillation curve with ≥ 3 points, teacher compute included |
| Distil intermediate representations | internal features | weights | layer mapping, matching widths or projections | mixer behaviour | as above plus alignment stages | MOHAWK: hidden-state alignment added about 8 points at fixed tokens |
| Several teachers | complementary skills | several sets of weights or APIs | tokenizer and representation conflicts | resolving conflicts | sum of teachers | Must beat the best single-teacher student; FuseLLM beat its sources on BBH but not on code |
| Prune a larger teacher | most of the teacher's knowledge | weights | none (same family) | recover from pruning | Minitron 8B → 4B: 94B tokens | Against the same-size from-scratch model and the teacher |
| Convert attention to recurrent layers | MLPs, embeddings, head; attention weights map into the recurrent layers | weights | layer mapping | the recurrent mixing | Llamba: 8–12B tokens | Retention per capability; losses concentrate in MMLU and recall |
| Combined (recommended) | prune from larger → convert → distil from the full teacher → train new mechanisms | Apache/MIT teacher weights | as above | our new mechanisms | §11 | Beats the same-size teacher-family model and approaches the larger teacher on held-out evaluation, at lower inference cost |

**Teacher licences (as read from model pages):**
- **Permitted:** Qwen2.5-7B/14B/32B, Qwen3-8B/14B/32B, OLMo-2-32B and Mistral 7B/NeMo/Small are Apache-2.0. DeepSeek-R1 is MIT and explicitly allows distillation.
- **Avoid as a base:**
  - Llama requires derived model names to begin with "Llama".
  - Gemma treats distilled students as derivatives bound by its terms.
  - Qwen2.5-3B uses a research licence.
  - MobileLLM is non-commercial.

**Risks:**
- **Teacher limits:** a student inherits its teacher's errors and preferred solution patterns.
- **Several teachers:** they can conflict.
- **Contamination:** teachers trained on benchmark-like data.
- **The capacity gap:** a much stronger teacher can produce a worse student, and for short training a smaller teacher is better (Gemma 3; Apple's distillation scaling laws).
- **Mitigation:** every inheritance experiment is scored on held-out problems, changed compositions and conditions the teacher data did not contain (§10).

**Accounting.** We report only *our* incremental compute: teacher inference, data generation, filtering, verification and student training. The upstream compute already embodied in a teacher is stated separately and never hidden. A claim such as "a 3.5B student approaches an 8B teacher" must name exactly what was inherited.

---

## 8. Architecture candidates

**Plain summary.** Three candidates, each aimed at a measured failure. Each part is labelled as established, our synthesis, or a new hypothesis. Names are working labels.

### Candidate 1: inherited hybrid with measured attention placement

- **Thesis.** Knowledge and computation can be separated. Inherit the knowledge, then replace most quadratic attention with recurrent layers. Place the few remaining attention layers where measurement shows they matter, rather than at a fixed ratio.
- **State.** Recurrent layers keep a fixed-size state per layer. Attention layers keep a KV cache.
- **Computation.**
  - **Recurrent layers:** a delta-rule update (Gated DeltaNet) initialised from the teacher's Q/K/V.
  - **Attention layers:** gated attention.
  - **Placement:** chosen by each layer's measured contribution to retrieval when it is converted.
- **Learning.** Prune a larger teacher (for example Qwen3-8B-Base, Apache) to 3.5–4B. Then three stages: align attention matrices, align hidden states, then end-to-end distillation from the full teacher, followed by ordinary training.
- **Parameters and compute.** About 3.5B parameters, all active; per-token compute is about that of a dense 3.5B. Long-context decoding is cheaper (fixed state), with an estimated 3–6× smaller cache.
- **Hardware.** Needs chunked-scan kernels for the recurrent layers: Pallas on TPU, the fla library on GPU.
- **Expected advantage.** Close to the teacher at about half its size, with much cheaper long-context inference.
- **Failure modes.** MMLU and recall losses (published as 7–14 points); pruning damage; licence constraints.
- **Labels:**
  - conversion and pruning: established;
  - prune-from-larger plus convert as one pipeline: our synthesis;
  - measured placement: a reasonable extension.
- **Falsification.** If the converted 3.5B does not beat a same-size model of the teacher family (for example Qwen3-4B-Base), the conversion adds no value at our target.

### Candidate 2: budgeted exact memory with teacher-guided writing

- **Thesis.** Recurrent layers compress context well but lose exact detail. Most tokens never need exact recall; a few always do. Keep a fixed number of tokens exactly. Learn which to keep from the teacher's own attention: tokens the teacher attends to heavily later are the ones worth storing.
- **State.** The recurrent state, plus an exact memory of M slots (key, value, position), with M fixed regardless of context length.
- **Update rule.**
  - **Writing:** a write head scores each token. The memory keeps the top-M by score, replacing the lowest.
  - **Reading:** attention over the M slots in designated layers.
- **Learning.** The write score is trained to predict the teacher's future attention mass on each token (a regression on teacher attention, available because we hold teacher weights), together with the language-model loss.
- **Parameters and compute.** Small parameter cost (the write head). Read cost is linear in M per token; memory is fixed.
- **Hardware.** Top-M selection and a small gather. Dense enough for both TPU and GPU.
- **Expected advantage.** Recovers most of the recall lost in conversion at a fixed, small memory, without the cost of full attention.
- **Failure modes.**
  - Teacher attention may not equal retrieval value (attention sinks, positional habits).
  - The write policy may overfit the teacher's patterns.
  - Discrete selection is hard to train.
- **Labels.**
  - Prior work: KV eviction and selection (H2O, SnapKV, NSA, DeepSeek Sparse Attention) and meta-tokens (Hymba).
  - **New hypothesis:** a write policy supervised by teacher future attention, with a fixed budget, inside a converted recurrent model.
- **Falsification.** At equal memory budget, if the learned policy does not beat a recency window and heuristic eviction (H2O) on recall tasks, the teacher signal adds nothing.

### Candidate 3: adaptive recurrent depth with verified halting

- **Thesis.** Some tokens need more computation than others. One shared reasoning block, applied repeatedly, can carry out multi-step composition with few parameters: our toy result showed whole-block reuse learning two-step chains where untied layers failed. Stop iterating when the block's update stops changing the state, like an iterative solver at convergence. Extra iterations at inference become a controllable budget.
- **State.** The hybrid's state, plus an iteration counter per token.
- **Update rule.**
  - **Iteration:** h(k+1) = h(k) + B(h(k), context).
  - **Halting:** stop when the residual ‖h(k+1) − h(k)‖ / ‖h(k)‖ falls below a threshold, or a cap is reached.
  - **Verifier:** a small head estimates whether the current state's prediction is reliable, and can extend the iterations.
- **Learning.**
  - **Training:** random iteration counts (as in Huginn), so the block learns to improve with more passes.
  - **Inherited models:** initialise the block from a teacher layer group.
- **Parameters and compute.** Parameters equal one block. Compute is variable and must be reported per token.
- **Hardware.**
  - **Cost of looping:** iterations are sequential, and recurrent depth does not shrink attention compute or the KV cache.
  - **Why the combination helps:** inside a recurrent hybrid most layers have a fixed state, which keeps looping cheap.
- **Expected advantage.** Better multi-step reasoning per parameter, and per FLOP if halting saves work on easy tokens.
- **Failure modes.**
  - Training instability (two of Huginn's runs collapsed).
  - Halting that never triggers.
  - Gains that vanish at matched FLOPs.
  - Looping hurts memorisation (2502.17416).
- **Labels:**
  - looped blocks: established;
  - per-token recursion (MoR): established in attention models;
  - per-token recursion over a recurrent state: open question 1;
  - residual-based halting with a verifier: new hypothesis.
- **Falsification.** At matched FLOPs, if the model does not beat an untied deeper model and fixed-depth recursion on held-out compositions, the mechanism has no advantage.

### Considered and not recommended first

- **Learned dynamic chunking (H-Net/BLT).** Promising at 8B, but it changes the tokenizer, which breaks weight and logit inheritance. It is the leading candidate only if the project chooses the from-scratch route.
- **Mixture-of-experts at 3–4B total.** Strong evidence, but it trades memory for compute in a way that suits a server more than a small model. Keep it as an option for the 7B ceiling.
- **Learned test-time memory (Titans, Atlas).** Unreplicated. Watch.
- **V5's transport and projection machinery.** No positive evidence above toy scale. Its evaluation contracts are adopted instead (§2.2).

---

## 9. Recommended direction and why

Combine Candidate 1 as the platform with Candidates 2 and 3 as the research contributions.

**Why this beats the alternatives:**
- **From scratch.** No mechanism closes a 100–1,000× compute gap. Our own architecture tests found nothing beyond noise.
- **Pure conversion.** Llamba-style conversion is not novel and loses knowledge and recall. Our contributions target exactly those losses.
- **Distillation from outputs alone.** It has no evidence of comparable savings when the tokenizer differs and no weights transfer.
- **MoE or Mamba combinations.** Generic MoE plus Mamba-style stacks are already shipped at our size (Qwen3.5, Qwen3-Next).

**What this direction gives:**
- capability inherited at modest cost;
- novelty where it can be measured (recall per byte of memory, reasoning per FLOP);
- a route to "beats larger models" by inheriting from a larger teacher;
- every step falsifiable at 0.5B before money is committed to 3–4B.

**The honest limit.** The platform's capability comes mostly from the teacher. The contribution has to be shown in the *margin*: what our mechanisms recover, or add, beyond a plain conversion of the same teacher.

---

## 10. Evaluation framework

**Plain summary.** Two tracks. Controlled comparisons isolate one change at a time with costs matched. Competitive comparisons place the model among strong current models. Both go far beyond multiple-choice accuracy, and every score sits next to the resources it cost.

**Controlled track:**
- Same data, tokenizer and teacher.
- Matched training compute, or the matched resource the question is about.
- A tuning budget for each arm.
- Three seeds where expected effects are under three times the seed spread.
- Paired bootstrap confidence intervals on the same items.
- Exploratory runs kept separate from registered confirmatory runs.
- One held-out evaluation set never used for selection.

**Competitive track: reference models.** Verified IDs, licences and training status are in the matrix report (§17).

| Group | Models (Hugging Face base IDs) |
|---|---|
| Around 0.5B | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen3-0.6B-Base`, `HuggingFaceTB/SmolLM2-360M` |
| 1–2B | `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen3-1.7B-Base`, `HuggingFaceTB/SmolLM2-1.7B`, `allenai/OLMo-2-0425-1B` |
| 3–4B target band | `Qwen/Qwen3-4B-Base`, `HuggingFaceTB/SmolLM3-3B-Base`, `Qwen/Qwen2.5-3B` (research licence), Llama-3.2-3B (gated), Gemma-3-4B (gated), Qwen3.5-4B (to be verified) |
| Larger models to beat | `Qwen/Qwen2.5-7B`, `Qwen/Qwen3-8B-Base`, `allenai/OLMo-2-1124-7B`, Llama-3.1-8B (gated); 14B and 32B Qwen as stretch references |
| Per-token diagnostics | Pythia (154 checkpoints), OLMo-2 (hundreds of checkpoints, open data), SmolLM2 and SmolLM3 intermediate checkpoints |

**Capabilities and measures:**

| Capability | Measure |
|---|---|
| Language modelling | Bits per byte on held-out text, decontaminated and with the code split by repository |
| Knowledge | MMLU (cloze and multiple-choice formulations, both reported), ARC, OpenBookQA |
| Generative reasoning | GSM8K and MATH subsets, answers checked |
| Coding | Execution-based tests (HumanEval or MBPP-style) |
| Instruction following | IFEval-style checks, instruction-tuned models only |
| Composition and out-of-distribution generalisation | Held-out recombinations of operations; problems in forms the teacher data lacks |
| Exact information use and long context | Phonebook/needle tasks; RULER-style suites from 4K to 64K |
| Memory revision | V5's correction and acquisition metrics, as a secondary evaluation |
| Reliability | Calibration (expected calibration error), selective-answer accuracy, recovery after an injected error |
| Domain | The project's scribing tasks where still informative |

**Rules:**
- Disclose base versus instruction-tuned status, shots, prompts, sampling, tools, retrieval and inference compute (recurrent passes and memory reads count).
- Compare models with different tokenizers on bits per byte or per-word measures.
- Report quality next to training FLOPs, inference FLOPs per token, latency, throughput, peak memory and energy (estimates labelled).
- **First fixes to our harness:**
  - reconcile the two scorers on one model;
  - run with and without the EOS prefix for both kinds of model;
  - compute logits in float32;
  - use bare-choice lengths for length normalisation;
  - remove contaminated items and report scores with and without them.

---

## 11. Compute economics

**Plain summary.** Training a 3–4B model from scratch costs several thousand dollars even at the minimum sensible budget. Inheritance cuts that to the low thousands, still above our current credit. Everything up to 1.5B fits a few hundred dollars. The flagship needs funding, such as Google's TPU Research Cloud.

Assumptions:
- **TPU v5e:** 197 TFLOP/s peak at our measured ~36% of peak, ≈ 2.6e17 FLOPs per chip-hour, at $1.20 per chip-hour on demand. Spot is cheaper; the spot rate has not been read from billing.
- **H100:** ≈ 4e14 FLOP/s effective (40% of dense peak, assumed), at $2.90 per hour.
- **Accounting:** student training is 6 × N × tokens; a teacher forward pass is 2 × N × tokens.

| Route, flagship scale | Our FLOPs | v5e chip-hours / cost on demand | H100 hours / cost |
|---|---|---|---|
| A. From scratch, 3.5B × 70B tokens (20 per parameter) | 1.5e21 | ≈ 5,700 / ≈ $6,800 | ≈ 1,000 / ≈ $2,900 |
| B. Teacher-written data (1B generated tokens) + 3.5B × 20B training | 4.2e20 + API $40–320 | ≈ 1,600 / ≈ $1,900 + API | ≈ 290 / ≈ $850 + API |
| C. Logit distillation from 8B, 3.5B × 20B | 4.2e20 + 3.2e20 | ≈ 2,800 / ≈ $3,400 | ≈ 510 / ≈ $1,500 |
| D. Prune 8B → 3.5B, convert, distil (Llamba/Minitron-like, 20–40B tokens) | ≈ 7e20 – 1.5e21 | ≈ 2,700–5,700 / ≈ $3,200–6,800 | ≈ 500–1,000 / ≈ $1,400–2,900 |
| E. Hybrid: D + 1B tokens of verified reasoning data + our mechanisms | D + ≈ 10% | as D | as D |

**Notes:**
- **Estimates only.** H100 figures assume 40% utilisation, not measured. Our TPU figures use measured throughput.
- **Small scale costs little.** The 0.5B validation work (§12) costs $150–250 in total.
- **Upstream compute.** The trillions of tokens embodied in the teachers are reported separately as inherited, not as our cost.

---

## 12. Experiment queue

**Plain summary.** Six experiments, in order of information per dollar. The first two are free and fix our measurement. The next four test the recommended direction at about 0.5B. Each has a decision it changes, a minimum meaningful effect and a stopping rule. Confirmatory versions are registered before running.

| # | Hypothesis / decision it changes | Baseline and controls | Smallest credible implementation | Evaluation and uncertainty | Minimum meaningful effect | Budget | Stopping rule | After positive / negative / inconclusive |
|---|---|---|---|---|---|---|---|---|
| **E0a** (free, local) | Our scorer is trustworthy; decides whether earlier comparisons stand | `eval_score.py` vs `eval_score_hf.py` vs official lm-eval figures | Export our 101M to HF format (400 MB); score with both scorers, with and without EOS; float32 logits; contaminated items removed | Paired differences with bootstrap CIs | Scorers agree within 1 SE per task | Laptop, hours | When agreement or the size of the bias is known | Correct the record / fix the scorer / extend to more tasks |
| **E0b** (≈ $5) | Where training time goes; decides kernel and engineering priorities | Current trainer at 386M per-chip load | One profiler trace on a v5e-1 spot machine for 15 minutes | Time and memory split by operation | — | ≈ $5 | One trace | Target the largest cost |
| **E1** (≈ $40–80) | Conversion works with our stack and recipe; decides whether Candidate 1 is the platform | Teacher Qwen2.5-1.5B (Apache); student a pruned and converted Qwen2.5-0.5B-shape model, or a converted Qwen2.5-0.5B; controls: the unconverted 0.5B, same distillation; from-scratch equal-compute student | Convert 0%, 50% and 75% of attention layers to Gated DeltaNet; MOHAWK stages; 1B distillation tokens; 2 seeds | Full §10 suite; retention per capability; throughput | ≤ 5-point drop at 75% conversion on non-recall tasks | 8× H100-hours spot or v5e, ≈ $40–80 | After 1B tokens | Platform / try selective conversion / extend tokens once |
| **E2** (≈ $30–50) | Teacher-guided exact memory recovers recall; decides Candidate 2 | E1's 75%-converted model; memory M ∈ {64, 256}; policies: recency, H2O-style, learned (teacher attention) | Write head plus top-M memory in 2 layers; 0.3B tokens | Recall suite (phonebook, RULER-lite at 4–16K); paired bootstrap | Learned beats the best heuristic by ≥ 5 points at equal M | ≈ $30–50 | Pre-registered | Adopt / drop teacher supervision, keep budgeted memory / larger M sweep |
| **E3** (≈ $30–60) | Adaptive recurrent depth adds reasoning per FLOP; decides Candidate 3 | From scratch at 125M–350M on our stack: looped block vs untied model at equal FLOPs vs fixed recursion; 3 seeds | Shared block, random iteration counts in training, residual halting at inference | Bits per byte; held-out composition probes; GSM8K-style arithmetic chains; FLOPs per token | ≥ 1.5× compute efficiency on composition probes, no loss in bits per byte | ≈ $30–60 | Pre-registered | Port to the E1 hybrid / try learned routing (MoR) / stop |
| **E4** (≈ $15) | Our data pipeline is clean; needed for any from-scratch control | 101M current vs sampling without replacement plus fixed decontamination | Two runs, 0.8B tokens | Bits per byte with overlapping windows removed | Report the size of the difference | ≈ $15 | — | Fix the pipeline for every later control |

Totals:
- Stage 1 (E0b–E4): roughly **$150–250**.
- E0a: free.
- E1–E3 need moderate code (a GDN layer, conversion stages, memory, looped block). That counts as a major code change, so it is in the approval package (§16).

---

## 13. Staged path to the flagship, with gates

| Stage | Scale | Work | Cost | Gate to the next stage |
|---|---|---|---|---|
| 0. Measurement | existing models | E0a, E0b, harness extensions, corrections recorded | free–$5 | Scorer reconciled; contamination-free evaluation set fixed |
| 1. Mechanisms | ≈ 0.5B student, 1.5B teacher; 125–350M from scratch for E3 | E1–E4 | $150–250 | At least one of Candidates 2 and 3 meets its minimum effect, and conversion retains capability (E1) |
| 2. Second scale | ≈ 1.5B student from a 7B teacher (Qwen2.5-7B, Apache) | Integrate the winning mechanisms; registered confirmatory test; interactions between mechanisms | $600–1,000 | The integrated model beats the same-family 1.5B model on held-out evaluation at ≤ its inference cost, *and* each mechanism's effect persists from 0.5B |
| 3. Flagship | 3–4B from an 8B teacher (prune, convert, distil), plus a 32B teacher for reasoning data | Full training and evaluation | $3,000–7,000; needs TPU Research Cloud or other funding | Beats Qwen3-4B-Base and approaches 7–8B models on the broad suite at lower inference cost |
| 4. Ceiling | ≤ 7B | Only if stage 3 shows the design scales favourably | separate approval | — |

---

## 14. Risks, unknowns and falsifiable hypotheses

**Hypotheses, each with the result that would falsify it:**
- **H1 (platform).** A pruned-then-converted student keeps ≥ 90% of its teacher's broad-suite score at half the size and a quarter of the long-context memory. *Falsified if* retention is below 85% at 0.5B and 1.5B.
- **H2 (memory).** Teacher-guided writing beats heuristic eviction on recall at equal memory. *Falsified if* the difference is under 5 points at both memory sizes.
- **H3 (depth).** Adaptive recurrent depth gives ≥ 1.5× compute efficiency on composition at matched FLOPs. *Falsified if* it is under 1.2× at two scales.
- **H4 (the claim we care about).** The integrated 3–4B model beats 7–8B models on at least three capability groups at lower inference cost. *Falsified if* it does not at stage 3. In that case the project's contribution is the mechanisms, not the flagship.

**Risks:**
- **Novelty risk.** The combination may be valuable but the individual parts derivative; the contribution is then integration and evidence. Mitigation: report each mechanism's measured margin over plain conversion.
- **Teacher dependence.** Licence terms, teacher errors and contamination. Mitigation: Apache/MIT teachers; held-out, changed-composition tests.
- **Funding.** Stage 3 exceeds current credit.
- **Scale transfer.** Effects found at 0.5B may vanish at 3B. Gated by stage 2.
- **Engineering.** TPU recurrent kernels need Pallas work, and GPU kernels exist (fla). Choosing GPU for stages 1–2 avoids kernel work.

**Breakthroughs that would change the plan:**
- teacher-guided memory reaching full-attention recall at a small fixed budget;
- halting by convergence giving large savings on easy tokens with no quality loss;
- either effect growing with scale.

---

## 15. What would make Nano LM genuinely different

Most small models are smaller copies of large ones: same layers, more data, sometimes distillation. Nano LM's distinguishing bet is to treat **knowledge, memory and computation as separate resources**:
- **Knowledge** is inherited from models that already paid for it.
- **Exact memory** is a small budget spent only on what later reasoning needs, chosen by learning from the teacher's own attention.
- **Computation** is applied per token until the answer stops changing, rather than a fixed depth for every token.

If those three can be shown to deliver more capability per FLOP, per byte of memory and per joule than a same-size model of the teacher family, that is a contribution regardless of the flagship. If they cannot, the plan finds out at 0.5B for a few hundred dollars.

---

## 16. Approval package

**Free work continuing now, within current limits:**
- harness corrections and E0a (scorer reconciliation on our 101M checkpoint, 400 MB);
- recording the audit's corrections in `INVESTIGATION.md`;
- writing the confirmatory protocols for E1–E3.

**Needs approval (one bundle):**
1. **Code** for E1–E3: a Gated DeltaNet layer and conversion stages, budgeted memory, and a looped block with halting. Recommended on GPU (PyTorch with the fla kernels) to avoid TPU kernel work.
2. **Downloads:** Qwen2.5-0.5B (already present, 1 GB) and Qwen2.5-1.5B (3 GB) as teacher.
3. **Compute:** stage 1, about $150–250 (spot GPU or TPU), plus E0b at about $5.
4. **Funding route for stage 3:** a TPU Research Cloud application, which only the owner can submit.

---

## 17. Sources

**Investigation reports (2026-11-23):** the full texts are in this session's record. Key primary sources:
- **Distillation:** 2407.14679 and 2408.11796 (Minitron), 2310.06694 (Sheared-LLaMA), 2408.10189 (MOHAWK), 2502.14458 (Llamba), 2408.15237 (Mamba in the Llama), 2410.10254 (LoLCATs), 2502.08606 (distillation scaling laws), 2401.10491 (FuseLLM), 2507.00432 (reasoning transfer).
- **Architectures:** 2412.06464 (Gated DeltaNet), 2402.18668 (Based), 2406.07887 (NVIDIA Mamba-2 hybrid), 2510.26692 (Kimi Linear), 2604.03444 (Olmo Hybrid), 2502.05171 (Huginn), 2507.10524 (MoR), 2502.17416 (looped transformers), 2505.06708 (gated attention), 2412.09871 (BLT), 2507.07955 (H-Net), 2502.16982 (Moonlight/Muon).
- **Reference models:** Hugging Face model cards and `config.json` for every ID in §10. Qwen2.5 (2412.15115), Qwen3 (2505.09388), Gemma 3 (2503.19786), OLMo 2 (2501.00656), MobileLLM (2402.14905, 2509.24945).
- **Our records:** `INVESTIGATION.md`, `LEDGER.md`, `results/` (including `checkpoint-index.tsv`), and the V5 package on PR #195 (`research/frontier/ontogenesis/architecture/`).

---

## 18. Component coverage map (living; updated each cycle)

**Plain summary.** Every part of the system gets an explicit decision: retain, strengthen, redesign, replace or invent, each with its evidence and status. Status words mean: *proposed* (in documents only); *implemented* (code exists and passes tests); *executed* (run, with results recorded); *demonstrated* (an advantage under controlled conditions, with seeds).

| Area | Current implementation (386M control) | Limitation found | Decision and proposed change | Evidence | Status | Measured result | Next decision |
|---|---|---|---|---|---|---|---|
| Inputs and tokenization | Own 32k byte-level BPE, single digits, NFC | 1–6% fewer bytes per token than Pythia and SmolLM2 on benchmark text; 44% of LAMBADA targets split into several tokens | **Replace** on the inheritance route with the teacher's tokenizer (required for logit and weight transfer). Byte-level hierarchy (H-Net, BLT) stays the leading option only if a from-scratch line is chosen | Audit; 2412.09871, 2507.07955 | proposed | — | Fixed by the teacher choice (Qwen2.5/Qwen3 tokenizer, 152k) |
| Embeddings and outputs | Tied input/output, 32k × d | With a 152k vocabulary, embeddings take 25–40% of a 0.5–1.5B model | **Redesign:** keep tied; measure the output-layer cost; options are a factored or low-rank output and vocabulary trimming | Matrix report (Gemma-3-270M: 168M of 268M are embeddings) | proposed | — | Measure in E1; act if the output layer exceeds 15% of FLOPs |
| Internal representations | Residual stream, RMSNorm, QK-norm | No direct measure of what is represented | **Strengthen** measurement: probes for entities and positions; hidden-state alignment to the teacher (MOHAWK stage 2 added about 8 points) | 2408.10189 | proposed | — | Part of E1 |
| Overall architecture | Dense decoder, all layers attention | Long-context cost; no parameter reuse | **Redesign** into Candidate 1: an inherited hybrid with measured attention placement, plus Candidates 2 and 3 | §6, §8 | proposed | — | E1 |
| Information processing (mixing) | Softmax attention with GQA | KV cache grows with context; recurrent replacements lose exact recall | **Replace most layers** with Gated DeltaNet, keeping some attention. **Invent:** budgeted exact memory (Candidate 2) | 2412.06464; 2402.18668 | **implemented** (`nano_core.py`, tests 4/4) | MQAR pilot: attention 100%, GDN 8% (one seed) | Cycle-1 grid running |
| Position and time | RoPE, 512 context | Short training context; RoPE may hurt hybrids (Kimi Linear's gain vanished with RoPE) | **Redesign:** a short causal convolution for local order (adopted in `nano_core`); compare NoPE against RoPE in the attention layers of hybrids; train at 2k–4k | 2510.26692; cycle-1 pilot (no order information: nothing learned) | partly implemented (conv) | Without order information, attention reached 4% on MQAR; with the conv, 100% | NoPE vs RoPE study (open question 2) |
| Mathematical building blocks | SwiGLU, RMSNorm, pre-norm residual | Untested alternatives | **Retain** SwiGLU and RMSNorm (strong evidence, no credible replacement). **Strengthen:** gated attention output (2505.06708); hyper-connections only if depth grows | 2505.06708; 2512.24880 | proposed | — | Gated attention as a cheap add in E1 |
| Memory and state | KV cache only | No budgeted or editable memory; n-gram stores and caches failed at 14–101M | **Invent:** budgeted exact memory with a teacher-guided write policy. **Retire:** n-gram store and cache augmentation | Store gains 0.0096 / 0.0037 / 0.0017 bits/byte; cache below 0.01 at 40M | **implemented**; running | pending | Decide after the cycle-1 grid |
| Allocation of computation | Fixed depth for every token | No adaptive compute; whole-block reuse helped only on a toy task | **Invent:** adaptive recurrent depth with residual (fixed-point) halting (Candidate 3). Mixture-of-experts kept as an option at the 7B ceiling | Our nested-LR result; 2507.10524; 2502.05171 | proposed | toy result only | E3 (cycle 2) |
| Learning objectives | Next-token cross-entropy | Learns knowledge slowly | **Strengthen:** logit distillation and hidden-state alignment from the teacher; a write-policy auxiliary loss; multi-token prediction at ≥ 1B only (small-model benefit unverified) | §7 | partly implemented (write auxiliary loss) | pending | E1, E2 |
| Optimization and training | Muon + AdamW, cosine to 1%, batch 32k tokens | Learning rate chosen at 101M; gains decayed with weight decay; final state not saved | **Strengthen:** wd = 0 on gains, split gate/up matrices for Muon, save final state, μP-style transfer of hyperparameters | Audit; 2203.03466; 2502.16982 | proposed | — | Fix in the trainer before any new from-scratch run (E4) |
| Data and curriculum | Corpus v2, education-filtered; sampling with replacement | Validation overlaps training on code and maths (about 7%); decontamination checks only every 7th window; benchmarks not removed | **Redesign:** sample without replacement, hash every window (or MinHash), split code by repository, remove benchmark matches; inheritance route adds verified teacher-written reasoning data | Audit | proposed | — | E4 |
| Knowledge inheritance | Small-teacher distillation (+0.02 bits/byte at 14–40M) | Weak teacher; never tried at scale | **Invent the pipeline:** prune from larger → convert → distil → train new mechanisms | §7 | proposed | — | E1 |
| Adaptation and correction | None in the LM. V5 and G04/G06 toy studies | Toy only. V5's largest executed model has 840 parameters | **Retain as evaluation:** V5 correction metrics. **Strengthen later:** budgeted memory gives an editable store | V5 assessment | evaluation infrastructure only | — | Secondary evaluation from stage 2 |
| Generation and interaction | Greedy and likelihood scoring only | No generative evaluation, verification or tool use | **Strengthen:** generative reasoning with checked answers; a verifier head linked to Candidate 3's halting | §10 | proposed | — | Harness extension (free) |
| Runtime and hardware | JAX on TPU v5e, data parallel, about 36% of peak (estimated) | Bottlenecks unmeasured; fused attention gave 1.05×; recomputing activations is 17% slower | **Measure first** (E0b profile). GPU with fla kernels for the recurrent layers in stages 1–2 | Q14; benchmark | executed (benchmark) | 30.8k tokens/s per chip at 386M | E0b |

### V5 and APS, scoped

- **Implemented in native computation:** a standard decoder, with optional block reuse. 840 parameters is the largest *executed* V5 model; that figure describes the V5 package only, not other Nano LM models.
- **In evaluation infrastructure only:** correction and acquisition cohorts, the causal cache, equal-information contracts, and the validation receipts.
- **Proposed only:** APS transport, the projection controller, operation memory and AETHER.
- **Demonstrated:** nothing above toy scale.

**APS reconstructed.** APS's mechanism is to reorganise predictive state without losing earlier access, through learned transport between parameterisations and projection of updates. It assumes a compatible parameterisation (V5's own compatibility table) and gradient-free transport maps.

**Should APS get a prototype?** Not now. Its claimed benefit (continual correction without interference) is not on the critical path to capability per FLOP. The cheapest test of its core idea, update projection, was already run in G06/G06b and works only when protection is localised. Revisit if the flagship needs continual factual updates.

**Memory and cache negative results, exactly:**
1. An n-gram store mixed into predictions after training: gains 0.0096 / 0.0037 / 0.0017 bits/byte at 14M / 40M / 101M (leave-document-out). Below the registered 0.01.
2. An exact in-context n-gram cache at 40M: all domain gains below 0.01.
3. Store-M at 40M worsened (1.0481 → 1.0495).
4. Joint training with the store: 0.14 worse, but biased by same-document leakage, so unsettled.

All four are *non-learned, count-based* memories bolted onto a trained model. None says anything about a learned, budgeted, attention-readable memory inside the model, which is what cycle 1 tests.
