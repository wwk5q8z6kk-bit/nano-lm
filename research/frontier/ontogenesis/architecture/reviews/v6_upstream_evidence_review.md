# Nano LM V6 upstream evidence review


**V6 package relocation:** exact upstream source bytes are unchanged. Packaged read-only audit runners are in `analysis/reconcile_upstream_v6.py`, `analysis/reconcile_correction_v6.py` and `analysis/reconcile_context_v6.py`; their stdout is compared with saved audit results by the V6 release gate. Original review workspace commands below describe the audit session.

**Review cutoff:** `3a38b5f1529a718838b5bc83a010574bfa9a4c6e` in `wwk5q8z6kk-bit/nano-lm`.  
**Prior integration cutoff supplied for this review:** `c597d5fefdd378c3e5c66a51e7690cdb90853287`.  
**Published V5 architecture commit supplied for this review:** `beb156b279900d640c3d1478eb8cafec9a8899ac`.  
**Review date:** 2026-11-23 UTC.  
**Scope:** dense language models, benchmark interpretation, nested/shared computation, forward transfer, and measured resource behavior. G04/G06 correction studies belong to the parallel review and are not adjudicated here.

## 1. Executive decision brief

The newer evidence establishes a completed, internally coherent 386M dense baseline and a narrower reason to investigate block reuse. It does **not** establish a superior general architecture, reliable compositional transfer, a causal explanation for the gap to modern small models, or a theorem connecting repeated layers to systematic reasoning.

| Evidence area | Independently checked result | Permitted architectural consequence | What remains open |
|---|---|---|---|
| Q13 dense scale | 386,062,080 parameters; 7,698,841,600 processed tokens; final mixed BPB index 0.8049; Q2/Q3a pass and Q3b fails as worded | Keep this exact checkpoint recipe as the strongest completed in-house dense reference in this inspected set | Seed variance; component attribution; generative reasoning; equal-compute competitiveness |
| Modern references | 386M is below or equal to SmolLM2-135M on all seven plain-accuracy tasks; below SmolLM2-360M on all seven | Use external models as evaluation references while preserving the owner's own-model direction | Better token efficiency or eventual convergence to their quality cannot be inferred |
| Nested learning-rate sweep | Best-of-three-rate counts: whole-block reuse 4/5 versus untied 0/5; single-layer reuse 2/5 versus dense 1/5; registered combined outcome is **mixed** | Advance block reuse as a bounded experimental candidate | General reasoning, language-model scaling, optimal tuning, longer-budget behavior |
| Forward-transfer pilot after the earlier automation cutoff | None of six trained arms demonstrates transfer under the registered adaptation procedure; dense replay retains 100% of stream answers but gets 2.78% on the primary new-combination test | Separate retention, fitting, and transfer in the architecture and decision ledger | Better adaptation, episodic training, representational existence, non-synthetic transfer |
| Q14 fused attention | 1.05106× measured throughput versus a required 1.25×; interrupted after 65,536,000 completed tokens | Retain 64-dimensional heads and the current attention implementation for the registered decision | Quality parity and kernel stability are unmeasured |
| LDO/cache decision status | Corrected 101M LDO gain 0.0017; 40M cache gains below 0.01 on every domain | Replace pending gates with precisely scoped closure entries | Full-corpus stores, joint training with repaired gate, longer context and newer checkpoints |
| Provenance | 148 exact repository source snapshots, 912,805 bytes, with verified Git-blob SHA and SHA-256; historical Q13 trainer hash matches the run's code-hash prefix | Promote the **source and arithmetic audit**, with clear evidence limits | Checkpoint, tokenizer, corpus, prepared evaluation data, and per-example outputs are not part of this inspection |

### Material changes after the earlier automation review

The frozen head is 17 commits beyond the automation's reviewed cutoff `6df72ff1f69ae91de6159bce9a7bb005561b68dc`. Those changes include the merged forward-transfer development pilot and its post-hoc controls, Pythia-410M `step3000` and `step4000` comparison files, and the owner's explicit reassessment. The latter records the direction to build Nano LM's own foundation model and core architecture, with outside pretrained models used as references rather than adopted as the base. This is a source-recorded direction, not an authorization issued by this review to incur new compute costs. See [the frozen reassessment](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/INVESTIGATION.md) and [the forward-transfer report](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/FORWARD-TRANSFER-20261121.md).

## 2. What this review actually verified

All repository evidence cited below is frozen at the stated cutoff unless an explicit historical commit is supplied. The source snapshots are in `upstream_evidence/sources/`; historical trainer versions are in `upstream_evidence/historical_sources/`. `upstream_evidence/manifest.json` records every path, ref, exact byte count, Git-blob SHA, SHA-256, and local location. The deterministic `upstream_evidence/reanalyse.py` reads these copies, checks their identities, recomputes the relevant predicates, and writes `upstream_evidence/reanalysis.json`.

The work performed here consists of source inspection, identity checking, and arithmetic on committed results. No model was trained or scored again; no checkpoint was downloaded; no remote job, paid resource, protocol edit, raw-result edit, or repository write was performed. A stored result matching a scorer is stronger than a prose summary alone, but it is still evidence from the original execution rather than an independent training replication.

### Verification register

| Check | Outcome | Limit |
|---|---|---|
| Exact content identities | All 148 snapshots pass byte count, SHA-256, and Git-blob checks | Identity does not validate the experiment that produced a file |
| Q13 completion | `finished=true`; last curve step equals 234,950 planned steps; token arithmetic matches | Checkpoint bytes were not inspected |
| Q13 recorded trainer identity | Recorded prefix `547d5610d3a760c1` matches historical trainer SHA-256 `547d5610d3a760c14d28453b1606a12a82d7d13f1a8f1b8b7c28816b349070d6` | Run metadata stores a 16-hex prefix rather than the full hash |
| Q13 LR selection and gates | Reproduces selected 0.002 and Q2=true, Q3a=true, Q3b=false | The scorer does not bind benchmark files to a checkpoint hash |
| Nested sweep | All 40 new and 20 reused comparator files have the expected arm, rate, seed, 26,000 steps, and 26 evaluations | No device-level replication here |
| Nested learning outcome | Counts and registered mixed verdict reproduce; replacing any-evaluation success with the separately recorded final evaluation does not change these 60 binary outcomes | Does not remove rate-selection or small-seed uncertainty |
| Forward-transfer predicates | T1/T2/T7 true; T3/T4/T5/T6 false; primary means and T7 extrema reproduce | Inputs are rounded per-seed aggregates; not independent inference |
| Q14 | Recomputed ratio 1.0510569; incomplete status and last completed step verified | No long-run quality or stability result |
| Held-out shingle filter counterexample | A one-word shift defeats the implemented stride-seven overlap test despite 18 shared full 13-word windows | This demonstrates a coverage limitation, not an observed leak in the actual corpus |

For the trainer provenance see [Q13's exact historical trainer](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/12316003862ff4117c26583a65ea808e1ff1e2a8/research/frontier/ontogenesis/tpu/train_jax.py). Public primary reference pages were also checked on 2026-11-23 for SmolLM2 token counts, Pythia checkpoint exposure, and TPU v5e specifications; those checks are distinguished from the immutable repository evidence in section 9.

## 3. Dense model: specification, result, and attribution

### 3.1 The implemented dense reference

The completed Q13 model is a causal decoder with the following configuration. This is an implementation description from the trainer and run metadata, not a new proposed specification.

| Element | Implemented value |
|---|---|
| Vocabulary / context | 32,768 tokens / 512-token prediction context |
| Width / blocks | 1,280 / 20 distinct blocks |
| Attention | 20 query heads; 4 key/value heads; head dimension 64; causal mask; grouped-query attention |
| Position / normalization | RoPE with base 10,000; RMS normalization; query and key normalization |
| Feed-forward | SwiGLU form; intermediate width 3,456 |
| Embedding / output | Tied input embedding and output projection |
| Parameters | 386,062,080 |
| Numerical path | bfloat16 forward computation with float32 master parameters/optimizer handling in the trainer |
| Optimizer | Muon on block matrices; Adam-family handling for embeddings and other parameters; consistent-RMS setting 0.2; clipping at 1.0 |
| Learning-rate schedule | Peak 0.002; 500-step warmup; cosine schedule with 1% terminal floor |
| Weight decay | 0.1, configured for all parameter categories through the optimizer's separate handling |
| Batch and execution | 64 sequences × 512 targets; four TPU v5e chips; two micro-passes; seed 1000 |
| Corpus | Ten-billion-token corpus v2; uniformly sampled crops with replacement |
| Completed training | 234,950 steps; 7,698,841,600 processed target tokens |

The exact exposure is **19.94198 processed tokens per parameter**. Using the usual approximate `6ND` accounting gives **1.78334 × 10^19 FLOPs**; this is a derived approximation, not hardware-counter measurement. The tokens are exposures, not unique tokens. Under the source's homogeneous sampling approximation, expected covered corpus positions are about `1 − exp(−7.6988416/10) = 53.69%`. Correlated crops, document boundaries, and corpus duplication make this an estimate rather than a uniqueness certificate. Sources: [run metadata and curve](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/scale2/dense-XXL-lr0.002-s1000-tok7.7e%2B09-muon-dp4-fl0.01-hs-v2.json), [trainer](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/12316003862ff4117c26583a65ea808e1ff1e2a8/research/frontier/ontogenesis/tpu/train_jax.py).

### 3.2 Registered predictions and observed values

| Prediction | Registered predicate | Checked result | Verdict |
|---|---|---|---|
| Q1a | Best 101M rate is 0.004 | BPB index: 0.9540 at 0.002; 0.9544 at 0.004; 0.9577 at 0.008 | Fails, with observed winner 0.002 |
| Q1b | Tuned 101M improves old 0.9775 by at least 0.02 | 0.9775 − 0.9540 = 0.0235 | Holds |
| Q2 | 386M final mixed index in [0.79, 0.83] | 0.8049 | Holds |
| Q3a | Beat 211M on both plain ARC-Easy and PIQA | 0.5964 > 0.5215; 0.6779 > 0.6480 | Holds |
| Q3b | Within ±0.03 of Pythia-410M on both specified accuracies, and worse LAMBADA perplexity | ARC-Easy difference +0.0754; PIQA +0.0099; perplexity 32.06 > 10.8 | Fails on ARC-Easy's upper boundary |

Q3b must remain failed: outperforming the specified band does not retrospectively change the registered predicate. Conversely, it is a failed forecast rather than a quality regression. The 0.0004 LR difference does not by itself measure seed variance; the correct statement is that 0.002 was the lowest observed value among these one-seed runs. Source: [mechanical Q13 scorer](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/score_q13.py) and [recorded verdict](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/scale2/q13-verdict.json).

### 3.3 Meaning of the BPB scalar

| Domain | Q13 final BPB estimate | Fixed mixture weight |
|---|---:|---:|
| Web | 0.8877 | 0.62 |
| Textbook | 0.5307 | 0.13 |
| Mathematics | 0.8713 | 0.13 |
| Code | 0.6024 | 0.12 |

The trainer evaluates 20 batches of 16 sequences of 512 targets **per domain**, using the same seed-zero crop sampler at every evaluation. That is 163,840 sampled target positions per domain, 655,360 total. It converts average token negative log-likelihood using each validation domain's stored bytes-per-token, then computes the fixed weighted average of domain BPB estimates. Reweighting the rounded domain values gives 0.804922, consistent with the stored 0.8049.

This is a useful fixed index for comparisons made with the same tokenizer, held-out files, mixture and estimator. It should not be described as the loss over the entire held-out corpus or as an exact globally pooled byte rate. A globally pooled byte rate generally needs byte weights, while the configured mixture weights are token-domain mixture weights. The denominator definition should travel with every future chart.

The 211M reference finished at 0.8763; the observed improvement is 0.0714. That comparison changes size, token budget, optimizer and schedule together. It supports progress of the whole recipe. It does not identify the contribution of each component, nor fit a defensible scaling law from these few confounded points. At fixed 101M size and the older recipe, the raw corpus-v2 results do separately support 0.9775 → 0.9399 when exposure grows from approximately 0.8B to 2.0B tokens. Sources: [Q12/Q13 investigation](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/INVESTIGATION.md), [evaluation implementation](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/12316003862ff4117c26583a65ea808e1ff1e2a8/research/frontier/ontogenesis/tpu/train_jax.py).

## 4. Benchmark evidence and unresolved comparability

### 4.1 Scores with denominators

All cells below are plain accuracy from the same project prompt-formatting family, except the explicitly separate perplexity row. The references use their own tokenizers and model implementations.

| Task | Items | Nano 211M | Nano 386M | SmolLM2-135M | SmolLM2-360M | Nano386 − Smol135 |
|---|---:|---:|---:|---:|---:|---:|
| HellaSwag | 10,042 | 0.3076 | 0.3427 | 0.3548 | 0.4316 | −0.0121 |
| ARC-Easy | 2,376 | 0.5215 | 0.5964 | 0.6444 | 0.7037 | −0.0480 |
| ARC-Challenge | 1,172 | 0.2261 | 0.2602 | 0.2807 | 0.3652 | −0.0205 |
| PIQA | 1,838 | 0.6480 | 0.6779 | 0.6844 | 0.7209 | −0.0065 |
| WinoGrande | 1,267 | 0.5107 | 0.5201 | 0.5296 | 0.5896 | −0.0095 |
| OpenBookQA | 500 | 0.1840 | 0.2180 | 0.2180 | 0.2460 | 0.0000 |
| LAMBADA accuracy | 5,153 | 0.3262 | 0.3544 | 0.4289 | 0.5379 | −0.0745 |
| LAMBADA word perplexity | Same 5,153 | 49.01 | 32.06 | 19.26 | 9.44 | +12.80; lower is better |

**Correction required:** the seven task denominators sum to **22,348**, not the 30,328 stated in the investigation's harness overview. Per-task sizes match across the inspected Nano, SmolLM2 and early Pythia result files. Sources: [Q13 benchmark](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/scale2/dense-XXL-lr0.002-s1000-tok7.7e%2B09-muon-dp4-fl0.01-hs-v2.bench.json), [SmolLM2-135M result](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/lm/eval-reference-SmolLM2-135M.json), [SmolLM2-360M result](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/lm/eval-reference-SmolLM2-360M.json).

Length-normalized accuracy is a distinct metric: for Q13 it is 0.4102 HellaSwag, 0.5345 ARC-Easy, 0.2918 ARC-Challenge, 0.6763 PIQA and 0.3120 OpenBookQA. In particular, OpenBookQA's plain score is below its nominal four-option random baseline while its length-normalized score is above it. Tables and statistical claims must name the selected metric rather than silently switch between them.

### 4.2 Context controls: useful decomposition, limited causal interpretation

| Task | Nano386 context gain | Smol135 context gain | Smol360 context gain |
|---|---:|---:|---:|
| HellaSwag | +0.0404 | +0.0487 | +0.0891 |
| ARC-Easy | +0.2631 | +0.3157 | +0.3939 |
| ARC-Challenge | +0.0384 | +0.0606 | +0.1451 |
| PIQA | +0.0468 | +0.0571 | +0.0582 |
| OpenBookQA | +0.0920 | +0.0880 | +0.1020 |
| LAMBADA | +0.3544 | +0.4289 | +0.5379 |

Here gain means accuracy with the prompt minus accuracy of the continuation scored after an end-of-text marker alone. The WinoGrande control is excluded: removing its option-filled prefixes leaves identical suffixes, so the recorded 0.4957 reflects deterministic tie breaking and label frequency. It is not a meaningful context-free reasoning baseline. Sources: [Nano no-context result](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/scale2/dense-XXL-lr0.002-s1000-tok7.7e%2B09-muon-dp4-fl0.01-hs-v2.bench-nocontext.json), [reference no-context result](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/lm/eval-reference-SmolLM2-135M-nocontext.json), [scoring code](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/eval_score.py).

The source says all Nano386/Smol135 no-context differences are within 0.004. The stored ARC-Easy values differ by **0.0046**, so “within about 0.005” is accurate. The close baseline scores and larger prompt-conditioned gaps make prompt use a sensible diagnostic target. They do not prove equal knowledge priors or identify a deficient reasoning mechanism. The difference decomposition is algebraic, and stripping a prompt changes the scoring distribution; it does not hold every alternative cause constant.

### 4.3 Modern-model and early-checkpoint interpretation

The external model cards directly confirm 2T training tokens for SmolLM2-135M and 4T for SmolLM2-360M. The Nano386 run therefore uses about **1/259.78** of Smol135's token exposures while having about 2.86 times its named parameter count. The same approximate `6ND` accounting yields a roughly 90.84-fold compute difference. These are meaningful resource-context estimates. The models differ in corpus composition, tokenizer, training schedule, architecture and implementation, so “the gap is caused by training compute” is too strong. The evidence is consistent with limited compute being an important constraint; it does not isolate causality. Primary token-count sources: [Smol135 card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [Smol360 card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M).

The new Pythia-410M `step3000` and `step4000` files strengthen the claim that the Nano recipe is competitive with **those early public checkpoints**. Nano386 exceeds both on all seven recorded accuracies and has lower LAMBADA perplexity. At `step4000`, for example, ARC-Easy is 0.4280 versus Nano's 0.5964; PIQA 0.6137 versus 0.6779; LAMBADA perplexity 59.04 versus 32.06. EleutherAI specifies 2,097,152 tokens per step, giving 6,291,456,000 and 8,388,608,000 token exposures respectively. They are early checkpoints in a much longer training schedule, not models with independently optimized terminal schedules at those budgets. A claim of equal-budget optimality would therefore be unjustified. Sources: [step3000 result](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/lm/eval-pythia410m-step3000.json), [step4000 result](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/lm/eval-pythia410m-step4000.json), [Pythia card](https://huggingface.co/EleutherAI/pythia-410m).

The source mentions a Cerebras-GPT comparison as in progress. No completed result from that prospective comparison is incorporated into this review.

### 4.4 Statistical and harness limitations

The committed files hold rounded task aggregates, not per-example correctness or likelihood arrays. They do not support exact paired tests of model differences, paired bootstrap intervals, item-specific contamination exclusions, or reliable estimates of run-to-run training variability. The datasets are the same across models, so independent-binomial error bars omit covariance. The reanalysis includes an explicitly descriptive unpaired approximation to expose scale, not to manufacture a significance gate.

The reassessment's statement that three tasks differ by more than about three standard errors needs a metric and method. With **plain accuracy** and an independent-binomial approximation, HellaSwag is about 1.8 standard errors; ARC-Easy is about 3.4 and LAMBADA about 7.8. A HellaSwag normalized-accuracy calculation is a different comparison. Neither substitutes for item-paired uncertainty, and none addresses seed variation.

The Nano scorer prefixes an end-of-document token; the public-model scorer does not prefix nonempty contexts. The preparation path tokenizes Nano context and continuation separately, while the public scorer tokenizes their concatenation and splits by the context-token count. These are documented implementation differences. Calibration against Pythia-160M is encouraging on several tasks but does not make the implementations identical. The source itself reports a roughly 0.025 LAMBADA-accuracy calibration difference from official results. Pythia checkpoint revision names are now recorded, but the older Smol reference files lack immutable downloaded-weight hashes. See [preparation](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/eval_prep.py) and [public-model scorer](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/eval_score_hf.py).

The claim that the longest benchmark example has 259 tokens is source-reported. Prepared task/tokenizer artifacts or a length-distribution report were not located among the inspected committed sources, so this review does not independently certify the maximum. A tiny artifact containing tokenizer hash, task-data hash, item counts and per-task length maxima would close this gap without retraining.

### 4.5 Contamination and held-out separation

The separate benchmark overlap scan tests every 13-token window against the 10B-token corpus. Any-overlap rates are 2.08% HellaSwag, 1.52% ARC-Easy, 1.79% ARC-Challenge, 1.25% PIQA, 0% WinoGrande/OpenBookQA, and 0.23% LAMBADA. Answer-touching rates are 1.27%, 0.17%, 0.26%, 1.25%, 0%, 0%, and 0.02% respectively. These are overlap detector outcomes, not proof of benchmark training or proof of clean paraphrase/source separation. The scan does not retain per-item overlap identifiers in its output. Source: [v2 overlap result](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/lm/eval-contamination-v2.json), [detector](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/eval_contam.py).

A different mechanism filters training documents against held-out BPB documents. Its `ngrams13` checks 13-word windows starting every seven words, not every possible offset. A deterministic counterexample uses 30 distinct words and the same passage with one word prepended: the full sets share 18 windows; the two stride-seven sets share none. This refutes the stronger implementation claim that every shared 13-word window is removed. It does **not** establish actual train/validation leakage. The corpus builder also excludes the held-out source shard and checks held-out file hashes, which are useful independent controls. Source: [corpus builder](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/build_corpus_v2.py). Future corpus releases should specify sampled-shingle coverage accurately, use stable full-coverage checks when intended, and preserve the original corpus/results rather than silently relabeling them.

## 5. Nested computation: what the learning-rate control resolves

The registered experiment compares repeated stacks and untied depth on synthetic BINDINGS. The implementation reapplies the same block stack; it does not add a new learned halting mechanism, latent search algorithm, or equilibrium solver. Its event Transformer differs from the dense language-model implementation: LayerNorm/GELU and full attention in the small BINDINGS model versus RMSNorm/SwiGLU and GQA in Q13. Sources: [nested protocol](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/PROTOCOL-NESTED.md), [baseline code](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/baselines.py).

### 5.1 Exact registered estimand

Each cell has five seeds, 21–25, each run for 26,000 optimizer steps. A run counts as learned if its two-hop accuracy reaches 90% at **any** of 26 evaluations. Periodic evaluations use the same fixed seed and 2,048 mixed-hop queries; the separately recorded final evaluation uses a seed-dependent test with 4,096 mixed-hop queries. Success counts are run-level outcomes, not 26 independent trials per run. The primary rule takes the best count across the three rates **separately for each arm**.

| Arm | Shared parameters | Layer applications | Rate 1e−3 | Rate 3e−4 | Rate 3e−3 | Best |
|---|---:|---:|---:|---:|---:|---:|
| One layer repeated four times | 128,705 | 4 | 2/5 | 0/5 | 0/5 | 2/5 |
| Four distinct layers | 277,889 | 4 | 1/5 | 0/5 | 0/5 | 1/5 |
| Four-layer block repeated twice | 277,889 | 8 | 4/5 | 1/5 | 0/5 | 4/5 |
| Eight distinct layers | 476,801 | 8 | 0/5 | 0/5 | 0/5 | 0/5 |

All 60 primary files were inspected and the table recomputed. The protocol requires both pairs to show an advantage of at least two successes to declare the reuse advantage standing; it requires both pairs to differ by at most one to say learning rate explains the difference. One pair meets each condition, so the combined outcome is **mixed**. L1/L2, predicting at least three successes for the untied arms at 3e−4, fail. L3, predicting no improvement at 3e−3, holds. Sources: [LR protocol](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/PROTOCOL-NESTED-LR.md), [scorer](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/score_nested_lr.py), [verdict](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/nested-lr/verdict.json).

### 5.2 Correct scope for the architecture ledger

The lower rate did not rescue either untied arm within this budget. That weakens the specific explanation that trying 3e−4 would remove the observed gap. It does not establish optimizer-independent superiority: there are only three rates, five seeds, one synthetic task family and one fixed horizon. Rate choice reuses the outcome seeds. Lower rates may need longer budgets; the source itself identifies delayed transition as an unresolved alternative.

Equal layer-application count is a useful compute proxy at equal width and input shape. It is not equality of parameter count, optimizer state, peak memory, realized FLOPs or measured training time. Shared blocks have fewer distinct parameters and different optimization dynamics; either can contribute to results. A next experiment should state which budget it matches instead of saying merely “equal compute.”

The earlier nested protocol had device deviations, including laptop-memory failures and CPU reruns. Its deviation rule withholds formal success if the registered-device and device-consistent interpretations disagree. The later LR sweep does not retroactively erase that earlier status. It provides a new scoped result. See [deviation record](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/PROTOCOL-NESTED-DEVIATION.md).

Extra test-time loops did not reliably yield unseen three-hop performance. A declining relative state displacement on a sampled batch is not a contraction bound and cannot prove convergence, uniqueness of a fixed point, or semantic correctness. Reuse deserves a candidate branch and a falsifiable protocol; it does not yet deserve replacement of the dense reference.

## 6. New forward-transfer pilot: retention is not transfer

### 6.1 World and denominators

The synthetic world has 23 inputs and six hidden permutation primitives, plus two unused primitives for a negative control. Training presents 24 ordered two-primitive task families. The primary test adapts a new 16-number task vector on eight examples from each of eight held-out combinations and evaluates the 15 unshown inputs. That is **120 query predictions per seed** and three seeds, 9101–9103. The experimental unit remains the seed/world and family; 360 predictions are not 360 independent training replications.

Each training phase takes 300 steps. Replay arms train on all families encountered so far; sequential arms train only on the current family. Replay therefore processes 2,070,000 stream examples versus 165,600 for sequential arms across 24 phases: a 12.5× exposure difference at the same step count. All trained arms also retain 24 task vectors, **384 additional learned numbers** omitted from the printed shared-model parameter column. Replay stores 552 input/output examples. Source: [forward-transfer implementation and registration](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/forward_transfer.py).

| Trained arm | Shared model parameters | Stream retained | Primary new-combination accuracy | Per-seed primary values |
|---|---:|---:|---:|---|
| Dense | 24,599 | 0.0845 | 0.0444 | 0.0333, 0.0417, 0.0583 |
| Dense + replay | 24,599 | 1.0000 | 0.0278 | 0.0250, 0.0417, 0.0167 |
| Loop | 20,631 | 0.0852 | 0.0361 | 0.0333, 0.0333, 0.0417 |
| Loop + replay | 20,631 | 0.9982 | 0.0417 | 0.0583, 0.0417, 0.0250 |
| Menu | 8,464 | 0.6455 | 0.0278 | 0.0083, 0.0417, 0.0333 |
| Menu + replay | 8,464 | 0.8847 | 0.0417 | 0.0417, 0.0333, 0.0500 |

The uniform random-label reference is `1/23 = 0.04348`. The fresh-network control scores zero because learned predictions can concentrate on already observed outputs, which are wrong on unshown inputs of a permutation. It is a useful measured control, but the fresh score is not the chance level.

T1/T2/T7 hold; T3/T4/T5/T6 fail. The source explicitly defines this as a **development pilot with predictions, not a confirmatory pass/fail gate**. The architecture ledger should preserve that distinction even though the report calls it gate G07. Source: [mechanical verdict](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/forward_transfer/verdict.json).

### 6.2 Post-hoc controls and two overclaims to correct

The post-hoc true-primitive menu obtains about 71.7% at eight examples. The trained-family probe obtains roughly 41% for dense replay and 42% for loop replay at eight examples; both rise with sixteen examples. These show that the probe can recover some known-family structure and can detect a hand-constructed compositional solution. They also show substantial adaptation error even where a suitable solution is known to exist. Source: [control implementation](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/forward_transfer_control.py), [control results](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/forward_transfer/control.json), [oracle results](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/forward_transfer/oracle_menu.json).

Two conclusions in the narrative should be narrowed:

1. A 200-step optimizer failing to fit novel-family examples from the tested initialization does **not** prove that no setting of the task vector can fit them. State “the tested adaptation procedure found no useful task vector.” A representation nonexistence claim would require a global certificate or a complete search in a restricted model.
2. One true-primitive menu reaching 72% with a particular probe does **not** prove that no arm could reach the registered 90% bound. It demonstrates that the chosen positive control did not calibrate that bound as intended. The threshold should remain a failed prediction and be redesigned prospectively, not converted into an impossibility theorem.

The defensible conclusion is substantive: **perfectly retaining the experienced tasks did not produce measured forward transfer under the tested training and adaptation procedure**. This motivates testing objectives and adaptation, as well as structure. It does not eliminate dense or recurrent architectures from future consideration. The menu's structure and optimizer were changed after an unregistered-seed stream-fit check but before the registered runs, with the deviation recorded; it is a development-stage engineering decision, not a blind architecture comparison.

## 7. Hardware and operational results

### 7.1 Completed dense run and speed measurements

| Measurement | Checked value | Interpretation |
|---|---:|---|
| Q13 four-chip steady training rate | 105,898 tokens/s | Timed training chunks; excludes compilation and other un-timed overhead |
| Q13 time to last recorded evaluation | 20.7011 hours | Includes more of the execution path than chunk throughput |
| Derived tokens/time to last evaluation | About 103,307 tokens/s | Does not include initial provisioning/corpus download or final benchmark work |
| One-chip, micro=2 speed experiment | 30,774 tokens/s | Short same-model engineering benchmark |
| One-chip activation recomputation | 25,518 tokens/s | 17.08% lower throughput than the two-pass comparator |
| Four-chip scaling versus four times one-chip rate | About 86.03% | Cross-run estimate, not a controlled distribution of scaling trials |
| Approximate one-chip `6N × tokens/s / peak` | About 36.18% | Model-FLOP approximation; not a direct full-step profiler measurement |

The Q13 job log contains the actual one-pass OOM, an exit-1 record, successful two-pass start, final completion, upload success and benchmark exits. It is stronger support than the earlier lost first-machine log. The error attempted a 9.71G reservation with 9.68G available. This illustrates why the printed post-evaluation live-allocation value cannot stand for peak training memory. Source: [Q13 execution log](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/scale2/job-scale2b-log.txt), [speed summary](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/bench-speed/log.txt).

The trainer's `device_gb_in_use` is actually `bytes_in_use / 2^30` sampled after evaluation: a **GiB live-allocation snapshot**, not a peak GB measurement. Future ledgers should name it accurately and add allocator peak/reservation and compiler memory analysis where available. Google specifies 197 BF16 TFLOP/s and 16 GiB HBM per v5e chip, with some HBM reserved by the runtime. [Primary hardware specification](https://docs.cloud.google.com/tpu/docs/v5e).

For the earlier 211M v6e test, 300-step raw files record 132,810 tokens/s on one chip, 236,293 on two, and 483,615 on four. Four-chip scaling is about 91.03% relative to the one-chip rate; batch 256 reaches 494,758. Global batch, gradient accumulation and physical chip count must be retained alongside these figures: the 64-sequence runs use micro counts 4, 2 and 1 respectively. Short training-throughput measurements cannot certify serving latency, energy efficiency, or a general hardware ranking. Sources: the frozen [`results/scale1/throughput` files](https://github.com/wwk5q8z6kk-bit/nano-lm/tree/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/scale1/throughput).

### 7.2 Q14: a failed speed gate and missing quality results

Q14 planned five arms: existing standard-head seed1000, a second standard-head seed, two fused 128-head runs, and a 128-head standard-kernel comparator. Only the existing baseline and the interrupted first fused run are present as completed/partial evidence.

The fused run's last curve row is step 2,000, 65,536,000 tokens, BPB index 1.2098, throughput 461,538 tokens/s. Its metadata also contains `steps=24400` and `train_tokens=799539200`; those are **planned totals**, and `finished=false`. Treating that top-level token field as completed exposure would overstate the run by more than twelvefold. The ratio to baseline 439,118 tokens/s is 1.0510569, below W1's 1.25. W2 (two-seed quality) and W3 (kernel parity) are **unmeasured**, not failed or passed. Source: [partial fused result](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/q14/dense-L128-lr0.002-s1000-tok8e%2B08-muon-dp4-fl0.01-splash-hs-v2.json).

Keeping the existing heads follows the registered adoption rule because the speed requirement failed. It does not show the new heads worsen quality. The machine's unresponsiveness has no established cause in the inspected evidence; do not turn a possible spot reclaim or possible kernel hang into a verified diagnosis. The source's future requirement for spot supervision is a reasonable operational response. The broader inference that attention is not the dominant bottleneck still needs profiling, since this comparison changes both head organization and kernel while covering only an early segment.

### 7.3 Cost evidence

The investigation supplies list-price estimates and upper bounds, explicitly saying billing has not been read. Q13's approximately $110 compute upper bound excludes corpus-transfer egress; Q14's failed machine incurred reported hours beyond its intended useful run. These are historical source estimates, not independently verified invoices or current prices. The architecture's resource ledger should separate **planned budget, useful training, failed/restarted time, storage/egress, billed total, and evidence status**. This review accessed no billing account and makes no current-resource-deletion assertion.

### 7.4 Addendum: corrected 101M LDO and 40M cache gates are no longer pending

The [research decision note](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/theory/RESEARCH-DECISION-20261121.md), section 10, explicitly overrides its earlier broader store rule. Before the 101M reading, it replaced leave-position-out fitting with leave-document-out fitting and narrowed the conclusion below 0.01 BPB: close **this store as a complement**, not the entire n-gram-store line. The current frozen sources contain the completed result, not merely a planned evaluation.

| Complement study | Registered read point | Raw gains in BPB | Control and denominator | Updated source status |
|---|---|---|---|---|
| Corrected LDO store | 101M checkpoint; 0.01 lower continuation threshold | Web 0.0013; textbook 0.0012; math 0.0031; code 0.0028; mix **0.0017** | 48M stored tokens = 6% of the 800M-token corpus; 1,048,576 fitting tokens; shifted-store control with the **real fitted weights** gives −0.0028 mixed gain and is negative in every domain | Close this fixed 48M-token store as a useful complement at the registered threshold |
| Exact in-context cache | **40M**, not 101M; 512-token context | Web 0.0012; textbook −0.0003; math 0.0036; code 0.0059; mix **0.0018** | 524,288 fitting tokens; recorded causality checks pass; foreign-cache gain 0.0000; base score agrees within 0.0001 | Every domain is below 0.01: close the tested explicit-copy line at this scale/context |

The LDO results across sizes are 0.0096 at 14M, 0.0037 at 40M and 0.0017 at 101M. The old leave-position-out estimator gave 0.0044, −0.0014 and −0.0032. The corrected estimator therefore reverses the apparent harm at the two larger sizes while still failing the usefulness threshold. All gain arithmetic and negative controls were checked from the raw BPB cells. Sources: [LDO implementation](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/store_ldo.py), [101M raw LDO result](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/lm/store-ldo-L.json), [cache protocol and implementation](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/cache_exchange.py), [40M raw cache result](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/lm/cache-exchange-M.json).

Five limitations should accompany the closure entries:

1. These use one checkpoint per size from the first-version sampler, whose bias remains recorded. Their gates can be read on the observed checkpoints without claiming the sampler issue was repaired in them.
2. LDO repairs **post-hoc mixture fitting**, not the earlier jointly trained model. The earlier joint-training S2/S3 claims remain confounded/unsettled because both same-document leakage and gate optimization differed. Do not use the LDO result to mark a corrected joint-training experiment complete.
3. The store covers 6% of the corpus and has source-reported storage near 1.02 GB. Full-corpus memory is untested here. Closure is an engineering priority decision for this complement, not a theorem that external memory cannot help.
4. The cache tests exact copying from the model's own 512-token window. It says nothing about longer-context retrieval, a different pointer implementation, or the newer 386M checkpoint.
5. The cache's foreign control **refits its own mixture table**. Its zero gain is therefore weaker than LDO's fixed-real-weights negative control: optimization can suppress the foreign cache. The explicit cache-update order and causality check provide separate implementation evidence; zero foreign gain alone is not a comprehensive leakage certificate. The script also records that an earlier future-leaking cache result was voided; that invalid result must never enter a performance table.

Accordingly, V6 can replace V5's pending status with two scoped source-completed decisions, while retaining the distinct unresolved memory questions.

## 8. Recommended ledger updates and next discriminating work

### 8.1 Concrete decision entries

| Suggested entry | Decision or correction | Reason | Reopening evidence |
|---|---|---|---|
| Dense-reference status | Retain Q13 as completed in-house dense reference | Best completed inspected language result, with verified arithmetic/provenance | Reproducible improvement on registered quality and resource criteria |
| Core model ownership | Carry forward source-recorded own-foundation-model direction | Explicit owner reassessment at frozen head | Owner direction changes |
| Block-reuse candidate | Keep a candidate; global nested-LR result stays mixed | Whole-block contrast is large; single-layer contrast is small | Registered language experiment with budget accounting and seed uncertainty |
| Transfer evidence | Record no transfer under tested protocol; retention and transfer separate | Near-perfect retention coexists with near-chance new-combination scores | Calibrated adaptation plus prospective held-out transfer evaluation |
| Attention configuration | Keep 64-dimensional heads/current path for this decision | Q14 speed requirement unmet | New profiler-based hypothesis and a new registered threshold |
| Benchmark denominator | Correct total to 22,348; retain each task's n | Raw files agree and prose total does not | A versioned change to evaluation sets |
| Context interpretation | Use descriptive context-gain language | Control does not identify mechanism | Controlled causal ablations with item-level outputs |
| Data separation | State stride-seven shingle coverage and preserve old artifacts | Deterministic counterexample to all-window wording | Versioned full-coverage audit and observed affected-item counts |
| Result accounting | Distinguish planned and completed tokens | Q14 top-level metadata is planned | Schema validation that enforces both fields |
| Compute explanation | Record resource mismatch as a constraint, not an identified sole cause | Multiple cross-model differences remain | Matched data/tokenizer/schedule comparisons or stronger controlled evidence |
| Statistical claims | Specify metric, seed unit and paired uncertainty | Rounded aggregate files cannot settle several claims | Per-item outcomes and repeated training seeds |

### 8.2 Cheapest next steps, ordered by information gained

1. **Complete artifact identity and measurement contracts before spending.** Record checkpoint SHA-256, exact tokenizer and prepared-task hashes, reference model revision hashes, per-task item counts, length histograms/maxima, prompt/normalization versions and scorer hash. Preserve raw likelihoods or correctness for paired analysis. Add explicit `planned_tokens`, `completed_tokens`, `status`, and `last_completed_step` to the next schema. This review's exact-source manifest and arithmetic are already done; generating missing model-side artifacts requires their existing local files, not retraining.
2. **Prepare the owner's D evaluation as the first model-quality discriminator.** Freeze a small generative suite with exact-answer parsers, known-positive/negative fixtures, task-instance disjointness, output-length limits, decoding configuration, failure accounting and separate one-hop/two-hop reasoning strata. Prepare the evaluator before running it on existing checkpoints; preserve continuations and item-level outcomes. “Free” in the source means no new training, not literally zero inference resources.
3. **Calibrate forward-transfer adaptation before another mechanism claim.** On existing frozen models if checkpoints exist, recover known trained-family vectors, compare those vectors with the probe's outputs, and bound adaptation sensitivity with a predeclared tiny grid of starts/steps/rates. Calibrate on disjoint development families using the true-primitive control. Do not select hyperparameters on the final new-combination test. If usable checkpoints were not saved, make that a recorded blocker instead of silently retraining.
4. **Resolve benchmark contamination sensitivity at item level.** Save overlap IDs and score clean/flagged subsets without changing the original published aggregate. A 1.21-point HellaSwag plain-accuracy gap is smaller than the 2.08% any-window overlap rate in Nano's corpus, so a blanket “negligible for every comparison” claim is not justified from aggregates alone. Overlap is not proof of influence, and reference contamination remains separately unknown.
5. **Only then register the owner's B recurrent-versus-dense language test.** Define parameter count, physical forward/backward FLOP accounting, optimizer work, exposure count, terminal schedule and wall-time cap separately. Match the main fairness criterion explicitly. Use three independent seeds as proposed, pair data streams, lock a real-text reasoning probe before reading outcomes, and include fixed-depth controls. Success must include retained ordinary language quality, not only the preferred reasoning slice.
6. **Treat A continuation as a separate schedule experiment.** The 386M reference reached a 1% terminal LR floor. Continuing for another 1–2B tokens requires a new schedule/resume identity, not merely extending the prior step budget. Freeze whether optimizer state and sampler state continue, define the decay-data mixture, and preserve original checkpoint/results. If changed data quality and schedule are combined, label the combined intervention accordingly.
7. **Register C generated reasoning text with a matched comparison.** Specify token/exposure budget, generator identity, data provenance, deduplication, verification quality, and equal compute or equal wall time as the primary criterion. Keep teacher-generated text separate from impossible direct cross-tokenizer logit matching. Synthetic text can change data quality, length and difficulty; the experiment must measure those changes rather than crediting architecture automatically.

Steps 5–7 remain prospective research plans with the source's cost estimates, not authorized or executed jobs in this review. The present result is sufficient to write concrete protocols and prepare evaluation assets. It is insufficient to bypass budget and execution gates.

### 8.3 Formal claims that can safely enter the architecture

The exact arithmetic identities, scorer predicates, parameter-sharing implementation, and finite counterexample above are checkable statements. The experiments supply evidence about specified distributions and procedures. None supplies universal generalization, convergence, transfer, correction, or safety guarantees for the full language model.

For a shared stack `F_theta` repeated `R` times, the implementation establishes weight sharing and the forward composition `F_theta^R`. A contraction theorem would additionally require a specified domain and a verified Lipschitz constant below one; observed relative movement on one batch supplies neither. Likewise, finite failure to optimize a task vector establishes an algorithm's failure on those trials, not nonexistence of a vector. These distinctions should appear beside the theoretical ledger rather than being left to an appendix.

## 9. Primary references and provenance notes

Repository links throughout this note are immutable commit references. Every materialized repository file, including all 60 nested comparator runs and all forward-transfer result files, appears in `upstream_evidence/manifest.json`; this avoids a giant duplicated bibliography while retaining exact provenance.

| External primary reference checked on 2026-11-23 | Fact used | Boundary |
|---|---|---|
| [HuggingFaceTB SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 2T-token exposure and dataset-family description | Does not validate the project's locally scored checkpoint bytes |
| [HuggingFaceTB SmolLM2-360M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | 4T-token exposure | Same boundary |
| [EleutherAI Pythia-410M model card](https://huggingface.co/EleutherAI/pythia-410m) | 2,097,152 tokens per step; 143,000-step training; public intermediate revisions | Early checkpoint comparison does not equal terminal-budget optimality |
| [Google Cloud TPU v5e specification](https://docs.cloud.google.com/tpu/docs/v5e) | 197 BF16 TFLOP/s, 16 GiB HBM per chip and runtime reservation distinction | Peak specifications are not measured end-to-end utilization or current cost |

For research integration, prefer the exact findings, predicates, corrections and scoped implications above. Preserve the original protocols and result files as history; attach corrections through versioned review/ledger entries.
