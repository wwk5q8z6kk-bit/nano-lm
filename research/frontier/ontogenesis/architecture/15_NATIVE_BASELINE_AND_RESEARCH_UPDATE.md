# 15 — Native baseline, shared depth, transfer and resource evidence

**V6 review date:** November 23, 2026. **Main source cutoff:** `3a38b5f1529a718838b5bc83a010574bfa9a4c6e`. This chapter updates the research state beyond V5's late cutoff. It reports source inspection and arithmetic on committed results; no model training or checkpoint inference was rerun in this continuation. The detailed review and immutable source manifest accompany the package.

## Executive summary

The project now has a reported 386M native dense reference with a substantially better complete recipe than the earlier 211M reference. That progress is useful, while its causes remain confounded by size, optimizer, schedule and token exposure. Shared-depth BINDINGS experiments justify a candidate branch rather than architecture adoption. The transfer pilot separates perfect retention from successful transfer: one can occur without the other under the tested adaptation procedure. New store and cache files also close specific numerical gates that were pending in V5.

| Area | Current evidence | Architecture decision |
|---|---|---|
| Native foundation model | 386,062,080 parameters; 7,698,841,600 token presentations; 0.8049 fixed mixed BPB index | Use the exact reported recipe as the current dense reference |
| Seven-task benchmarks |386M exceeds 211M on every recorded plain accuracy; remains below/equal to Smol 135 and below Smol 360 | Preserve progress and remaining quality gap without causal overclaim |
| Shared-depth rate sweep | Whole-block reuse 4/5 versus untied 0/5; single-layer reuse 2/5 versus dense 1/5 | Registered combined verdict stays mixed |
| Forward transfer | All six trained arms near chance on new combinations; dense replay retains 100% of experienced tasks | Retention is not measured transfer; redesign adaptation prospectively |
| Hardware experiment |1.05106× speed versus 1.25× requirement; only 65,536,000 tokens completed | Speed gate fails; quality and parity remain unmeasured |
|101M learned store |Mixed gain 0.0017, below 0.01 stopping threshold | Close this tested store complement; broader store questions remain open |
|40M causal cache |Every domain gain below 0.01 | Close this explicit-copy line at this scale/context; retain provenance qualifications |

All repository claims in this chapter are supported by the [detailed upstream audit](reviews/v6_upstream_evidence_review.md), [source manifest](evidence/source_manifest_v6.json) and [reanalysis](results/upstream_reanalysis_v6.json). Exact primary Git sources are linked throughout the audit. External reference token counts are independently grounded in [P-44–P-45](04_EVIDENCE_AND_REFERENCES.md).

## 1. Source boundaries and changing authority

The main research cutoff is 17 commits after the daily review cutoff`6df72ff1f69ae91de6159bce9a7bb005561b68dc`. New evidence includes forward-transfer work, post-hoc controls and early Pythia 410M comparisons. Commit counts are a history measure, not experiment counts. The source investigation records the owner's direction to build Nano LM's own foundation model and core architecture, with external pretrained models serving as references. This continuation carries that direction forward without treating it as a new compute authorization.

The source audit verifies exact text/result bytes and scorer arithmetic. It does not recover all trained checkpoint weights, corpus bytes, tokenizer artifacts, prepared benchmark items or per-example predictions. Run completion and quality figures remain reported empirical evidence with those limits. G04/G06 are separate pinned branches reviewed in [17](17_CORRECTION_MEASUREMENT_AND_PROTECTION.md); their status is not silently promoted into the main branch.

## 2. Exact native recipe and BPB estimand

The 386M model has width 1280, 20 blocks, 20 query heads, 4 KV heads, 64 dimensions/head, FFN3456 and vocabulary 32768. It uses pre-RMSNorm, normalized Q/K, RoPE, grouped KV attention, gated SiLU feed-forward units and a tied embedding/head. [The full architecture](00_FULL_ARCHITECTURE.md#3-the-native-decoder) gives the tensor inventory and parameter formula.

The run records 234,950 steps with 64×512 targets per step, seed 1000, two micro-passes and four devices. Muon updates block matrices and an Adam-family optimizer updates embedding/gain tensors; peak rate 0.002,500 warm-up steps, 0.1 weight decay and 1% terminal rate are recorded. The old trainer prose describing a 10% floor is not the actual override in this run. Its historical trainer SHA256 matches the run's recorded 16-hex prefix. The result's completed flag and last curve step agree with the planned training steps.

The total 7,698,841,600 is **processed target exposure**, not distinct corpus tokens. Uniform host-sampled crops can repeat. The ratio to 386,062,080 parameters is 19.94198. The rough`6ND` accounting yields 1.78334×10^19 FLOPs, explicitly an estimate rather than a hardware measurement. Neither this estimate nor a nominal corpus size establishes unique-data coverage.

| Domain | Final sampled BPB | Fixed weight |
|---|---:|---:|
| Web |0.8877|0.62|
| Textbook |0.5307|0.13|
| Mathematics |0.8713|0.13|
| Code |0.6024|0.12|

The evaluator uses 20 batches of 16 sequences of 512 targets per domain with fixed seed-zero crops: 163,840 sampled target positions/domain and 655,360 total. It estimates domain BPB using stored bytes/token and averages those domain values with the above weights. Recombining the rounded values gives 0.804922, consistent with reported 0.8049. This is a fixed sampled index; it is not whole-corpus pooled BPB. The index improves by 0.0714 relative to 211M's 0.8763, but the comparison changes multiple recipe components.

## 3. What the benchmark scores support

These are plain accuracies from the project harness. They are not copied from model-card leaderboards and should not be mixed with length-normalized scores.

| Task | Items | Nano 211M | Nano 386M | SmolLM2-135M | SmolLM2-360M |
|---|---:|---:|---:|---:|---:|
| HellaSwag |10,042|30.76%|34.27%|35.48%|43.16%|
| ARC-Easy |2,376|52.15%|59.64%|64.44%|70.37%|
| ARC-Challenge |1,172|22.61%|26.02%|28.07%|36.52%|
| PIQA |1,838|64.80%|67.79%|68.44%|72.09%|
| WinoGrande |1,267|51.07%|52.01%|52.96%|58.96%|
| OpenBookQA |500|18.40%|21.80%|21.80%|24.60%|
| LAMBADA |5,153|32.62%|35.44%|42.89%|53.79%|
| **Total items** |**22,348**| | | | |

The investigation's 30,328 total is an arithmetic error. The correction changes the reporting denominator, not the saved per-task scores. LAMBADA word perplexity improves 49.01→32.06 for Nano 211M→386M; the recorded Smol 135/360 values are 19.26/9.44. Lower perplexity is better.

The registered Q13 numeric predicates reproduce: Q2 holds, Q3a holds and Q3b fails. Q3b required being within a symmetric±0.03 band of a specified Pythia comparison on two metrics, so exceeding the upper band still fails that forecast. It is not a quality regression and must not be relabeled a pass after seeing the result. The learning-rate prediction that 0.004 would win failed; 0.002 is the observed lowest of three one-seed values 0.9540, 0.9544 and 0.9577. The 0.0004 difference is not a seed-variance estimate.

SmolLM2's official cards report 2T tokens for 135M and 4T for 360M. Nano 386 uses about 1/260 of Smol 135's nominal token exposures, with a larger named parameter count. Data mixture, tokenizer, optimization, context and architecture differ. The comparison suggests that exposure is an important research variable; it does not prove that training compute alone caused the quality gap. [P-44–P-45]

Nano 386 also exceeds the newly recorded Pythia 410M step 3000/4000 plain accuracies. These are early checkpoints from a longer schedule, not optimized terminal runs at those budgets. Pythia is valuable for controlled study of training trajectories, but checkpoint-step comparisons do not establish globally optimal token efficiency. [P-48]

### 3.1 Context controls and remaining harness work

No-context scoring removes the question/prefix while preserving candidate continuation scoring. Nano 386 and Smol 135 have similar no-context accuracy on several tasks; their prompt-conditioned gains differ. ARC-Easy's no-context difference is 0.0046, so “within 0.004 on every task” is literally too tight. WinoGrande becomes identical suffixes after prefix removal, making its 49.57% outcome a tie/label-frequency control rather than evidence about reasoning without context.

Subtracting context-free scores is a useful descriptive decomposition. It does not identify the causal mechanism of the remaining gap or prove identical knowledge. Native and HF scorers also differ in BOS behavior and context/continuation tokenization. This is a documented comparability risk, not proof all scores are wrong. The source's maximum-length 259 claim lacks an inspected item-length manifest with tokenizer and checkpoint identities. [The context review](reviews/v6_context_audit_review.md) and [chapter 16](16_GENERATIVE_REASONING_EVALUATION.md) specify the missing receipts.

Rounded aggregate scores cannot recover per-item pairing or exact paired confidence intervals. A future harness should retain item IDs, prompt/token hashes, both raw and normalized likelihoods, selected choices, correctness, context length and overlap flags. Seed variability must also be measured separately from finite-item uncertainty. No significance gate is manufactured from these aggregates.

## 4. Recurrent depth: bounded evidence, mixed registered outcome

The synthetic BINDINGS model is a separate LayerNorm/GELU/full-attention implementation. It is not the 386M RMSNorm/SwiGLU/GQA decoder. Each arm has five seeds and 26,000 updates at three rates. A run counts as learned when two-hop accuracy reaches 90% at any of 26 periodic evaluations; the experiment then takes the best count across rates for each arm.

| Arm | Parameters | Layer applications | Best successful seeds |
|---|---:|---:|---:|
| One layer reused four times |128,705|4|2/5|
| Four distinct layers |277,889|4|1/5|
| Four-layer block reused twice |277,889|8|4/5|
| Eight distinct layers |476,801|8|0/5|

The combined rule required both pairs to show at least two more successes for reuse. Only the whole-block pair does, so the verdict is mixed. All 60 files were inspected. The lower tested rate did not rescue either untied arm within the budget; this narrows one proposed explanation without establishing optimizer-independent superiority. Rate selection uses the same outcome seeds. Equal layer applications do not imply equal parameters, optimizer state, memory, wall time or energy. The next native comparison should report both equal-experience and measured-compute views, with fresh outcome seeds.

Extra loops have not established unseen three-hop capability. Declining relative state change is not a contraction proof. It does not guarantee a unique fixed point, semantic correctness or useful adaptive computation.

## 5. Forward transfer: a useful negative result with an incomplete probe

The development pilot trains 24 task families constructed from hidden permutations over 23 inputs. A new 16-number task vector is adapted on eight examples from each of eight held-out combinations and tested on 15 remaining inputs: 120 predictions per seed, three seeds. Replay sees 12.5× more stream examples than the sequential arm at the same update count, stores 552 examples and retains 384 task-vector numbers in addition to the printed shared-model count.

| Arm | Experienced-task retention | New-combination accuracy |
|---|---:|---:|
| Dense |8.45%|4.44%|
| Dense + replay |100.00%|2.78%|
| Loop |8.52%|3.61%|
| Loop + replay |99.82%|4.17%|
| Menu |64.55%|2.78%|
| Menu + replay |88.47%|4.17%|

Uniform chance is 1/23≈4.35%. A fresh-network control scores 0%, but that measured baseline is not the mathematical chance rate. The predictions T1/T2/T7 hold and T3/T4/T5/T6 fail. The source labels this a development pilot, so these are retained prediction outcomes rather than a confirmatory architecture verdict.

Post-hoc probes recover only about 41–42% on some already trained families at eight examples, while a true-primitive menu reaches about 71.7%. A suitable solution is known to exist in the latter control. Thus finite optimizer failure cannot prove that no task-vector setting or representation could solve the task. The correct conclusion is that the tested training/adaptation procedure did not produce useful measured new-family transfer. Retention and stream fitting are insufficient by themselves under that procedure. A stronger future probe must demonstrate that it can recover an available solution, use a prospectively fixed target and retain its adaptation cost.

## 6. Hardware, completion and cost semantics

Q14 compares a fused attention variant and a changed head configuration with the existing baseline. The measured speed ratio is 461,538/439,118=1.0510569, below the registered 1.25 threshold. The partial result has `finished=false`; its last row is step 2000, or 65,536,000 completed targets. The top-level 799,539,200-token field is a planned total. Treating it as completed would overstate exposure by more than 12×.

W1 therefore fails. W2, two-seed quality, and W3, kernel parity, are unmeasured. Keeping the existing implementation follows the adoption rule; it does not establish that the variant harms quality. An unresponsive machine has no established cause in the inspected evidence, and a compound kernel/head comparison cannot by itself identify the bottleneck.

Q13's recorded steady rate is 105,898 tokens/s; its time to the last recorded evaluation is 20.7011h. The derived exposure/time rate is about 103,307 tokens/s and still omits provisioning, corpus transfer and final benchmark work. Live allocation after evaluation is not peak training memory. Historical cost estimates are not bills, and this review does not present them as current prices or independently verified spending.

## 7. New store/cache results supersede V5's pending status

The corrected 101M learned-store result reports mixed gain 0.0017, below its 0.01 stopping threshold, and a foreign control gain−0.0028. This closes the tested 48M-token, approximately 1.02GB store complement to that frozen model. It does not close every external-memory architecture or the separate joint-training stages S2/S3. The tested LDO fitting is post-hoc, not corrected joint native training. Its context and provenance constraints remain part of interpretation.

The explicit causal-cache gate was at the 40M scale. Gains are web 0.0012, textbook−0.0003, mathematics 0.0036 and code 0.0059 BPB: each below 0.01. This supports closing that registered explicit-copy line at the tested 512-token context and checkpoint scale. Its foreign-cache control separately refits weights, so it is not identical in strength to applying fixed real-store weights to the LDO negative control. First-version sampler and historical provenance qualifications remain.

Neither result rehabilitates the historical invalid 0.62 cache claim. V5's pending statements remain visible as history and receive a new D-058 event linking to this update. Future memory work needs a changed, justified hypothesis and new protocol rather than relabeling a stopped arm as a success.

## 8. Next experiments and dependencies

| Direction in source plan | Why it is informative | Prerequisite and status in V6 |
|---|---|---|
| A: Continue 386M with decay | Tests additional exposure and terminal schedule within a known baseline | Exact resume/data recipe and compute decision; not launched here |
| B: Recurrent versus dense 101M | Tests whether toy block reuse carries to native language quality | Tune controls, choose exposure/compute budget, fresh seeds; pending |
| C: Generated reasoning training data | Tests data/objective contribution separately from architecture | Hold out families/templates, prevent evaluator leakage; pending |
| D: Generative reasoning evaluation | Directly tests use of visible facts and compositional chains | Implemented instrument inV6; real checkpoint binding and scoring pending |

The next step that can be completed without claiming a new training result is D's exact evaluation interface. Its primary result must remain **NOT_RUN** until a real model produces predictions with valid tokenizer, checkpoint, decoding and context receipts. [16]
