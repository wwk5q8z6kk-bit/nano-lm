# Mechanism cycles: results log (exploratory unless marked confirmatory)

## Cycle 1: budgeted exact memory on MQAR (exploratory), 2026-11-23

Setup: `tpu/mech_mqar.py` at commit 83c54dc. 16 key-value pairs, 8 queries, vocabulary 128, sequence 48 tokens. Models: d=64, 4 heads, 2 layers, short causal conv; 4,000 steps of AdamW. 3 seeds (0, 1, 2), the same data stream for every arm within a seed; evaluation on 16,384 answers per run. Raw results: `mqar-cycle1.json`.

Mean exact-answer accuracy (number of seeds that learned the task, accuracy > 0.9):

| Arm | M = 8 | M = 16 | M = 32 |
|---|---|---|---|
| attn (no budget) | 1.000 (3/3) | | |
| gdn (no memory) | 0.660 (2/3) | | |
| mem, recency | 0.379 (1/3) | 0.082 (0/3) | 0.082 (0/3) |
| mem, learned (teacher-guided) | 0.972 (3/3) | 0.788 (2/3) | 0.381 (1/3) |
| mem, oracle | 1.000 (3/3) | 0.737 (2/3) | 0.140 (0/3) |

Learned policy, share of kept tokens that are value tokens (0.5 = indifferent): M=8: 0.65, 0.56, 0.58; M=16: 0.59, 0.55, 0.54; M=32: 0.50 (all pair tokens fit).

**Reading: inconclusive.**
- **Outcomes are bimodal.** Each run either learns (about 0.9–1.0) or stays near 0.08. With 3 seeds, the means mostly count successes.
- **The pilot was misleading.** The single-seed "GDN 8%" figure was an unlucky seed; GDN learned in 2 of 3 seeds. So the recall gap between attention and GDN is *not* established at this size.
- **Larger budgets did worse for the oracle** (M=32: 0/3). That shows the outcome is driven by optimisation difficulty in the read layer, not memory capacity.
- **One hint.** At M=8 the learned policy succeeded in 3/3 seeds against recency's 1/3, and it kept value tokens more often than chance. Not significant with 3 seeds.

**What changed in the design:**
- Success rates must be measured with many seeds.
- The memory read layer needs an easier optimisation path.
- **Next (cycle 1b):** 8 seeds, 8,000 steps, M in {8, 16}; report the success rate (with a Wilson interval) and accuracy among successful runs.

## E0a: scorer conditions on our 101M checkpoint (2026-11-23)

Checkpoint `scale1-v6e/dense-L-lr0.001-s1000-tok8e+08-dp4-hs-v2.npz` (sha256 8745652c49cf…). Task file sha256 ec09b3fca312…, the same file as on the bucket (CRC32C match). Run on the laptop CPU with JAX 0.6.2; the stored reference was scored on a TPU. Raw files: `results/e0a/`.

| Change | Largest effect observed | Scope |
|---|---|---|
| CPU against TPU, both bf16 | ≤ 0.003 accuracy; LAMBADA perplexity 141.69 vs 141.76 | this checkpoint, these 7 tasks |
| float32 against bf16, same CPU | up to 0.008 (WinoGrande +0.008; ARC-C acc_norm −0.005) | same |
| No end-of-text prefix | LAMBADA accuracy −0.012 and perplexity 141.7 → 152.6; other tasks ≤ 0.004 | same |
| lm-eval length rule (choice without leading space) | ARC-E acc_norm −0.008; OBQA acc_norm +0.006 | acc_norm only |

**Item-level, bf16 against float32:**
- WinoGrande: 22 of 1,267 items (1.7%) change their argmax. ARC-Challenge: 7 of 1,172 (0.6%).
- The flipped items are near-ties: median float32 margin 0.028 and 0.012 nats, against medians of 0.51 and 1.93 nats over all items.
- bf16 shifts log-probabilities by a median of about 0.03 nats (99th percentile about 0.13).

**Reading:**
- **Hardware is not the main source.** The earlier "0.003 floor" is withdrawn. Precision moves scores more than hardware does, and how much depends on how many near-ties a benchmark has.
- **The end-of-text prefix is a real bias in our favour** on LAMBADA (about 1.2 points of accuracy) when comparing with the Hugging Face models, which were scored without it.
- **Sources of uncertainty stay separate:** implementation and precision (this table), seed variation (measured per experiment), benchmark sampling (bootstrap), and evaluation configuration (prefix, length rule).

**Decisions:**
- Score reference comparisons in float32.
- Score our models both with and without the prefix when comparing with public models.
- Report acc_norm with the lm-eval length rule.

## E0a, combined condition (no end-of-text prefix + float32 + lm-eval lengths)

Compared with the default, every task moves by at most 0.010, except LAMBADA: accuracy −0.010 and perplexity +10.7, which comes almost entirely from the prefix. This is the configuration that matches how the public models were scored, and it is the one to use for comparisons with them.

## Cycle 1c: oracle-memory diagnostic (exploratory), 2026-11-23

Kaggle T4, JAX 0.11.1, optax 0.2.8. 8 seeds (0–7), 4,000 steps; task and model as cycle 1. "Learned" means evaluation accuracy > 0.9; intervals are Wilson 95% over seeds. Raw files: `kaggle-cycle1c/mqar-diag.json`.

| Arm | Learned | Mean, all seeds | Mean, learned seeds only | First step with batch accuracy > 0.9 |
|---|---|---|---|---|
| oracle-only (8 readable tokens) | 7/8 (0.53–0.98) | 0.917 | 1.000 | 268–1,407; one never |
| oracle-recency-8 | 7/8 (0.53–0.98) | 0.917 | 1.000 | 265–1,407; one never (same seed) |
| oracle-recency-16 | 7/8 (0.53–0.98) | 0.894 | 0.987 | 928–3,288; one never |
| oracle-recency-32 | 3/8 (0.14–0.69) | 0.607 | 0.972 | 2,022–3,504; five never |
| oracle-random-32 | 3/8 (0.14–0.69) | 0.631 | 0.954 | 2,538–3,083; five never |

Per-seed accuracies are in the raw file.

- **The distraction hypothesis is rejected.** Random filler fails as often as recent filler.
- **The slot-count hypothesis is supported but not settled.** Success falls 7/8 → 3/8 from 8 to 32 readable slots, and time to learn rises. On 8 seeds, 7/8 vs 3/8 is about p ≈ 0.1, so "luck" is not excluded.
- **Implementation check passed.** oracle-only and oracle-recency-8 select identical tokens and gave identical outcomes, seed for seed.
- **Design consequence (hypothesis to test):** read layers over many slots learn slowly and unreliably. Next test: a curriculum on the memory budget (start M small, grow it), or a sharper read (top-k sparse read), against fixed M = 32.

## Cycle 2: looped reasoning (pointer chasing), first attempt: no arm learned (task failure, not a result)

Kaggle T4, 3 seeds, 4,000 steps. Every arm, including the untied baseline, stayed at chance (accuracy about 0.08 for every hop count, k = 1 included). Loss settled near 2.67 ≈ ln 13, so the models learned only "the answer is one of the 13 chain nodes", not the lookup.

Cause: one supervised answer per sequence (64 per step) is too sparse a signal for this task in this budget. **No conclusion about looping can be drawn.**

Fix: six supervised queries per sequence, with the same 1–4 hop training range. Re-run after a pilot shows the untied baseline learns k = 1.

## Cycle 1b: learned memory against recency, 8 seeds (exploratory), 2026-11-23

`mech_mqar.py`, commit 83c54dc. 8 new seeds (10–17), 8,000 steps (twice cycle 1), M ∈ {8, 16}, same task and models as cycle 1. All 64 runs completed. Success = evaluation accuracy > 0.9 (the threshold written in the cycle-1 log before this run). Raw files: `mqar-cycle1b.json`, `.log`.

| Arm | M | Learned | 95% interval | Mean accuracy, all seeds | Mean accuracy, learned seeds only | First checkpoint with batch accuracy > 0.9 (per seed, steps) |
|---|---|---|---|---|---|---|
| attention (no budget) | — | 8/8 | 0.68–1.0 | 1.000 | 1.000 | 500–1,500 |
| Gated DeltaNet, no memory | — | 3/8 | 0.14–0.69 | 0.586 | 0.964 | 3 learners at 3,000, 4,000 and 6,000; five never |
| memory, recency | 8 | 7/8 | 0.53–0.98 | 0.848 | 0.956 | 2,000–5,500; one never |
| memory, recency | 16 | 6/8 | 0.41–0.93 | 0.762 | 0.962 | 2,500–4,000; two never |
| memory, learned (teacher-guided) | 8 | 8/8 | 0.68–1.0 | 0.985 | 0.985 | 1,500–2,500 |
| memory, learned (teacher-guided) | 16 | 8/8 | 0.68–1.0 | 0.982 | 0.982 | 1,000–5,500 |
| memory, oracle (uses future queries) | 8 | 8/8 | 0.68–1.0 | 0.999 | 0.999 | 500–5,000 |
| memory, oracle | 16 | 8/8 | 0.68–1.0 | 0.999 | 0.999 | 1,500–3,500 |

Per-seed accuracies are in the raw file. The 8 runs left unresolved at 8,000 steps: gdn seeds 11, 13, 14, 16 and 17 (five), recency M=8 seed 15, recency M=16 seeds 12 and 13.

**Comparisons (all exploratory; budgets pooled, so the 16 runs per arm come from only 8 seeds and are not independent):**
- Learned against recency, success rate: 16/16 vs 13/16, Fisher p = 0.23. **Not distinguishable.**
- Learned against recency, time to first checkpoint above 0.9: median 2,000 vs 3,500 steps. Paired by seed and budget, learned is faster in 12 pairs, slower in 2, tied in 2 (sign test p = 0.013). Resolution is 500 steps. **Suggestive of faster learning, not of higher reliability.**
- Learned against oracle: identical success (16/16), time to learn indistinguishable (permutation p = 0.69).
- Any memory arm against Gated DeltaNet alone: learned 16/16 vs 3/8 (Fisher p = 0.001); recency 13/16 vs 3/8 (p = 0.065). A readable memory makes learning more reliable. Cycle 1's pilot "recurrent gap" (8%) was one unlucky seed; at this size the gap is a learning-reliability gap, not a capability ceiling: successful runs score 0.96.
- Cycle 1 (4,000 steps, 3 seeds) and this run (8,000 steps, 8 seeds) cannot be pooled. Longer training resolved many of cycle 1's failures, which supports the delayed-learning explanation.

**Why this run does not show the teacher-guided write rule works (limits, stated before anyone cites it):**
1. **The task gives a causal write policy nothing to learn.** Queried pairs are drawn at random after all pairs are written. At write time no causal policy can know which pair will be asked about, so it cannot beat chance on selection. The oracle sees future queries, so it is an upper bound, not a usable policy.
2. **Selection precision is not demonstrated.** The learned head kept value tokens in 49–91% of slots (M=8: 0.51, 0.49, 0.58, 0.80, 0.53, 0.61, 0.55, 0.91; M=16: 0.52–0.68), which is indifference for most seeds. A pair needs only one of its two tokens kept, because the convolution and the recurrent layer carry the other.
3. **The recurrent layer shares the storage burden.** M=8 slots cannot hold 8 random queried values from 16 pairs, yet accuracy is 0.985, so the recurrent state and the memory together answer.

**What changed in the design:** the next test needs a task where salience is predictable at write time.
- **Salience-marked MQAR:** a marker token precedes every pair that will later be queried; unmarked pairs are never queried and make up half of the pairs, placed so that recency discards the marked ones.
- **Arms:** recency; learned from teacher attention; learned from random targets (control); oracle; no memory.
- **Prediction:** recency fails whenever unmarked pairs arrive last, and learned-from-teacher succeeds only if the teacher's future attention carries salience. Learned-from-random-targets must fail, or the teacher signal adds nothing.
- Pre-register the success threshold (0.9), seeds (≥ 8) and steps (8,000) before running.

## Cycle 2 v2: looped reasoning on pointer chasing, six supervised queries per sequence: inconclusive (task unresolved at budget end), 2026-11-23

`mech_hops.py` commit 567c0f3 (sha256 prefix f75ae1c7911a032b), Kaggle T4, 4 seeds (0–3), 6,000 steps, 3 arms, 12 runs; all completed. Raw: `kaggle-hops2/hops-g0.json`, `hops-g1.json`.

| Arm | k=1 accuracy (4 seeds) | k = 2–8 accuracy | Runs learning k ≥ 2 |
|---|---|---|---|
| untied-6 (7 distinct layers) | 0.64–0.66 | 0.04–0.10 (chance 1/13 = 0.077) | 0/4 |
| looped-6 (one shared block × 6) | 0.47–0.61 | 0.05–0.11 | 0/4 |
| looped, random depth 2–8 (R=6, R=12, halting) | 0.60–0.68 | 0.06–0.10 at every R and under halting | 0/4 |

Training hop range was 1–4, so hops 2–4 were *in distribution* and still at chance for every arm. Final training losses: untied 2.08–2.16, looped 2.29–2.43, random-depth looped 2.17–2.33. Chance-level loss for 13 candidates is 2.56.

**Reading.**
- **No test of the looping hypothesis is possible.** The baseline never learned the multi-hop task in this budget.
- **Extra test-time iterations changed nothing.** R=12 and halting give the same accuracies as R=6, but without any learned computation to extend that says nothing about recursion.
- **Halting behaviour is not informative here.** Mean iterations under halting were 9.96–11.13 of 12: the residual rule almost never triggers when nothing was learned.

**Why this probably happened, and what it implies (hypotheses):**
- The earlier toy task showed delayed learning (jumps at 6,000–21,000 steps out of 26,000). 6,000 steps may sit before the jump.
- One-hop lookup was only learned to about 0.65, so sequential composition has no reliable first step to build on.

**Next, smallest controlled step:** a *difficulty and budget pilot* of the untied baseline alone, one seed per setting, before any looping comparison:
1. a hop curriculum (k = 1 first, adding one hop each quarter of training);
2. 20,000 steps;
3. success defined in advance as accuracy ≥ 0.9 at k = 2, 3 and 4.

If the baseline cannot learn k = 2 within 20,000 steps with the curriculum, the task is replaced (not tuned further) by the registered BINDINGS task from the earlier nested experiment, where delayed learning is already characterised and 26,000-step budgets learned in 14 of 20 reuse runs.

## Stage A, conversion of SmolLM2-135M attention to Gated DeltaNet: registered, not yet run (2026-11-24)

`tpu/convert_stage_a.py` carries the registered criteria in its docstring; they are not to be changed after seeing results.
- **C1:** at least 90% of layers (27 of 30) reach trained-inherit error ≤ 0.1 × the untrained random-initialisation error. Error is ||y − t||² / ||t||² on 16 held-out sequences of 1,024 tokens, per layer, after 300 steps.
- **C2:** no non-finite loss in any training run.
- **C3:** conversion-training throughput and peak memory reported.
- **Descriptive only:** language-model loss of six models (teacher; all layers converted; 22 converted with the highest-error 8 layers kept as attention; random 22/8 split; all converted from random initialisation; all converted but untrained).
- **Expectation, written before the run:** C1 is a strict bar and may fail, because the converted layer has no rotary positions and is trained on 65k tokens for 300 steps. A fail would say the strict bar is too high for this budget; it would not show the pipeline is broken. The inherit-versus-random comparison and the language-model losses carry the information.
- **Local debug run (not a result):** 3 steps on CPU with a naive recurrence, three-step errors near 1; this checked only that the hooks, the grouped-query weight mapping (verified exact against the teacher's projections), training loop and hybrid assembly run.

## Stage A result: SmolLM2-135M attention → Gated DeltaNet, layer by layer (2026-11-24)

Kaggle T4, fla 0.6.0, `convert_stage_a.py` at commit 64a4b7d; 30 layers; per layer 300 steps on 65,536 teacher-tokenised training tokens, 16,384 held-out tokens; error = ||y − t||² / ||t||². One seed per layer and arm. Raw: `results/stage-a/stage_a.json`.

**Registered criteria**
- **C1: FAIL.** 3 of 30 layers (0, 1 and 24) reached ≤ 0.1 × the untrained-random error; the bar was 27 of 30.
- **C2: pass.** No non-finite loss in any of the 60 training runs.
- **C3:** conversion training ran at a median 154,535 tokens/s per layer module, peak memory 1.18 GiB (single-layer training, not a full model).

**Per-layer fit (descriptive)**
- Median relative error after training: inherited initialisation **0.323**, random initialisation **0.388**; untrained errors are 1.06 and 1.00.
- Inheriting is better in **28 of 30 layers** (one-sided sign test p ≈ 4×10⁻⁸). The advantage is modest, a median 17% lower error, and both arms are far from exact.
- 17 of 30 inherited layers start *worse than outputting zero* (error > 1.05): copying the projections does not reproduce the attention layer, as expected without positional encoding, with normalised keys and with the recurrent update.
- Errors vary a lot by depth: layers 0, 1 and 24 are easy (0.016, 0.061, 0.046); layer 23 is the hardest (0.577); most others sit between 0.2 and 0.42.

**Language-model loss without any end-to-end training** (nats per token, 16,384 held-out tokens, paired on the same tokens, one seed):

| Model | Layers converted | Loss | Perplexity | Gap to teacher |
|---|---|---|---|---|
| Teacher | 0 | 2.910 | 18.4 | — |
| 22 converted, 8 *random* layers kept as attention (one draw) | 22 | 3.506 | 33.3 | +0.595 |
| 22 converted, the 8 *highest-error* layers kept as attention | 22 | 3.741 | 42.1 | +0.831 |
| All converted, inherited, trained | 30 | 4.031 | 56.3 | +1.121 |
| All converted, random init, trained | 30 | 4.514 | 91.3 | +1.604 |
| All converted, inherited, untrained | 30 | 13.187 | 533,492 | +10.277 |

**Reading, scoped to this run:**
- **Inheritance helps both ways it was measured:** lower per-layer error in 28 of 30 layers, and 0.48 nats lower language-model loss for the all-converted model (4.03 vs 4.51). This is layer-wise alignment only, one seed, and says nothing yet about equal-token end-to-end training (stage B).
- **The pipeline works,** but layer-wise alignment alone leaves a large gap (0.6 to 1.1 nats), so end-to-end training is needed, as expected.
- **Data-driven placement by alignment error is not supported:** keeping the highest-error layers as attention (3.74) did *worse* than one random draw (3.51). One random draw has unknown variance, so this is "not supported", not "refuted". Layer-wise relative error on teacher-forced inputs is a poor guide to which layers matter for the model's loss.

**What changed in the design:** placement is measured directly by *leave-one-out importance* (convert one layer, measure the loss change) with several random draws for the baseline spread. That extension, plus a rerun that reproduces the stage-A numbers (a check on run-to-run variation), is in `convert_stage_a2.py`. It is exploratory and does not alter the registered criteria above.

## Pointer chasing: baseline difficulty pilot, result (2026-11-24)

`mech_hops.py --curriculum`, untied baseline, one seed, 20,000 steps, hops 1 → 1–2 → 1–3 → 1–4 by quarters; registered success: accuracy ≥ 0.9 at k = 2, 3 and 4.
- One-hop accuracy reached 0.84; accuracy for **k = 2…8 stayed at chance (0.07–0.09)**, including the trained range 2–4. Training loss fell to 0.36 during the one-hop phase and returned to about 1.97 once multi-hop queries entered.
- **Criterion failed**, so under the rule written in advance the task is **replaced, not tuned further**. The looping hypothesis moves to the registered BINDINGS task, where delayed learning is already characterised (looped and whole-block reuse learned two-step chains within 26,000 steps). This was one seed on a 0.3M-parameter model; that scope limits the conclusion to "this task and model size do not support a baseline", not to anything about looping.
