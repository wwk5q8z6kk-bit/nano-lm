# Research decision note, 21 November 2026

First delivery requested by the unified mission of the same date. Evidence-linked; no architecture is proposed or frozen. Everything cited as [Observed] is in this branch (`research/ontogenesis-phase1`) under `research/frontier/ontogenesis/`, with predictions committed before the runs. Literature statements are marked [Published, unverified] until the verification workflow reports; its findings will be appended as the last section and override anything above them.

Labels: [Observed] measured here · [Derived] follows from stated assumptions · [Published] · [Hypothesis] · [Contradicted] · [Unknown].

---

## 1. Status of every registered run

| Experiment | Protocol or script | Status | Result against its registered rule | Raw files |
|---|---|---|---|---|
| Looping replication, 13,000 steps | `PROTOCOL-LOOP.md` | Finished | Mixed: looped 2 of 3, dense 0 of 3 | `results/loop/` |
| Looping follow-up, 26,000 steps | `PROTOCOL-LOOP2.md` | Finished | Mixed: looped 3 of 5, dense 1 of 5 | `results/loop2/` |
| Nested dense-loop | `PROTOCOL-NESTED.md` + `PROTOCOL-NESTED-DEVIATION.md` | **Running**, third execution; the first two were stopped by me with no result written | None yet | bucket `runs/nested-cpu-1/` |
| First acquisition benchmark | `cab.py` | Finished | Predictions held, but the benchmark is passed by a lookup table, so it measures storage only | `results/cab/` |
| Structured acquisition benchmark | `cab_s.py` | Finished | 3 of 4 predictions held; forgetting was mild, against prediction | `results/cab_s/` |
| What is forgotten, plus two controls | `forgetting_split.py` | Finished | Corrections cause the loss of old regular facts; exceptions do not | `results/forgetting_split/` |
| Routing with an oracle | `routing_test.py` | Finished | Gain reproduced; later shown to be label-noise removal | `results/routing/` |
| Routing without an oracle | `routing_loo.py` | Finished | Recovers a third of the inference gain; registered bar (half) not met | `results/routing_loo/` |
| Fixed-address memory saturation | `memory_saturation.py` | Finished, one seed | 3 of 4 held; the failed one shows optimiser state is a main cause | `results/memory_saturation/` |
| Writing a correction | `correction_interference.py`, `correction_margins.py` | Finished | See section 3 | `results/correction_interference/` |
| Optimiser control for corrections | `correction_optimizer.py` | Finished | Prediction held: no gradient-only write avoids the damage | `results/correction_interference/optimizer-*.json` |
| Text model with an exact store | `lm_store.py` | Finished, one seed | Post-hoc store helps (0.058 bits per byte); training with the store hurts (0.14 worse); registered claim failed | `results/lm/store-pilot.json` |
| Dense against looped on text | `lm_compare.py` | Finished, one seed | Looped slightly worse, as predicted | `results/lm/` |
| Dense baseline, PyTorch, 40M parameters, full corpus | `kaggle/baseline_dense/` | **Running** on Kaggle | None yet | Kaggle notebook output |
| Dense baseline ladder, JAX, 14M to 211M parameters | `tpu/train_jax.py` | **Running** on two Cloud TPU machines | Partial curves only | bucket `runs/ladder-v5e/`, `runs/ladder-v6e/` |

Nothing above marked Running is treated as a result anywhere in this note.

## 2. Corrections to claims in circulation

- **M21 against dense (merged PR 194).** [Observed, verified by reading the PR] Dense 4.65 against M21 5.02 bits per token at equal training time, ahead on all three seeds; M21 keeps a 0.36 advantage on copyable tokens and loses 0.97 elsewhere. Limits stated in the PR: time matching drifted by up to 30% per seed, 5M parameters, one code distribution, unoptimised implementations. So: dense is the stronger whole backbone in that setting; the copy mechanism is worth isolating; nothing more.
- **DEMOS learns faster.** [Contradicted] The lead came from a mis-tuned baseline.
- **Exceptions damage regular knowledge.** [Contradicted] Exceptions alone left regular facts at 99.6%; corrections alone reproduced the loss.
- **Routing irregular facts raises inference by ten points.** [Narrowed] True only with oracle labels, and the effect is label-noise removal. Without an oracle, 3.8 of 11.3 points.
- **Fixed addresses trade retention for acquisition.** [Contradicted as stated] With the optimiser state reset each phase, the same addresses acquire 86% and retain 83%. The collapse needs both concentrated addresses and carried-over optimiser state.
- **Training the core through an exact store carries the routing idea to text.** [Not supported] It lost to adding the same store afterwards by 0.14 bits per byte. A recorded design flaw (same-document leakage) biases against it, so the idea is unproven, not disproven.
- **"Never forgetting equals order-independence" and "bounded size, no forgetting, rich features: pick two".** [Withdrawn] Neither is a theorem as I stated them.
- **"The additive-statistics learner is better because of its write rule."** [Weakened] It also fits exceptions, is not order-free, and differs mainly in re-solving an entity's address from all its facts at once.

## 3. The deepest unresolved question

**What must be present at the moment a learner changes itself for the change to leave everything else it knows intact, and how much of that must be carried explicitly, as opposed to recovered from the learner itself or re-supplied by the data that keeps arriving?**

### Formulation

State θ, behaviour f_θ. At time t the learner answers correctly on a set of queries Q_t. A new observation (q*, a*) arrives. Define the feasible set

F = { θ' : f_θ'(q*) = a*, and f_θ'(q) = f_θ(q) for all q in Q_t }.

A harmless write needs three separate things:
1. **Existence.** F is not empty. This is a capacity question.
2. **Selection.** The write rule lands in F. That requires information that identifies F, at least about the queries whose answers depend on the parameters the write touches.
3. **Persistence.** At every later write that touches those parameters, the same information is needed again.

The information for 2 and 3 can only come from three places: carried explicitly (stored facts, replay); read back from the learner (its own current answers); or re-supplied by the world (later data that implies the old answers).

### What our evidence says, in one world and one model family [Observed]

| Source of the constraints | At the moment of writing: entity's other facts kept | 20+ phases later: old regular facts on corrected entities | Correction itself, 20+ phases later |
|---|---|---|---|
| None, gradient on the one fact | 64.7% | 59.6% | 66.2% |
| None, smallest possible edit | 35.3% | 34.2% | 23.8% |
| Other entities' facts (equal-sized replay) | 66.6% | 68.8% | 67.0% |
| The entity's own facts, carried explicitly | 100% | 99.5% | 62.3% |
| The learner's own current answers (computed edit) | 98.0% | 83.4% to 89.7% | 21.4% to 35.1% |
| The world re-supplies them (regular facts, no corrections at all) | — | 99.6% to 99.9% | — |
| Exact store, model untouched | 99.9% | 99.5% | 100% by construction |

- Existence holds: an address-only edit that changes none of the entity's other seven answers exists for 99.84% of corrections.
- Selection fails without constraints, and smallness does not substitute for them: the smallest edit is the worst.
- Persistence fails for anything the stream does not re-assert, whatever the margin: only the store held corrections.

### Competing explanations

| Explanation of the damage | Prediction | Status |
|---|---|---|
| Capacity: no state holds both old and new | No harmless edit exists | [Contradicted] 99.84% feasible |
| Step size or path: a gentler write would be harmless | The smallest write keeps the other facts | [Contradicted] 35.3% |
| Missing constraints | The entity's own facts protect; other entities' facts do not; the learner's own answers protect at once | [Observed] 100% against 66.6%; 98.0% |
| Optimiser artefact (Adam's per-coordinate steps, stale moments, too many steps) | Plain gradient descent stopped at first fit keeps the other facts | [Contradicted] it keeps 14%; fresh Adam 40% to 63%. Optimiser state **was** a main cause in the memory-saturation result, so the two failures have different causes |
| Drift under later training, separate from the write | Decay continues after a harmless write; margins slow it little; a store is immune | [Observed] |

### Counterexample to the weakest assumption

The weakest assumption in our earlier reasoning was "if a write damages something, the model lacked room, so separate storage is necessary" (and its cousin, "be gentler"). A two-dimensional case refutes both. [Derived; checked numerically]

Address u = (2, 1). Relation 1 has values A, B with weights (1, 0) and (0, 1). Relation 2 has values C, D with weights (1, 0) and (0, 0.4). The entity currently answers (A, C). Correct relation 1 to B with a 3-logit lead.
- Following the gradient of that one fact reaches (0, 3): answers (B, **D**). Relation 2 is broken.
- The smallest edit reaches (2, 5): relation 2 is tied, so it is lost.
- A state with both exists: (3.67, 6.67) answers (B, C) with leads of 3 and 1. It is farther away (size 7.3 against 4) and in a direction neither the gradient nor the minimal edit takes.

Related, and due to the owner's reviewer: for any invertible g, the encoder g∘E with decoder D∘g⁻¹ computes the same function. So fixed internal coordinates are not necessary for fixed behaviour. That refutes "addresses must be frozen"; it does not supply a learning rule that maintains the compensation.

### What is and is not derived

- [Derived] In a linear readout, exact preservation of old outputs confines a write to the directions old keys do not span; the room left for new learning is the dimension of that complement. A projector of d² numbers is a sufficient summary; no raw data is needed. Capacity is bounded by d.
- [Derived] For n independent arbitrary values from V possibilities, lossless recall needs about n·log₂V bits somewhere. Regular content costs only its rule.
- [Hypothesis, not derived] The explicitly carried information a learner needs is roughly the off-rule content that shares parameters with whatever keeps being updated; content the stream re-implies costs nothing; content in parameters never touched again costs nothing.
- [Published, unverified] Rehearsal, pseudo-rehearsal from the model's own outputs, null-space projection and locality-constrained model editing each implement one of the three sources. I claim no novelty for the framing.

## 4. Recommendation: what to do next, by information per unit of resource

1. **Map the dense reference on real text first** (running now). Parameters from 14M to 211M, one and two passes, two learning rates, extra seeds. It costs a few chip-hours per point and every later decision needs it: the seed noise sets the threshold for any mechanism claim, and the slopes say whether we are limited by parameters or by data.
2. **Then measure what closes the gap between small and large on text**, in three separable parts: stored information (the same exact store added to each size), computation (more parameters at fixed data), and search (same model, longer or better training). Each mechanism we have discussed attacks exactly one of these, and we do not know their sizes.
3. **Close the correction question on the toy** with the optimiser control (running) and stop there. It has given its answer; further polishing would be the easy-task trap.
4. **Do not start an architecture tournament yet.** Without the noise floor from step 1 it cannot be judged.

The largest practical finding of today is not about architecture: **the corpus is the binding constraint.** At measured speeds the remaining credits could train a 100M model on well over 100 billion tokens, or a 350M model on tens of billions. We have 0.8 billion.

## 5. Reference models, parameter counts, licences, hardware

Filled 2026-11-22 from the literature-verification workflow. A verifier opened every source; no one ran these models. Full tables, with the published evidence behind each experiment: `theory/REFERENCE-MODELS-20261121.md`. [Published, verified by the workflow]

| Model | Parameters | Pattern | Licence | Hardware or kernels | Evidence under 500M parameters |
|---|---|---|---|---|---|
| Qwen3.5-4B-Base | 4,659,865,088 shipped; language model alone 4,205,751,296 | 32 = 8 × (3 Gated DeltaNet + 1 gated attention); hidden 2560; vocabulary 248,320, tied | Apache-2.0 | pure-PyTorch fallback; optional fla / causal_conv1d kernels | none |
| Qwen3.5-2B-Base / 0.8B-Base | 2,274,069,824 / 873,438,784 shipped | 24 = 6 × (3 + 1) | Apache-2.0 | same | none; at 0.8B the embedding table is 33.8% of the language model |
| Kimi-Linear-48B-A3B | 49.1B total, 3B active | 20 KDA : 7 full (2.86:1) | MIT | fla-core ≥ 0.4.0, remote code | none; its only exact 3:1 ablation (653M active, MoE, Muon) beats 1:1 by 0.01 perplexity, no seeds |
| Gated DeltaNet paper | 1.3B main; 400M and 500M ablations | sliding-window hybrids | no weights; reference code non-commercial | 32 GPUs for about 4 h at 0.4B / 15B | mixer ablations only |
| Hybrid study, arXiv 2507.06457 | 340M class and 1.3B | ratios labelled 24:1 to 3:1 are really 23:1 to 2:1 | no checkpoints declared | AdamW, 4K context | ratio moved scores by under 1%; pure Gated DeltaNet 0.442 vs Transformer 0.444 |
| Mamba-3 | 180M to 1.5B | pure recurrent; 5:1 hybrid for retrieval at 1.5B | Apache-2.0 | Linux, NVIDIA, CUDA kernels | 180M: 16.59 vs Transformer 16.89 perplexity; no seeds, no wall-clock |
| SmolLM3-3B (dense) | 3,075,098,624 | 36 full-attention layers, GQA 16/4 | Apache-2.0 | 384 H100 for 24 days, 11.2T tokens | none |
| Looped LMs (Schwethelm et al.) | up to about 1B | shared block looped 1 to 8 times | MIT | — | looped trails untied by 0.03 to 0.12 nats at matched compute |

**Where a 3:1 recurrent-to-attention reference is not appropriate** (workflow synthesis):
1. As the starting reference for 10M to 500M models trained on under 1B tokens: nothing tests it there, and the nearest evidence (340M, 20B tokens) shows the ratio does not move language-model scores.
2. As a quality prior at any scale: the only exact ablation is 0.01 perplexity, one seed, mixture-of-experts, trained with Muon.
3. For short-context training under equal wall-clock: published recurrent kernels are 1.3 to 4.1 times slower than FlashAttention-2 at 1K to 2K tokens.
4. On TPU with JAX or on Apple silicon: no optimised kernel is listed.
5. In shallow models whose task needs several retrieval steps in one pass: 3:1 at 4 to 8 blocks leaves one or two attention layers. This is an inference across sources and is untested.
6. As something the commonly cited models use: their patterns are positional-encoding changes only (SmolLM3), sliding-window hybrids (Gated DeltaNet), 2:1 (the hybrid study), 20:7 (Kimi) and 5:1 (Mamba-3).
7. As a copy of the Qwen head layout at small width: the recurrent mixer does not shrink with width (at hidden 1024 it is as large as the feed-forward block).

Parameter ceiling: by the plan's rule (4.0B over every learned component), the illustrative 32-block, d = 2560 design with a 248,320-entry vocabulary and Qwen-sized mixers is 4,205,751,296: over the ceiling before any head. With a 65,536 vocabulary it has about 262M of headroom.

## 6. One inexpensive, falsifiable test, registered here before it is run

**Question.** Does an exact store still add anything once the model is larger and fully trained, or does a well-trained model absorb what the store knows?

**Inputs.** The finished ladder checkpoints for 14M, 40M and 101M parameters (one pass, 800M tokens, seed 1000). The same 4-gram store as the pilot, built from the same 48M-token training subset. The same held-out evaluation crops as every other run.
**Procedure.** For each checkpoint: compute the model's probability of the true next token on the held-out crops and on 1M tokens of training text; fit the 25-entry mixing table on the training text only, with leave-one-out counts and the model frozen; report bits per byte with and without the store.
**Controls.** With the mixing weight forced to zero the script must reproduce the checkpoint's reported bits per byte to four decimals. With the store's probabilities shuffled across positions the gain must be zero within 0.002.
**Expected effect.** The pilot gain was 0.058 at 14M parameters after 33M tokens. I expect it to shrink to about 0.02 to 0.03 at 14M after 800M tokens and to 0.01 to 0.015 at 101M.
**Decision rule, fixed now.** Gain at 101M of 0.02 or more: the store is a durable complement; build the leave-document-out training variant and a larger store. Between 0.01 and 0.02: complement too small for its 1 GB; keep only as a baseline. Below 0.01: close the n-gram store line.
**Stopping rule.** One evaluation per checkpoint. No tuning on held-out text. No further seeds unless the result falls within 0.003 of a threshold.
**Resources.** Laptop CPU, under an hour. No paid compute.

## 7. Route to one native model

| Gate | Evidence required to pass | Scale | Limiting resource |
|---|---|---|---|
| 0. Reference map (now) | Curves, slopes, seed noise, cost per run, on our corpus | 14M to 211M | none; running |
| 1. Bottleneck decomposition | Sizes of the storage, computation and search parts on held-out text | 14M to 101M | laptop and a few chip-hours |
| 2. Mechanism trials | A mechanism beats dense at equal wall-clock and equal tokens by more than twice the seed noise, on two seeds, on text and code | about 100M | implementation time; about 1 to 2 chip-hours per arm |
| 3. Combination | Two winning mechanisms still win together | about 100M | — |
| 4. Pilot | The combination holds at 300M to 500M | 300M to 500M | **a corpus of 10 to 50 billion tokens**, licensed and deduplicated |
| 5. Native model | Funded plan from measured throughput | 1B to 2B, under 4B | data and several thousand dollars of compute |

Rough cost from today's measured speeds, for planning only: a 1B-parameter model on 100 billion tokens is on the order of 1,200 fast-chip-hours. That is beyond the present credits and well within reach of a modest budget. The data is not.

## 8. An original idea, or why none has survived

**No original idea of ours has yet survived criticism.** Six independent reviews removed, in turn: the particle model's advantage, the bridge account of the jump, tied-beats-untied, the organisation measures, the two "limits" on continual learners, and the collateral-damage mechanism.

Two things remain possibly open, neither established:
- **How weight tying changes the distribution of jump times across seeds.** The cloud CPU run adds 35 runs to the 16 we have. I would abandon it if tied and untied distributions overlap, or if the published all-seed successes turn out to depend on a training ingredient we lack.
- **The three-source account of what a harmless write needs** (section 3), with its hypothesis about how little must be carried explicitly. I would abandon it as a contribution if the literature check shows the same decomposition with the same quantities, which I think likely; it would remain useful to us as a design test either way.

## 9. Compute used for this delivery

Paid: two Cloud TPU machines, both self-deleting with hard time limits (10 and 6 hours), authorised by the owner in their own words on the day. Estimated total for this cycle: 55 to 65 dollars of credit. Free: one Kaggle GPU session. No RunPod spend.

---

## 10. Corrections after independent review (2026-11-22). These override everything above.

Four independent reviewers attacked this note: prior art, mathematics, experiments, and baselines and cost. Their full output: `/Volumes/Express1M2/nanolm-lab/workflow-review-decision.json`. Their findings are below, with what I did about each. I checked those that rest on our own code or files. Prior-art findings rest on the reviewer's reading, marked [Published, per review].

### Section 3: what does not survive

- **"The smallest edit is the worst" is false.** Plain gradient descent on the address, stopped at first fit, kept the other facts better (14.3% damage) than the smallest edit (35.3%). The 64.7% row is not comparable: it writes elsewhere. The supported statement: *smallness is not the property that protects; constraints are.*
- **The table overstates what was measured.**
  - The 100% for "the entity's own facts" is accuracy on the very facts rehearsed. The entity's five unshown answers are never scored in any gradient arm.
  - The label "20+ phases later" is wrong for 47% to 49% of the scored facts.
  - The computed-edit row mixed arms. Its true ranges are 96.9% to 98.5% at writing, 75.7% to 89.7% later for neighbours, and 21.4% to 35.1% for the correction. The arm with the most durable corrections has the worst neighbour retention: a trade-off the row hid.
  - The 100% against 66.6% contrast is close to forced by the architecture: other entities' facts have no gradient path to the edited row.
- **The two-dimensional example** is arithmetically right and does refute "a damaging write means there was no room". It refutes "be gentler" only at the 3-logit lead I chose: below a 1.29-logit lead the gradient write is harmless, and the registered code asks for 1. The "smallest edit" (2, 5) is one of several minimisers, and it ties. "Size 7.3 against 4" is 4.03 like-for-like, and reverses (1.0 against 2.67) under the note's own reparametrisation. The measured arms carry the claim; the example only illustrates it.
- **The feasible set as written is empty for a correction**, because the corrected query q* is itself in Q_t. It should read: f_θ'(q) = f_θ(q) for q in Q_t \ {q*}.
- **"The information can only come from three places" is an assertion, not a result.** The store row uses no information about old answers: it avoids the write instead of constraining it. The persistence bullet is contradicted by the constrained arm. There the edited address received no gradient, yet the correction fell from 99.8% to 21.5% through the shared decoder. So persistence fails through parameters the write never touched.
- **Derived bullets.**
  - "Capacity is bounded by d" fails for the argmax readout we actually use: 192 arbitrary facts fit in 24 dimensions.
  - A projector cannot write a correction, because it only preserves.
  - "Regular content costs only its rule" omits the per-entity type (10,240 bits against a 640-bit rule).
  - The hypothesis bullet commits to no prediction.
- **Optimiser row.** "[Contradicted]" and "the two failures have different causes" are **not established**. The control wrote the address alone, with the decoder frozen, as a separate step, under a new stopping rule. No arm reproduced the original 64.7%, and the original arm's stale Adam state (about three times the step on a long-idle row) was never removed.
- **Prior art** [Published, per review]. All seven findings of this cycle are published in substance:
  - same-subject damage is a standard editing metric (relation specificity);
  - constraining the model's own outputs is Learning without Forgetting, De Cao et al., and ROME's subject-vector optimisation;
  - the formulation is the Ideal Continual Learner (ICML 2023);
  - the projector bullet is OWM (2019);
  - edit decay under later training is reported in two 2025 papers;
  - the failed store-training arm reproduces Neubig and Dyer (2016), cause included.

  **The three-source account is struck from "possibly open" in section 8.** Section 8's headline stands: no original idea has survived.

### Section 4: "the corpus is the binding constraint" is withdrawn

No curve shows it:
- training and held-out loss agree within 0.03 nats per token in every run;
- at matched tokens, more parameters still win;
- 0.8B tokens was a builder quota, not a scarcity.

What was measured: *at the measured speeds, compute for models near 100M parameters is cheap relative to our 0.8-billion-token corpus; whether more unique text helps more than more passes is untested.* Corpus v2 (same pinned sources and recipe, 10B tokens; it reproduces v1's held-out sets and training prefix exactly) is being built so that the test can be run.

**Cost arithmetic** [Derived, corrected]. "1,200 fast-chip-hours for 1B parameters on 100B tokens" reproduces (845 to 1,429), but with today's code that is one chip for about 50 days. One run cannot yet use several chips. "Well over 100 billion tokens for 100M parameters" holds only on the slower chip, with every dollar spent on one run. More chips buy wall-clock time, not tokens per dollar. On value, the v5e beats the v6e at 101M by about 1.4 to 1.5 times.

### Section 6: the store test needs a corrected estimator before its rule is read

- It was already run at 14M (+0.0044) and 40M (−0.0014). Both are now recorded in `INVESTIGATION.md`. The first 14M run used the shuffled-store control, which I later replaced by a valid one.
- **The estimator is biased against the store.** It fits the mixing table on training text with leave-one-position-out counts, and same-document repeats make the store look 1.2 to 3.5 times more reliable there than on held-out text. The review measured this; leave-document-out removes the gap. So the fitted weights over-trust the store, and the held-out gain is understated. This may explain the negative value at 40M.
- **Amendment, before the 101M checkpoint is scored** (revised 03:55 UTC, before any run of it, following the merged review). The table is refitted on the same training windows with **leave-document-out** counts: every count used for fitting excludes the whole current document. Leave-document-out reproduces held-out reliability in all four domains, as the review measured, and it keeps the registered rule of no tuning on held-out text. Cross-fitting on held-out halves, my first plan, would have broken that rule.
- **A control that can fail** replaces the uninformed-store control, whose own fitted weights were all zero, so it could not fail. The real fitted weights are applied to the uninformed store. That must make the loss worse; a gain would reveal leakage.
- All three sizes are reported under both estimators. The decision rule (0.02 / 0.01) is read on the leave-document-out estimate at 101M. Script: `tpu/store_ldo.py`.
- **Scope of the rule, narrowed.** The store covers 48M of the 800M training tokens (6%), and published corpus-scale stores still help models trained on the same corpus. So a gain under 0.01 closes only *this store as a complement*, not the n-gram store line. Closing the line would need a store over the full training corpus.

### Sections 4 and 7: the dense reference is a default recipe, not a tuned baseline

- One learning rate at 14M. At 101M and 211M the better rate sits at the edge of the grid.
- Weight decay 0.1 applies to every parameter, including normalisation gains (the common convention excludes one-dimensional parameters).
- No noise floor yet. The only extra seed ran on the other chip type with a different micro-batch count. The v6e runs sit 0.002 to 0.006 above the v5e run at every common checkpoint, and that cannot be separated from seed noise.
- Every first-version TPU run also oversampled the first 37% of the corpus by about 12%; fixed in the second version.

Until three-seed baselines on one chip type exist, **any difference under about 0.006 bits per byte is uninterpretable.** Results from different chip types are not pooled. Gate 2 in section 7 requires three seeds on one device, not two.

### Section 9: the machines were not reliably self-deleting

As written, deletion was a single unchecked request sent after an upload with no time limit, and the time limit lived in one controller's shell. **Fixed 2026-11-22.** Each machine now has:
1. a persistent systemd timer, which survives the death of any controller and a reboot, sends the delete request first and retries until the API confirms;
2. a Cloud Scheduler job in the project that sends the delete request from Google's side at the deadline (rehearsed with a read-only call: HTTP 200).

Deadlines: machine A 11:06 / 11:08 UTC, machine B 07:33 / 07:35 UTC, 22 November. Worst case for this cycle is now bounded at about $125 to $140 (v5e-4 for 10 hours plus v6e-4 for 6 hours at list prices). The expected cost is about $80 to $90: machine A stays until the CPU runs finish, around 09:45 UTC, and trains a further two batches meanwhile.
