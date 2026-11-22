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
| Optimiser control for corrections | `correction_optimizer.py` | **Running** | None yet | — |
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
| Optimiser artefact (Adam's per-coordinate steps, stale moments, too many steps) | Plain gradient descent stopped at first fit keeps the other facts | [Unknown] control running; note that optimiser state **was** a main cause in the memory-saturation result |
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

Pending the verification workflow. To be appended.

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
