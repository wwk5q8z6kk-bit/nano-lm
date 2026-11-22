# First-principles investigation: running record

Categories used throughout: Established · Published evidence · Observed · Derived · Hypothesis · Speculation.
References are from memory unless marked checked.

## Question 1. What changes when a system learns?

**Why it matters.** Every design choice so far assumed an answer without stating it. Memory mechanisms assume learning is storing; looping assumes it is acquiring procedures; plasticity work assumes it is keeping the ability to change.

**Three things that can change, and they are mathematically different.**

1. *Information about the world.* Bits the system holds about its data source. Bounded by the data: no computation increases the information a variable carries about its source (data-processing inequality). [Established]
2. *Accessibility.* The cost of computing useful answers from information already held. Shannon information does not see this at all. A theorem follows from its axioms and adds zero information, yet a system that can produce the proof is more capable than one that cannot. Formal treatments: time-bounded Kolmogorov complexity and logical depth (Bennett); information usable by a bounded family of predictors, which computation can increase (Xu et al., "usable information"). [Established definitions / Published]
3. *Inductive bias.* The prior over what to try next. Generalisation bounds depend on the distance between what was learned and the prior (PAC-Bayes); learning to learn is improving the prior so later tasks need fewer samples (Baxter). [Established as theory]

**Direct answers.**
- Capability can grow with no new information: yes. Self-play systems reach strong play from the rules alone; everything they learn was implied by the rules. [Published] What grows is accessibility.
- Information can grow with no new capability: yes. An unindexed or unstructured store holds facts it cannot use. [Derived]
- Acquiring information and acquiring the ability to use it are distinct: the first is statistical, the second computational. [Derived from 1 and 2]

**Competing accounts of what learning fundamentally is.**
- A. Inference: updating beliefs about hypotheses; computation is an implementation detail.
- B. Compression: finding the shortest description of experience (minimum description length, Solomonoff). [Established theory, uncomputable in general]
- C. Amortised computation: building procedures that make answers cheap; a learned function is a compiled shortcut.
- D. Change in adaptive dynamics: what matters is the set of behaviours the system can reach and how fast.

These predict different things. A says a task whose answers are fully determined by the input has nothing to learn. C says it has a learning curve, and that what is learned transfers to new inputs.

**What our experiments show about this.** [Observed, one seed] A looped model stayed at guess rate on two-step chains for seven minutes and reached 98% by fifteen. The inputs contained all the information throughout. [Inferred] That pattern fits C (a procedure being found) better than A; it does not exclude B.

**Evidence that would distinguish them.** Pair two tasks on one learner: one that needs only stored facts, one whose answers are computable from the input alone. Measure what transfers to unseen facts. Accounts A and C diverge on the second task.

**What is not understood.** [Unknown]
- No accepted theory prices computation on both sides: the learner's cost of building a procedure and the later cost of using it.
- No accepted measure of capability gained per unit of experience.
- Whether "what makes information reusable" has a general answer. Hypothesis: information is reusable when it is stored in parts that match the compositional structure of the tasks that will need it, and is addressable from those tasks. Untested.

**Candidate measurable definition, for later scrutiny.** [Hypothesis] Fix a compute budget. Learning is the increase in information usable within that budget about future tasks. Learning to learn is a fall in the experience needed to reach a given level on a new task. Cumulative learning would be both rising together over a long sequence.

## Register

**Unresolved questions**
1. What is retained when knowledge is retained: values, access paths, or the ability to reconstruct? (next)
2. Why does shared, repeated computation help, if it does?
3. Is forgetting a property of distributed representations or of the optimisation method?
4. What makes a representation reusable and compositional?
5. Does the continuous acquisition benchmark measure cumulative learning or sequential memorisation?

**Assumptions needing tests**
- That forgetting in the sparse memory was caused by address drift (one experiment, one alternative ruled in, none ruled out).
- That the looped model's advantage is about reuse and not size, optimisation or seed.
- That synthetic fact tasks say anything about language.

**Contradictory or fragile evidence**
- DEMOS "advantage" vanished with a tuned baseline.
- Dense learned the one-step task at one learning rate and not at the two either side.
- All chain results are single-seed until the replication reports.

## Question 2. What is retained when knowledge is retained?

**Why it matters.** It decides how the forgetting results should be read, and what "persistent knowledge" has to preserve.

**Three different things can survive.**
1. *Values:* the stored content still exists somewhere in the parameters.
2. *Access:* the system's ordinary forward computation still reaches that content.
3. *Reconstructability:* the content can be recovered at low cost, for example by a few steps of relearning ("savings").

They can come apart. Information can be present and unreachable (1 without 2), or absent in usable form yet cheap to rebuild (3 without 2). This is the accessibility distinction from question 1 applied to memory. [Derived] Published work reports that a network's internal features often survive when its outputs have forgotten, recoverable by refitting only the last layer. [Published evidence, from memory]

**What we believed.** "The learned-address memory forgot because its addresses moved." [Hypothesis]

**What was measured.** Learned addresses: 3.4% retention. Frozen addresses: 38–39%. Three seeds, 200 tasks, chance 1.6%. [Observed]

**Why that does not establish the explanation.** Freezing addresses changed several things at once:
- the address map stopped moving (the claimed cause);
- queries stayed spread across slots, where learned keys may have collapsed onto a few slots and overwritten each other's values;
- the input embeddings stopped changing, so old inputs still looked the same;
- learning became faster early on, which changes how hard each task presses on the stored values.

**Competing explanations.**
- Drift: values intact, path moved.
- Overwrite: values destroyed by later tasks sharing slots.
- Both.
- Neither is "forgetting" in the strong sense if relearning is much faster than learning.

**Evidence that distinguishes them** (`retention_probe.py`, readings committed before the run). Read each old task with a snapshot of the addressing taken when it was learned, against today's values. High accuracy means the values survived and only access was lost. Chance means the values were destroyed. Separately, compare steps to relearn an old task with steps to learn a new one.

**Is stable addressing required?** Not by any theorem I know. [Unknown] If a representation changes by a transformation the system can track, old entries could be re-keyed instead of frozen. Whether that can be learned cheaply is open. [Speculation]

**Is forgetting always destructive?** No in principle: merging two memories into one abstraction loses detail and can gain reach. We have measured nothing that separates harmful loss from useful reorganisation. [Unknown]

## Audit of our own conclusions

| Conclusion | Directly measured | Inferred or hypothesised | Not ruled out | Can the design support it? |
|---|---|---|---|---|
| A plain network loses plasticity | Steps per task rose from about 65 to never-within-300 over 200 tasks, three seeds | That this is the published "loss of plasticity" | Step cap too low late on; one learning rate; one optimiser | Supports "slower under these settings", not a general law |
| Normalisation and weight decay preserve learning speed | 33 → 37 steps, three seeds | Which of the two matters | Not separated; interaction untested | Supports the pair, not either alone |
| Learned-address memory fails | Slow and no retention, three seeds | That learned addressing is the cause | Slot count, temperature, top-k, learning rate all untuned | Weak: one configuration |
| Frozen addresses improve retention | 3.4% → 38% | That drift was the cause | Overwrite through key collapse; frozen input embeddings | No: several variables changed at once (probe running) |
| Looping beats dense on two-step chains | 98.2% vs 21.7% at 15 minutes | That reuse of weights is the cause | Seed; model size; learning rate tuned on dense only at one step | No: one seed (replication running) |
| DEMOS has no learning-speed advantage | Tuned dense 99.8% vs DEMOS 98.3% | — | DEMOS itself untuned | Supports "no advantage shown", not "worse" |
| DEMOS cannot follow chains | At guess rate in every variant, 7 minutes | That the hop step is the reason | Budget; the erasing variant also failed | Weak: derivation shows a limit of one variant only |
| Learned discrete keys do not train | Guess rate after 7 minutes, one seed | That exact match gives no gradient | Budget; codebook size; estimator | Weak: one configuration, one seed |
| Learning speed and retention are separate | Normalised network fast and forgetful | That they are independent in general | Only one task family | Supports "can dissociate", which is all it claims |

Common weaknesses: single seeds in several places; one hyperparameter setting per arm; synthetic fact tasks only; short budgets on tasks known to show sudden late jumps.

## Question 3. Why might repeated, shared computation help?

**Why it matters.** One seed showed a looped model learning two-step chains that a dense model did not. Before treating reuse as a principle, the explanations have to be separated.

**A distinction that organises the question.** [Derived, by counting] Some knowledge is compressible: a procedure such as "look up, then look up again" has a short description however many inputs it applies to. Some is not: arbitrary facts need storage in proportion to their number, whatever the architecture. Shared operations can only help with the first kind. For the second kind, sharing removes capacity.

**Competing explanations for the looping result.**

| Explanation | What it predicts |
|---|---|
| E1 Shared learning signal: every step of the chain trains the same weights | Advantage grows with the number of hops; two layers looped twice falls between |
| E2 Restricted hypothesis class: tied models can only express "the same function applied repeatedly", which is right for this task and smaller, so it needs fewer examples | Advantage disappears on tasks whose steps need different operations |
| E3 Transfer inside the model: once one hop is learned, the second application comes free | Two-hop accuracy jumps soon after one-hop mastery in the looped model; a long lag in the dense one |
| E4 Size or optimisation: fewer parameters train more easily | A dense model narrowed to the same parameter count does as well |
| E5 Chance: these tasks are learned in sudden jumps whose timing varies | Disappears across seeds |

**What the running replication can and cannot decide.** Three seeds address E5. The learning curves address E3. The single unlooped layer does not address E4, because one pass may be unable to do two hops at all. **Missing control: a four-layer dense model narrowed to the looped model's parameter count.** E1 and E2 need tasks with more hops and a task with unlike steps; neither is in the protocol.

**Relevant mathematics.** A tied model's functions are a subset of an untied model's; smaller classes need fewer samples when the target lies inside them. [Established, generic] Iteration to a fixed point allows more steps at test time than in training. [Established] Short programs are favoured by a simplicity prior, which links this to the compression account of learning in question 1. [Established theory]

**Published evidence that cuts the other way.** [Published, from memory, unchecked] In language modelling, sharing weights across layers has generally cost some quality at equal compute while saving parameters. That fits the distinction above: language carries a great deal of incompressible knowledge.

**Prediction recorded before the language-model pilot reports.** At equal steps, the looped model with fewer parameters will be no better than dense on bits per byte, and probably slightly worse. If it is clearly better, the two-kinds view is wrong or incomplete.

**Not understood.** Why gradient descent finds the iterative solution when it does. Whether natural tasks separate cleanly into procedures and facts. Whether a system can be given capacity for facts and reuse for procedures without being told which is which.

### Question 2: results of the retention probe (2026-11-20)

80 tasks, two seeds, chance 1.6% (`retention_probe.py`, `results/pilots/retention-probe.json`). [Observed]

| Learner | Old tasks, read normally | Old tasks, read with their original addressing | Relearn an old task | Learn a new task |
|---|---|---|---|---|
| Sparse memory, learned addresses | 1.4–1.6% | 50–55% | 300 steps (cap) | 300 steps (cap) |
| Plain network with normalisation and weight decay | 1.6–2.1% | not applicable | 34–35 steps | 33–35 steps |

- **Memory: the pre-stated reading "both" applies.** About half of the stored values survived and became unreachable because the addressing moved; the other half was destroyed. Address drift is a real cause, and it is not the whole cause.
- **Plain network: no savings.** Relearning an old task takes as long as learning a new one, so by this measure nothing reusable was left.
- **The output-layer probe proved nothing, and a control shows why.** Refitting only the output layer recovered 65% of old tasks, which looked like surviving features. The same refit on tasks the network had never seen recovered 64%. With 256 facts and wide features the output layer can simply memorise. The probe is discarded.
- Savings for the memory are uninformative: both hit the step cap.

Lesson for method: a probe needs its own control before its result is read.

## Question 4. Is forgetting caused by distributed representations, or by how we optimise them?

**Why it matters.** If forgetting is built into shared parameters, the answer is a different substrate. If it comes from the update rule, the answer is a different way of learning. The remedies differ completely.

**What one gradient step does to old knowledge.** [Derived, first order] A step on a new task changes the parameters by −η∇L_new. The old task's loss changes by about −η ∇L_old·∇L_new. So:
- if the two gradients point the same way, learning the new task also improves the old one: transfer;
- if they oppose, it damages the old one: interference;
- if they are orthogonal, neither.

**Transfer and interference are one quantity with opposite signs.** Any design that removes interference by keeping tasks in separate parameters also removes transfer between them. This is why the frozen-address memory retains more and composes nothing, and it is a trade-off, not an engineering defect. [Derived; the trade-off is discussed in published continual-learning work]

**Forgetting is not forced by distributed representations.** The same network, trained on all tasks at once, holds all of them. The capacity is there. What the sequential learner lacks is information about what the old tasks still require. [Derived] Ordinary large-scale training avoids forgetting by showing everything repeatedly in shuffled order, which is replay at full scale.

**The exact answer is known, and it is expensive.** Exact Bayesian updating never forgets: yesterday's posterior is today's prior, and it records which parameter directions past data has pinned down. [Established] Gradient descent keeps a single point and throws that record away. Forgetting is the cost of that approximation. Published methods approximate the record (importance weights per parameter, variational posteriors) and work partially. A published result states that optimal continual learning in general requires perfect memory and is computationally hard. [Published, from memory, unchecked] So there is no free solution in the worst case; any practical one must exploit structure in what is being learned.

**Alternatives to updating shared numbers.** [Published, from memory]
- Local learning rules (Hebbian, predictive coding, equilibrium propagation, forward-only methods): updates touch fewer parameters, which lowers interference and transfer together.
- Structural growth: add capacity for each new task; no forgetting by construction, no improvement of old skills, growth without bound.
- Learning operations instead of numbers: systems that add reusable programs to a library do not overwrite earlier entries. Knowledge of that kind is cumulative by construction; the cost moves to searching for programs.

**Assumptions rarely questioned.**
1. One point estimate of the weights, with no record of how firmly each is determined.
2. Shuffled, repeated data as the normal condition for learning.
3. Fixed capacity.
4. The same update rule for facts and for skills.
5. The same learning rate for a parameter whatever past data depends on it.

**A reading of our own result.** [Hypothesis] Weight decay kept the plain network able to learn by continually shrinking weights, which is also a mechanism of erasure. If so, the plasticity it bought and the retention it lost are the same effect. Untested: we never ran normalisation without decay.

**Evidence that would distinguish the explanations.** Measure the gradient inner product between old and new tasks for each learner. Run normalisation and weight decay separately. Add a learner that projects each update to be orthogonal to stored old gradients. If forgetting tracks the inner product across all of them, the first-order account is sufficient.

**Not understood.** Whether a representation can be learned in which related tasks share directions and unrelated tasks do not, so that transfer is kept and interference removed. That is question 5.

### Question 3: result of the pre-registered looping replication (2026-11-21)

Protocol: `PROTOCOL-LOOP.md`, committed before the fits. 13,000 steps, three seeds, one learning rate for all arms. Raw files: `results/loop/`. [Observed]

Two-step accuracy at the end (guessing is about 21%):

| Arm | Parameters | Seed 11 | Seed 12 | Seed 13 | Seeds that learned |
|---|---|---|---|---|---|
| dense, 4 layers | 278k | 20.7% | 22.6% | 22.4% | 0 of 3 |
| looped, 1 layer × 4 | 129k | 20.5% | 98.1% | 95.1% | 2 of 3 |
| looped, 2 layers × 2 | 178k | 96.5% | 95.6% | 22.2% | 2 of 3 |
| looped wide, 1 layer × 4, width 96 | 239k | 20.9% | 22.9% | 22.3% | 0 of 3 |
| one layer, one pass | 129k | 20.6% | 22.2% | 21.7% | 0 of 3 |

One-step accuracy was 98–100% in every run.

**Verdict by the pre-registered rules: mixed.** "Replicated" needed looped to win in all three seeds; it won in two. "Not replicated" and "it is model size" both fail as well.

**What the data do and do not say.**
- [Observed] Every run is either at guessing or above 95%. Where learning happens it is a jump that starts between 5,000 and 7,000 steps, never gradual. The outcome at this budget is a matter of whether the jump has happened yet.
- [Observed] Weight-tied arms at width 64 made the jump in 4 of 6 runs; untied dense in 0 of 3. With counts this small the difference is suggestive, not established (Fisher exact, 4/6 against 0/3, p is about 0.17).
- [Observed] The wide looped model failed in all three seeds. This contradicts the simple reading "tying weights helps". It also had the same learning rate as the narrow one, which may be too high for it; the protocol named the lack of per-arm tuning as a limitation.
- [Observed] One layer applied once never learned two steps, so E4 in its simplest form (smaller is easier) is not supported. The narrow four-layer dense control is still missing.
- E5 (chance) is not eliminated: jump timing varies by seed, and a seed-dependent jump is exactly what E5 describes. What survives is a weaker claim: tying may make the jump more likely or earlier at this budget.
- E3 predicted the two-step jump soon after one-step mastery. **Corrected 2026-11-21 after re-reading the curves; the first version of this line was wrong.** One-step accuracy itself sat at guessing for 3,000 to 5,000 steps and then jumped (above 90% at step 4,000 to 6,000 in all fifteen runs). In the four runs that learned two steps, that jump began within 0 to 3,000 steps of one-step mastery. So E3 is consistent with the successes, but it does not explain the looped runs that mastered one step and never made the second jump.

**Status change.** "Looping learns two-step chains that dense does not" is downgraded from pilot observation to: mixed under pre-registration, two of three seeds. It must not be cited as a demonstrated advantage.

**What would settle it.** Longer budget for all arms (does dense jump later, or never), ten or more seeds to estimate jump probability and timing per arm, a learning-rate sweep for the wide arm, and the narrow dense control. These are cheap on the laptop.

### Question 3: result of the pre-registered follow-up, `PROTOCOL-LOOP2.md` (2026-11-21)

26,000 steps, five new seeds, raw files in `results/loop2/`. [Observed]

| Seed | Dense two-step | Looped two-step | Dense first at 90% | Looped first at 90% |
|---|---|---|---|---|
| 21 | 19.7% | 98.8% | never | 21,000 |
| 22 | 23.1% | 98.8% | never | 6,000 |
| 23 | 21.7% | 99.2% | never | 23,000 |
| 24 | 22.0% | 22.3% | never | never |
| 25 | 90.1% | 22.9% | 25,000 | never |

Looped learned in 3 of 5, dense in 1 of 5. **Verdict by the registered rules: mixed.** "More likely" needed looped in at least 4; "only slower" needed dense in at least 3; "no difference" needed counts within 1.

- **Dense does make the jump.** Seed 25 jumped at step 25,000. "Dense cannot learn two-step chains" is refuted at this budget; "dense is later or less likely" survives as a possibility.
- Across both registered tests: looped 5 of 8, dense 1 of 8, at two different budgets. That is not a pooled estimate and no test is attached to it.
- Published work reports weight-tied models succeeding on all seeds on related tasks (referee's search, section 10 of the nested note). Ours fail on 3 of 8. The difference is unexplained and points at our optimiser settings before it points at anything about tying.

## Question 6. Does our acquisition benchmark measure cumulative learning?

**Answer: no.** [Derived] `cab.py` uses arbitrary facts. A plain lookup table scores 100% on acquisition, retention and correction, and 100% on composition by iterating its lookups. So the benchmark cannot tell learning from storage, and my prediction P3 for it ("no learner gets all four above 80%") is false for a learner I did not include.

**A definition that excludes the lookup table.** [Proposed] A learner is cumulative if, as the stream grows: (1) what it held stays correct; (2) new things get easier because of old things (forward transfer); (3) cost per new item stays bounded. A lookup table has 1 and 3. A network trained by gradient may have 2 and loses 1. Property 2 can only be measured in a world with structure to discover.

**A candidate principle for having all three.** [Derived; components published, see below] Interference is the price of sharing. With a linear read, writing one fact changes the answer for another in proportion to the overlap of their addresses. If addresses are fixed, sequential least squares already gives exact, order-independent, non-forgetting writes. The unsolved part is changing addresses. That is cheap and exact when:
- every item has its own address, so changing one address touches one item;
- everything shared is a **sum** of per-item statistics, so an item's contribution can be subtracted and re-added exactly;
- the raw facts sit in an exact store, which is authoritative for anything seen.

Gradient steps are neither additive nor reversible. Sums of statistics are both.

**Test.** `cab_s.py`: 16 hidden types, 8 relations, 32 values, 10% exceptions; 40 phases of 64 new entities with 3 of 8 relations shown; 16 corrections per phase. Predictions P1–P4 were committed (`cdfae59`) before any run. Hyperparameters were tuned on seed 4000 only; results below are seeds 4001–4003, last ten phases. Raw files: `results/cab_s/`. [Observed]

| Learner | Infer, new entities | Infer, old entities | Retain (model alone) | Retain (with store) | Correct | Seconds |
|---|---|---|---|---|---|---|
| Lookup table | 13.5% | 13.5% | 13.1% | 100% | 100% | 0.4 |
| Gradient, current phase only | 79.4% | 69.5% | 93.5% | 93.5% | 100% | 86 |
| Gradient, replay of everything seen (lr 0.003) | 78.2% | 71.7% | 99.9% | 99.9% | 60.4% | 96 |
| Gradient, replay of everything seen (lr 0.01) | 76.4% | 70.3% | 100% | 100% | 97.5% | 92 |
| Additive statistics with exact store | 86.7% | 82.8% | 89.4% | 100% | 100% | 5.8 |

Chance is 3.1%; the ceiling for inference is about 90%. Seed-to-seed spread was under one point for every learner.

**Predictions against results.**
- P1 (table retains, cannot infer): held.
- P2 (gradient learner forgets heavily; old-entity inference at least 20 points below new): **failed.** It kept 93.5% and the gap was 10 points. Forgetting was mild in this world.
- P3 (additive: exact retention, inference at least 75%, old within 10 points of new): held.
- P4 (full replay comes within 10 points of additive on inference): held, at 8.5 points. By the rule set in advance this means **the additive write buys exactness and cost, not a decisive accuracy gain.** It was nonetheless ahead on every seed.

**Limits.**
- The world matches the learner's model exactly (a bilinear rule). Every learner used that same model, so the comparison is fair, but nothing here shows the principle survives when the shared part needs deep features. Then a change to the shared encoder touches every item and exact re-addition costs time proportional to everything stored.
- Timings compare a closed-form solver with 200 Adam steps per phase; they show order of magnitude only.
- The ridge strength chosen was at the edge of the three values tried.
- The components are published: online alternating least squares, decremental least squares, frozen-key stores for continual learning. No novelty is claimed. What this adds for us is a learner that meets all three parts of the definition on a structured stream, and a benchmark a lookup table fails.

**Next question.** Can the shared part be made deep while keeping writes additive and local? Candidates: fixed random features under the statistics; a shared encoder that is re-fitted only when the store has doubled, so the re-addition cost per fact stays constant on average.

### First acquisition benchmark (`cab.py`), full run (2026-11-21)

Seeds 3001–3003, 60 phases, raw file `results/cab/cab-v0.json`. Run with 8 relations after the sampler fix recorded in the file header. Means over the last 15 phases; seed spread was a few points at most. [Observed]

| Learner | Acquire new facts | Retain old facts | Apply corrections | Still answers the stale value | Two-step chains by iterating | Two-step chains in one pass |
|---|---|---|---|---|---|---|
| Plain network | 100% | 4.7% | 6.6% | 0.2% | 8.3% | 2.3% |
| Network with 10% replay | 100% | 19.5% | 21.6% | 0.2% | 21.4% | 1.8% |
| Fixed-address memory | 2.5% | 60.7% | 10.8% | 75.8% | 2.0% | 0.1% |

Fixed-address memory in the first 15 phases: acquire 69%, retain 86%, iterate 68.5%, stale answers 91%.

- P2 held: the plain network kept almost nothing, replay helped without fixing it, and the fixed-address memory retained most but kept answering stale values.
- P3 held as worded (no learner got all four above 80%), but is empty: a lookup table would get all four (Question 6).
- P1 held only where one-step facts were retained. Iterating a one-step read composes untrained chains exactly as well as the facts themselves are remembered; the one-pass read never composes. That is nearly true by construction.
- **Not predicted: the fixed-address memory stopped acquiring.** New-fact accuracy fell from 69% early to 2.5% late, with 7,680 facts in 16,384 slots read 8 at a time. It did not forget so much as fill up and refuse. Retention and acquisition traded off inside one learner.
- This benchmark uses arbitrary facts, so it shows only the storage side of the problem. Its replacement is `cab_s.py`.

### Language-model store pilot: threats to validity recorded before any arm result was read (2026-11-21)

Written while `lm_store.py` was still training its first arm; no arm result had been printed or read. These do not change the registered predictions S1–S4 or the 0.01 bits-per-byte threshold. They limit what either outcome can mean.

1. **Same-document leakage in training.** On training text the store's counts leave out only the current position. Other occurrences of the same n-gram in the same document stay in. On held-out text the whole document is absent from the store. So during training the store looks more reliable than it will be at test time, and a jointly trained core may learn to lean on the store for within-document repetition that the store cannot supply later. This biases **against** the joint arms. If joint training fails S2, that is not yet evidence that routing fails on text; a leave-document-out store is needed first. If it passes, it passes despite this handicap.
2. **Static, not causal.** The store is built from the whole training subset, including text that comes after a given position. That is legitimate for a fixed training-set model, in the same way that weights are a function of the whole training set. It is not an online or streaming memory and must not be described as one.
3. **Held-out separation.** The store contains training text only. The mixing table for the post-hoc arms is fitted on training text with the core frozen, not on held-out text. No model selection uses the held-out sets. Near-duplicate documents between training and held-out sets were removed when the corpus was built (the manifest records skipped overlaps) but this was not re-audited here.
4. **The mixture is a proper probability model.** The gate depends only on the context (order matched and context count), so the reported bits per byte are valid for the combined system.
5. **System cost.** The store for 48M tokens is 1.02 GB against about 57 MB of model weights. Any gain is a system-level gain and must be reported with that storage.
6. **One seed.** Differences under about 0.01 bits per byte are not evidence, as registered.
7. **Operational.** The run shared the laptop with eleven other jobs and the machine was swapping heavily; wall-clock times from this run are not usable as efficiency measurements.

## Question 7. Why does the fixed-address memory stop acquiring? (2026-11-21)

Diagnostic `memory_saturation.py`, predictions committed in `7132497` before any run. One seed (3001), so effect sizes matter and small differences do not. Raw files: `results/memory_saturation/`. [Observed]

Measured without training: 16,384 slots, 4,619 ever addressed, effective number of slots 526; two queries with the same relation share 0.60 of 8 slots against 0.004 for a uniform hash.

| Arm | New-fact accuracy, phases 1–15 | 16–30 | 31–45 | 46–60 | Retention, phases 46–60 |
|---|---|---|---|---|---|
| Original addresses | 88.1% | 28.9% | 6.9% | 3.4% | 63.9% |
| Uniformly hashed addresses | 100% | 100% | 100% | 99.9% | 89.2% |
| Original addresses, learning rate ×10 | 99.9% | 94.9% | 76.6% | 59.9% | 86.3% |
| Original addresses, optimiser state reset every phase | 99.3% | 96.4% | 91.2% | 86.1% | 82.7% |

- D1 (the collapse reproduces): held, 3.4%, without the chain loss that `cab.py` also trains.
- D2 (hashing keeps acquisition above 90% and retention above 80%): held.
- D3 (a higher learning rate helps but stays at least 20 points below hashing): held.
- D4 (resetting the optimiser does not restore acquisition): **failed.** Resetting Adam's state each phase took late acquisition from 3.4% to 86.1% and retention from 64% to 83%, with the addresses unchanged.

**Reading.** The memory was never full: in every arm with the original addresses the same 4,549 slots were in use. Two interventions each remove most of the collapse, so it has two necessary ingredients: queries concentrated on a few slots, **and** optimiser state carried across phases on those slots. Neither nominal capacity nor address concentration alone explains it.
**Mechanism, not yet tested.** [Hypothesis] Adam's running second-moment estimate stays large on rows that receive a burst of large gradients at the start of every phase, which shrinks the effective step so new facts cannot be fitted in 200 steps. Predicts: a faster-forgetting second moment (smaller β2) or plain gradient descent restores acquisition without resets.
**Limits.** One seed. Hashing changes the query representation as well as the address distribution, so its success supports congestion without excluding other causes. Loss of plasticity through optimiser state, and resetting it as a remedy, are reported in the continual-learning literature [Published, from memory, unchecked]; this is a diagnosis of our own learner, not a discovery.
**Consequence for an earlier claim.** "Fixed addresses trade retention for acquisition" (first acquisition benchmark) was an over-reading. With the optimiser reset, the same fixed addresses acquire 86% and retain 83%.

## Question 8. Does the routing gain need an oracle? (2026-11-21)

`routing_loo.py`, predictions committed in `7132497`. Seeds 7001–7003. Raw files: `results/routing_loo/`. [Observed]

The detector judges each fact before fitting it: fit a temporary address to the entity's other facts with the decoder frozen, and ask whether this fact is predicted. No labels, no look-ahead.

| Arm | Inference on unseen relations | Old regular facts, entities with an irregular write | Old regular facts, clean entities | Sent to exact store |
|---|---|---|---|---|
| Everything trains the model | 77.6% | 78.4% | 100% | 0% |
| Oracle routing | 88.9% | 99.8% | 100% | 16.2% |
| Leave-one-out detector | 81.4% (84.5, 81.9, 77.7) | 95.0% | 96.0% | 17.4% |

Detector quality: recall 88.3% on truly irregular facts, false alarms 6.5% on regular ones.

- L1 (the oracle gap reproduces): held, 11.3 points.
- L2 (the detector recovers at least half the inference gap): **failed.** It recovered 3.8 of 11.3 points, about a third, and one seed was within a point of the baseline.
- L3 (old regular facts on irregular-carrying entities stay at 95% or more): **not met**, 94.997% on average and below 95% on one seed.
- L4 (recall at least 70%, false alarms at most 15%): held.
- The registered failure condition (within 2 points of the baseline on inference) was not triggered on the mean, but was on one seed.

**Reading.** A write-time detector with no oracle keeps most of the retention benefit and about a third of the inference benefit. It also has a cost the oracle does not: clean entities lost 4 points, because a false alarm takes away one of only three facts that identify the entity.
**Unresolved.** The spread across seeds suggests a feedback loop: a weak model detects poorly, so it trains on more noise, so it stays weak. Untested.
**Scope.** Exceptions and corrections only. Nothing here separates a one-off exception from the first example of a new rule, or from a rule that has changed; a detector of this kind would send all three to the store, and for the last two that is the wrong decision.

### Language-model store pilot: result (2026-11-21)

`lm_store.py`, predictions S1–S4 committed in `6e26dc2`; validity threats committed in `5cac076` before any arm result was read. One seed, 48M-token training subset, 2,000 steps. Raw file: `results/lm/store-pilot.json`. Bits per byte on held-out text, lower is better. [Observed]

| Arm | Parameters | Web | Textbook | Mathematics | Code | Weighted mix |
|---|---|---|---|---|---|---|
| 8 blocks, no store | 14.3M | 1.499 | 1.242 | 1.704 | 1.543 | 1.497 |
| 8 blocks, store added after training | 14.3M | 1.451 | 1.168 | 1.639 | 1.457 | 1.439 |
| 8 blocks, trained with the store | 14.3M | 1.533 | 1.238 | 1.868 | 1.902 | 1.583 |
| 4 blocks, no store | 11.3M | 1.526 | 1.272 | 1.776 | 1.612 | 1.536 |
| 4 blocks, store added after training | 11.3M | 1.476 | 1.194 | 1.704 | 1.521 | 1.475 |
| 4 blocks, trained with the store | 11.3M | 1.536 | 1.242 | 1.853 | 1.884 | 1.581 |

The store is 1.02 GB for 48M tokens of text, against about 57 MB of weights for the 8-block model.

**Against the registered predictions.**
- S1 (adding the store after training helps): **held.** 0.058 bits per byte better on the mix at 8 blocks, 0.062 at 4 blocks, on every domain.
- S2 (training with the store beats adding it afterwards by at least 0.01): **failed decisively.** Training with the store was 0.14 worse than adding it afterwards, and 0.085 worse than using no store at all.
- S3 (4 blocks trained with the store match 8 blocks alone): **failed**, 1.581 against 1.497.
- S4 (the store helps most on code and least on web): **held** for the post-hoc arms (code 0.087, web 0.048).

**Verdict by the registered rule.** Training the core through the store adds nothing beyond mixing the store in afterwards; in this implementation it does harm. The claim that routing memorisable content away from the core during training carries over to text is **not supported**.

**What the pre-recorded caveat allows and does not allow.** Before any result was read I recorded that same-document leakage biases this test against the joint arms. The pattern fits: the damage is concentrated in code (+0.36 over no store) and mathematics (+0.16), where repetition inside a document is heaviest, and is near zero on textbook text. That is a reading made after seeing the pattern, so it is a hypothesis. It does not rescue S2. It does mean the registered failure cannot yet be read as "the principle is false for text": a store that excludes the whole current document during training is the control that would separate the two.

**Not registered, but visible in the table.** Four blocks with the store added after training (1.475) beat eight blocks without it (1.497). So the capacity claim behind S3 holds for the post-hoc store, which is the published inference-time effect, and fails for joint training. One seed, and the margin (0.023) is above the 0.01 noise floor I set but not by much.

**Cost.** Any gain here is a system-level gain bought with 1.02 GB of exact storage. Per stored byte that is a poor exchange: doubling the blocks costs about 12 MB and buys 0.039; the store costs 1,022 MB and buys 0.058. Wall-clock times are unusable (the machine was swapping).

## Question 9. How should a correction be written, and does a harmless write exist? (2026-11-21)

`correction_interference.py`, predictions K1–K6 committed in `0443ef6` before any registered run. Seeds 8001–8003. World with corrections only (no exceptions). Raw files: `results/correction_interference/`. Model alone unless stated. [Observed]

| How the correction is written | Correction right, at once | Entity's other facts, at once | Same-type facts for that relation | Old regular facts on corrected entities, at the end | Old regular facts on other entities | Correction still right 20+ phases later | Inference on unseen relations |
|---|---|---|---|---|---|---|---|
| Trained alone among that entity's facts | 100% | 64.7% | 95.7% | 59.6% | 99.9% | 66.2% | 98.0% |
| Trained with the entity's other shown facts | 100% | 100% | 99.7% | 99.5% | 99.9% | 62.3% | 98.6% |
| Trained with as many facts from other entities | 100% | 66.6% | 96.5% | 68.8% | 100% | 67.0% | 97.9% |
| Trained with the entity's address frozen | 100% | 84.9% | 56.7% | 54.7% | 60.9% | 12.1% | 66.7% |
| Not trained; kept in an exact store | 2.7% (100% with the store) | 99.9% | 99.9% | 99.5% | 99.9% | 2.8% (100% with the store) | 99.9% |
| Not trained; computed edit of the address only | 99.8% | 98.0% | 98.2% | 83.5% | 99.9% | 21.5% | 99.7% |

The linear programme found an address-only edit that changes none of the entity's other seven answers for **99.84%** of corrections (all but one of 624 in each seed).

**Against the registered predictions.**
- K1 (training the correction alone reproduces the damage): held. A third of the entity's other facts break at the moment of writing.
- K2 (a harmless address-only edit exists for at least 80% of corrections): held, 99.84%. The model can represent the corrected state. The damage is not a capacity limit.
- K3 (the computed edit: harmless at once, correction taken, and old facts on those entities at 95% or more at the end): the first two parts held (98.0%, 99.8%); **the third failed** (83.5%).
- K4 (rehearsing the entity's own facts protects it; equal-sized replay of other entities' facts does not): held. 100% against 66.6%.
- K5 (freezing the address protects the entity, but the write fails or the rule is damaged): **the first part failed.** The entity was not protected (84.9%), because the decoder absorbed the correction by bending the rule for the whole type. The second part held, severely: same-type facts fell to 56.7%, uncorrected entities' old facts to 60.9%, inference to 66.7%.
- K6 (a correction written into the model decays under later training): held in every arm that writes into the model: 100% to 62–67% for gradient writes, 99.8% to 21.5% for the computed edit.

**What this establishes, in this world and model.**
1. **The collateral damage of a correction is caused by the write rule, not by the representation.** A state exists in which the correction holds and nothing else of that entity changes, in 99.8% of cases. Training on the one fact does not reach it.
2. **What is missing is the entity's own facts, specifically.** Adding them removes the damage completely. Adding the same number of facts from other entities removes none of it. So this is not "replay helps" in general; it is a missing constraint on one address.
3. **Where the correction is written matters more than how hard.** Forced into the shared decoder (address frozen), one correction per 13 new entities destroys the rule for everyone.
4. **There are two separate problems, and they have different fixes.** Writing without damage is solved by same-entity rehearsal or by a computed edit. Keeping an off-rule fact alive is not solved by either: later training on rule-following data pulls it back in every arm. Only the exact store held corrections.
5. **A minimal edit is fragile.** The computed edit satisfies its constraints with the smallest margins that work, so later drift undoes both the correction (21.5%) and some neighbours (83.5%). Larger margins, or re-solving the edit when the decoder has moved, are untested.

**Rejected explanations.** "The model cannot hold exceptions without separate storage" (rejected for the moment of writing by K2; not rejected for durability). "Any replay protects" (rejected by the random-replay arm). "Protect the address and let the decoder learn the exception" (rejected by the frozen-address arm).

**Limits.** One world, one bilinear model, Adam only. The store arm's perfect retention is by construction. Correction locality, damage to a subject's other facts, and memory-based editing with a scope decision are established topics in model editing [Published, verification in progress]; this is a clean reproduction and decomposition in our own harness, not a discovery.

### Question 9, follow-up: is it the size of the write, or missing constraints? (2026-11-21)

`correction_margins.py`, predictions M1–M4 committed in `8d3b03e`. Same three seeds and streams as the arms above. All arms write the correction by a computed edit of the entity's address only. Raw files: `results/correction_interference/margins-*.json`. [Observed]

| Edit | Size of the edit (L1) | Entity's other facts, at once | Old regular facts on corrected entities, at the end | Correction still right 20+ phases later |
|---|---|---|---|---|
| Smallest edit that installs the correction, no other constraint | 12.7 | 35.3% | 34.2% | 23.8% |
| Constrained, margin 1 (the registered arm; reproduced exactly) | 21.8 | 98.0% | 83.4% | 21.5% |
| Constrained, margin 3 | 55.5 | 97.3% | 78.7% | 31.1% |
| Constrained, margin 6 | 110.0 | 96.9% | 75.7% | 35.1% |
| Constrained, margin 3, other relations keep 90% of their margin | 62.4 | 98.5% | 89.7% | 21.4% |

For comparison, from the table above: trained alone 64.7% / 59.6% / 66.2%; trained with the entity's other facts 100% / 99.5% / 62.3%.

- M1 (the smallest write still damages): held, and more strongly than predicted. **The smallest possible write is the most damaging of all** (35.3%), worse than gradient training on the fact alone (64.7%). The reading "the write is just too big; a minimal write would be harmless" is refuted. What protects the entity's other answers is constraints, not smallness.
- M2 (a larger margin makes the correction last, reaching 60% at margin 6): direction held (21.5% → 31.1% → 35.1%), **size failed**. Larger margins also make the entity's other facts more fragile (83.4% → 75.7%).
- M3 (keeping the neighbours' margins holds them at 93% or more): **failed**, 89.7%. It helps (from 78.7% at the same margin) but does not solve it.
- M4 (no margin makes a written correction permanent): held. The best computed edit keeps 35%; gradient training kept 62–67%; only the exact store keeps 100%.

**Two conclusions that now have direct evidence in this world.**
1. *At the moment of writing*, harm comes from absent constraints. Supplying them, either as the entity's own facts or as the model's own current answers, removes it. Making the write smaller does not.
2. *Afterwards*, anything the later data does not re-assert is eroded, whatever the margin. A constraint has to be present at every later update that can disturb it, not only at the moment of writing.

### Question 9, second follow-up: is it the optimiser? (2026-11-21)

`correction_optimizer.py`, prediction O1 committed in `27cc27c`. Same three seeds. The correction is written by gradient steps on the entity's address only (decoder frozen), using only the corrected fact. [Observed]

| Write | Steps taken | Correction right, at once | Entity's other facts, at once | Old regular facts on corrected entities, at the end |
|---|---|---|---|---|
| Plain gradient descent, stopped at a 1-logit lead | about 750 | 99.4% | 14.3% | 12.0% |
| Adam with fresh state, stopped at a 1-logit lead | about 4,180 | 99.0% | 39.8% | 39.1% |
| Adam with fresh state, 200 steps | 200 | 47.9% | 62.5% | 60.9% |

- O1 (every gradient-only write damages the entity's other facts, 85% or less): held for all three. None came near 95%.
- **The optimiser is not the cause of the correction damage.** Plain gradient descent stopped the moment the correction takes is the most destructive write we have measured. Fresh optimiser state does not help. Fewer steps help only by not completing the write.
- This differs from the memory-saturation result (Question 7), where carried-over optimiser state was a main cause. The two failures look alike and have different causes.

**Explanations of the correction damage, final status in this world.** Capacity: rejected (a harmless state exists for 99.8%). Size of the write: rejected (the smallest write keeps 35%). Optimiser: rejected (14% to 63%). Replay in general: rejected (other entities' facts keep 67%). Missing constraints on the written address: supported (the entity's own facts 100%; the model's own current answers 98%).

## Question 10. Is a small model limited by its capacity or by its training signal? (registered 2026-11-21, before any run)

**Why.** Between 14M and 40M parameters the dense model improves by 0.096 bits per byte on the same 800M tokens. Two mechanisms we tested on real text (an exact n-gram store, an exact in-context cache) add under 0.005 to the trained 14M model. Before testing further mechanisms we should know whether the 14M model could do better at all with the same parameters and data, given a better target to learn from.

**Test.** Distil the finished 101M model into fresh 14M and 40M models: loss = α · (cross-entropy on the data) + (1 − α) · KL(teacher ‖ student). Same data, token budget, schedule and seed as the baselines. Script: `tpu/train_jax.py`, second version; job: `tpu/job_kd.sh`. Teacher fixed in advance: the 101M model at learning rate 1e-3, seed 1000.

| Arm | Size | α |
|---|---|---|
| Baseline, second-version sampler | 14M, seeds 1000 and 1001; 40M, seed 1000 | — |
| Distilled | 14M | 0.5 |
| Distilled, teacher only | 14M | 0.0 |
| Distilled | 40M | 0.5 |

**Rules, fixed now.** Read on the 14M model, α 0.5, against its same-sampler baseline:
- gain of 0.03 bits per byte or more (about a third of the 14M-to-40M gap): the training signal is a major lever for small models;
- under 0.01: the 14M model is capacity-limited on this data; a better signal is not the lever;
- between: partial.
A difference counts only if it exceeds twice the difference between the two baseline seeds.

**Predictions.** 14M gains 0.01 to 0.03 ("partial"). 40M gains under 0.015. Teacher-only training is no better than α 0.5.
**Controls.** The teacher, evaluated inside the student run, must reproduce its own reported score to 0.001. The second-version baseline at 14M shows what the sampling fix changed: I expect it within 0.005 of the first-version result (1.1441).
**Not tested.** Distillation with a different temperature, a larger teacher, or more training tokens.

## Corrections and status changes, 2026-11-22 (UTC)

### The store pilot's S2/S3 verdict is confounded; status changed from "failed" to "unsettled"

Found by the literature-verification workflow and confirmed by reading `lm_store.py:112-116`. [Verified: mechanism. Inferred: size of its effect.]

In the arms trained through the store, the gate table started at zero (mixing weight 0.5). It was trained by the model's own AdamW optimiser: learning rate 1e-3, 200 warm-up steps, then cosine decay to 10% over 2,000 steps. Adam moves a parameter by at most about the learning rate per step, and the schedule sums to about 1.1. So the gate logit could move at most about ±1.1. **Wherever the store had a match, it kept between 0.25 and 0.75 of the mixture, even where it was wrong.** At a position where the store gives the true token probability 0, the mixture can reach at most 0.75 even if the core is certain, which costs at least 0.42 bits there. The after-training arms fitted the same table with Adam at learning rate 0.05, with nothing holding the weight in.

So the joint arms carried a handicap the post-hoc arms did not have, in addition to the same-document leakage recorded before the run. This alone could produce a loss of the size observed (0.14). S2 and S3 are therefore **not evidence** that training through a store hurts. The S1 and S4 results (post-hoc arms) are unaffected.

A valid re-test needs both a leave-document-out store and a free gate: a separate gate learning rate, or a gate initialised closed. It is low priority. The n-gram store adds at most 0.004 bits per byte to models trained on 800M tokens (below), and the registered rule for this line is read on the 101M checkpoint.

### All first-version TPU baseline scores carry a sampling skew; status "provisional"

Found by the independent code review of `tpu/train_jax.py`. [Verified by reading the code; the effect on scores is unmeasured.]

In the first version, crop starts came from `jax.random.randint` over about 8e8 positions. Without 64-bit mode in JAX 0.6.2 that draw is not uniform. The first 294,969,861 start positions each had 6 chances in the draw, the rest 5. So 41.2% of crops began in the first 36.9% of the training file: that region was sampled about 12% more often than the rest. The domains are interleaved in 64k-token blocks, so the domain mix is unaffected. The model still saw an uneven sample of documents.

This affects every run in the bucket under `runs/ladder-v5e/`, `runs/ladder-v5e-2/` and `runs/ladder-v6e/`, including the 14M result (1.1441) and the 40M result (1.0481), and the checkpoints used by the store-exchange and cache tests. Comparisons inside that set are like-for-like, because every run shares the skew. Comparisons with runs that sample uniformly are not.

Fixed in the second version of the script (commit `96c1863`). Crops are now drawn on the host with numpy, as in the PyTorch harness, and run names end in `-hs`. The effect is measured directly by Question 10's baselines: 14M seeds 1000 and 1001 and 40M seed 1000, with the fixed sampler, the same seeds and the same recipe. Until that reading, no first-version score should be cited as settled.

### In-context cache test on the 40M checkpoint: registered rule read

`tpu/cache_exchange.py`, rule registered in its header before any run. Raw files: `results/lm/cache-exchange-S.json`, `results/lm/cache-exchange-M.json`. [Observed: one checkpoint each, first-version sampler.]

| Model | Web | Textbook | Mathematics | Code | Mix | Same-window tokens the cache could copy |
|---|---|---|---|---|---|---|
| 14M | 0.0023 | −0.0005 | 0.0068 | 0.0114 | 0.0036 | web 13%, textbook 9%, maths 33%, code 45% |
| 40M | 0.0012 | −0.0003 | 0.0036 | 0.0059 | 0.0018 | same windows |

Gain in bits per byte from mixing an exact causal cache over the same 512-token window into the trained model. Controls: causality check passed; a cache filled from a different window gives 0.0000; the model alone reproduces the training run's own scores to 0.0001.

**By the registered rule:** under 0.01 on every domain at 40M, so the explicit-copy line is closed at this scale and context length. Up to 45% of code tokens repeat something earlier in the window. The trained model already predicts those tokens about as well as an exact cache does, and the gain halves from 14M to 40M.

Limits. The cache sees only the model's own 512-token window, so this says nothing about copying from beyond the context. These are first-version checkpoints, but a sampling skew of this size cannot plausibly move a gain of 0.006 past 0.01.

M21's 0.36-bits-per-token advantage on copyable code tokens was measured against a 5M model trained for minutes. It does not survive against dense models trained on 800M tokens.

### Question 10: amendment before any run (2026-11-22, 03:15 UTC; the batch had not started)

The independent review found two weaknesses in the registration above. Both are corrected before any run:

1. **Micro-batch confound.** The 14M baselines were set to 2 micro-batches and the distilled arms to 4. Accumulation gives the same update in exact arithmetic but not in floating point, and the review measured unexplained offsets of 0.002 to 0.006 between runs that differed in micro-batch count and chip. Now every arm uses 4 micro-batches on the same chip type (v5e).
2. **Noise rule.** "Twice the difference between two baseline seeds" passes a null effect about 30% of the time. Now there are three 14M baseline seeds (1000, 1001, 1002). A difference counts only if it exceeds t(0.975, 2 df) · s · √(1 + 1/3) ≈ 4.97 s, where s is the standard deviation of the three baseline seeds. The decision thresholds (0.03 major lever, under 0.01 capacity-limited) are unchanged.

Every result file now records the chip type and a hash of the training script. The sampling-fix check compares the mean of the three 14M seeds with the single first-version run (1.1441). It is informative only: one first-version seed. Job file: `tpu/job_kd.sh` as of this commit.

## Question 11. Is our dense recipe close to the best available at 14M, and does Muon beat AdamW here? (registered 2026-11-22, 03:45 UTC, before any run)

**Why.** Every mechanism claim will be judged against the dense baseline. The independent review and the literature workflow both found that it is a default recipe, not a tuned one:
- one learning rate at 14M;
- decay applied to the normalisation gains;
- a 10% floor on the learning rate;
- AdamW only.

Published results say recipe effects are larger than architecture effects at this scale. The first-version ladder has since finished. At 40M, 101M and 211M the higher rate (1e-3) beat 5e-4 at the end of training by 0.005 to 0.007, so the best rate sits at the top edge of the tested grid. At 14M, rates of 1e-3, 2e-3 and 3e-3 are within 0.002 of each other in partial curves.

**Design.** 14M model, one pass of the 800M-token corpus, second-version script, v5e chips, 4 micro-batches, seed 1000. Job `tpu/job_recipe.sh`; list and selection `tpu/recipe_select.py`.
- *Phase A* (20 runs):
  - AdamW at learning rate {1e-3, 2e-3, 4e-3} × weight decay {0.1, 0.5} × final rate floor {10%, 1%}. The default cell (1e-3, 0.1, 10%) is not rerun: it is Question 10's three-seed baseline, from the same script, chip and micro-batch setting.
  - The default cell with decay on matrices and embedding only.
  - Muon on the blocks' matrices, with AdamW on the embedding and gains (optax 0.2.9, consistent-RMS scaling 0.2, so AdamW's rate and decay carry over): rate {1e-3, 2e-3, 4e-3} × decay {0.1, 0.5} at floor 1%, plus Muon at the default cell.
  - The default cell with 2 micro-batches (micro-batch control).
- *Phase B*, chosen mechanically by `recipe_select.py` from phase A's seed-1000 results:
  - seeds 1001 and 1002 for the best AdamW cell (none if it is the default cell) and for the best Muon cell;
  - both winners at 40M, learning rate fixed at 1e-3, to test whether the rest of the recipe transfers.

**Predictions.**
- P1: the 1% floor beats 10% in at least 5 of the 6 matched AdamW pairs, by 0.003 to 0.010.
- P2: decay 0.5 against 0.1 shows no consistent sign, mean |difference| under 0.005.
- P3: the best AdamW rate is 2e-3.
- P4: the best Muon cell beats the best AdamW cell by 0.01 to 0.03 bits per byte.
- P5: decaying matrices only changes the result by under 0.003.
- P6: 2 against 4 micro-batches differ by no more than one seed standard deviation.

**Rules, fixed now.** s is the standard deviation of the three default-cell seeds.
- A single-seed cell differs from the default only if |Δ| > 4.97 s, a t-test with 2 degrees of freedom (Question 10's rule).
- A setting replaces the default for every later dense run if its three-seed mean beats the default's three-seed mean by more than 2.27 s_p (two-sample t, 4 degrees of freedom, 95%).
- Muon replaces AdamW only if its three-seed mean beats the best AdamW cell's three-seed mean by the same margin, **and** at 40M it beats AdamW with the same settings, with the same sign and at least half the 14M gain.
- Results are reported at equal tokens. Throughput per arm is reported beside them, and a gain is also stated at equal wall-clock, read from the curve, wherever an arm's step time differs by more than 5%.

**Not tested.** Other schedules (warm-up, stable, decay), batch size, longer training, other model sizes beyond the 40M transfer check.

### Question 11, 40M extension (registered 2026-11-22, 03:40 UTC, before any run)

The second TPU machine (v6e) would otherwise delete itself at about 04:15, so it runs the same grid at 40M. Job `tpu/job_recipe_m.sh`, list `recipe_select.py gridm`, 22 runs. Settings: 2 micro-batches, seed 1000, its own default cell at seeds 1000, 1001 and 1002; AdamW and Muon cells as at 14M. Its deadline moves from 07:33 to 08:30 UTC, adding at most about $13.

This grid is compared only within itself (one chip type, one micro-batch setting), with the same rules as at 14M. s is the standard deviation of its own three default-cell seeds. The phase B check of the 14M winners at 40M runs on the v5e with 4 micro-batches, and is read separately.

Added prediction:
- P7: at 40M the best rate is no higher than at 14M, for each optimiser.
- P1 to P5 are predicted to hold at 40M as stated for 14M.

## Evaluation harness for later models (2026-11-22)

Bits per byte cannot be compared with published small models, so the scale-up runs will also be scored on standard zero-shot benchmarks. Formats follow lm-evaluation-harness conventions.
- `tpu/eval_prep.py` downloads and tokenises the benchmarks once (HellaSwag, ARC-Easy, ARC-Challenge, PIQA, WinoGrande, OpenBookQA, LAMBADA; full standard sets, 30,328 examples).
- `tpu/eval_score.py` scores a checkpoint with JAX. Every context starts with the end-of-document token, as every document did in training.
- `tpu/eval_contam.py` measures verbatim overlap with the training corpus.

**Harness checks on the 14M model** (seed 1000, first-version sampler, 1,000 examples per task). [Observed]

| Condition | HellaSwag acc / acc_norm | ARC-Easy acc / acc_norm |
|---|---|---|
| Trained model | 0.317 / 0.364 | 0.402 / 0.388 |
| Untrained model, same harness (must be at chance) | 0.268 / 0.267 | 0.253 / 0.252 |
| Trained model, continuations scored without their context | 0.288 / 0.276 | 0.301 / 0.319 |

Standard error is about 0.015 at n = 1,000. The harness gives chance with an untrained model. Part of the trained score is a preference for the more fluent ending, which the no-context row measures. The remainder (+0.03 to +0.10) is use of the context. These figures are **not** comparable with published numbers until a public checkpoint has been scored through this same harness. Benchmark scores will always be reported beside both controls.

**Contamination of the 800M-token training corpus** (13-token windows of context plus correct answer; `results/lm/eval-contamination-v1.json`). [Observed]
- Examples sharing any window with the training text: HellaSwag 0.37%, ARC-Easy 0.42%, ARC-Challenge 0.43%, PIQA 0.44%, WinoGrande 0%, OpenBookQA 0%, LAMBADA 0.14%.
- A planted 20-token copy is found exactly, so the check works.
- Verbatim contamination is negligible. Paraphrased or same-source overlap (for example WikiHow pages behind HellaSwag) is not measured by this check.

## Store test with the corrected estimator (decision note, section 10)

`tpu/store_ldo.py`; raw files `results/lm/store-ldo-{S,M,L}.json`. Same store (48M tokens, 6% of the corpus), same fitting windows, same evaluation crops. [Observed: one checkpoint per size, first-version sampler]

| Model | Gain, leave-document-out fit (corrected) | Gain, leave-position-out fit (as registered) | Control that can fail (must be negative) | Model alone vs its training run |
|---|---|---|---|---|
| 14M | 0.0096 (web 0.0074, textbook 0.0087, maths 0.0164, code 0.0142) | 0.0044 | −0.0119, passed | 0.0001 |
| 40M | 0.0037 (web 0.0029, textbook 0.0030, maths 0.0067, code 0.0054) | −0.0014 | −0.0054, passed | 0.0001 |
| 101M | 0.0017 (web 0.0013, textbook 0.0012, maths 0.0031, code 0.0028) | −0.0032 | −0.0028, passed | 0.0001 |

The review was right. The registered estimator understated the store by more than half at 14M, and turned positive gains negative at 40M and 101M.

Reliability check at 101M (the store's mean probability of the true token, gate cells 1 to 8):
- leave-position-out on training text: 0.17 to 0.36;
- leave-document-out: 0.15 to 0.22;
- held-out text: 0.17 to 0.24.

Leave-position-out over-trusts the store by a factor of 1.3 to 1.6. Leave-document-out is about 10% pessimistic, so the corrected gains are slight underestimates.

**Registered rule, read on the 101M checkpoint (scope as amended before the run): 0.0017 < 0.01, so this store is closed as a complement.** The gain falls about 2.5-fold for each 2.5-fold increase in parameters (0.0096 → 0.0037 → 0.0017). The exchange rate is poor: 1.02 GB of counts buys 0.0017, while going from 101M to 211M parameters (about 0.2 GB more weights in bfloat16) buys 0.034. Whether a store over the full training corpus helps remains **open**: published corpus-scale stores do, and this store held 6% of the corpus. That test is not scheduled; its expected value is low on this trend.

## Question 12. How much does fresh data improve a fixed-size model, and what is the best model one machine-session buys? (registered 2026-11-22, 04:30 UTC, before any run)

**Why.** The first-version ladder trained every size on the same 0.8B tokens. The 211M model saw under 4 tokens per parameter and still beat 101M, so the larger models were limited by training length. Corpus v2 (10B tokens, same sources and recipe) is built. Whether more unique text beats more passes was untested; the review asked for this test.

**Design.** v6e machine, queued after the 40M grid. One run spans all four chips (`--dp 4`, verified on simulated devices to match single-chip losses to rounding). Default recipe: AdamW 1e-3, decay 0.1, floor 10%, batch 64 × 512, seed 1000. The tuned recipe is not available yet, so this measures data at a fixed recipe. Job `tpu/job_scale1.sh`.
1. Throughput: one 211M run for 300 steps on 1, 2 and 4 chips at 16 sequences per chip, and on 4 chips at batch 256. Reports scaling efficiency and memory. Engineering only.
2. Runs on corpus v2, in this order: 211M on 4.2B tokens (20 per parameter); 101M on 2.0B; 101M on 0.8B (the same budget as the first ladder).
3. Every finished model is scored on the benchmark harness, with the no-context control. Contamination of corpus v2 is measured separately (running).

The sampler draws crops uniformly with replacement, as in every earlier run, so a budget of D tokens covers about 1 − e^(−D/corpus) of the corpus. For 0.8B tokens on the 0.8B-token corpus v1 that was about 63%.

**Predictions.**
- P1: 101M at 2.0B beats 101M at 0.8B (both corpus v2) by 0.025 to 0.045 bits per byte.
- P2: 211M at 4.2B tokens reaches a mix of 0.90 or lower (first ladder at 0.8B: 0.952).
- P3: four chips give at least 80% of four times the single-chip throughput at 16 sequences per chip.
- P4: on HellaSwag and ARC-Easy, 211M-4.2B > 101M-2.0B > 101M-0.8B, each above its no-context control.
- P5: 101M at 0.8B on corpus v2 lands within ±0.01 of the first ladder's 101M (0.9857), which used the first-version sampler on corpus v1.

**Use of the result.**
- The 211M-4.2B model becomes the reference "best model so far" and a teacher candidate.
- The gain per added token at 101M and 211M sets the size and token plan for the next scale.
- None of this is a mechanism claim.

**Cost.** About 4 hours of a four-chip v6e (about $52 at $3.24 per chip-hour). Machine B's deadline moves to 12:30 UTC.

### Question 9: statements withdrawn after the independent review (2026-11-22)

The full list is in `theory/RESEARCH-DECISION-20261121.md`, section 10. In short:
- "The smallest edit is the worst" is false: gradient descent on the address alone did less damage (14.3%) than the smallest edit (35.3%).
- The 100% for same-entity rehearsal is accuracy on the rehearsed facts themselves. The corrected entity's unshown answers were never scored in any gradient arm.
- The "20+ phases later" columns mislabel 47% to 49% of the scored facts.
- The computed-edit ranges mixed arms and hid a trade-off between neighbour retention and correction durability.
- The optimiser control did not test the write that did the damage, so "the optimiser is not the cause" is **not established**.
- All findings of this question are published in substance.

Two cheap checks are queued:
- score every other answer of each corrected entity before and after the write, in every arm;
- repeat the damaging write itself, changing one thing at a time.

### Question 9, review checks (registered 2026-11-22, 05:05 UTC, before any run of `correction_entity.py`)

Same world, model, seeds and arms as `correction_interference.py`, which is unchanged. Two additions:
1. Every corrected entity's answers on all eight relations are read just before and just after its phase's write.
2. A new arm, `single_rowscale`: `single` with one change. Before each phase's training, the corrected rows' Adam state is set to that of recently trained rows: first moment 0, second moment the median of the previous phase's new entities.

**Predictions.**
- E0: `single` reproduces its recorded `collateral_now` and `corrected_now` exactly. Same code path and random draws; the added reads use no randomness.
- E1: in `single`, the unshown answers are damaged about as much as the other shown ones: kept_unshown within 0.10 of kept_shown.
- E2: in `entity_replay`, the rehearsed shown facts are kept (≥ 0.97) but the unshown answers are not fully kept (kept_unshown ≤ 0.90). Rehearsal protects what it rehearses, not the whole entity.
- E3: `constrained` keeps at least 0.97 of all seven other answers. The edit constrains every relation; only the phase's other training adds drift.
- E4: `store` (drift only) keeps at least 0.97 of all seven.
- E5: `single_rowscale` against `single`:
  - kept_all7 higher by 0.15 or more: stale optimiser state is a main cause of this write's damage;
  - within 0.05: it is not.

## Results of 22 November (all registered before their runs)

All files are under `results/kd/`, `results/recipe/` (14M on v5e; `m-v6e/` for 40M), `results/scale1/` and `results/nested/`. Every TPU run used the second-version script (uniform host sampling). [Observed]

### Question 10: distillation

The teacher, the 101M model, reproduced its own score inside the student runs exactly (0.9857). Baselines: 14M, three seeds, 1.1460, 1.1450, 1.1444, so s = 0.0008 and the noise guard is 4.97 s = 0.004.

| Arm | Mix | Gain over baseline | Wall-clock |
|---|---|---|---|
| 14M baseline (mean of 3) | 1.1451 | — | 0.42 h |
| 14M, α 0.5 | 1.1273 | **0.018** | 1.13 h |
| 14M, teacher only (α 0) | 1.1264 | 0.019 | 1.07 h |
| 40M baseline | 1.0466 | — | 1.00 h |
| 40M, α 0.5 | 1.0252 | **0.021** | 1.55 h |

**Rule: a gain between 0.01 and 0.03 is "partial".** The training signal is a real lever at equal tokens, but not a major one.
- Predictions held: 14M gain between 0.01 and 0.03; teacher-only no better than α 0.5.
- Prediction failed: the 40M gain (0.021) was larger, not smaller.
- Sampling-fix check: the 14M second-version mean is 1.1451 against the first version's 1.1441, so the skew moved the score by about 0.001.
- Caveat: distillation cost 1.5 to 2.6 times the wall-clock here, because the teacher runs at every step. At equal compute it has not been shown to beat a longer plain run.

### Question 11: recipe

At 14M, s = 0.0008 (default cell, three seeds). At 40M on v6e, s = 0.0012 (1.0464, 1.0485, 1.0486).

**Learning-rate floor.** P1: direction held, size underpredicted. A 1% floor beat 10% in **6 of 6** matched pairs at 14M, by 0.004 to 0.031 (mean 0.017), and in 6 of 6 at 40M, by 0.006 to 0.046.

**Decay 0.5.** P2 failed. It is worse than 0.1 in every cell at both sizes, by 0.04 to 0.075.

**Best rate.** P3 failed: the best rate is 4e-3, the top of the grid, for both optimisers at both sizes. **The optimum is not bracketed**; 8e-3 is untested.

**Decay mask and micro-batches.** P5 and P6 held: decaying matrices only changes the result by −0.002, and 2 against 4 micro-batches by +0.0003.

**Phase B** (chosen mechanically): best AdamW cell (4e-3, decay 0.1, floor 1%) and best Muon cell (the same settings).

| 14M, three seeds | Mix |
|---|---|
| Default AdamW (1e-3, 10% floor) | 1.1451 |
| AdamW 4e-3, 1% floor | 1.1278 (1.1267, 1.1285, 1.1281) |
| Muon 4e-3, 1% floor | **1.1192** (1.1193, 1.1183, 1.1199) |

Muon against the best AdamW: **0.0086**. The criterion was 2.27 s_p = 0.002, so it holds. P4 predicted 0.01 to 0.03, so the size is slightly under.

At 40M (v5e, 4 micro-batches, rate 1e-3, 1% floor), Muon beat AdamW by **0.0125**: same sign and more than half the 14M gain. **By the registered rules, Muon (consistent-RMS 0.2) with decay 0.1 and a 1% floor replaces the default.**

The 40M grid on v6e agrees:
- best AdamW 1.0312 (4e-3, 1%);
- best Muon **1.0203** (4e-3, 1%);
- default 1.0478 (mean of 3).

Total recipe gain at 40M: **0.0275 bits per byte**. That is about 45% of what going from 40M to 101M parameters bought on the first ladder (0.062).

Muon's step cost: 1% slower at 14M and 1.4% slower at 40M, so the equal-wall-clock gain is about the same.

### Question 12: corpus v2, four chips per run

| Run (default recipe, corpus v2) | Mix | Web | Code | Hours (4 × v6e) |
|---|---|---|---|---|
| 101M, 0.8B tokens | 0.9775 | 1.0450 | 0.7892 | 0.39 |
| 101M, 2.0B tokens | 0.9399 | 1.0118 | 0.7448 | 0.71 |
| **211M, 4.2B tokens** | **0.8763** | 0.9534 | 0.6770 | 2.59 |

- P1 held: 101M improves by 0.038 from 0.8B to 2.0B tokens.
- P2 held: 211M reaches 0.876; the first ladder had 0.952 at 0.8B tokens.
- P5 held: 101M at 0.8B on v2 is within 0.01 of v1 (0.9775 against 0.9857).
- P3 held: four chips run at 483.6k tokens per second against 4 × 132.8k, **91% efficiency**. Each chip used only 2.7 GB of its 32 GB; batch 256 was barely faster (494.8k).

**Benchmarks, full standard sets.** Our harness adds an end-of-document prefix; calibration against Pythia-160M is running.

| Task | 101M-0.8B | 101M-2.0B | 211M-4.2B | 211M, no context | Pythia-160M, 300B tokens (EleutherAI official, lm-eval) |
|---|---|---|---|---|---|
| ARC-Easy acc | 0.468 | 0.477 | **0.522** | 0.314 | 0.435 |
| ARC-Challenge acc_norm | 0.236 | 0.255 | 0.257 | 0.271 | 0.233 |
| PIQA acc | 0.603 | 0.625 | **0.648** | 0.606 | 0.627 |
| WinoGrande acc | 0.530 | 0.515 | 0.511 | 0.496 | 0.531 |
| LAMBADA acc (perplexity) | 0.237 (142) | 0.257 (104) | **0.326 (49)** | 0 | 0.328 (38) |
| HellaSwag acc_norm | 0.294 | 0.308 | 0.346 | 0.292 | — |
| OpenBookQA acc_norm | 0.276 | 0.278 | 0.302 | 0.278 | — |

P4 held for HellaSwag and ARC-Easy: the order is 211M > 101M-2.0B > 101M-0.8B, each above its no-context control. On ARC-Challenge and WinoGrande nothing is clearly above chance or above the control.

The 211M model on 4.2B tokens roughly matches Pythia-160M trained on 300B tokens: ahead on ARC-Easy and PIQA, level on LAMBADA accuracy, behind on WinoGrande and LAMBADA perplexity. That comparison holds only if the calibration run confirms our formats.

**Corpus v2 contamination** (13-token windows). Examples sharing any window: HellaSwag 2.1% (1.3% touching the answer), ARC-Easy 1.5%, ARC-Challenge 1.8%, PIQA 1.25%, WinoGrande 0, OpenBookQA 0, LAMBADA 0.2%. Low, but about five times v1's.

### Nested dense-loop protocol (`PROTOCOL-NESTED.md` and its deviation record)

Two-step chains learned (≥ 90% at any evaluation), out of 5 seeds, all arms on the same CPU machine:

| Arm | Weights reused? | Applications | Learned | First 90% (steps) |
|---|---|---|---|---|
| dense 4×1 | no | 4 | 1 | 21k |
| looped 1×4 | yes | 4 | 2 | 8k, 16k |
| 2×2 | yes | 4 | **4** | 6k to 17k |
| 4×2 | yes | 8 | **4** | 9k to 16k |
| 8×1 | no | 8 | **0** | — |
| 1×8 | yes | 8 | **4** | 7k to 14k |
| 4×1, width 40 | no | 4 | 0 | — |

Registered laptop-GPU arms: dense 1/5, looped 3/5.

**Rules.** R1 (4×2 ≥ 4 and 8×1 ≤ 1) holds in **both** evaluations. R2 holds device-consistently (2×2 4 against looped 2) but not as registered (2×2 4 against GPU looped 3). R3 and R4 fail in both. Because the two evaluations disagree, the result is **device-sensitive**, and by the deviation record's clause no rule is formally declared met. Device sensitivity is visible directly: looped seed 21 learned at 8k on CPU and 21k on GPU, and seed 23 learned only on GPU.

**What stands descriptively.**
- Arms that reuse weights learned in 14 of 20 runs; arms that do not, in 1 of 15.
- More compute without reuse (8×1) did not help, and neither did matching the looped model's size (width 40).

**Caveats.**
- All arms share one learning rate (1e-3). The closest published study reports untied models failing at 1e-3 but succeeding at 3e-4, so this may be tolerance of a single learning rate rather than an effect of reuse. The per-arm learning-rate sweep is the needed control.
- Extra loops at test time do not buy three-step chains: three-step accuracy stays 0.29 to 0.44 and often falls with more loops.

**Cost of the cycle** (list prices; billing not yet read): machine A, v5e-4, 00:57 to 08:53, about $38; machine B, v6e-4, 01:25 to 11:15, about $127. **Total about $165.** Both machines deleted themselves on the normal path, and Cloud Scheduler later received 404.

### Question 9, review checks: results

`correction_entity.py`, seeds 8001 to 8003. Files: `results/correction_interference/entity-*.json`. [Observed]

| Arm | Other shown facts kept | Unshown (rule-implied) answers kept | All seven kept | Correction right after | Correction 20+ phases later |
|---|---|---|---|---|---|
| single (as recorded) | 0.658 | 0.426 | 0.492 | 1.000 | 0.662 |
| single, stale optimiser state rescaled | 0.768 | 0.523 | 0.593 | 1.000 | 0.645 |
| entity_replay | 0.987 | **0.682** | **0.768** | 1.000 | 0.623 |
| random_replay | 0.670 | 0.448 | 0.511 | 1.000 | 0.670 |
| frozen_address | 0.909 | 0.834 | 0.855 | 1.000 | 0.121 |
| constrained edit | 0.983 | 0.966 | **0.971** | 0.998 | 0.215 |
| store (model untouched) | 0.999 | 0.976 | 0.982 | (store holds it) | (store holds it) |

**Verdicts.**
- E0 held: `single` reproduced its recorded numbers exactly.
- E1 **failed**: the unshown answers are damaged more than the shown ones (0.43 against 0.66 kept).
- E2 held: same-entity rehearsal keeps what it rehearses (0.99) but only 68% of the entity's unshown answers. **The earlier "100%" for rehearsal was a property of the rehearsed facts, not of the entity.**
- E3 held narrowly: the constrained edit keeps 97.1% of all seven, the only write into the model that protects the whole entity.
- E4 held: the store keeps 98.2%.
- E5 in between: rescaling the stale optimiser state raises whole-entity retention by 0.10. Stale state contributes but is not established as the main cause.

Corrected summary: protecting the whole entity at write time needs constraints on all of its answers. The model's own current answers supply them (computed edit, 97%); rehearsing the facts one happens to have stored does not (77%). Durability is a separate problem that no in-model write here solves (at most 67% of corrections survive). This summary is still published in substance.

### Harness calibration against a public model (2026-11-22)

`tpu/eval_score_hf.py` scored Pythia-160M through our prompt texts, with lm-evaluation-harness encoding and no prefix. Compared with EleutherAI's official zero-shot file for the same checkpoint (`results/lm/pythia-160m-step143000-zero-shot.json`):

| Task | Ours | Official | Difference in standard errors |
|---|---|---|---|
| ARC-Easy acc / norm | 0.436 / 0.401 | 0.435 / 0.397 | 0.1 / 0.4 |
| ARC-Challenge acc / norm | 0.195 / 0.235 | 0.188 / 0.233 | 0.6 / 0.1 |
| PIQA acc / norm | 0.623 / 0.621 | 0.627 / 0.616 | 0.4 / 0.4 |
| WinoGrande acc | 0.513 | 0.531 | 1.3 |
| LAMBADA acc (perplexity) | 0.354 (37.3) | 0.328 (38.1) | **4.0** (perplexity agrees) |

**Our formats reproduce the official harness on ARC, PIQA and WinoGrande.** LAMBADA accuracy reads about 0.025 high, probably a text-normalisation difference in the target word; LAMBADA accuracy is therefore not compared with published numbers, but perplexity is.

With that, Question 12's comparison stands for ARC, PIQA and WinoGrande. One known difference remains: our own models get an end-of-document prefix and Pythia does not.

The 211M model trained on 4.2B tokens of corpus v2:
- is ahead of Pythia-160M (300B tokens) on ARC-Easy (0.522 against 0.435) and PIQA (0.648 against 0.627);
- is level on ARC-Challenge;
- is behind on WinoGrande (0.511 against 0.531) and on LAMBADA perplexity (49 against 38).

## Question 13. The next scale: a 386M dense model on corpus v2 with the adopted recipe (registered 2026-11-22, 15:10 UTC, before any run; owner approved, about $100)

**Why.** Question 11 adopted Muon (consistent-RMS 0.2), decay 0.1 and a 1% floor. Question 12 showed that fresh tokens still pay at 101M and 211M. This run tests whether the gains hold together at the largest size the credit buys, and gives the best model so far.

**Design.** v5e-4 machine (the v5e gave about 1.5 times more training per dollar than the v6e at 101M), one run across four chips, batch 64 × 512, seed 1000, corpus v2. Job `tpu/job_scale2.sh`; controller `tpu/vm_run.sh`.
1. **Rate check at 101M**, 0.8B tokens, Muon, 1% floor: rates 2e-3, 4e-3 and 8e-3. The largest-model rate is chosen mechanically: the best of the three, capped at 4e-3, because no rate above 4e-3 has been tested at any size beyond 101M.
2. **386M model** (`XXL`: width 1280, 20 layers, 386,062,080 parameters) on 7.7B tokens (20 per parameter). If it diverges, it is rerun once at half the rate.
3. Benchmarks on the finished model, with the no-context control. Bits per byte on the same held-out sets as every run.

**Predictions.**
- Q1: the best 101M rate is 4e-3. The 101M Muon run at 0.8B tokens beats Question 12's default-recipe 101M at 0.8B (0.9775) by at least 0.02.
- Q2: the 386M model reaches a held-out mix between 0.79 and 0.83 (211M at 4.2B tokens, old recipe: 0.876).
- Q3: on our calibrated harness, it beats our 211M model on ARC-Easy and PIQA. Against Pythia-410M (300B tokens; official: ARC-Easy 0.521, PIQA 0.668, WinoGrande 0.537, LAMBADA perplexity 10.8), it is within 0.03 on ARC-Easy and PIQA and worse on LAMBADA perplexity.

**Cost bound.** About 19 to 20 hours of a v5e-4 at $4.80 per hour, about $90 to $95. Deadline 2026-11-23 12:00 UTC (on-machine timer; Cloud Scheduler at 12:02; laptop watcher at 12:10). Worst case about $101.

**Question 13, operational note (15:20 UTC).** Measured: four v5e chips give 439k tokens per second at 101M, so each rate check takes about 30 minutes. Scaled to 386M, the main run takes about 18.6 hours and would finish about 20 minutes before the 12:00 deadline. All three deadlines were therefore moved two hours later: on-machine 14:00, Cloud Scheduler 14:02, laptop watcher 14:10 on 23 November. The bound rises to about $111, with about $95 expected. Nothing in the design changed.

**Question 9 summary, amended (2026-11-22).** "Durability is a separate problem that no in-model write here solves" is **superseded**, by G06 (session d5, draft PR #202, `NULLSPACE-DURABILITY-20261122.md`; predictions committed before its runs).
- Its unprotected arm reproduces this branch's `constrained` arm exactly, seed by seed.
- Projecting later decoder updates away from old entities' address directions (OWM, published 2019) keeps computed-edit corrections at 98.5% after 20+ phases, against 21.5% without.
- The price is the rule: new entities' unshown relations are inferred at 56–66% instead of 99.7%.
- Rehearsing the corrections alone keeps them (97.4%) but drops the corrected entities' other old facts to 40%.

Corrected summary: in this model, durable in-model corrections and rule inference for new entities trade off through the shared decoder. Each method found so far buys one at the cost of the other; only the external store has both. [Reported by PR #202 and checked against its committed files; not re-run here.]
