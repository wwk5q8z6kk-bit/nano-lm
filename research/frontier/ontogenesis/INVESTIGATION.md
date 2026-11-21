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
