# Independent research agenda: what makes a small system capable

Date: 2026-11-21. Author: the research agent, on its own judgement. Separate from, and not altering, `PROTOCOL-LOOP.md`, `PROTOCOL-LOOP2.md`, `PROTOCOL-NESTED.md`, the acquisition benchmark, or their results. A prior-art critic reviews this note; its corrections will be appended as the last section and override the text above.

Labels: **[Established]**, **[Published]** (from memory unless marked checked), **[Observed]** (ours), **[Derived]** (here), **[Hypothesis]**, **[Speculation]**.

---

## 1. Is the premise right?

"Can a small system match a large one through better organisation?" has a part that is impossible, a part that is already true, and a part that is open. They must be separated before anything is designed.

**Impossible.** [Established, counting] A system that can answer N independent facts of b bits each must hold at least N·b bits somewhere. No organisation removes this. [Published] Measured capacity of language models is of the order of 2–4 bits per parameter. So a system that *knows* what a large model knows cannot be small in total size. "Small" can only mean a small **processor**; the knowledge has to live somewhere, and must be counted.

**Already true.** [Published] Moving knowledge out of weights into a store that is searched (retrieval, nearest-neighbour language models, memory layers, sparse experts) works and is standard. Very small recursive models solve some hard puzzle tasks. Neither is ours to discover.

**Open.** Three things, in my judgement:
1. Whether gradient descent can **reliably learn procedures** in a small reused core. Our own runs show it is a seed-dependent jump even for two steps.
2. Whether the **division** between stored knowledge and computed procedure can be learned rather than hand-built. [Observed] Our learned-address store stayed at guessing while the token-address store reached 100%.
3. How far actual trained networks are from the **smallest system** that does the same job, and why.

**Parameter count is the wrong measure.** [Derived] What a deployed system pays for is information stored, bytes moved per step, and operations per step. Parameters conflate all three, because in a dense network every parameter is stored, moved and used on every token. That conflation is a property of dense networks, not of computation.

## 2. A resource model with content

The owner's A = F(P, M, C, B, E, T, I, O) is a list until it yields inequalities. Three hold for any system. [Derived from Established results]

- **Information.** bits(P) + bits(M) ≥ K, where K is the description length of what the task requires beyond the input. Organisation cannot lower K.
- **Depth.** Sequential steps ≥ the task's depth requirement. [Published] One Transformer layer cannot compose two lookups in general (communication-complexity bound); k-step chains need depth of order log k with untied layers.
- **Traffic.** Time per step ≥ (bytes read from outside the cache) / B, and energy per step ≥ arithmetic energy + movement energy. [Published, hardware measurements] Reading a word from main memory costs on the order of a hundred to a thousand times the energy of one arithmetic operation.

**Organisation, O, made measurable.** [Derived; definitions proposed here] Two numbers per system and task:

- **Locality** ℓ = (bytes touched per step) / (bytes stored). Dense network: ℓ = 1. Sparse experts: 1/k. A store read by key: tiny.
- **Reuse** ρ = (operations per step) / (distinct parameters touched per step). Untied stack: ρ ≈ 1. Loop of T passes: ρ ≈ T.

| System | ℓ | ρ |
|---|---|---|
| Dense stack | 1 | 1 |
| Tied loop | 1 | T |
| Dense stack + retrieval | below 1 for the store, 1 for the model | 1 |
| Small looped core over a keyed store | far below 1 | T |

"Improve O holding the rest fixed" then means: lower ℓ and raise ρ at fixed K and fixed accuracy. The information bound says what cannot be gained (K). The traffic bound says what can: cost per step can fall from proportional to stored bits to proportional to touched bits. Whether accuracy survives is empirical.

**What this model does not give.** It bounds cost, not learnability. Every bound above is met by systems nobody knows how to train. The scarce thing is not a better diagram; it is a learning procedure that finds low-ℓ, high-ρ solutions reliably.

## 3. Five directions

### Direction 1. Why composition is learned in a jump, and what controls it

- **Problem.** A reused core is only useful if gradient descent learns procedures reliably. [Observed] Two-step chains: flat at guessing, then a jump; dense 0 of 7 runs so far, looped 5 of 7, wide looped 0 of 3.
- **Why current approaches may be inadequate.** Scale hides this: large models see enough data that every layer learns every needed step. [Published] Work on implicit multi-hop reasoning reports delayed, abrupt acquisition and a "bridge entity", and at least one paper argues that sharing across layers should help. **So my bridge hypothesis is probably largely published.** The referee is checking. Its value to us is as an instrument, not a discovery.
- **Competing explanations and what separates them.**

| Explanation | Distinct prediction |
|---|---|
| X1 Multiplicative saddle between two circuits (bridge) | bridge entity becomes decodable just before the jump; jump times heavy-tailed; copying the lookup circuit into later layers *after* one-step mastery triggers the jump in dense |
| X2 Smaller hypothesis class | advantage vanishes on chains with unlike steps; unaffected by copying circuits |
| X3 Learning-rate mismatch per arm | each arm has a learning rate at which it jumps; wide looped recovers at lower rate; ranking changes under per-arm tuning |
| X4 Signal dilution | two-step queries are half the loss; raising their share speeds dense as much as tying does |
| X5 Implicit curriculum | removing one-step queries prevents the jump in every arm |
| X6 Symmetry at initialisation | untied layers initialised as copies of each other (random, before learning) behave like tied early on |
| X7 Discrete algorithm switch | before the jump the model uses a different strategy with measurably different attention patterns, not a weaker version of the final one |
| X8 Chance | tied and untied jump-time distributions coincide given enough seeds |

- **On the copied-initialisation control: my earlier proposal was under-specified.** [Derived] Copying weights at step 0 copies random weights. It makes the untied model compute the same function as the tied one at step 0 only; the gradients then differ (the tied update is the sum over passes, each untied layer gets only its own term), and with Adam the difference is not a rescaling. So copy-at-initialisation tests X6, not X1. To test X1 the copy must happen **after** one-step lookup is learned, and from the layer that actually implements the lookup, which must first be located by probing. Valid only if: same optimiser state handling is declared (reset or copied), the copied layer's input format matches what later layers receive, and a sham control (copying a non-lookup layer) is run. A cleaner family is **soft tying**: untied layers with a penalty λ·Σ‖θ_i − θ̄‖², swept from 0 to large, plus "tie then untie at step s" and its reverse. That measures *when* sharing matters.
- **Smallest experiment.** Probing and attention analysis on checkpoints through training for dense and looped; soft-tying sweep; query-mix sweep. Needs a reduced task where a run takes about two minutes, fixed in advance.
- **Resources.** Laptop, about a day.
- **Reasons not to pursue.** Likely to reproduce published findings. A toy with 512 symbols may not transfer to language.

### Direction 2 (departure). Compile first: find the smallest system that has the capability, then ask why learning misses it

- **Problem.** The owner's central question is "what is the smallest system that produces a specified capability". Training cannot answer it: a trained network gives an upper bound loosened by whatever the optimiser needed.
- **Why current approaches may be inadequate.** Machine learning almost never measures the gap between the size it needed and the size the task needs.
- **Hypothesis.** [Hypothesis] For algorithmic capabilities, the minimal network is orders of magnitude smaller than the smallest one that gradient descent trains reliably, and the excess is the price of search, not of representation. If so, the lever for small capable systems is the search procedure, not the architecture.
- **Mathematics.** Description length: K(task) ≤ bits of the smallest correct network. Define **search overhead** Ω = (parameters of smallest reliably trained network) / (parameters of smallest hand-constructed network). Define the **basin** of a constructed solution: the largest weight-noise radius from which training returns to it.
- **Existing research.** [Published] Hand-coded and compiled Transformers (RASP, Tracr); lottery-ticket results showing small trainable subnetworks exist inside large ones; overparameterisation theory.
- **What may be new.** Using Ω and basin size as the dependent variables of an architecture comparison: does tying, nesting or a store change Ω? I have not seen this done; unverified.
- **Predictions that could fail.** (a) A looped solution to k-step BINDINGS exists with far fewer than 129k parameters; if the minimum is near 129k the hypothesis is wrong for this task. (b) Started from the constructed solution plus noise, training holds it; started far away at the same size, it fails. (c) Ω is smaller for tied than untied.
- **Smallest experiment.** Construct by hand the minimal looped Transformer for BINDINGS; verify it scores 100%; shrink until it breaks; then train from perturbed copies. Almost no compute.
- **Reasons not to pursue.** Hand construction does not scale beyond algorithmic tasks. Ω on a toy may say little about language.

### Direction 3. A small learned program over a store that is written, not trained

- **Problem.** [Observed] Gradient updates overwrote earlier knowledge in every learner we tried; frozen addresses retained 38–39%; learned addresses retained 3%.
- **Hypothesis.** [Hypothesis] Knowledge should enter by a write (one step, local, no interference), and gradients should train only the procedure that reads, writes and composes. Then acquisition cost is independent of what is already known, and ℓ is tiny.
- **Mathematics.** Exact Bayesian updating never forgets; a keyed write is its simplest case. Interference under gradient descent is first-order the inner product of task gradients; a write to a fresh key has inner product zero by construction.
- **Existing research.** [Published] Complementary learning systems, memory networks, differentiable computers, retrieval augmentation, model editing.
- **What may be new.** Possibly nothing in the mechanism. The open part is whether the **address function can be learned** so the store generalises (related keys share structure) without re-introducing interference. That is where our learned-address arm failed.
- **Predictions that could fail.** On the acquisition benchmark the written store should hold retention and composition by iteration together; it should lose on corrections unless writes overwrite by key; a learned address should collapse without a constraint that keeps old addresses fixed.
- **Smallest experiment.** Already built (`cab.py`); first full result pending. Next arm: write-by-rule store with overwrite.
- **Reasons not to pursue.** Risk of rebuilding retrieval augmentation under another name. The owner's earlier warning, "don't build a database with a language model attached", applies.

### Direction 4 (departure). Traffic, not arithmetic: measure what organisation costs on real hardware

- **Problem.** Every efficiency claim so far counts parameters or layer applications. Neither is what the machine pays.
- **Hypothesis.** [Derived, then Hypothesis] At batch size one a dense model must read all its weights per token, so speed is bounded by memory bandwidth divided by model size. A tied block small enough to stay in cache pays that once, not T times. So reuse has a hardware benefit that is independent of learning: at equal layer applications, a tied loop that fits in cache should run faster and cooler than an untied stack that does not.
- **Mathematics.** time per token ≥ max(ops / throughput, bytes outside cache / B). Crossover when untied weights exceed last-level cache while the tied block does not.
- **Existing research.** [Published] Bandwidth-bound inference is well known; weight-shared language models reported no speed gain, but on batched accelerators where arithmetic dominates.
- **What may be new.** Probably little. It is a measurement we need, not a discovery: it decides whether "reuse" is worth anything in deployment even if it buys nothing in learning.
- **Predictions that could fail.** No speed difference between tied and untied at any size on the laptop CPU at batch one would refute it for this hardware.
- **Smallest experiment.** Time forward passes of tied and untied stacks of equal depth, widths chosen to straddle cache sizes, batch one, idle machine. Minutes. Must wait until the current runs finish, or the timings are contaminated.
- **Reasons not to pursue.** Accelerators with batching erase the effect; relevant mainly to on-device use.

### Direction 5. Automated discovery: not yet, and not by reinforcement learning

- **Question asked.** Can reinforcement learning, evolution or automated experiment selection find organisations we would not design?
- **Assessment.** [Derived] A search method needs many evaluations. Our evaluation is a 30-minute run with a two-valued, seed-dependent outcome: roughly one bit per half hour per run. Reinforcement learning over architectures needs thousands to millions of evaluations. [Published] Its record in architecture search is poor per unit of compute; evolutionary search guided by a language model has found real algorithms, but only where a candidate can be scored in seconds by an exact checker.
- **What does help now.** Sequential experimental design. With two-valued outcomes, a pre-declared stopping rule on a Beta-Binomial posterior reaches the same decision with fewer runs than fixed-seed designs.
- **Precondition for search.** A task scored exactly in seconds, and a search space of short programs over a small set of primitives with a description-length penalty. Direction 2 produces both as by-products.
- **On objectives.** A search agent cannot discover that its objective is inadequate from inside the objective. That needs held-out capabilities it was not optimised on; those must exist first.
- **Reasons not to pursue yet.** No fast evaluator; high risk of producing a search system instead of a finding.

## 4. Ranking

Scored on importance, impact, feasibility, and information per unit of compute.

| Rank | Direction | Why here |
|---|---|---|
| 1 | Direction 2, compile first | Directly answers the owner's central question for one capability; almost free; separates "cannot be represented small" from "cannot be found small", which decides whether architecture or search is the lever |
| 2 | Direction 1, the jump | Every small-core idea depends on it; evidence in hand; cheap. Ranked below 2 because much of it is probably published |
| 3 | Direction 3, written store | Highest impact if the address problem yields; that problem has beaten us once |
| 4 | Direction 4, traffic | Minutes to run and it corrects our accounting; modest scientific weight |
| 5 | Direction 5, automated discovery | Preconditions absent |

**Nested adaptive computation is not in the list as an architecture.** It adds no function class at equal compute (theory note, D1) and every member is published. Its arms remain useful as conditions inside Direction 1, and the running nested test will be reported as registered.

## 5. Corrections to my own earlier methodology

- **Twenty seeds per cell is not enough for a factorial on jump time.** [Established, Schoenfeld's formula] Detecting a hazard ratio of 2 at 80% power needs about 66 observed jumps across the two groups; twenty per cell detects only ratios near 3 or more. Either the task gets cheap enough for 40 or more runs per cell, or the factorial is cut to the few cells that separate explanations.
- **Jump times are interval-censored**, known only to the nearest 1,000 steps, and right-censored at the budget. Fixed-budget censoring does not bias survival estimates; coarse intervals blur them. Evaluate more often.
- **One learning rate for all arms biases the comparison** toward whichever arm it suits. Tune per arm on one set of seeds, confirm on fresh ones.
- **The unit is the run.** Runs share an evaluation set but not data order or initialisation; treat them as independent and say so.
- **A jump needs a definition fixed in advance**: first evaluation at or above 90% that is not followed by a fall below 50%.

## 6. Disagreements with the directive

- "Do not assume neural networks": agreed, and Direction 2 starts from programs. But learning from data is the only route we have to capabilities nobody can write down, so a non-learned system is a yardstick, not a destination.
- Seven parallel theory agents by discipline: not done. In this project five designs produced on paper were reviewed and none survived; the yield of more paper theories is low. One critic with a literature brief is used instead. If the owner wants the full fan-out it can be run, at a token cost.
- "Self-modifying structures", "morphology": no mathematical content identified yet. Not scheduled.

## 7. What starts without further approval

Theory and laptop-only work, within existing authorisation: the hand construction for Direction 2; the reduced fast task; the traffic measurement once the machine is idle. No paid compute.
