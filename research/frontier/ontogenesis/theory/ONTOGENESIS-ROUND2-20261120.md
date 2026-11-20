# ONTOGENESIS round 2: the missing principle, 2026-11-20

Status: theory, plus three numerical checks of mathematical identities (`ontogenesis-checks/prefix_structure_checks.py`, all pass). No architecture is implemented or trained. Labels: [T] theorem, [V] verified numerically here, [E] published empirical result, [S] speculation, [U] novelty unverified. A referee pass on this document is pending.

## 0. Corrected core of round 1

Round 1 (`ONTOGENESIS-THEORY-20261120.md`, section 9) stands corrected as follows.

- A local learned PDE, discretised, is a tied-weight residual convolutional network when the update is a pointwise network on the stencil. [T]
- Exact storage of N tokens needs N·log₂V bits; a fixed-size state cannot be exact beyond its capacity. [T]
- Phase-space volume preservation (not conservation of a quantity) excludes asymptotically stable sets in finite dimension. [T]
- Contraction implies exponential forgetting (contraction theory). [T]
- Sequential training of nonlinear recurrences is a cost, not an impossibility: Newton-type parallel solvers exist. Exact one-pass parallel training needs transition maps that are closed under composition with a compact representation. [T/E]
- Softmax attention is not a Wasserstein gradient flow because of row normalisation; convergence proofs do not require gradient structure. [T]
- DEMOS: with a fixed price κ the Lyapunov function is E − κ·Σm_i, between discrete events and with no conservative term. [T]

## 1. Which conventions are necessities

| Convention | Required by mathematics? | Actual constraint |
|---|---|---|
| Sequential layers | No. A layer stack is one fixed-depth circuit. | Some problems are believed inherently sequential (P-complete, if P ≠ NC); iteration to a fixed point is the alternative to fixed depth. |
| Fixed-dimension vectors | No. State can be a set, a measure, a graph, a table. | Accelerators want static shapes. This is engineering. |
| Memory in slots or caches | No form is required. | Memory size must be at least the information retained (bit count). For prediction, the floor is the past–future mutual information (excess entropy) E(N), not N. Measurements on text suggest E(N) grows sublinearly (Hilberg's conjecture, roughly N^0.5). [E, conjecture] So a linear-size cache is not a necessity. |
| Same compute per token | No. Surprisal varies per token. | Batched hardware charges for the slowest element; adaptive compute mostly saves inference. |
| Fixed structure after training | No. | Discrete structure changes have no gradient. |
| Similarity lookup | No. | Exact maximum-inner-product search in high dimension has no known truly sublinear algorithm (conditional hardness under SETH). Exact-match lookup on discrete keys is O(1) expected with memory proportional to the number of keys. [T] |

The last row is the opening. Attention retrieves by similarity, which is inherently a scan. Retrieval by discrete key is not.

## 2. DEMOS against the neural field, and against conventional networks

**Amari field.** τ·u_t = −u + w∗f(u) + I on a domain of dimension D.

**When the particle description is equivalent.** Projecting the field dynamics onto the translation mode of each bump gives ż_i = a·(input overlap) + Σ_j g(z_i − z_j). This holds when: (i) bumps are separated by much more than the range of w; (ii) inputs are weak and slow; (iii) the kernel is such that each bump is linearly stable. [T, standard perturbation theory]

**Where round 1 overstated.** I wrote that the two give "the same equations". They do not:
1. A stable Amari bump has one soft mode, its position. Its width and amplitude are fixed by the kernel and threshold. DEMOS's mass m_i has no counterpart; the mass equation comes from a different source (replicator / Fisher–Rao dynamics).
2. The particle coordinate z lives in the field's own domain. DEMOS uses z ∈ R^d with d of order 64 or more. A field on a 64-dimensional grid cannot be simulated. So the particle system is not an economical form of that field; in high dimension it is the only form.
3. A scalar bump carries about log₂(M/width) bits (its position). Content needs a multi-component field.

**What particles lose.** Sub-threshold accumulation of weak evidence before a bump forms; nucleation and splitting produced by the dynamics themselves; radiation between bumps; strong interactions that deform bumps.

**Higher-order interactions, composition, hierarchy.** A symmetric three-body term keeps the Lyapunov structure at cost O(n·k²). Short-range attraction with a repulsive core gives bound pairs and clusters of clusters (established physics). Direction (a→b versus b→a) needs role coordinates, because a symmetric potential cannot encode it.

**The decisive question: is there an efficiency separation?** No. One DEMOS transport step is
  ż_i = −∇V(z_i) − Σ_{j∈N(i)} m_j·∇₁W(z_i, z_j)
which is a sum over neighbours of a function of (z_i, z_j), followed by a pointwise map. That is one sparse message-passing layer. The mass update is a gate. A weight-tied sparse message-passing network with a learned eviction gate reproduces every DEMOS step at the same asymptotic cost, and DEMOS adds constraints (symmetric W, gradient form). Constraints cannot increase what is computable per unit cost. **DEMOS is a restricted looped sparse Transformer with token eviction.** Its possible benefits are guarantees and inductive bias, not capability. Both ingredients (looping, learned eviction) are published.

Sub-quadratic particle interaction has the same obstacle as attention: finding neighbours in high dimension is the similarity-search problem. Tree codes work only in low dimension.

**Verdict.** DEMOS is demoted. It does not answer the guiding question.

## 3. The principle

**Prefix-monoid principle.** A memory mechanism trains in parallel across sequence positions, in one pass and O(log N) depth, when its state after t tokens is a prefix product s_t = x_1 ⊕ x_2 ⊕ … ⊕ x_t in a monoid whose elements have a compact representation. [T, parallel prefix; sufficient, not necessary]

Known instances:
- Attention: the monoid is set union. Elements are not compact (size t), hence cost N per token.
- Linear recurrent and state-space models: the monoid is composition of affine maps. Compact, parallel, but lossy by superposition (capacity d).

The space between these two is where current architectures do not look: **monoids that are compact and exact for the queries that matter.** Data-structure theory already classifies them as mergeable summaries [T]: structures with a merge operation such that summary(A ∪ B) = merge(summary(A), summary(B)). Three have exact, useful queries:

| Monoid | Element | Exact query | Size |
|---|---|---|---|
| Right-biased map union | dictionary key→value | latest value bound to a key | number of live keys |
| Semiring matrix closure over a time filtration | relation matrix of arrival times | is b related to a at time t, through any chain | entities² |
| Merge forest (union-find over a filtration) | partition history | are a and b the same thing at time t | entities |

Each gives one proposal below. Approximate members of the same family (count sketches, Bloom filters, min-hash) give fixed-size "gist" channels with proven error bounds.

**Why this could matter for language.** An induction head, the best-understood in-context mechanism in Transformers, computes "find the previous occurrence of the current token and return what followed" [E]. That is a dictionary lookup keyed by the token. A Transformer implements it by scanning N positions. The archive agrees: M21's whole gain came from exact copying (1.79 against 2.15 bits per token on copyable targets).

**Interpretations that are precise.**
- Conservative memory: between writes to a key, the stored value is exactly constant. There is no crosstalk between keys.
- Dissipation: information is destroyed only by an overwrite or delete. Each such event erases a countable number of bits; the count can be priced.
- Topological memory: the merge forest is the 0-dimensional persistent homology of the time filtration. [T]
- Gauge structure: rotary position encoding is a U(1) connection (relative phase is parallel transport), and the archive's rotation registers are a holonomy. These supply invariances. They were measured in the archive (about 8× slower, under 1% gain) and are not a new source of compute.

## 4. Three proposals

All three assume tokens are mapped to discrete symbols. That is a shared risk, stated once: learning discrete keys from a language-modelling loss (straight-through estimation, codebook collapse) is the hardest unsolved part.

### K. KEYSTORE — parallel conservative memory (functions)

**Principle.** Exact keyed storage with last-write-wins; the right-biased union of maps is associative.

**State.** G tables T^(g): code → (value ∈ R^d, position).

**Equations.** For token t with hidden state h_t:
  write key c_t^(g) = VQ_g(K_g·h_t), value v_t = V·h_t, gate w_t ∈ {0,1}
  T_t^(g)[c] = v_j,  j = max{ j ≤ t : c_j^(g) = c, w_j = 1 }
  read q_t^(g) = VQ_g(Q_g·h_t);  r_t = Σ_g T_{t−1}^(g)[q_t^(g)]  (a learned "absent" vector if unbound)
  delete = write of a null value.

**Parallel form.** "Most recent earlier position with the same code" for all positions is one stable sort by (code, position) and a shift. [V] Arbitrary (code, time) queries add a binary search. Total O(G·N·log N); then one gather.

**Learning.** Values receive exact gradients through the gather. Codes use vector quantisation with a straight-through estimator. A first version needs no learned codes: keys are hashes of the last 1–3 token ids (this is a cache or n-gram memory, which is published).

**Retrieval.** O(G) per token at inference by hash lookup. Chains: feed a retrieved value back as a query; h hops are h gathers.

**Complexity.** Training O(G·N·log N + N·W·d) with a local window W. Inference per token O(W·d + G). Memory O(live keys · d), not O(N·d·L).

**Stability.** No dynamics, so nothing to diverge. Stored values are bit-exact until overwritten.

**Limits.** Exact match only: a near-miss key retrieves nothing. G independent groups give multi-probe behaviour with collision probabilities as in locality-sensitive hashing. No reasoning by itself.

**Hardware.** Sort and gather are native on accelerators with static shapes.

**Relatives.** Neural Episodic Control's differentiable dictionary; discrete key-value bottleneck; product-key memory and its fast-weight form; Reformer (sort-based bucketing); cache and infinite-n-gram language models; M21's ConceptSlot. Learned in-context discrete keys with last-write-wins and parallel-prefix training: [U].

**Experiment most likely to disprove it.** Bind values to keys that later appear in a different surface form. If learned codes for the two forms agree less than 90% of the time, KEYSTORE collapses to a token-keyed cache, which already exists.

### R. RELATOR — computation in relationships (relations)

**Principle.** Relations are the state; composing relations is the computation. Reach doubles with each squaring.

**State.** For n entities and ρ relation types: arrival times τ ∈ (N ∪ {∞})^{ρ×n×n} written by tokens, and a learned composition table Θ ∈ {0,1}^{ρ×ρ×ρ} (Θ_abc = 1 means type b followed by type c yields type a).

**Equations.** In the (min, max) semiring:
  D ← D ⊕ ⊕_{b,c: Θ_abc=1} (D_b ⊗ D_c),  (X ⊗ Y)_ij = min_k max(X_ik, Y_kj),  ⊕ = min
iterated from D = τ to the least fixed point.

**Filtration theorem.** For one relation type, b is reachable from a using only facts stated by time t if and only if D_ab ≤ t. So a single closure answers every prefix of the sequence. [T, minimax paths; V] This removes the per-position recomputation that would make relational state sequential.

**Convergence.** The operator is monotone on a finite lattice (entries take values in the finite set of arrival times), so iteration reaches the least fixed point; ⌈log₂ n⌉ squarings suffice for one type. [T]

**Learning.** Θ and the write decisions are discrete; training uses a smooth semiring with temperature annealing. The archive's Gödel model used the same max–min algebra on synthetic graphs.

**Complexity.** Dense: O(ρ²·n³·log n) per sequence; state O(ρ·n²). Lower bound: transitive closure is as hard as Boolean matrix multiplication, so n³ (or n^ω) cannot be avoided for dense relations. [T] n is the number of entities, not tokens.

**Expressivity.** Pairwise state with composition is strictly stronger than node message passing (it distinguishes graphs that the 1-WL test cannot). [T] A node state of d numbers cannot hold a reachability set over n > d·b nodes; the relation matrix can.

**Limits.** Monotone: no negation, no counting. Deletion breaks the filtration (a fact with a lifetime is not one number). Soft strengths and times together do not form a single semiring.

**Relatives.** Edge Transformer and triangular attention; provably powerful graph networks; Neural Bellman–Ford; differentiable Datalog; Tropical Attention. The filtration used to train causal relational memory in parallel, and a learned composition table inside a language model: [U].

**Experiment most likely to disprove it.** Branching reachability with chains 8× longer than in training, against a looped Transformer given the same operation count. If the looped model is within 5 points, RELATOR has no advantage.

### F. FILTRA — a different substrate (equivalences)

**Principle.** The primitive operation is unification: declaring two things the same and merging everything known about them. It is none of attention, recurrence, message passing, a neural field, or a mixture of experts. It is a union-find structure over learned symbols.

**State.** A merge forest over n symbols with merge times.

**Operations.** unify(a, b) at time t; find(a, t) = representative of a's class at time t; class values stored in a KEYSTORE keyed by representative.

**All prefixes at once.** The symmetric (min, max) closure of unify times gives class membership at every time; it equals the union-find result at each prefix. [T; V] Equivalently: sort unify events and build the merge tree once; find(a, t) is an ancestor query in O(log n).

**Why it differs in cost.** An alias chain of any length is one find. Cost is independent of chain length h, where a looped model needs about log h steps and a fixed-depth model needs depth. With congruence closure (if a ≡ b then f(a) ≡ f(b)), it performs equational reasoning in O(n log n). [T]

**Limits.** Only symmetric, transitive structure. No splitting without versioning. Unify decisions are discrete and must be learned. Union-find is pointer-chasing; on accelerators it becomes label propagation in O(log n) rounds.

**Relatives.** Entity-centric language models and recurrent entity networks; coreference-aware models. A differentiable union-find trained inside a language model: [U].

**Experiment most likely to disprove it.** Alias chains in code-like text with no supervision of the unify events. If accuracy is not flat in chain length, or the events cannot be learned from the prediction loss alone, FILTRA fails.

## 5. What the limits allow

| Question | Lower bound (theory) | Engineering limit |
|---|---|---|
| Exact retrieval sublinear in N? | Yes for exact-match keys: O(1) expected, memory ∝ keys. No known sublinear method for exact similarity search. | Learning the keys. |
| Particles without quadratic cost? | Same as similarity search; needs bucketing. | — |
| Dynamic topology learned efficiently? | Closure costs matrix multiplication in the number of entities. | Discrete decisions; soft-to-hard gap. |
| Adaptive computation without a sequential bottleneck? | — | Batches pay for the slowest element; saves inference, rarely training. |
| Long-range information without storing every token? | Floor is the excess entropy E(N), believed sublinear for text. | No constructive scheme reaches the floor. |
| Reproducible by a looped Transformer? | Yes in accuracy, with enough steps. | Not in cost: it scans N positions per step. The advantage, if any, is on the cost axis. |

## 6. Archive verdicts

| Idea | Verdict | Reason |
|---|---|---|
| S2 recurrent memory | Baseline only | Fixed-state bound; half speed |
| M21 copy pointer | Evidence, then replace | Proves exact recall has value; its scan is what KEYSTORE removes |
| Gödel max–min model | Keep | Its algebra is exactly what makes RELATOR parallel |
| CLOSURE | Merge into RELATOR | Same algebra; gains the filtration theorem |
| LEDGER | Discard as standalone | Largely published; soft eviction collapses on incompressible retrieval |
| DYADIC | Discard | Published; routing capacity risk |
| LEXICON | Discard for now | Published relative shows quality loss; its symbol-learning problem is shared by all three proposals |
| METRIPLEX | Discard as standalone | Its parallel form is linear attention with complex features |
| DEMOS | Demote | No efficiency separation from a looped sparse Transformer with eviction |
| ALGEBRA-EQUILIBRIUM | Discard | Cost and capacity claims inconsistent; sequential relaxation |
| Mesh, ERN, holonomy registers | Discard | Lost to dense where measured |
| QGT Companion | Not relevant | Self-published physics site; no content on computation; no peer-reviewed references |

## 7. Experimental matrix (tiny scale, no credits)

**Task suite BINDINGS.** A stream of statements among distractors: set(k, v), update(k, v′), delete(k), alias(k₁, k₂), rel(a, r, b). Queries:

| Probe | What it tests |
|---|---|
| get(k) | single-fact retrieval |
| get after update | conflicting facts: latest wins |
| get after delete | deletion |
| chain of h gets | multi-hop |
| alias chain of length h | unification |
| relational path with branching | composition |
| distance between fact and query | long-range dependency |
| distractor load N | distractor resistance and cost scaling |
| h and m beyond training (8×) | generalisation |

Fact counts must exceed the fixed-state capacity of the recurrent baselines (thousands of facts), or that axis tests nothing.

**Arms.** Dense Transformer; looped Transformer with adaptive steps at matched compute; S2-style recurrent; M21; a gated recurrent network with an external differentiable memory; DEMOS (as the restricted looped model it is); KEYSTORE with token-hash keys; KEYSTORE with learned keys; RELATOR; FILTRA.

**Rules.** Same data, parameter budget, optimizer and learning-rate grid; equal training wall time on one device; three seeds minimum; prefix-causality audit before any score is read; report accuracy, operations per query, throughput, peak memory and parameters. Protocol committed before any fit.

**The language test that can sink the whole principle.** A 10–20M model with a short local window plus prefix-monoid memories against a dense model with full context, equal training time, ordinary text and code, 8k context. If the gap exceeds 0.03 bits per token, the claim that long-range dependence in language is mostly keyed plus gist is false.

## 8. Recommendation

1. **KEYSTORE first.** It has the clearest measurable advantage (keyed recall and update in O(1) per token, memory proportional to live keys, parallel training), it is supported by the archive's one robust finding (copying), and its first version needs no learned discrete keys.
2. **RELATOR second,** only for the branching-relation probe where a node-state model has a capacity argument against it.
3. **FILTRA third,** as the high-risk idea with the sharpest cost prediction (independence from chain length).

The answer to the guiding question, stated as a claim to be tested: *latest-binding recall, relational closure and identity merging over a context can be computed with cost independent of context length and trained in parallel, where attention pays a scan and recurrent state pays in capacity.* This is a claim about cost at equal accuracy. It is not yet a claim about language-modelling quality.

Three victories are needed; the status of each:
- Mathematical: the prefix-monoid principle and the filtration results are sound and checked. Whether the framing is new is unverified.
- Engineering: sort, gather and small matrix closures are accelerator-friendly. Unmeasured.
- Intelligence: no evidence yet.

## 9. Referee corrections (2026-11-20), overriding the text above

The three identities hold (the referee re-ran the checks and added its own). The following statements were wrong or too strong.

1. **The principle misdescribes its own algorithms.** Associativity plus "compact elements" is not sufficient: materialising N prefix dictionaries of K live keys costs N·K. None of the three mechanisms is a prefix scan. KEYSTORE is an offline batched predecessor query; RELATOR and FILTRA store the arrival time as the value. Corrected statement: one-pass parallel training holds when (a) element size is independent of N, or (b) all N queries can be answered offline in near-linear work without materialising any prefix.
2. **The principle is already published in its associative form** ("Sequential-Parallel Duality in Prefix-Scannable Models", ICLR 2026; affine aggregators only). What may be unpublished is the use of exact data structures (dictionaries, time-valued closures, merge forests) in that role. Confidence low to medium.
3. **RELATOR with several relation types.** The log₂ n bound holds for one type only. With a composition table the same iteration needed n−2 rounds for a simple rule and can need many more; the O(ρ²n³ log n) cost is wrong in general. Some rule sets are inherently sequential (P-complete).
4. **Lower bound.** "n³ or n^ω cannot be avoided" is not a theorem. Best known (min, max) product is about O(n^2.69).
5. **Smooth training semiring leaks the future.** The per-prefix equivalence holds only for hard min/max and only when a position reads the indicator 1[D ≤ t]. Under a log-sum-exp relaxation the full-sequence closure differs from the prefix closure. RELATOR's proposed training method therefore violates causality and needs a different one.
6. **Memory floor.** Excess entropy E is a lower bound; the minimal memory for optimal prediction is the statistical complexity C_μ ≥ E. Recent estimates of the Hilberg exponent are nearer 0.8 than 0.5, which weakens the argument that a linear cache is unnecessary.
7. **Similarity search.** "Inherently a scan" is too strong: approximate attention runs in near-linear time in a bounded-entry, low-dimension regime and is conditionally hard outside it.
8. **DEMOS.** "Constraints cannot increase what is computable per unit cost" is false in general. The simulation argument shows no expressivity-per-cost separation; it says nothing about sample efficiency or optimisation. DEMOS could still win on those, and the experiment must be able to show it.
9. **FILTRA.** The merge forest determines the H0 barcode, not the reverse. find(a, t) in O(log n) needs binary lifting or union by size. With congruence closure the problem can be P-complete, so the O(log n)-round claim holds only without it. Log-depth Transformers already solve connectivity and k-hop at depth Θ(log k); that is the baseline to beat.
10. **RELATOR expressivity.** Drop the 1-WL comparison; keep only the capacity argument.
11. **Amari.** D translation modes in D dimensions; reduction condition is separation much larger than bump width but within the kernel tail.
12. **KEYSTORE.** Keys must not depend on same-layer reads. Tombstones count as live keys, so memory can be O(N) under heavy deletion.

**Strategic weaknesses the referee raised, with evidence.**
- The prize may be small: context beyond 2k tokens improves prediction for a small set of tokens, mostly copyable ones. This supports the keyed-dependence bet and also caps the gain; the 0.03-bit criterion could pass without showing an advantage.
- Exact match gives no learning signal on a miss; codebook collapse is documented, and working published memory layers use soft top-k, not hard equality.
- Retrieval gains have not transferred: kNN-LM improves perplexity but not generation or reasoning.

**Novelty after review.** Principle: published (associative form). KEYSTORE: partly published. RELATOR: classical mathematics, application not found. FILTRA: not found inside a language model. The last two searches were shallow.
