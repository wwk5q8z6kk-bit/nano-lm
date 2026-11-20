# ONTOGENESIS phase 2: a unified substrate, 2026-11-20

Status: theory. Nothing here is implemented. Citations are from memory unless a referee section says otherwise; a literature check is pending. Labels: [T] theorem, [V] checked numerically in this project, [E] published empirical result, [S] speculation, [U] novelty unverified.

## 1. One substrate behind the three structures

KEYSTORE, RELATOR and FILTRA are not three inventions. They are three settings of one object that programming-language research already has: **a store of timestamped facts, closed under rules, with conflicts resolved by a merge function.**

- **Fact.** A tuple (r, a, b, t): relation type r, arguments a and b (learned discrete symbols), arrival time t, and a continuous payload.
- **Rule.** "If these facts hold, that fact holds." Evaluation repeats rule application until nothing new appears (least fixed point).
- **Merge.** What happens when two facts collide on a key.

| Structure | Rule set | Merge |
|---|---|---|
| Dictionary (KEYSTORE) | none | latest timestamp wins |
| Relation with chains (RELATOR) | r(a,b), r(b,c) ⇒ r(a,c), or a composition table | keep both |
| Equivalence (FILTRA) | symmetric, transitive, and congruence | union of classes |
| Tree or order | transitive, antisymmetric | keep both |
| Operation | apply(f, x) = y stored as a fact keyed by (f, x) | latest wins |

The published form of this unification is Datalog combined with union-find (equality saturation, "egglog"). [T/E]

**Consequence for "learn the data structure".** A data structure is not chosen from a menu. It is a relation type together with its rules and its merge function. Learning the structure means learning, per relation type, a few parameters: is it functional (φ_r), symmetric (σ_r), transitive (τ_r), and which types compose into which (Θ_{abc}). Structure learning is rule learning.

## 2. The principle, corrected twice

**Monotonicity is what makes it parallel.** A computation is monotone if more input facts never retract an output. The CALM theorem states that a program has a coordination-free distributed implementation exactly when it is monotone. [T] Read for sequence models: a memory computation can be evaluated for all positions without stepping through them in order when it is monotone in the set of facts.

**Time as data restores monotonicity.** Overwrite and delete are not monotone. Appending (fact, time) to a log is. So: keep the log, and answer "what was true at time t" as a query.

**General filtration theorem.** For any monotone rule set, evaluate the rules once in the (min, max) semiring over arrival times: a derived fact gets the time min over its derivations of the max time among the facts used. Then a fact is derivable from the first t tokens if and only if its time is ≤ t. [T: provenance semirings, (min, max) is a distributive lattice; V for composition tables by the referee, hard semiring only] One evaluation serves every prefix.

**With deletion and update the saving shrinks.** Truth then holds on sets of times, not on "from t onwards". The lattice of time sets under union and intersection still gives exact answers [T], but the cost scales with the number of distinct change points. The saving is proportional to how compressible the history of changes is. There is no free version.

**Limits that remain.** [T]
- Rule evaluation is polynomial in the number of entities, cubic for transitive rules.
- The number of rounds is not logarithmic in general; some rule sets are inherently sequential (P-complete).
- With function symbols (rewriting that creates new terms) evaluation need not terminate, so rounds must be capped.
- Exact match gives no gradient on a miss.

## 3. The missing operation: transformation

Three candidates, from cheapest to most general.

### T1. Composition by prefix product (state tracking)

Each token emits a small matrix A_t ∈ R^{d×d}. The state is the product P_t = A_1·A_2·…·A_t, computed for every t by a parallel prefix scan, because matrix multiplication is associative. [T]

- **Why it is a transformation mechanism.** Every finite-state process is a prefix product in its transition monoid. [T] Tracking how a sequence of operations changes a state (permutations, edits, moves) is exactly this.
- **What it can do that fixed-depth Transformers cannot** (conditionally): compose permutations of five elements, the canonical hard case for constant-depth models. [T, conditional on TC⁰ ≠ NC¹]
- **Cost.** O(d³) per combine, O(log N) depth; d is small (8–32) with many heads.
- **Stability.** Products of orthogonal matrices keep norm 1; general matrices need normalisation.
- **Status.** Published line: input-dependent non-diagonal state-space models and products of Householder reflections. Not new.
- **Archive link.** The rotation registers in the holonomy line are this mechanism with rotations. They solved group composition (100% against under 6%) and ran about 8× slower. Rotations are associative, so they could have been computed by a parallel scan. Whether the archive implementation did so is unverified; if it did not, its speed penalty was avoidable.

### T2. Rule application on the store

New facts are derived by joining the newly arrived facts against the existing store (semi-naive evaluation). Work is proportional to the new derivations, not to the context. [T] This is the direct answer to "transform knowledge without reprocessing the context". Rules are a learned composition table Θ plus the per-type flags of section 1.

### T3. Rewriting

Rules l → r over terms built from learned operator symbols, applied non-destructively so that all equivalent forms are kept (equality saturation). Most general; matching is NP-hard in pattern size, the store can grow exponentially under associativity and commutativity, and termination is not guaranteed. [T/E] Too expensive and too hard to learn as a first mechanism.

**Recommendation within this section.** T1 and T2 are complementary: T1 composes operations along the sequence; T2 composes facts across the store. T3 is deferred.

## 4. A non-Transformer architecture: STRATA

**Central engine:** the timestamped fact store with learned rules. Neural parts only read tokens and predict.

1. **Perception.** A small local causal network over the last W tokens (convolution or windowed network). For each token it emits: zero or more facts (r, a, b) with payload, and zero or more queries.
2. **Symbols.** Arguments are discrete codes from a learned codebook. A first version uses token ids and hashes of short token spans, which need no learning.
3. **Store.** All facts of the sequence, with times. Dictionary reads by sort and predecessor search [V]; closures by semiring evaluation.
4. **Transformation.** T1 scans for sequential state; T2 rules for derived facts.
5. **Prediction.** Local features plus retrieved payloads give the next-token distribution, including direct copy of stored tokens.

| Requirement | Achievable? | Condition or trade-off |
|---|---|---|
| Parallel training | Yes | Monotone rules over the timestamped log; keys must not depend on reads in the same layer |
| Adaptive memory | Yes | Memory ∝ facts written; deletions leave tombstones unless compacted |
| Exact retrieval | Yes, for exact-match keys | No similarity generalisation; multi-probe keys trade exactness for recall |
| Compositional reasoning | Within the rule language | Monotone only: no negation, no counting, unless stratified by time |
| Dynamic pathways | Yes | Derivations differ per input; depth can be inherently sequential |
| Efficient inference | Yes for lookups; polynomial for closures | Closure cost grows with entities, not tokens |
| Learned structure | **Open** | Discrete keys, types and rules have no exact gradient |

The last row decides whether this is a learning system or a database. Everything above it is known to work in databases.

## 5. The smallest experiment that tests learning, not lookup

**SCHEMA DISCOVERY.** The store has R untyped relation slots with learnable flags (φ, σ, τ) and a composition table Θ. Event streams use opaque relation tokens. The hidden structure behind each token differs: a dictionary with updates, an alias relation, a transitive order, and a pair of relations that compose. Supervision is answers to queries only.

- **Pass:** the learned flags and table match the hidden schema for every relation token, in at least 4 of 5 seeds, and accuracy holds on chains 8× longer than in training.
- **Controls:** the same store with the correct schema hand-set (upper bound); with a random fixed schema (lower bound); looped Transformer; dense Transformer.
- **What a pass shows:** that the system can discover which structure a pattern of information calls for.
- **What a fail shows:** that it is a database needing a hand-written schema, and the direction should stop there.

## 6. Language-model validation plan

Each ability is tested alone before any combination, on ordinary text and code at 8,000 tokens of context, at equal training time, three seeds.

| Stage | Ability | Model | Criterion |
|---|---|---|---|
| L0 | Size of the prize | Dense with 512 context against dense with 8,000 | Measure the gain and which tokens carry it |
| L1 | Exact recall | Local window + dictionary keyed by recent tokens | Recovers ≥ 70% of the L0 gain at ≤ 50% of dense's cost per token |
| L2 | Entities across contexts | L1 with learned keys | Beats L1 on tokens whose antecedent appears in a different surface form |
| L3 | Updates | Latest-wins against first-wins and keep-all | Latest-wins best on tokens following a redefinition |
| L4 | Relations and composition | Add T2 rules | Gain on multi-hop probes written in natural text |
| L5 | State tracking | Add T1 scans | Gain on code with reassigned variables |

The earlier 0.03 bits-per-token test stays, but it is not sufficient alone: the referee noted that long context helps only a small set of tokens, so a window model may pass it without showing any advantage. L0 measures that directly, and L1's criterion is stated as a share of the measured gain.

## 7. Risks, stated plainly

- **This is neuro-symbolic AI.** Learned rules and symbols inside neural systems have been studied for decades (differentiable logic programming, neural theorem provers). The record at language scale is poor.
- **The general lesson of the field** is that built-in structure tends to lose to scale and data. A structured store has to earn its place on cost, since a large Transformer can imitate its behaviour.
- **Retrieval gains have not transferred.** Nearest-neighbour language models improve perplexity without improving generation or reasoning. [E]
- **The archive's own pattern:** synthetic wins that did not carry to text (rotation registers, the Gödel model).

## 8. Recommendation

1. Finish phase 1 as authorised.
2. Run L0 and L1. They are cheap, need no learned symbols, and measure whether the prize is large enough to pursue.
3. Run SCHEMA DISCOVERY. It decides whether structure can be learned.
4. Build STRATA only if both pass.

RELATOR does not enter confirmatory testing until it is trained with the hard semiring and a causal read (the indicator time ≤ t), since its smooth form leaks future facts.
