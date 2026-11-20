# One model: the unified design (working name PATHWAY), 2026-11-20

Status: design draft, unreviewed. Nothing here is implemented or measured. [T] theorem, [E] published result, [S] speculation, [U] novelty unverified.

## 1. The unifying mathematics

Memory, relations, retrieval, transformation and iteration can be written as operations on one object: **a weighted graph over stored items, with weights in a semiring.** Three operations suffice, and they are the three operations of a Kleene algebra, the standard algebra of programs:

| Operation | Symbol | Meaning for computation | Realised as |
|---|---|---|---|
| Choice / merge | + | combine alternatives | attention (soft), latest-wins (hard) |
| Composition | · | do one thing after another | ordered products along the sequence |
| Iteration | * | repeat until nothing changes | closure over the stored graph |

Kleene algebra with tests is complete for reasoning about while-programs, and matrices over a Kleene algebra form a Kleene algebra, so the algebra applies to graphs of items. [T]

**How a Transformer fits.** Ignoring the nonlinearities, L residual attention layers compute (I + A_L)…(I + A_1): a polynomial of fixed degree L in its attention matrices. It uses + and ·, at fixed depth. It has no iteration operator and no persistent graph: every layer rebuilds its links from scratch.

**What the unified design adds.** One persistent graph that layers add to, and the star operation on it. For a causal graph (links only to earlier items) A is strictly lower-triangular, so A* = I + A + A² + … is a finite sum and always exists. [T]

## 2. State

For one context:
- **Items** i: a vector h_i, a key, an arrival time. Tokens create items; so do derived results.
- **Links** A_ij between items, of a few learned types. A link is created whenever an item reads another. Links persist across layers.

## 3. Operations and equations

For layer ℓ with item states H:

1. **READ** (choice). r_i = Σ_j κ_β(q_i, k_j)·v_j over candidates j < i. The sharpness β is learned per read. Low β is ordinary attention over a local window. High β with a recency tie-break is an exact latest-value lookup, computed by sort instead of a scan. Each read adds its strongest links to A.
2. **COMPOSE** (composition). Items may emit small matrices M_i; the prefix product P_i = M_1·…·M_i is computed for all i by a parallel scan.
3. **STAR** (iteration). x = b + γ·A·x, solved by T rounds of x ← b + γ·A·x on the stored links. T is fixed in training and may stop early at inference when x stops changing. Each round is one sparse multiply; no links are recomputed.
4. **WRITE-BACK.** Results of STAR and COMPOSE that are read again may be stored as new items or links, at a learned price. A later query then finds the answer in one READ.

Layer update: H ← H + MLP([READ(H), COMPOSE(H), STAR_A(H)]); A ← A + links(READ).

This is still a neural network trained by gradient descent on next-token prediction. Attention remains, as the general case of READ.

## 4. How the capabilities depend on one another

| Loop the directive asks for | Mechanism here | Status |
|---|---|---|
| Memory improves reasoning | STAR runs over stored links | designable now |
| Reasoning creates memories | WRITE-BACK of derived results | designable now |
| New memories change later computation | later READs find stored results in one step | designable now |
| Operations become reusable | stored composed matrices and derived links | designable; value unproven |
| Learning reorganises the architecture | which reads are sharp, which links persist, are learned | weak form only |

## 5. The ten capabilities, honestly

| Capability | What the design offers | Honest status |
|---|---|---|
| 1 Memory with exact retrieval | sharp READ, persistent items | mechanism known; learned keys unproven |
| 2 Multi-step reasoning | STAR, COMPOSE | mechanism known; advantage over a looped Transformer unproven |
| 3 Self-organisation | learned link persistence and read sharpness | within a context only |
| 4 Adaptive computation | early stopping of STAR at inference | saves inference, not training |
| 5 Knowledge discovery | derived links written back | only what the link algebra can express |
| 6 Updating knowledge without retraining | within a context: yes. Across contexts: no | **no known reliable method**; continual learning is an open field |
| 7 Long-context efficiency | no global dense attention; keyed reads | depends on capability 1 |
| 8 Compositional generalisation | composition is an operation, not an emergent behaviour | unproven |
| 9 Efficiency per parameter and per token | fewer recomputed links | unmeasured |
| 10 Improving its own computation | memoisation is the only concrete mechanism | mostly aspiration |

Capabilities 6 and 10 in their strong form are not delivered by any architecture I can derive. They should be treated as research questions, not design requirements for the first model. Following instructions and general competence come from training data and scale, not from the architecture.

## 6. Cost

Per token per layer: local attention O(W·d), keyed reads O(G), STAR O(T·k·d) for k links per item, scans O(d_s³) amortised. None grows with context length N. Memory grows with items and links kept.

Training is parallel across positions: soft reads as in a Transformer, sharp reads by sort, scans by parallel prefix, STAR by T masked sparse multiplies.

## 7. The strongest simpler alternative

A weight-tied (looped) Transformer with learned token eviction. It can imitate every operation above, with more expressive steps. PATHWAY's only possible advantages are cost (it reuses links instead of recomputing attention, and stores results instead of rederiving them) and length generalisation (iteration is an operation, so more rounds can be run at test time). If the looped model matches it at equal compute, PATHWAY has no reason to exist.

## 8. Smallest experiment for the unified claim

Task: BINDINGS with updates, alias chains and repeated queries.

- **Arms:** PATHWAY; PATHWAY without STAR; without sharp READ; without WRITE-BACK; looped Transformer with eviction at matched compute; dense Transformer.
- **Claim 1, interplay:** the full model beats every ablation on chained queries by at least 10 points at chains 8× longer than in training.
- **Claim 2, reasoning becomes memory:** the measured cost of answering a chain a second time is at most half the first, with equal accuracy. The looped Transformer should show no such drop.
- **Claim 3, against the simpler alternative:** equal or better accuracy than the looped Transformer at no more than half its wall-clock cost per query.
- Three seeds, protocol committed before fits. Any claim that fails is reported as failed.

Then language: L0 (size of the long-context gain) before any language build.

## 9. Known relatives

Attention diffusion and personalised-PageRank attention; deep equilibrium and Universal Transformers; non-diagonal state-space models; memory layers; egglog. The combination as one Kleene-algebra language model: [U].

## 10. Route to a frontier-scale model

1. Phase 1 results and L0. Decide whether structured memory has a prize worth pursuing on our data.
2. The experiment in section 8, laptop scale.
3. If claims 1–3 hold: a 10–20M language model against dense at equal time.
4. If that holds: 125M on TPU (needs approval), three seeds.
5. 350M, then 1B, then the 2B decision. Each step only on a repeatable win at the previous one.

At the measured TPU speed (65k tokens/s for a 336M dense model on one chip), the credits cover the 125M and 350M stages comfortably. They do not cover a 2B model trained to frontier quality; that needs a separate compute decision if the evidence earns it.
