# Four architecture proposals, 2026-11-20

Status: design draft. Nothing here is measured. Novelty is "unverified" for every proposal: one literature search each found close published relatives (listed). Speed figures are estimates, not measurements.

Notation: N context length, d width, L layers. Dense reference: per token per layer O(N·d) lookup + O(d²) MLP; cache O(N·d·L).

Fresh names are used on purpose; none of these is Gen3/4/5, S2, M-anything or Omega.

---

## P1. LEDGER — a priced, write-once, addressable store

**Replaces attention's lookup with:** an index lookup into one shared store, with a learned price on every write.

**State.** One store for the whole network (not one per layer): entries (k_j, v_j, x_{j+1}, w_j) with key k_j ∈ R^{d_k}, value v_j ∈ R^{d_v}, the next token id, and a retention weight w_j ∈ (0,1].

**Write.** w_j = σ(g(h_j)); k_j = K h_j; v_j = V h_j. Written once, never recomputed.

**Read (training form, differentiable).** For query q_i = Q h_i:
  a_ij = softmax_j( q_i·k_j / τ + log w_j ),  j < i
  r_i = Σ_j a_ij v_j
The log w_j term makes retention a prior on each memory: w_j → 0 removes entry j smoothly.

**Objective.** L = L_LM + λ · mean_j(w_j). This is a rate–distortion trade: λ is the price of memory, and the model learns which tokens are worth it.

**Exact recall.** p(x) = (1−c_i)·p_vocab(x) + c_i·Σ_j a_ij·1[x_{j+1}=x], with copy gate c_i = σ(u·h_i). Same scores, no extra machinery.

**Inference form.** Keep entries with w_j > θ (fraction ρ of tokens). Keys are product-quantised (two sub-codebooks of size K), so a read scores O(B) candidates from matching buckets instead of N.

**Local processing.** Windowed attention of width W (e.g. 256) in every layer; the ledger is read at R of the L layers.

**Cost.** Per token: O(L·(W·d + d²) + R·B·d_k) at inference, independent of N. Memory: O(ρ·N·(d_k+d_v)), about L/ρ times smaller than a dense cache. Training within a window computes R full N×N score matrices instead of L, so the lookup saving in training is about L/R, not N.

**Estimated speed.** About 1.0–1.3× dense at 2k context; advantage should grow roughly linearly with N beyond that. Unmeasured.

**Under static shapes (XLA).** Training uses the dense masked form above; top-k and bucket lookup appear only at inference, with fixed k and B.

**Archive relative.** M21's EvidPtr (copy pointer) and ERN's surprise cache. ERN's eviction was a surprise heuristic and scored 2/16; here retention is trained end to end against a price.

**Published relatives.** Product-key memory (Lample 2019), sparse read/write memory (Rae 2016), kNN-LM and Memorizing Transformers, write-gated retrieval (arXiv 2603.15994).

**Claimed new part.** Retention as log w_j inside the score, priced by λ, with copy taken from the same scores.

**Falsifiable.**
- F1a: at ρ ≤ 0.25, loss within 0.01 bits/token of ρ = 1. If it needs ρ > 0.5, "learns what to retain" is false.
- F1b: learned retention beats recency-only and random retention at equal ρ by ≥ 0.02 bits/token.
- F1c: hard-index inference loses ≤ 0.01 bits/token against the soft training form.

**Main risk.** The model keeps everything unless λ is high, then forgets what it later needs; soft-to-hard gap at inference.

---

## P2. CLOSURE — reasoning as a fixed point over a sparse causal graph

**Replaces attention's lookup with:** inheritance. Each token links to a few earlier tokens and inherits everything they have already derived.

**State.** For each token i, a relational state z_i ∈ [0,1]^r, final once computed.

**Graph.** k edges per token to earlier tokens: structural (previous token, dyadic skips) plus content edges chosen by top-k key match. Edge weights A_ij ∈ [0,1]^r.

**Update (max–min semiring, per channel).**
  z_i = e_i ⊕ ⊕_{j∈N(i)} (A_ij ⊗ z_j),  ⊕ = max, ⊗ = min, e_i = σ(E h_i)
Because edges point only backwards, the graph is acyclic, and z_i computed from its neighbours' final states already equals the full transitive closure: z = A*·e. One pass gives unlimited hops. Idempotence (a ⊕ a = a) means extra iterations change nothing.

**Training form.** Smooth semiring: ⊕ = (1/β)·logsumexp(β·), ⊗ = soft-min, with β annealed upwards. Parallel in time by T rounds of relaxation (T ≈ log₂ of the longest chain, or pointer doubling); exact one-pass DP at inference gives the same answer once β is large.

**Mixing.** z_i is projected back into h_i and followed by a standard MLP, because a monotone semiring alone cannot express negation.

**Cost.** Per token per layer O(k·r + d²), independent of N at inference. Training: O(T·k·r) per token. Memory O(N·r).

**Estimated speed.** About 0.6–0.9× dense at 2k context (gathers are slow on accelerators). Unmeasured.

**Dynamic computation.** Depth of reasoning is set by the data (chain length), not by layer count. No halting controller is needed.

**Archive relative.** The Gödel model (V5): 0.845 vs 0.39 on 5-hop synthetic graphs at 19K parameters; never run on language. New here: the causal one-pass form and its use as a language-model layer.

**Published relatives.** Tropical Attention (arXiv 2505.17190, NeurIPS 2025) realises tropical closure by stacking layers on algorithmic tasks; "Geometry of Thought" (arXiv 2601.09775) shows softmax attention tends to max-plus at zero temperature. Neither search hit is a language model.

**Falsifiable.**
- F2a: with 2 layers, solves ≥ 8-hop binding chains written in text (e.g. variable aliasing in code) at ≥ 90%, where a 2-layer dense model of equal size is below 50%.
- F2b: replacing half the attention layers of the dense baseline with CLOSURE costs ≤ 0.03 bits/token on ordinary text at equal training time. Worse than that and it is a puzzle solver, not a language mechanism.
- F2c: accuracy after annealing to hard max–min within 2 points of the smooth model.

**Main risk.** Sparse gradients through max/min; no benefit on ordinary text, where long chains are rare. It does nothing for exact retrieval.

---

## P3. DYADIC — a persistent multiscale tree with exact descent

**Replaces attention's lookup with:** O(log N) summaries plus a tree search when exact detail is needed.

**State.** A binary tree over the context. Leaf ℓ₀(t) = (key, value, token) of position t. Internal node for block [a, a+2^m): s = Merge_m(s_left, s_right), computed once when the block completes.

**Read.** The prefix [1, t) splits into at most ⌈log₂ t⌉ complete blocks (Fenwick decomposition). Token t attends over those block summaries: O(d·log N).

**Exact descent.** For query q, at a node score both children by q·key(child) and follow the best b (beam). After m steps the search reaches leaves: exact tokens, used for a copy distribution as in P1. Cost O(b·d·log N). Descent runs only when a gate asks for it.

**Why this shape.** Resolution falls as 1/distance: recent text exact, old text coarse. Each node is computed once and amortised merges are O(1) per token, so memory is persistent rather than recomputed.

**Cost.** Per token per layer O(d·log N + d²); total O(N log N). Memory O(N·d) for the tree (leaves are needed for descent). Static shapes throughout.

**Estimated speed.** About 0.7–1.0× dense at 2k; ahead at 16k. The published log-linear work needed custom kernels. Unmeasured here.

**Archive relative.** None direct.

**Published relatives.** Log-Linear Attention (arXiv 2506.04761, ICLR 2026) uses the same Fenwick partition over linear-attention states and reports matching or beating Mamba-2 and Gated DeltaNet, with a remaining gap to Transformers on many tasks. The multiscale read is therefore NOT new.

**Claimed new part.** Routable summaries and tree descent to exact tokens, i.e. precise retrieval in O(log N) from the same structure.

**Falsifiable.**
- F3a: descent with beam 4 retrieves the correct token at distance 8k with ≥ 90% accuracy; kill if more than 10 points below flat top-k over all keys.
- F3b: at 8k context and equal training time, loss within 0.02 bits/token of dense.
- F3c: measured time per token grows no faster than log N from 2k to 32k.

**Main risk.** A d-dimensional key cannot say what 4,096 tokens contain, so descent misroutes at high levels. This is an information-capacity limit and may be fatal; F3a tests it directly.

---

## P4. LEXICON — the model grows its own units

**Changes the representation itself:** repeated material becomes a single learned unit, so the sequence the network processes gets shorter as it reads.

**State.** A dictionary D of phrase entries, each (embedding e_p, expansion = token span). Starts empty for each context.

**Step.** At each position the model emits either a vocabulary token or a dictionary entry:
  p(next) = softmax over [W_vocab·h ; E_D·h]
Emitting entry p consumes its whole span in one step (an LZ78-style back-reference). After each step a write gate may add the span just completed: e_p = Compose(states of the span).

**Why.** Minimum description length: the best code for repeated structure is a reference to it. Exact recall is then the native emission, not an add-on, and compute per original token falls where text is predictable.

**Cost.** If the dictionary compresses the stream by factor c, the processed length is N/c and any backbone's cost shrinks accordingly (dense: (N/c)²). Dictionary scoring adds O(|D|·d) per step, bounded by capping |D|. Memory O(|D|·d).

**Estimated speed.** Unknown. On ordinary prose c may be near 1; on code and structured text it could be 2–4.

**Under static shapes.** Hard: variable-length steps. Training would teacher-force a segmentation (greedy longest match against the growing dictionary) and pad to fixed step count; the write gate is trained with a price as in P1.

**Archive relative.** ConceptSlot (16-slot overwrite memory) loosely.

**Published relatives.** Byte-latent and dynamic-chunking models (BLT, H-Net), MegaByte, pointer-generator networks, classical LZ/grammar compression. Persistent in-context dictionaries with copy-as-emission: not found in one search.

**Falsifiable.**
- F4a: achieves c ≥ 1.5 on the code corpus and c ≥ 1.15 on ordinary text with loss per original token no worse than dense at equal time.
- F4b: tokens per second per original token ≥ 1.2× dense on code.
- F4c: learned dictionary beats a fixed greedy LZ dictionary by ≥ 0.02 bits/token.

**Main risk.** The least mature of the four; segmentation is discrete and training may not be stable. It adds nothing to reasoning.

---

## Considered and set aside

**Exact least-squares memory** (replace the delta rule's one gradient step with the optimal online solution). Published and scaled already: MesaNet (arXiv 2506.05233) trained 140M–940M models. Use it as a published reference point, not as our proposal.

## Against the five goals

| | Remember | Reason | Retrieve exactly | Adapt compute | Scale with N |
|---|---|---|---|---|---|
| LEDGER | yes (priced store) | nothing new | yes | partly (retention) | yes at inference |
| CLOSURE | partly (states persist) | yes | no | yes (data-set depth) | yes |
| DYADIC | yes (tree) | partly | yes (descent) | partly (descent on demand) | yes, N log N |
| LEXICON | yes (dictionary) | nothing new | yes (emission) | yes (variable step) | only if text compresses |

No single proposal covers all five. LEDGER or DYADIC plus CLOSURE would; that is a later question.

**Shared risks.** P1, P3 and P4 all rely on a hard selection trained through a soft relaxation. P1, P3 and P4 share the copy mixture. P2 with content edges needs a key index like P1's.

## Test protocol (to be committed before any fit)

- **Stage 0, free:** deeper literature check per proposal; write the shared harness.
- **Stage 1, free (Kaggle or Mac), PyTorch:** about 10–20M parameters, three seeds per arm, same tokenizer and data, same optimizer (AdamW), learning rate chosen per arm from the same three-value sweep, equal wall-clock time on the same device.
  - Arms: strong dense (pre-norm, rotary positions, full context), one arm per proposal. S2 as a reference if time allows.
  - Data: ordinary text and the screened code corpus; contexts 2k and 8k.
  - Probes: recall at distance, multi-hop binding, loss versus context length, tokens per second, peak memory.
  - Prefix-causality audit on every arm before any score is read.
- **Gate to Stage 2:** the proposal's own F-tests pass, and loss on ordinary text at 8k context is within 0.02 bits/token of dense at equal time. At most two proposals advance.
- **Stage 2, Google TPU:** 125M, three seeds, equal time, contexts 2k and 16k. Port only survivors to JAX. TPU hourly price to be verified first.
- **Stage 3:** 350M, then decide on 1B+.
