# Nested adaptive computation: theory before implementation

Date: 2026-11-21. Status: theory and proposed controls for agreement. Nothing here is implemented beyond what `PROTOCOL-NESTED.md` already runs. An independent referee reviews this document; corrections will be appended as a final section and will override the text above them.

Labels: **[Established]** textbook mathematics. **[Published]** in the literature; cited from memory unless marked checked. **[Observed]** measured by us. **[Derived]** follows here from stated assumptions. **[Hypothesis]** testable, untested. **[Speculation]** not yet testable as stated.

---

## 1. The question, sharpened

Owner's question: is capability set by how transformations are organised across space, depth and time, and can passes reinforce one another so that each pass changes what the next can compute?

Four resources, with symbols used throughout:

| Symbol | Resource | Meaning |
|---|---|---|
| P | parameters | description length of the model |
| L | inner depth | distinct transformations composed inside one pass |
| T | outer steps | number of passes |
| S | state | amount of information carried between passes |

Compute per input is C ∝ L·T·(cost of one layer). An untied stack of depth D has P ∝ D and T = 1. A nested model has P ∝ L and depth L·T.

## 2. What follows from the definitions alone

**D1. A tied loop never computes anything an untied stack of the same total depth cannot.** [Derived] Copy the block T times into an untied stack of depth L·T; it computes the same function. So for a fixed compute budget, nesting cannot add expressive power. Whatever it adds must come from one of three places: needing fewer examples (a smaller hypothesis class), being easier to optimise, or running for more steps at test time than in training. Every claimed benefit should be assigned to one of these three.

**D2. The interaction measure I is positive by construction on the obvious task family.** [Derived] Take chains of k steps where each step needs depth at least L₀ to compute. Then a model with L < L₀ fails for any T, and a model with T = 1 fails once k outgrows its depth, while (L ≥ L₀, T ≥ k) can succeed. So I = A(L,T) − A(L,1) − A(1,T) + A(1,1) > 0. This is true and uninformative: an untied stack of depth L₀·k also succeeds. H1 as an expressivity claim is nearly a tautology. The scientific content is only in the comparison against the untied model of equal compute, and against the equal-parameter one.

**D3. "Each pass changes what the next pass can compute" has exactly two readings.** [Derived]
- (a) The weights are fixed and only the state changes. Then the next pass computes the same function F on a different input. Any state-dependent choice of operation is already inside F if F is expressive enough. A controller that selects among shared transformations, h_{t+1} = h_t + F_{θ(h_t)}(h_t, x), is a particular parameterisation of a fixed F, not a larger class. It can still be a better inductive bias or cheaper to run. [Published: hypernetworks, mixture-of-experts layers, mixture-of-experts inside a universal Transformer]
- (b) The pass writes something that later passes read as if it were parameters: a persistent store. Then the effective function really does change from pass to pass, and S grows. This is the only reading in which the sentence is literally true. [Published: fast weights, memory-augmented networks, scratchpads]

**D4. Time and state are different resources, and a loop over a fixed-size state gives only one of them.** [Established, from the theory of computation] T steps over a state of size S can do what a machine with space S and time T can do. A looped Transformer with fixed width has bounded S per position. A model that writes tokens to a scratchpad gains both T and S. [Published: scratchpad and chain-of-thought results showing constant-depth Transformers gain power with a polynomial number of written steps]

**D5. What parameter reuse buys, stated exactly.** [Established concept, applied here] A tied loop has a short description (small P) and a long running time (large T). The functions it suits are those with a short program that takes many steps: low description length, high computational depth (Bennett's logical depth). An untied stack suits long programs with short running time. Arbitrary facts have high description length whatever the architecture. So:
- reuse across time can help on procedures, by at most the factor by which the procedure repeats itself;
- it cannot help on facts, and removes capacity for them.

This is a prediction, and we have two observations on each side of it. [Observed] Looped models learned two-step chains in 5 of 7 runs where dense learned in 0 of 7. [Observed] On real text the looped model was slightly worse than dense (1.519 against 1.499 bits per byte, one seed, pre-predicted).

**D6. Consequence for the "ambitious question".** [Derived from D4, D5] "More capability without more parameters, by organising reuse across time" is true for the class in D5 and false outside it. A general model needs both: large storage for facts whose cost per step does not grow with its size (sparse access), and a small reused program run for a variable number of steps over a state that can grow. That is the organisation of a stored-program computer. It is not new as a description. What is not solved is learning it by gradient descent, reliably, from data that does not label which knowledge is which.

## 3. Mathematical specification of the candidate family

State h_t ∈ ℝ^{n×d} (n positions). Input x. Block F_θ = f_L ∘ … ∘ f_1, each f_i a Transformer layer.

| Member | Update | New free parameters | Status of the idea |
|---|---|---|---|
| N0 untied | h = f_{LT} ∘ … ∘ f_1 (x) | — | baseline |
| N1 nested | h_{t+1} = F_θ(h_t) | none | [Published]; running in `PROTOCOL-NESTED.md` |
| N2 with input re-injection | h_{t+1} = F_θ(h_t + W x) | W | [Published as necessary for stable long iteration] Our current loop does **not** re-inject x; that is a gap between the code and the owner's Φ(h, x) |
| N3 gated residual | h_{t+1} = h_t + α_t ⊙ (F_θ(h_t, x) − h_t), α_t = σ(a_ψ(h_t)) | a_ψ | [Published: highway, ReZero, gated recurrent units] |
| N4 halting | N3 plus stop probability p_t = σ(q_ω(h_t)); loss L_task + λ·E[T] | q_ω | [Published: ACT, PonderNet] |
| N5 state-conditioned block | F_{θ(h_t)} from K shared transformations mixed by a small controller | controller | [Published in several forms]; by D3(a) not a larger function class |
| N6 writable store | pass t reads and writes a persistent memory M_t; h_{t+1} = F_θ(h_t, read(M_t)); M_{t+1} = write(M_t, h_t) | read/write heads | [Published in several forms]; the only member where D3(b) holds |

Note on N3: the owner's form h + αF(h) and the form above are the same family; the convex form keeps the state bounded when α ∈ [0,1].

**Jacobians.** [Established] For N1, J_t = ∂h_{t+1}/∂h_t = ∂F_θ/∂h at h_t. The gradient of the loss with respect to the shared weights is a sum over passes:

∂ℒ/∂θ = Σ_{t=0}^{T−1} (∂h_{t+1}/∂θ)ᵀ · (J_{T−1} ⋯ J_{t+1})ᵀ · ∂ℒ/∂h_T.

Two consequences. Every pass contributes signal to the same weights, which is the "shared learning signal" explanation (E1) written down. And the terms are scaled by products of Jacobians, so the singular values of those products decide whether early passes receive any signal: far below 1 they vanish, far above 1 they explode. [Published: dynamical isometry, edge-of-chaos analyses]

**Energy.** [Established] An update h_{t+1} = h_t − η g(h_t) is gradient descent on some energy E only if the Jacobian of g is symmetric everywhere (on a simply connected domain). So "does the learned loop follow an energy landscape" has a direct test: measure the asymmetry ratio ‖J_g − J_gᵀ‖ / ‖J_g‖ along trajectories. Near zero means an energy description is available; far from zero means it is not, and rotation-like dynamics matter. [Observed, ours] When we imposed conservative dynamics by construction (DEMOS), the hop step could not change the norm of the state and gradients were badly behaved. That is evidence against imposing an energy; it says nothing about whether one emerges.

## 4. The sudden jumps: a concrete mechanistic hypothesis

[Observed] In 25 completed runs, every accuracy curve is flat at guessing and then jumps. One-step accuracy itself waits 3,000–5,000 steps. Two-step jumps came between step 5,000 and step 23,000, or not at all.

[Published] Abrupt transitions after plateaus are documented for in-context learning circuits and for delayed generalisation; saddle-point dynamics in deep linear networks give plateaus followed by fast transitions.

**H-bridge.** [Hypothesis] Answering a two-step query needs two circuits in sequence: (1) look up the first value and write it somewhere readable (the "bridge" entity); (2) use it as the key of a second lookup. The gradient that builds circuit 2 is proportional to how well circuit 1 already delivers the bridge in the format circuit 2 expects, and conversely. Near initialisation both factors are small, so the two-step loss sits on a saddle: growth is multiplicative and the escape time is roughly log(1/ε)/λ, highly sensitive to the initial alignment ε. That gives seed-dependent jump times.

In a tied loop, circuit 2 **is** circuit 1. Once one-step lookup works, the second lookup exists; only the routing (write the result where the key is read) remains to be learned. In an untied stack, later layers must learn lookup again, with signal only from two-step queries.

What it predicts, and what would refute it:

| Prediction | Refuted if |
|---|---|
| Before the jump, the bridge entity is not linearly decodable at the query position after pass 1; it becomes decodable shortly before or at the jump | decodability is high long before the jump, or absent after it |
| In failed looped runs, one-step works but the bridge is never written where the key is read | failed runs show the bridge decodable at the right place |
| Dense jumps much later rather than never; a dense model whose later layers are initialised as copies of its first layer jumps about as early as looped | copied-initialisation dense still never jumps |
| The wide looped model's failure (0 of 3) is an optimisation effect: it recovers with a lower learning rate | it fails across a learning-rate sweep |
| Jump time distribution across seeds is heavy-tailed (roughly log-normal) | times cluster tightly |

Competing explanations kept open: E2 (the tied class is simply smaller and the target is inside it), which predicts no help when steps need different operations; and pure chance (E5), which predicts equal jump-time distributions for tied and untied given enough seeds.

## 5. Measurement definitions that fit what we actually see

**The interaction measure must not be computed on mean final accuracy.** [Derived from Observed] Outcomes are two-valued (about 21% or above 95%). Mean accuracy over a few seeds is then just an estimate of "fraction of seeds that jumped by the cut-off", with a hidden dependence on the cut-off. Use instead:

- q(L, T; B) = probability of having jumped by step budget B, with a confidence interval;
- τ(L, T) = jump time, analysed as time-to-event with runs that never jump treated as censored (Kaplan–Meier curves, log-rank test).

Interaction on the log-hazard scale: does adding inner depth change the jump rate more when T is larger? With 5 seeds per cell only very large effects are detectable; 20 or more seeds per cell are needed for a factorial, so the task must be made cheap enough for that (smaller symbol set, shorter sequences) and that reduced task must be fixed before the runs.

**Other quantities.**
- Cost: parameters, layer applications per input, wall-clock; always all three.
- Length extrapolation: accuracy on chains longer than any trained, at trained T and at larger T.
- Jacobian: top singular values of J_t and of the product across passes, by training step.
- Settling: relative move per pass (already recorded).
- Energy test: asymmetry ratio from section 3.
- Bridge decodability: linear probe accuracy, by layer, pass and training step.

## 6. Proposed staged experiments and controls (for agreement; not started)

Stage 0 is already running and is not changed by this document: `PROTOCOL-LOOP2.md`, `PROTOCOL-NESTED.md`.

| Stage | Question | Arms | Decisive control | Gate to next stage |
|---|---|---|---|---|
| 1 Mechanism of the jump | Is H-bridge right? | existing dense and looped checkpoints through training, plus copied-initialisation dense | the refutation column of section 4 | any outcome is informative; proceeds regardless |
| 2 Factorial on jump time | Is there an interaction between L and T beyond equal-compute untied? | L ∈ {1,2,4} × T ∈ {1,2,4,8}, untied depth-matched stacks, narrow parameter-matched stacks; 20 seeds on a reduced task | untied stack of depth L·T; learning-rate sweep per arm | interaction on log-hazard with interval excluding zero |
| 3 Unlike steps | Does tying help only when steps repeat? (separates E1 from E2) | chains whose steps alternate between two different operations; tied, nested 2×T, untied | nested 2×T should win if "a block = one period of the procedure" | — |
| 4 Extrapolation and halting | Do extra passes at test time buy longer chains; does learned halting improve accuracy per unit compute? | N2, N3, N4 with randomised T in training | fixed-T model given the same average compute | only if stage 2 or 3 shows a tied advantage |
| 5 Writable store | Does D3(b) give what D3(a) cannot? | N6 against N2 at equal compute, on chains over facts that arrive over time | the continuous acquisition benchmark, where iterating a one-hop read already composed untrained chains in the smoke test | only after the benchmark's first full result |

N5 (state-conditioned block) is not scheduled. By D3(a) it adds no function class, and no observation of ours points at it. It should enter only if stage 3 shows that a fixed block cannot serve unlike steps.

## 7. Where a new principle could and could not be

- **Not in the architecture diagram.** Every member of the family in section 3 is published in some form. [Published]
- **Not in the interaction measure as an expressivity statement.** D2.
- **Possibly in the learning dynamics.** If H-bridge holds, the statement "weight sharing converts a multiplicative saddle into an additive one for compositions of a repeated operation" would be a precise, checkable account of when and why reuse makes composition learnable. Whether that account already exists in the literature is unknown to me and must be checked before any claim. [Hypothesis]
- **Possibly in joining the two halves of D6.** A reused program reading a store that grows (N6) is where this line meets the cumulative-learning objective. Our only evidence is a smoke test. [Hypothesis]

## 8. Disciplines: which connections carry weight and where they break

| Field | Connection that is mathematically real | Where it breaks |
|---|---|---|
| Dynamical systems | the loop is an iterated map; stability is governed by Jacobian spectra | useful computation is often transient, not at a fixed point; convergence is not correctness |
| Physics: energy | symmetric-Jacobian test decides if an energy exists | attention updates are generally not gradient fields; imposed conservation hurt us |
| Physics: phase transitions | plateaus then jumps resemble saddle escapes | "criticality" needs a scaling law across sizes before the word is earned; we have none |
| Algorithms | time and space as resources; short programs with long running times | theory says what is computable, not what gradient descent finds |
| Geometry | singular values, decodability, trajectory distances are measurable | "manifold" language adds nothing until a specific quantity predicts success |
| Information theory | what is decodable from the state at each pass | information present is not information usable by the next layer |
| Morphology | — | no mathematical content identified yet; treated as a label for N5 [Speculation] |

## 9. Decisions needed from the owner

1. Agree the order in section 6, in particular doing the jump mechanism (stage 1) before any new architecture.
2. Agree that outcomes are analysed as jump probability and jump time, not mean accuracy.
3. Agree a reduced task for the 20-seed factorial, fixed before running.
4. Agree that N5 stays unscheduled unless stage 3 calls for it.
