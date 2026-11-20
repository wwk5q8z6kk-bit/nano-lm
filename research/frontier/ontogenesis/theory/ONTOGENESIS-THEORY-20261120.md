# ONTOGENESIS: theoretical investigation, 2026-11-20

Status: theory only. **An independent referee pass found errors; section 9 lists the corrections and overrides the text above wherever they differ.** Nothing is implemented or measured. Citations are from memory and are being verified separately; treat any citation as unverified until that pass is recorded. Results marked [T] are established theorems, [E] established empirical findings, [S] speculation.

## 0. What "a different substrate" can and cannot mean

1. **Computability does not change.** Any system simulated on a digital computer computes a Turing-computable function. A field, a mesh or a population cannot compute something a Transformer with enough steps cannot. [T]
2. **What can change:** (a) resource scaling (compute and memory as functions of context length N, facts retained m, reasoning steps h); (b) which functions are reachable at fixed depth; (c) what can be proven about the system (stability, conservation, convergence); (d) inductive bias and sample efficiency.
3. **Costs are real costs.** In a simulation, a physical "energy" is a number in a loss. It saves nothing unless it reduces operations, memory or wall time. Every claim below is stated in those units.

A "new capability" is therefore defined as: solving a task family inside resource bounds where dense Transformers, S2 and M21 provably or measurably cannot.

## 1. The five ideas: what is established

### 1.1 Computation as a field

Assumptions: Φ(x,t) ∈ R^c on a lattice of M sites; F local and translation-invariant.

Discretise ∂Φ/∂t = F(Φ, ∇Φ, ∇²Φ, I; θ) with forward Euler and centred differences:

  Φ_{t+1}(x) = Φ_t(x) + Δt · f_θ( Φ_t(x−1), Φ_t(x), Φ_t(x+1), I_t(x) )

This is exactly a weight-tied residual convolutional network with kernel width 3 (a neural cellular automaton). [T, by construction; cf. Ruthotto–Haber, Chen et al. neural ODEs]. So idea 1 is not outside neural networks; it is the subclass with tied weights and local kernels.

Consequences that can be proven:
- **Light cone.** With stencil radius r, information moves at most r sites per step. Linking two sites at distance D needs T ≥ D/r steps. [T]
- **Diffusive fields forget.** For ∂Φ/∂t = D∇²Φ, mode k decays as e^{−Dk²t}; spreading goes as √t, so reaching distance D takes T ∝ D². Fine detail is destroyed first. [T]
- **Wave fields remember exactly, as a delay line.** For Φ_tt = c²Φ_xx with input at x=0, Φ(x,t) = I(t − x/c). A medium of M sites holds the last M/c inputs with zero interference, addressed by position. This is a shift register, i.e. an orthogonal recurrent network. [T]
- **Capacity.** M sites with c channels at b bits hold at most M·c·b bits. [T]
- **Universality is uninformative.** Reaction–diffusion systems and cellular automata are Turing-complete (Rule 110) [T]; that says nothing about whether gradient descent finds the program.
- **Flows preserve topology.** A neural ODE is a homeomorphism of its state space, so it cannot represent maps that change topology without extra dimensions (Dupont et al.). [T]

### 1.2 Memory as a conserved structure

- **Liouville's theorem.** A Hamiltonian flow preserves phase-space volume. Hence a conservative system has no asymptotically stable attractors. [T]
- **Attractors need dissipation.** Convergence of many states to one is contraction; contraction erases information (Landauer: at least kT ln 2 per bit in physical systems). [T]
- **Contradiction C1.** "Memory as a conserved quantity" and "memory as an attractor" cannot hold in the same degrees of freedom. The consistent resolution is to split the state: conservative coordinates carry information, dissipative coordinates make decisions. The general form is metriplectic (GENERIC, Grmela–Öttinger):
  ż = L(z)∇E(z) + M(z)∇S(z),  L = −Lᵀ, M = Mᵀ ⪰ 0,  L∇S = 0, M∇E = 0
  which gives dE/dt = 0 and dS/dt ≥ 0. [T]
- **Exact recall needs growing state.** Storing N tokens from a vocabulary V exactly needs N·log₂V bits. A fixed-size state cannot do it for N beyond its capacity, whatever the dynamics. [T] This is the measured S2-versus-M21 gap, stated as a theorem.
- **Resonant retrieval.** A bank of oscillators a_k(t+1) = e^{−(γ_k + iω_k)} a_k(t) + b_k·I(t) with geometrically spaced γ_k stores a Laplace transform of the history (Shankar–Howard; Legendre memory units; HiPPO). This is a diagonal linear state-space model: the S4/Mamba family. Resonance is a matched filter; d modes hold d numbers. [T/E] So this route arrives at a known architecture.
- **Associative attractor capacity.** Pairwise Hopfield: about 0.138·n patterns. Order-p interactions: about n^{p−1} (Krotov–Hopfield). Exponential interactions: exponential capacity (Demircigil et al.), and one update step of that system is softmax attention (Ramsauer et al.). [T] So the high-capacity attractor route arrives at attention.
- **Stable with use, without interference.** Cascade models with geometrically spaced hidden timescales (Benna–Fusi) reach memory lifetime that scales nearly linearly with the number of synapses with power-law forgetting. [T/E] There is a proven trade-off: bounded synapses cannot have both fast learning and long retention; cascades approach the bound, they do not remove it. "No interference" is not achievable at fixed size (see capacity).

### 1.3 Topology instead of layers

- **Attention already is input-dependent topology.** Each layer builds a weighted complete graph per input. What it lacks is persistence and a variable node count.
- **Rigorous form of "nodes are born and die".** Write a network as a measure over units: f(x) = ∫φ(x;ω)dμ(ω). Training is a gradient flow of μ. Adding a reaction term gives the Wasserstein–Fisher–Rao flow:
  ∂_t μ = ∇·(μ∇Ψ) − α(Ψ − Ψ̄)μ,  Ψ = δE/δμ
  The first term moves units; the second grows or kills them by fitness. Global convergence holds in the mean-field limit under stated convexity conditions (Chizat–Bach; Rotskoff et al.). [T]
- **Non-differentiability (C7).** Discrete births, deaths and merges have no gradient. Known routes: continuous masses (above), relaxations, or search. Each has a soft-to-hard gap. [E]
- **Sheaf and topological methods.** Sheaf Laplacians generalise graph diffusion and are established [T/E]. Persistent homology gives descriptors and regularisers. Category theory currently supplies language, not algorithms. [S as a source of capability]

### 1.4 Competing physical principles

- **Free energy is variational inference.** Minimising F = E − TS over a belief is the evidence lower bound. The information bottleneck and rate–distortion are the same family. [T] These are objectives, not architectures.
- **Stability against memory (C4).** A contractive system has the fading-memory property: the influence of old inputs decays exponentially (Boyd–Chua). [T] A stability penalty that enforces contraction therefore removes long memory. Long memory requires marginal stability, which is the conservative subsystem again.
- **Symplectic integration.** Leapfrog steps for a separable Hamiltonian are exactly symplectic and reversible for any learned potential. Reversibility allows activations to be recomputed backwards, so training memory does not grow with the number of steps. Volume is preserved (determinant 1), so gradients cannot vanish in all directions; they can still grow if the dynamics are chaotic. [T]
- **Equilibrium propagation.** For an energy E_θ(u) with loss L, let u⁰ minimise E and u^β minimise E + βL. Then
  lim_{β→0} (1/β)·[∂E/∂θ(u^β) − ∂E/∂θ(u⁰)] = dL/dθ   (Scellier–Bengio) [T]
  Learning needs two relaxations and only local quantities. It is a genuinely different training principle. Its demonstrated scale is small. [E, to be verified]
- **Fixed-point models.** Deep equilibrium models define the output as a fixed point; existence and uniqueness are guaranteed for monotone operators (Winston–Kolter). [T]
- **Symmetry cost (C6).** An energy requires symmetric couplings. Softmax attention with separate query and key matrices is not the gradient of any symmetric interaction energy, so Transformers have no Lyapunov function, and energy-based models give up directed interactions to get one.

### 1.5 A model that invents its mathematics

- **Embedding theorems.** Every finite group has a faithful matrix representation (Cayley); every finite-dimensional associative algebra is a matrix subalgebra; every finite-dimensional Lie algebra is (Ado). [T] On a digital computer, "beyond vector spaces" always reduces to vectors plus learned multilinear maps. The content of idea 5 is therefore: which maps are learned, under which constraints, and whether they change with experience.
- **Learned algebra.** A product (a∘b)_k = Σ_ij C_ijk a_i b_j with learned structure tensor C. Associativity is the polynomial identity Σ_m C_ijm C_mkl = Σ_m C_jkm C_iml, imposable as a penalty. Published relative: parameterised hypercomplex layers. [E]
- **Learned geometry.** A learned convex potential G_θ defines a Hessian metric; the mirror flow u̇ = −(∇²G_θ)^{−1}∇E is natural-gradient descent in that geometry. [T] Learned curvature and pulled-back metrics are published. [E]
- **Learned operators.** Koopman theory: nonlinear dynamics are linear in a suitable space of observables. [T]
- **Standard networks already discover algebra.** Networks trained on modular arithmetic learn Fourier and group-representation features. [E]
- **Speculative part.** That learned algebraic structure yields capability beyond better sample efficiency on algebraic tasks. [S]

## 2. Contradictions the project must respect

| # | Contradiction | Consequence |
|---|---|---|
| C1 | Conservation excludes attractors (Liouville) | Split conservative and dissipative coordinates |
| C2 | Fixed state excludes exact recall (bit count) | Memory must grow with retained information |
| C3 | Local dynamics have a light cone | Long range costs steps, or needs non-local structure |
| C4 | Contraction implies fading memory | Stability penalties destroy long memory |
| C5 | Persistent nonlinear state is sequential in time | Training cannot be parallelised over the sequence unless the update is associative (linear); this is why S2 ran at 0.48× and M21 at about 0.2× |
| C6 | Energy functions need symmetric coupling | Convergence proofs cost directed computation |
| C7 | Topology changes are discrete | Soft-to-hard gap |
| C8 | Simulated energy is not compute | Only operation counts matter |
| C9 | Universality is not learnability | Turing-completeness proves nothing useful |
| C10 | Self-organisation needs a training signal | Backprop through the dynamics makes it a neural network with tied weights; local rules have unproven scaling |

C5 is the most serious for a frontier model. Every "continuously evolving" design is a recurrent system, and every recurrent design in this repository lost at equal training time.

## 3. Three architectures

### O1. METRIPLEX — conservative carrier, dissipative decision

State z = (q, p, s): carrier (q,p) ∈ R^{2M}, decision s ∈ R^m.

  H_θ(q,p) = ½‖p‖² + ½ qᵀΩ²q + U_θ(q)
  q̇ = p,  ṗ = −Ω²q − ∇U_θ(q) + B·I(t)      (leapfrog, symplectic)
  ṡ = −Γ ∇_s F_θ(s; q),  F bounded below

Provable: carrier map symplectic and reversible; for frozen q, dF/dt = −‖∇_sF‖²_Γ ≤ 0, so s converges to critical points (LaSalle). With Ω giving a dispersion relation and U_θ = 0 the carrier is an exact delay line.
Cost per token: O(M) for the diagonal linear part plus O(M·w) for a local U_θ; memory O(M); training memory independent of step count.
Capacity: at most M numbers (C2). With fixed M it is in S2's class and will lose on exact long recall. With M ∝ N, reading by resonance is a correlation over N modes, which is attention.
Verdict: the cleanest mathematics and the least likely to give a new capability. Best used as a component.
Archive relative: the port-Hamiltonian block in the holonomy line (not promoted).

### O2. DEMOS — a population that moves, merges, is born and dies

State: μ_t = Σ_{i=1}^{n_t} m_i δ_{z_i}, with z_i ∈ R^d, mass m_i ≥ 0, and an attached exact payload (token id).

  E_θ[μ] = ∫V_θ dμ + ½∬W_θ(z,z′) dμ dμ′,  W symmetric
  Ψ(z) = V_θ(z) + ∫W_θ(z,z′)dμ(z′)
  ż_i = −∇Ψ(z_i) + J∇Φ_θ(z_i)       (dissipative + conservative part, J = −Jᵀ)
  ṁ_i = −α m_i (Ψ(z_i) − κ)          (κ is the price of memory)

Each token is injected as a particle. Particles with m_i < ε are removed; particles closer than δ merge and their masses add. Output is a readout ∫φ dμ plus a copy distribution over payloads.

Provable (with J = 0): dE/dt = −Σ m_i‖∇Ψ(z_i)‖² − α Σ m_i(Ψ(z_i) − κ)(Ψ(z_i) − κ) − … ≤ 0 in the mass-conserving form (κ = Ψ̄), so E is a Lyapunov function; mean-field convergence results apply under their conditions.
Relation to Transformers: with all m_i = 1, no death and a softmax-normalised kernel, one transport step is a self-attention layer (Sander et al.; Geshkovski et al.). Transformers are the mass-conserving, fixed-population special case. Mass dynamics and a variable population are the new degrees of freedom.
Cost per token: O(τ·n·k) for τ flow steps, n live particles, k neighbours; memory O(n·d). Both depend on retained information, not on N. This is the only one of the three that answers C2.
Failure modes: pure gradient flows cluster (proven for attention dynamics), which merges distinct facts; hence the conservative J-term, which removes the Lyapunov guarantee. The flow cannot create mass at new locations without nucleation. Training is backprop through time (C5).
Archive relative: the interaction mesh (lost to dense, 0.516 vs 0.521) had fixed nodes and no mass dynamics. LEDGER from the previous proposal set is the static limit of DEMOS.

### O3. ALGEBRA-EQUILIBRIUM — computation as relaxation in a learned algebra

State u ∈ R^n; bounded activation ρ; learned convex G_θ; symmetric W; fully symmetric structure tensor C of rank R.

  E_θ(u) = Σ_i G_θ(u_i) − ½ Σ W_ij ρ(u_i)ρ(u_j) − ⅓ Σ C_ijk ρ(u_i)ρ(u_j)ρ(u_k) − bᵀρ(u)
  u̇ = −(∇²G_θ)^{−1} ∇E_θ(u)

Context writes fast couplings: C ← C + η·a⊗b⊗c for observed triples, so composition rules seen in context become part of the energy.
Provable: E bounded below and non-increasing, so trajectories converge to equilibria (Cohen–Grossberg, extended to higher order); cubic energies have capacity of order n² patterns; equilibrium propagation gives exact gradients in the β → 0 limit with local updates.
Cost: O(n·R) per relaxation step; T steps per token with T set by the instance. Reasoning depth is not fixed.
Failure modes: spurious minima; critical slowing; symmetric coupling (C6); sequential per token (C5); equilibrium propagation unproven at language scale.
Published relatives: dense associative memories, Energy Transformer, deep equilibrium models.
Archive relative: the "attractor" candidate in the UII arena (dense won 0.621 vs 0.573).

## 4. Comparison

| | Dense | S2 | M21 | O1 METRIPLEX | O2 DEMOS | O3 ALG-EQ |
|---|---|---|---|---|---|---|
| State size | N·d·L | fixed | fixed + pointer over N | M (fixed) | n ∝ retained facts | n fixed + fast C |
| Per-token cost | O(N·d·L) | O(d²·L) | O(d²·L + N) | O(M) | O(τ·n·k) | O(T·n·R) |
| Exact recall | yes | no beyond capacity | yes (copy) | window M only | yes if payload kept | up to capacity |
| Reasoning steps | bounded by depth | bounded by depth | bounded by depth | bounded by steps | τ adaptive | T adaptive |
| Parallel training over sequence | yes | partly | no | yes if linear, else no | no | no |
| Proven guarantees | none | none | none | symplectic, Lyapunov for s | Lyapunov (J=0) | convergence, capacity |
| Evidence | strong | small edge, one seed at 350M | wins by copy, loses on time | none | none | none |

## 5. The smallest decisive experiment: KEEP-AND-CHASE

Fixed-depth Transformers with log-precision arithmetic lie in the circuit class TC⁰ (Merrill–Sabharwal) [T]; iterated function composition is hard for them without more depth or intermediate output (Peng et al.). Fixed-state models are bounded by C2. That locates one task family where each baseline should fail along a different axis.

**Task.** A stream of N tokens contains m facts of the form a→b (a random function on symbols), scattered among distractors. A query gives a start symbol and a hop count h; the answer is f^h(start).

**Three independent axes.** N (distractor load), m (facts to retain), h (reasoning steps). Train on h ≤ 4, m ≤ 16, N ≤ 512. Test up to h = 32, m = 64, N = 4,096.

**Predictions, fixed before any run.**
- Dense, depth 4–6: accuracy falls towards chance for h beyond what its depth supports; cost grows with N.
- S2-style recurrent: falls as m exceeds state capacity.
- M21: one-hop copy works; chained hops fail.
- Controls that decide whether a new substrate is needed: a weight-tied Transformer with adaptive iteration, and the CLOSURE layer. If either meets the criterion, the capability comes from iteration, not from fields or populations, and ONTOGENESIS is not required for it.

**Success criterion for an ONTOGENESIS model (O2 first, then O3).** At ≤ 1M parameters, three seeds: accuracy ≥ 90% at h = 32, m = 64, N = 4,096, with measured operations per query growing with m·h and not with N (less than 1.5× from N = 512 to N = 4,096), while every baseline is below 50% on at least one axis.

**Failure reads.** Accuracy falling with m means particles are merging (clustering collapse). Falling with h means the flow does not iterate. Cost growing with N means retention is not selective.

**Scale.** Runs on a laptop in hours. No credits needed.

**What success would not show.** That the mechanism helps language modelling. The archive already has two synthetic wins that did not transfer (rotation registers on group composition; the Gödel model on 5-hop graphs). A pass here earns a 10–20M language test under the equal-time protocol, nothing more.

## 6. Assessment

- Ideas 1 and 2, followed rigorously, lead to known families: tied-weight convolutional networks, linear state-space models, and attention. The new content is the metriplectic split, which gives guarantees rather than capability.
- Idea 3 has a rigorous core (birth–death measure flows) that is not exploited by current language models. It is the best-founded route to something different, and DEMOS is built on it.
- Idea 4 contributes a different training principle (equilibrium propagation) and convergence proofs, at the price of symmetry and sequential relaxation.
- Idea 5 reduces to learned multilinear structure and learned metrics; whether that buys capability is open.
- This repository has already tried physics-inspired fields, meshes, attractors and cellular automata at toy scale, and dense won each time. What is different here is the order of work: theorems first, one pre-registered capability test, and controls that can show the new substrate is unnecessary.

## 7. The self-organizing neural field (owner's highlighted idea)

The established mathematical form is the Amari neural field (1977):

  τ ∂u/∂t = −u(x,t) + ∫ w(x−y) f(u(y,t)) dy + I(x,t)

with short-range excitation and longer-range inhibition in w. [T/E]
- **Disturbances become persistent structures.** Localised "bumps" are stable solutions that outlast their input; this is the standard model of working memory in computational neuroscience.
- **Capacity.** Roughly M / (bump width) bumps on M sites; each bump holds a position and an amplitude. Bounded, as C2 requires.
- **Failure modes.** Bumps drift under noise (they sit on a continuous attractor, which is only marginally stable), merge when close, and die under strong inhibition.
- **Reduction to particles.** Writing each bump by its centre z_i and amplitude m_i gives ordinary differential equations for (z_i, m_i) with pairwise interaction: the same form as DEMOS. Simulating the full field costs O(M) per step for M much larger than the number of bumps; the particle form costs O(n·k). DEMOS is therefore the economical form of the self-organizing field, not a different idea.
- **"New information changes the geometry."** In field terms this is plasticity of w; in particle terms, of W_θ and the fast couplings.

The linked "QGT Companion" page is a self-published physics site with no peer-reviewed references and no content on neural computation; nothing here depends on it.

## 8. Literature check on the earlier four proposals (2026-11-20)

LEDGER: partly published (TRIM-KV arXiv 2512.03324; Selective Attention 2410.02703; Expire-Span). Incompressible retrieval collapses under eviction in TRIM-KV. DYADIC: largely published (Log-Linear Attention 2506.04761; tree indexer 2610.42112). LEXICON: partly published (zip2zip 2506.01084), with quality losses reported. CLOSURE: causal one-pass semiring closure inside a language model not found (confidence about 50%). MesaNet vs Transformer perplexity: 16.05 vs 15.81 (145M), 12.34 vs 12.40 (400M), 10.83 vs 10.86 (1B).

## 9. Referee corrections (2026-11-20), overriding the text above

Errors:
1. **DEMOS dissipation.** With symmetric W: dE/dt = −Σ m_i‖∇Ψ_i‖² − α Σ m_i Ψ_i(Ψ_i − c).
   - c = Ψ̄ (mass-weighted mean): dE/dt = −Σ m_i‖∇Ψ_i‖² − α Σ m_i(Ψ_i − Ψ̄)² ≤ 0; mass conserved; E is Lyapunov.
   - c = κ (fixed price, the form written in O2): E is not monotone. The Lyapunov function is E − κ·Σ m_i, with derivative −Σ m_i‖∇Ψ_i‖² − α Σ m_i(Ψ_i − κ)² ≤ 0.
   - Conditions: functional bounded below (e.g. W positive semidefinite); holds only between injection, prune and merge events; only with J = 0.
2. **C1.** Volume preservation, in finite dimension, excludes asymptotically stable sets. Conserving a quantity does not: GENERIC conserves E and has attractors. O1 as written is a one-way cascade, not GENERIC.
3. **C5 was too strong.** Exact one-pass parallel scans need transition maps closed under composition with a compact form (e.g. affine with input-dependent gates). General nonlinear recurrences can be trained in parallel by Newton-type iterations (DEER, quasi-DEER/ELK, ParaRNN, which reports 7B-parameter nonlinear RNNs), at the cost of extra iterations and no guarantee for chaotic dynamics. Sequential training is a cost, not an impossibility.
4. **C6 and "Transformers have no Lyapunov function".** Wrong as stated. The obstruction to a Wasserstein gradient structure is softmax row-normalisation; attention is a gradient flow in a modified metric under symmetry conditions (Geshkovski et al.); Energy Transformer has an energy with separate query and key matrices; convergence can be proven without gradient structure (monotone operators). Correct statement: gradient dynamics need a symmetric Jacobian in some metric; convergence proofs do not.
5. **"Transformers are the special case of DEMOS".** Not exactly. The unit-mass, no-death limit with symmetric W is unnormalised or Sinkhorn attention, not softmax attention.
6. **Wave field.** Forward Euler with centred differences is unstable for wave equations. The exact delay line is first-order advection, upwind, at Courant number 1, on a half-line or with an absorbing end. A finite shift register is nilpotent, not orthogonal.
7. **Fading memory.** "Contraction implies exponential forgetting" is contraction theory / the echo-state property, not Boyd–Chua (whose theorem is Volterra approximation).
8. **O3.** Capacity of order n² needs tensor rank of order n²; a rank-R tensor stores O(R) patterns, so O(n·R) cost and n² capacity cannot both hold. Fast writes must be symmetrised, and the Lyapunov claim holds only between writes.

Conditions to add: capacities (0.138n is for random binary patterns with a few per cent errors; n^{p−1} costs n^p weights; exponential ≈ 2^{n/2}); mean-field convergence needs convexity in μ and concerns training over units, not token dynamics; TC⁰ hardness is conditional on TC⁰ ≠ NC¹; Peng et al. is unconditional for one layer only; symplectic maps always have a direction with gain ≥ 1; LMU and HiPPO are not diagonal.

Experiment (section 5) must change:
- Depth Θ(log h) suffices for h hops (Sanford–Hsu–Telgarsky), so a depth-6 dense model can represent 32 hops. A dense failure would be a failure to extrapolate from h ≤ 4, not a depth limit.
- Looped Transformers with adaptive steps already extrapolate in hop count to roughly 2–3× the training range (Saunshi et al.; Fan et al.; arXiv 2609.33144). The control arm will probably pass in that range; 8× is open.
- 64 facts is under 1 kbit, so the fixed-state bound C2 does not bite. m must exceed the baselines' actual state capacity for that axis to mean anything.
- Equilibrium propagation now reaches full ImageNet with a 10-layer convnet; "small scale" should read "vision convnets only".

Novelty: learned forgetting and variable token counts are published (Expire-Span, token merging and pooling, DNC allocation, learned retention). DEMOS's claim narrows to mass dynamics coupled to transport with a Lyapunov structure. No birth–death token-dynamics language model was found in three searches; that is not proof of absence.
