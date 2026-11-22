# Independent formal and implementability audit — Nano LM

Reviewed 2026-11-21. Baseline: the recovered V2 package's architecture, experiment
plan, gap review and mathematical checks. Corrections were implemented only in
the assigned working architecture document and new forward-reference files.
This audit is an intermediate review artifact; no remote branch was modified.

## Result

The core algebra in V2 is useful, but several conclusions were broader than
their premises and the neural architecture was not yet an executable contract.
The current revision fixes those statements and adds a small implementation.
It establishes conditional mathematics and deterministic forward correctness;
it supplies no trained-model quality or efficiency result.

| ID | Finding | Precise repair | Disposition |
|---|---|---|---|
| FA-01 | An arbitrary bijective latent map was too close to a claim of an equivalent fixed Transformer parameterization | Separate functional conjugacy from closure of an operator family; give normalization, RoPE, attention and tied-head conditions | Implemented in 01 §3.2 |
| FA-02 | Covariance-preserving migration could be confused with prediction-preserving migration | Isotropic covariance survives arbitrary rotation while a fixed classifier may change; require prediction/task target and held-out evaluation | Implemented in 01 §3 |
| FA-03 | Per-step logit bounds had no precise sequence extension | State a uniform prefix-wise premise and derive a finite-horizon joint-log-probability/KL bound | Implemented in 01 §3.3 |
| FA-04 | A projected raw gradient may cease to be feasible after Adam, decay or coordinatewise clipping | Project or otherwise constrain the realized complete displacement; report damping/rounding residual | Implemented in 01 §4.1 |
| FA-05 | First-order preservation can protect false facts intended for correction | Version the protected target set and remove/replace corrected targets; preserve explicit new-task acquisition evaluation | Implemented in 01 §4.1 |
| FA-06 | “Preserves earlier as-of answers” is false under backdated insertions into a one-time index | Use immutable recorded-time cutoff and fixed policy/view; supply an explicit counterexample | Replaced 01 §6 |
| FA-07 | Controller overhead and utility had ambiguous units and shared costs | Use the difference of full branch costs; sunk/common controller cost cancels; add finite-horizon/budget assumptions | Replaced 01 §7 |
| FA-08 | A failed candidate was said to falsify an existential architectural conjecture | Retain existential statement as ambition; make the operational hypothesis finite and candidate/protocol-specific | Replaced 01 §8 |
| FA-09 | Input reinjection was mandatory without a theorem; cache/token clocks were ambiguous | Make reinjection an ablation; consume observed token once; distinguish token, refinement and training clocks | Replaced 01 §2 |
| FA-10 | Named auxiliary objectives were not defined algorithms | Set all undefined terms to zero; initial objective is explicit masked causal CE | Replaced 01 §2.3; added 07 §4 |
| FA-11 | Architecture diagrams omitted logical-depth cache ownership and exact counts | Implement a forward-only dense/shared-block reference with one cache per logical layer; enumerate parameters | Added 07 and prototype scripts |
| FA-12 | Forward-transfer slope criterion in 03 is unnecessarily restrictive and confounded by task order | Define transfer as a paired improvement on the same new task under different prior experience; counterbalance family difficulty | Root integration recommendation |
| FA-13 | G02 arms and primary endpoints disagree across tables | Enumerate exact discrete arm IDs and distinguish primary endpoint from guardrails; do not call an uncommitted template preregistered | Root integration recommendation |
| FA-14 | Unique-parameter accounting can misprice recurrence | Compute operations from logical depth and executed shapes; 6ND is only a dense proxy, not a shared-weight theorem | Added 07 §3.1; root should align 03 |

## 1. Exact transport conditions

### 1.1 Functional identity versus architectural realization

The equation $D'=D\circ T^{-1}$ is a definition of a function. It does not show
that $D'$ belongs to the same constrained network class, that its parameter
count is unchanged or that it can be learned cheaply.

For a linear readout, $W'=WA^{-1}$ is constructive. A whole transition requires
$\Phi'=A\circ\Phi\circ A^{-1}$. Nonlinear operations, cached values, query/key
coordinates and embedding ties must be addressed at their actual boundaries.
An explicit inverse wrapper is valid but has storage, arithmetic and numerical
conditioning costs. A wrapper that exactly cancels its forward map provides
no new expressive power by itself.

### 1.2 Attention and RoPE

For column-vector queries and keys, $q'=A^{-\top}q$ and $k'=Ak$ preserve the
dot product. Applying $A$ to both gives $q^\top A^\top Ak$, which is generally
different.

If fixed RoPE rotations are then applied, the transformed score is

$$
(R_tA^{-\top}q)^\top(R_sAk)
=q^\top A^{-1}R_t^\top R_sAk.
$$

It equals the original score for every q/k when A commutes with all relevant
relative rotations. This condition applies to the stated query/key-coordinate
transformation. It does **not** imply that RoPE blocks every residual-stream
basis change: Q/K projections may absorb a residual transform while leaving
head coordinates and rotary operators unchanged.

### 1.3 Normalization and ties

For gain-free RMS normalization $N(z)=z/\sqrt{\|z\|^2/d+\epsilon}$, orthogonal
Q obeys $N(Qz)=QN(z)$. Generic invertible A changes the norm anisotropically.
The example $z=(1,1), A=\operatorname{diag}(2,1),\epsilon=0$ yields distinct
$N(Az)$ and $AN(z)$.

Learned gains and neighboring projections require explicit accounting;
an arbitrary unchanged gain matrix need not commute with the transform.
One must not turn that observation into a blanket impossibility statement:
in some architectures a gain can be absorbed into an adjacent linear matrix.

With column-layout input embeddings E and tied output $E^\top$, transforming
input embeddings to AE forces output $E^\top A^\top$, while exact readout
transport needs $E^\top A^{-1}$. For full-row-rank E, universal equality requires
orthogonality. Untying or adding an adapter changes the implementation/resource
contract and must be reported.

### 1.4 Optimization after a representation change

Prediction equivalence at a checkpoint does not imply identical training.
Under parameter coordinates $\theta'=A\theta$, the gradient transforms as
$g'=A^{-\top}g$. Plain SGD in transformed coordinates uses
$-\eta A^{-\top}g$, whereas mapping the original step gives $-\eta Ag$.
They generally differ. Adam's coordinatewise moment operations are not
rotation-equivariant either. A reset is a new optimizer intervention, not
proof that optimization history was coherently transported.

## 2. Approximate preservation

For finite logits with infinity-norm displacement at most epsilon, the
log-normalizer displacement is at most epsilon, so each log probability
changes by at most $2\epsilon$. Therefore the old-to-new KL is at most
$2\epsilon$. The bound is valid though not always sharp.

For two autoregressive distributions at common horizon T, suppose the same
condition holds at **every relevant prefix**, with bounds $\epsilon_t$.
Factoring their joint probabilities yields

$$
|\log P(x_{1:T})-\log Q(x_{1:T})|\le2\sum_t\epsilon_t,\qquad
D_{\rm KL}(P\Vert Q)\le2\sum_t\epsilon_t.
$$

This is a conditional sequence statement, not a guarantee inferred from a
finite anchor evaluation. The error bound can grow with horizon. Small average
anchor error says nothing about an unmeasured rare prefix without a coverage
or generalization argument. Deterministic greedy equivalence additionally
requires a sufficient margin along every common reached prefix.

Covariance alone cannot replace these prediction conditions: if $z$ is
isotropic and a fixed classifier is $w^\top z$, a rotation preserves covariance
but generally changes the classifier's values.

## 3. Realized optimizer displacements

For $J=(1,1)$ and $g_\perp=(1,-1)$, $Jg_\perp=0$. A preconditioner
$D=\operatorname{diag}(1,2)$ changes this to $JDg_\perp=-1$. Therefore even
perfect raw-gradient projection does not ensure a feasible Adam-style step.
Weight decay can independently add a component outside the nullspace.

The Euclidean nearest feasible displacement to the full proposal u is

$$
\Delta^*=u-J^\top(JJ^\top)^+Ju.
$$

This follows from orthogonal decomposition. With a Lipschitz Jacobian on
the whole update segment,

$$
\|f(\theta+\Delta)-f(\theta)\|_2
\le\|J\Delta\|_2+\tfrac L2\|\Delta\|_2^2.
$$

For damped inverse $(JJ^\top+\gamma I)^{-1}$, the residual is explicitly
$\gamma(JJ^\top+\gamma I)^{-1}Ju$, not zero. Floating-point realization,
rounding and later transforms must be checked on the actual committed
displacement. A matrix-free approximation needs a residual and cost receipt,
not a dense P-by-P projector.

Protecting every old target is incompatible with correcting an old false
target. Declare the protected target set and correction/supersession policy.
The remaining constraints can still block new learning, so retention without
acquisition is not a successful edit method.

The source audit independently identified AlphaEdit as direct fixed-key
nullspace prior art. A layer identity $(W+\Delta P)K_0=WK_0$ when $PK_0=0$
is exact for those fixed keys. It is distinct from the nonlinear whole-model
Jacobian bound. Approximate SVD thresholds, changed upstream keys and changing
multiple layers require additional analysis; the root's source register
contains that evidence and should control its bibliographic details.

## 4. Historical queries and correction semantics

Use immutable event records with valid interval, recorded sequence, authority,
action and ID. Fix the query's permission view and conflict-policy version.
Select only records valid at t and recorded by r. The maximum under a declared
total order is invariant to replay permutation **when replay preserves the
same timestamps and IDs**. Assigning new timestamps on replay changes the
problem.

If a newly appended event always has recorded sequence greater than r, it
cannot change the old query's eligible set. This is the precise historical
invariance. A late correction may intentionally change the current
reconstruction of old valid time. A single-time latest-event index cannot
provide both meanings of history.

Explicit outcomes are a value, retracted-in-scope, or unknown-in-scope.
Permission denial does not prove absence. A deterministic authority ordering
is a declared resolver, not a claim that its answer is factual truth.

## 5. Adaptive computation

The one-step continuation condition is
$\mathbb E[\Delta U\mid s]>\lambda\mathbb E[c_{continue}-c_{stop}\mid s]$.
Common control costs already paid cancel; additional costs must be converted
to utility with the declared exchange rate. A cost in joules cannot be added
to dimensionless utility without a conversion.

A multi-step Bellman recursion needs a sufficient state/history representation,
known or estimated transitions, a terminal utility counted once, and a finite
horizon or positive minimum cost. The identity does not give an imperfect
controller oracle access to counterfactual utility. State-dependent cost can
matter as much as the number of steps.

For the implemented shared-depth Transformer, tokenwise depth skipping raises
a separate cache problem: a past token may have no key/value at a depth needed
later. Start with fixed per-request depth or a pointwise output refiner whose
backbone cache is always populated. A stop flag alone is not a complete
variable-depth streaming architecture.

## 6. Hypotheses and experimental logic

An existential conjecture cannot be disproved by a single failing candidate.
Preregister a finite candidate's code/configuration, task distribution,
development search, independent unit, test seeds, primary endpoint, minimum
useful effect, regression tolerances, full cost budget and decision rule.
Retire or revise that tested claim after failure. A wide interval is
inconclusive, not evidence of equivalence.

### Forward transfer

The V2 criterion requiring a decreasing slope of examples-to-threshold over
successive tasks is not necessary for forward transfer. Suppose the baseline
requires 100 examples on every novel task and prior experience reduces this
to 60 on every matched task. Both slopes are zero, but the experienced learner
has a persistent sample-efficiency advantage.

Define a paired contrast on the same held-out new family, comparing prior
experience conditions under the same resource budget. Counterbalance task
order/difficulty and exclude in-context old demonstrations. Specify censoring
when a learning threshold is not attained; learning-curve area can be an
additional endpoint. Retention is a separate guardrail.

### G02 and update of G03

G02's arm descriptions should become a single explicit list: “original
frozen/learned” is not one unambiguous level, and beta2 sweeps are not one
fixed optimizer setting until specified. Distinguish baseline/reset/alternative
moment settings from a separately declared SGD comparator; declare the
tuning budget for each. Select one primary outcome and make other safety
measures named guardrails or explicitly handle multiple co-primary outcomes.

The parent discovered newer three-seed Q9 correction results on the research
branch. Those supersede the claim that the basic G03 locality controls are
unrun. They must be integrated with pinned raw evidence. Immediate corrected
fact accuracy, same-entity collateral effects and durable retention after
later updates are distinct outcomes; do not promote one into the others.

## 7. Concrete implementation delivered

Files added to the working architecture package:

- 07_IMPLEMENTATION_CONTRACT.md.
- prototype/native_reference.py.
- prototype/test_native_reference.py.

The implementation is a NumPy float64 causal Transformer with bias-free
attention, interleaved RoPE, RMSNorm, SwiGLU and tied input/output embeddings.
Dense uses four unique/four logical blocks; shared uses two unique/four logical
blocks. Each logical block owns a separate KV cache.

Observed checks: **12/12 pass** in NumPy 2.3.5. Full-prefix versus incremental
maximum absolute logit error is **6.140921104957897e-16** for dense and
**1.4155343563970746e-15** for shared. Tiny parameters are **10,896** and
**5,712**; each arm's batch-2, length-11 float64 KV storage is **22,528 bytes**.

The nominal configuration V=32768, d=256, h=4, f=768, L=8 has
**15,208,704** parameters when U=8 and **11,798,784** when U=4.
These counts are verified without allocating the nominal models. Both have
eight logical caches and the same mathematical forward depth, but actual
training/serving cost equality is unmeasured.

The reference has no optimizer, trained weights, data pipeline, retrieval,
learned transport, adaptive controller or production kernel. Correct gradients,
training stability and real language-model evidence remain open dependencies.

## 8. Primary sources checked for operation definitions

- Farajtabar et al. (2020), Orthogonal Gradient Descent for Continual Learning:
  https://proceedings.mlr.press/v108/farajtabar20a.html
- Zhang and Sennrich (2019), Root Mean Square Layer Normalization:
  https://arxiv.org/abs/1910.07467
- Su et al. (2021; revised 2023), RoFormer:
  https://arxiv.org/abs/2104.09864
- Banino et al. (2021), PonderNet:
  https://arxiv.org/abs/2107.05407

Primary landing pages/abstracts were opened. The explicit algebra above is an
independent derivation; no source is being represented as establishing Nano
LM's performance. The root source-audit agent owns the wider bibliography.
