# NANOLM Ω — Mathematical Formulations & Theorems

---

## Theorem 1 — Exact Quaternion Reverse-Mode Adjoint & $O(1)$-Activation Recurrence (`HolonomyScanFn`)

Let inputs be transition rotors $g_t \in \mathbb{H}$ and bound sources $s_t \in \mathbb{H}$ for $t \in \{0, \dots, T-1\}$ with initial state $\Omega_{-1} \in \mathbb{H}$, evolving via non-commutative Hamilton multiplication $\star$:
$$\Omega_t = g_t \star \Omega_{t-1} + s_t.$$

**Claim:** Let $\delta_t = \frac{\partial \mathcal{L}}{\partial \Omega_t} \in \mathbb{H}$ be the upstream Euclidean gradient with respect to $\Omega_t$ viewed in $\mathbb{R}^4$. Defining the adjoint variable $\lambda_t = \frac{d\mathcal{L}}{d\Omega_t}$ (with terminal condition $\lambda_T = 0$), the exact reverse-mode adjoint recurrence is:
$$\lambda_t = \bar{g}_{t+1} \star \lambda_{t+1} + \delta_t,$$
and exact parameter gradients are:
$$\frac{\partial \mathcal{L}}{\partial s_t} = \lambda_t, \qquad \frac{\partial \mathcal{L}}{\partial g_t} = \lambda_t \star \bar{\Omega}_{t-1}, \qquad \frac{\partial \mathcal{L}}{\partial \Omega_{-1}} = \bar{g}_0 \star \lambda_0.$$

**Proof Sketch:** Using the standard inner product $\langle u, v \rangle = \operatorname{Re}(u \star \bar{v}) = \sum_{i=0}^3 u_i v_i$ on $\mathbb{H} \cong \mathbb{R}^4$, the identity $\langle q \star a, b \rangle = \langle a, \bar{q} \star b \rangle = \langle q, b \star \bar{a} \rangle$ holds for all quaternions $q, a, b \in \mathbb{H}$ because conjugation reverses Hamilton product order ($\overline{u \star v} = \bar{v} \star \bar{u}$) and $\operatorname{Re}$ is cyclic. Applying this identity to $\Omega_t = g_t \star \Omega_{t-1} + s_t$ immediately transposes left multiplication by $g_{t+1}$ into left multiplication by $\bar{g}_{t+1}$, so $\lambda$ obeys a **time-reversed holonomy scan** using the *exact same Triton kernel* while saving only **1** trajectory tensor ($\Omega_{0:T-1}$) instead of $O(\log_2 T)$ Blelloch merge buffers. **Verified numerically** in `test_holonomy.py` via `torch.autograd.gradcheck` (`float64`, `atol=1e-5`) and CUDA Triton equivalence (`rel_err < 1e-4` at `L=31, 300`).

---

## Theorem 2 — Abelian Group Homomorphism Ceiling ($S_4/A_4 \cong \mathbb{Z}_2$) vs Faithful $2O \subset SU(2) \to SO(3)$ Holonomy

**Part A (Abelian Collapse):** Let $G$ be any non-abelian finite group with non-trivial commutator subgroup $[G, G]$. Any representation or multiplicative state accumulator mapping generators $g \in G$ into an **abelian** Lie group $U(1)^k$ (such as complex diagonal SSMs) satisfies $\phi(xyx^{-1}y^{-1}) = \phi(x)\phi(y)\phi(x)^{-1}\phi(y)^{-1} = 1$, hence must factor through the abelianization $G^{\text{ab}} = G / [G, G]$. For $S_4$, $[S_4, S_4] = A_4$ (order 12), so:
$$S_4^{\text{ab}} = S_4 / A_4 \cong \mathbb{Z}_2 \implies |\phi(S_4)| \le 2,$$
capping uniform state-tracking accuracy over $|S_4|=24$ elements near $\le 2/24 = 8.3\%$ (empirically `3.1%–6.0%`). Standard constant-depth self-attention falls in uniform $TC^0$ and also cannot evaluate iterated non-solvable/non-abelian word problems at OOD horizons (`2.1%–5.2%`).

**Part B (Double-Cover $SU(2) \to SO(3)$ & Lie-Exp Equator Geometry):**
The rotational symmetry group of the cube embeds $S_4 \hookrightarrow SO(3)$. Under the double cover $\pi: SU(2) \twoheadrightarrow SO(3)$, $S_4$ lifts to the **binary octahedral group** $2O \subset SU(2)$ of order $48$, represented faithfully by unit quaternions. However:
1. Every element permutation $P \in S_4 \subset SO(3)$ has **two** antipodal quaternion preimages $\{+q, -q\} \subset SU(2)$, so linear classification directly on $q \in \mathbb{R}^4$ suffers destructive $\pm q$ interference (`H1`: `3.1%–4.4%`).
2. Every transposition and order-2 permutation in $S_4$ lifts to pure imaginary unit quaternions ($w = \operatorname{Re}(q) = 0$), which lie at $\theta = \pi/2$ where the rational Cayley map $\frac{1+i\omega}{1-i\omega}$ has $\omega \to \infty$ coordinate distortion (`H2`: `36.5%` at `L=32`, decaying to `3.6%` at `L=512`).
3. Combining the **Lie Exponential**
   $$q(\omega) = \exp_{\mathfrak{su}(2)}(\omega) = \cos\|\omega\| + \frac{\sin\|\omega\|}{\|\omega\|}\omega$$
   with the **$\mathbb{Z}_2$-Invariant Projective Adjoint** $R(q) \in \mathbb{R}^{3 \times 3}$ ($R(q)v = q v \bar{q}$, satisfying $R(-q) = R(q)$ and $R(q_2 \star q_1) = R(q_2)R(q_1)$) yields an exact, singularity-free homomorphism onto $SO(3)$.

---

## Theorem 3 — Spectral Nullspace Orthogonal Projection & Zero Principal Interference

Let frozen pretrained weight matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ have thin SVD $W_0 = U \Sigma V^\top$, partitioned into dominant singular subspace $V_k \in \mathbb{R}^{d_{\text{in}} \times k}$ (top-$k$ right singular vectors with orthonormal columns $V_k^\top V_k = I_k$) and orthogonal projector:
$$P_\perp = I_{d_{\text{in}}} - V_k V_k^\top, \qquad P_\perp^2 = P_\perp = P_\perp^\top.$$
Defining `SpectralNullspaceLoRALinear` forward residual $\Delta y = \frac{\alpha}{r} B A (P_\perp x)$ gives effective weight update $\Delta W = \frac{\alpha}{r} B A P_\perp$. For **every** trainable update $(A, B)$:
$$\Delta W V_k = \frac{\alpha}{r} B A (I - V_k V_k^\top) V_k = \frac{\alpha}{r} B A (V_k - V_k I_k) \equiv 0.$$
Consequently, any pure input signal aligned with dominant pretrained representations $x \in \operatorname{span}(V_k)$ undergoes **strictly zero distortion** ($W_{\text{eff}} x = W_0 x$), while $\frac{\alpha}{r} B A$ retains full $r$-dimensional expressive rank across the $(d_{\text{in}} - k)$-dimensional residual manifold.

---

## Theorem 4 — Precision-Weighted Free-Energy Predictive-Coding Descent (`PredictiveCodingCell`)

Given normalized representation $z \in \mathbb{R}^d$, latent state $c^{(0)} = \sigma_{\text{silu}}(W_{\text{enc}} z)$ evolves over $K$ refinement steps under Gaussian generative prediction error $\epsilon^{(k)} = \operatorname{RMSNorm}(z - W_{\text{dec}} c^{(k)})$:
$$c^{(k+1)} = c^{(k)} + \sigma(\log \pi) \cdot W_{\text{err}} \epsilon^{(k)}, \qquad y_{\text{PC}}(z) = W_{\text{dec}} c^{(K)}.$$
Up to step-size scaling $\alpha = \sigma(\log \pi)$, when $W_{\text{err}} \approx W_{\text{dec}}^\top$, each step performs preconditioned gradient descent on quadratic variational free energy $\mathcal{F}(c) = \frac{1}{2}\|z - W_{\text{dec}} c\|_2^2$, explaining why iterative error feedback yields consistent BPB and state-tracking gains over feedforward `SwiGLU`.

---

## Theorem 5 — Roofline VRAM, Compute-Time & Bandwidth-Time Selection Model (`ComputePlanner`)

For architecture with $N$ total parameters, $N_{\text{train}}$ trainable parameters in FP32 AdamW ($16\text{ B/param}$ for master copy, grad, $m, v$), $b_w$-byte weights, batch $B$, context $L$, chunk limit $L_{\text{eff}}$, training tokens $D_{\text{tr}}$, decode tokens $D_{\text{dec}}$, hardware peak $\Phi_{\text{gpu}}$ (FLOP/s), and HBM bandwidth $\mathcal{B}_{\text{gpu}}$ (B/s):
$$\text{VRAM}_{\text{peak}} \approx M_{\text{weights}} + 16 N_{\text{train}} + M_{\text{state}}(B, L) + M_{\text{act}}(B, L_{\text{eff}}, N_{\text{train}} > 0) + M_{\text{ctx}},$$
$$T_{\text{wall}} \approx T_{\text{boot}} + \frac{6 N D_{\text{tr}}}{\eta_{\text{mfu}} \Phi_{\text{gpu}}} + \frac{D_{\text{dec}}(M_{\text{weights}} + M_{\text{state}})}{\eta_{\text{bw}} \mathcal{B}_{\text{gpu}}}.$$
Notice why **holonomic state $M_{\text{state}} = O(1)$ dominates Transformers $M_{\text{KV}} = O(L)$** at long sequences: in Decode, $T_{\text{dec}}/{\text{tok}}$ is linear in $M_{\text{weights}} + M_{\text{state}}$, so $O(L)$ KV reads overtake arithmetic cost as $L \to 32,768+$.

---

## Theorem 6 — Deep Iterated Prefix-Product Obstruction, $\mathfrak{su}(2)$ Exponential Splitting, & `res_mix_gate` (`Gen4_Omega_SO3_PredCode`)

In an $L$-layer causal LM, stacking non-abelian $SU(2)$ scans creates iterated non-commutative prefix products:
$$H_t^{(\ell)} = H_{t-1}^{(\ell)} A_t^{(\ell)}\big(X_{1:t}^{(\ell-1)}\big), \qquad A_t^{(\ell)} \in SU(2).$$
Because small perturbations in $H_t^{(0)}$ alter every subsequent step $t$ of layer $\ell \ge 1$ nonlinearly, full residual mixing across deep stacks triggers chaotic prefix-of-prefix interference on non-abelian word problems (unbroken $S_4$), while adaptive optimizers restore downscaled projections in $< 10$ steps.

`HSPAFamily.GEN4_OMEGA_SO3_PREDCODE` resolves this in an end-to-end multi-layer causal LM via three coupled invariants:
1. **Unitary / Episodic Register Bifurcation ($r_0 = \lfloor R/2 \rfloor$):** Holonomy registers $r < r_0$ obey $g_{t,r} = \exp_{\mathfrak{su}(2)}(\omega_{t,r}) \in SU(2)$ (`su2_lie_exponential`) with strictly zero episodic coupling ($s_{t,r} = 0$) and normalized identity anchor (`scan_state[:, :r_0, :, 0] = 1.0`), ensuring exact norm conservation $\|H_{t,r}\|_F = \|H_{0,r}\|_F$. Episodic registers $r \ge r_0$ retain adaptive decay/binding ($s_{t,r} \neq 0$).
2. **Sign-Invariant Projective $SO(3)$ Adjoint Readout:** Mapping unit spinor $q \in S^3$ via `so3_projective_adjoint` to its $3 \times 3$ rotation matrix $R(q) \in SO(3)$ guarantees $\mathbb{Z}_2$ antipodal invariance $R(-q) = R(q)$, eliminating sign-flip discontinuities across readouts.
3. **Scalar Residual Highway Gate (`res_mix_gate`):** Initializing scalar gate $\gamma_0 = 1.0$ on layer $0$ and $\gamma_\ell = 0.0$ on layers $\ell \ge 1$ isolates exact group-state composition at layer $0$ (**`100.00%`** at $L \in \{32,64,128\}$ and **`93.06%`** at $8\times$ OOD $L=256$ in `EXP-Ω4`), while deeper layers engage selectively as gradient signal warrants (`0.3775` vs `0.3824` Transformer BPB, `-14.3%` VRAM vs `Gen1_H2`, and `9.03%` math probe in `EXP-Ω5`; though trailing all-active `Gen1_H2` `0.3701` BPB on short 800-step general text).

---

## Theorem 7 — Zero-Overhead Nullspace Weight Folding & Static CUDA-Graph Decode

Because orthogonal nullspace projection $P_\perp = I - V_k V_k^\top$ is linear, `SpectralNullspaceLoRALinear` forward evaluation
$$y = x W_0^\top + \alpha\, (x P_\perp) A^\top B^\top = x \Big(W_0 + \alpha\, B (A P_\perp)\Big)^\top = x W_{\text{merged}}^\top$$
can be folded **exact to floating-point precision** (`6/6` exact greedy match at `1.747B`) into `base.weight` via `merge()` / `unmerge()` (CPU base backup = `0 MB` extra GPU VRAM), preserving nullspace orthogonality ($\|\Delta W_\perp V_k\|_F / \|\Delta W_\perp\|_F \le 7\times 10^{-7}$). Eliminating dual-path GEMM + Python kernel-launch overhead via static-buffer `torch.cuda.CUDAGraph` (`StaticGraphHSPADecoder`, $L_{\text{dec}}=1$) yields **`7.55x`** nullspace linear speedup (`252.24 -> 33.42 us`), **`1.61x`** end-to-end HF decode speedup (`8.73 -> 14.09 tok/s`), **`3.10x`** on `1.808B` (`158.413 -> 51.023 ms/tok`), and **`2.35x`** on flagship `3.841B <= 4B` (`236.105 -> 100.291 ms/tok`).

---

## Theorem 8 — Discrete Port-Hamiltonian Sheaf Interaction Stability & Givens Symplectic Contraction (`first_principles.py`, `PortHamiltonianFFNCell`)

Let node states $x \in \mathbb{R}^{N \times d}$ evolve over a typed bitemporal interaction graph $G_t = (V, E_t)$ with $SO(3)^{d/3}$ edge transport maps $P_{ij}$ and discrete Port-Hamiltonian update:
$$x_{k+1} = \exp\big(\Delta t (J - R)\big) x_k + \Delta t\, B u_k, \qquad J^\top = -J, \quad R \succeq 0.$$
Because $J$ is skew-symmetric ($v^\top J v = 0$) and $R$ is positive semidefinite ($v^\top R v \ge 0$), the quadratic Hamiltonian $\mathcal{H}(x) = \frac{1}{2}\|x\|_2^2$ obeys the strict autonomous ($u_k = 0$) Lyapunov contraction invariant:
$$\mathcal{H}(x_{k+1}) - \mathcal{H}(x_k) \le 0 \quad \forall k \ge 0.$$
In `PortHamiltonianFFNCell`, partitioning channels into planar pairs $(q_j, p_j)$ with exact planar Givens symplectic rotation $G(\theta_j) \in SO(2)$ and strictly positive damping $\gamma_j = \operatorname{softplus}(\alpha_j) > 0$ enforces $\|G(\theta_j) e^{-\gamma_j \Delta t}\|_2 = e^{-\gamma_j \Delta t} < 1$, preventing recurrent energy blow-up across arbitrary depths (`autonomous_lyapunov_max_energy_delta = 0.0` verified in `EXP-FP1` and `test_first_principles.py`).

---

## Theorem 9 — Information-Bottleneck Causal Source-Copy Duality (`CausalSourceCopyGate`)

Let a finite-state recurrent backbone compress history $X_{1:t}$ into bounded state $S_t \in \mathbb{R}^{d_s}$ ($d_s \ll t \log |V|$). By the Data Processing Inequality, for arbitrary key-value facts with entropy exceeding $H(S_t)$, any purely recurrent readout suffers irreducible exact-copy distortion (`0.00%` observed across `B0_Transformer` without copy anchor, `B2_Gated_DeltaNet`, `Gen1_H2`, and `Gen4` in `EXP-FP2B`, and `0.48%` in `gen5-fp-nocopy-micro`). Augmenting the readout with an $O(d)$-parameter causal source-copy gate over admissible ledger/prefix tokens $\alpha_t = \operatorname{softmax}_{\text{fp32}}(q_t^\top K_{1:t} / \sqrt{d_h})$:
$$P_{\text{out}}(y \mid X_{1:t}) = (1 - \lambda_t) P_{\text{gen}}(y \mid h_t) + \lambda_t \sum_{s \le t} \alpha_{t,s}\, \mathbb{I}[x_s = y]$$
decouples **lossy state-space reasoning** ($P_{\text{gen}}$) from **lossless verbatim transport** ($\sum \alpha_{t,s}\mathbb{I}[x_s=y]$). Empirically in `EXP-FP2B`, this `+2 d d_h + d + 1` gate lifted bitemporal exact value recall **`201x`** (`0.48% -> 96.67%`) while also lowering validation BPB (`0.3789 -> 0.3756`).
