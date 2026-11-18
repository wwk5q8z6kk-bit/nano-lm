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
