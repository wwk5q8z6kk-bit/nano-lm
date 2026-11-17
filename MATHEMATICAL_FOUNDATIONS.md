# Mathematical Foundations of `HSPA`

All algebraic primitives below are implemented in [spinor_algebra.py](nanolm_holonomy/spinor_algebra.py) and cell transitions in [architecture.py](nanolm_holonomy/architecture.py), verified programmatically to `<= 1e-5` numerical equivalence in [test_holonomy.py](nanolm_holonomy/tests/test_holonomy.py).

---

## 1. Non-Abelian $SU(2)$ Spinor Representation & Hamilton Gauge Algebra

A hidden fiber width $d_{\text{fiber}} = 4G$ is partitioned into $G$ quaternions $p = (p_0, p_1, p_2, p_3) \in \mathbb{H}^G \cong \mathbb{R}^{G \times 4}$.

### 1.1 Exact Hamilton Product & Conjugate Unbinding
Given two spinors $p, q \in \mathbb{H}$:
$$p \star q = \begin{pmatrix} p_0 q_0 - p_1 q_1 - p_2 q_2 - p_3 q_3 \\ p_0 q_1 + p_1 q_0 + p_2 q_3 - p_3 q_2 \\ p_0 q_2 - p_1 q_3 + p_2 q_0 + p_3 q_1 \\ p_0 q_3 + p_1 q_2 - p_2 q_1 + p_3 q_0 \end{pmatrix}$$
- **Strict Non-Commutativity:** $p \star q \neq q \star p$ generically in the $(i,j,k)$ vector components.
- **Associativity:** $(a \star b) \star c = a \star (b \star c)$ holds identically, enabling exact parallel prefix scans.
- **Norm Preservation:** $\|p \star q\|_2 = \|p\|_2 \|q\|_2$.
- **Exact Unbinding via Conjugate $\bar{q} = (q_0, -q_1, -q_2, -q_3)$:**
  $$\overline{H \star q} \star (H \star q \star v) = \bar{q} \star \bar{H} \star H \star q \star v = \|H\|_2^2 \|q\|_2^2 v.$$

### 1.2 Algebraic Cayley Lie Map ($\mathfrak{su}(2) \to SU(2)$)
Instead of trigonometric $\exp(\boldsymbol{\omega})$ calls, `cayley_rotor` maps any unconstrained generator vector $\boldsymbol{\omega} \in \mathbb{R}^3$ smoothly to $SU(2)$:
$$u(\boldsymbol{\omega}) = \frac{(1, \, \omega_1, \, \omega_2, \, \omega_3)}{\sqrt{1 + \omega_1^2 + \omega_2^2 + \omega_3^2}}, \qquad \|u(\boldsymbol{\omega})\|_2 \equiv 1, \quad u(\mathbf{0}) = (1,0,0,0).$$

---

## 2. Parallel $\lceil \log_2 T \rceil$ Non-Abelian Holonomy Scan & $O(1)$ Decode

Across time $t = 1, \dots, T$, each register $r \in \{1, \dots, R\}$ evolves via left-acting affine $SU(2)$ transport:
$$\Omega_{t,r} = g_{t,r} \star \Omega_{t-1,r} + s_{t,r},$$
where
$$g_{t,r} = \max(10^{-4}, \, 1 - w_{t,r} \beta_{t,r}) \, u\big(m_{t,r} \boldsymbol{\omega}_{t,r}\big), \qquad s_{t,r} = w_{t,r} \big(u(\boldsymbol{\rho}_{t,r}) \star \phi_t\big).$$

### Associative Composition Law
Composing transition tuple $A_1 = (g_1, s_1)$ followed temporally by $A_2 = (g_2, s_2)$ yields:
$$A_2 \circ A_1 = \big(g_2 \star g_1, \; g_2 \star s_1 + s_2\big).$$
Crucially, **left-multiplication order $(g_2 \star g_1)$ must be strictly preserved** because $SU(2)$ multiplication is non-commutative. The binary Hillis-Steele prefix scan evaluates all $T$ tokens in $\lceil \log_2 T \rceil$ vectorized GPU tensor operations during training/prefill (`max_abs_diff < 1e-5` vs sequential recurrence), while incremental generation updates a constant-shape register tensor $\Omega_r \in \mathbb{R}^{G \times 4}$ in exact $O(1)$ memory per token (`step_holonomy`).

### Conjugate Query Unbinding & Competitive Readout
At query time $t$, register $r$ is rotated into the query's local gauge frame via $\bar{q}_{t,r} = \overline{u(\mathbf{q}_{t,r})}$ and pooled by soft read distribution $\pi_t \in \Delta^{R-1}$:
$$y_t = W_o \, \text{RMSNorm}\!\left(\sum_{r=1}^R \pi_{t,r} \big(\bar{q}_{t,r} \star \Omega_{t,r}\big)\right).$$

---

## 3. Matched Abelian $U(1)$ Control Formulation

To isolate non-abelian $SU(2)$ order sensitivity from general complex-phase dynamics at identical real state width ($4G$ reals $= 2G$ complex scalars), `B1_Abelian_SSM` sets $\mathbb{C}^{2G}$ fibers with scalar angles $\theta_{t,r} \in \mathbb{R}^{2G}$ and commutative multiplication $(p_0 q_0 - p_1 q_1, \, p_0 q_1 + p_1 q_0)$. Here $u(\theta_2) u(\theta_1) \equiv u(\theta_1) u(\theta_2)$, making consecutive rotations over an unweighted interval order-invariant.

---

## 4. Precision-Weighted Predictive-Coding Cell (`PredictiveCodingCell`)

Given post-mixer normalized residual $z \in \mathbb{R}^d$, `PredictiveCodingCell` uses the exact same `3`-matrix parameter count (`3 * d * d_pc`) as SwiGLU (`gate`, `up`, `down`):
1. Initial bottom-up latent state: $c^{(0)} = \text{SiLU}(W_{\text{enc}} z) \in \mathbb{R}^{d_{\text{pc}}}$.
2. Iterative top-down free-energy revision for $k = 0, \dots, K-1$ (`K = 2`):
   $$\varepsilon^{(k)} = z - W_{\text{dec}} c^{(k)}, \qquad \gamma_k = \sigma\!\left(\log \tau + \log \sqrt{\tfrac{1}{d}\|\varepsilon^{(k)}\|_2^2 + 10^{-6}}\right),$$
   $$c^{(k+1)} = c^{(k)} + \gamma_k \, \text{SiLU}\!\big(W_{\text{err}} \, \text{RMSNorm}(\varepsilon^{(k)})\big).$$
3. Output projection: $\text{PC}(z) = W_{\text{dec}} c^{(K)}$.

Intuitively, tokens whose top-down reconstruction $W_{\text{dec}} c^{(k)}$ exhibits large surprise norm $\|\varepsilon^{(k)}\|_{\text{rms}}$ receive larger correction step $\gamma_k$, concentrating nonlinear refinement on unpredictable reasoning/state-transition tokens while keeping parameter count constant.
