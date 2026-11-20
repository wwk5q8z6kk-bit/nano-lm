# Mathematical Foundations of `HSPA` & `HSPA-Ω`

Full proofs and derivations are unified here and in [NANOLM_MATHEMATICAL_FORMULATIONS.md](NANOLM_MATHEMATICAL_FORMULATIONS.md).

---

## 1. Non-Abelian Parallel Holonomy Recurrence
Given $g_{t,r} = (1 - w_{t,r}\beta_{t,r}) q(\boldsymbol{\omega}_{t,r}) \in \mathbb{H}^G$ and $s_{t,r} = w_{t,r} (q(\boldsymbol{\rho}_{t,r}) \star \phi_t) \in \mathbb{H}^G$, transport follows associative Hamilton-product composition:
$$(g_2, s_2) \circ (g_1, s_1) = \big(g_2 \star g_1,\; g_2 \star s_1 + s_2\big).$$
Both chunked parallel prefill and $O(1)$-memory single-token decode evaluate the identical mathematical trajectory (`max_abs_diff < 2.2e-6`).

## 2. Exact Quaternion Reverse-Mode Adjoint Theorem (`triton_holonomy.py`)
For $\Omega_t = g_t \star \Omega_{t-1} + s_t$ under Euclidean upstream gradients $\delta_t = \partial \mathcal{L} / \partial \Omega_t$:
$$\lambda_t = \bar{g}_{t+1} \star \lambda_{t+1} + \delta_t \quad (\lambda_T = 0), \qquad \frac{\partial \mathcal{L}}{\partial s_t} = \lambda_t, \quad \frac{\partial \mathcal{L}}{\partial g_t} = \lambda_t \star \bar{\Omega}_{t-1}, \quad \frac{\partial \mathcal{L}}{\partial \Omega_{-1}} = \bar{g}_0 \star \lambda_0.$$
Because Hamilton conjugation satisfies $\overline{u\star v} = \bar{v}\star\bar{u}$, reversing time and conjugating $g$ converts the exact VJP into a second pass of `_triton_holonomy_fwd_kernel` while saving only **1** trajectory tensor instead of $\log_2 T$ tree buffers (`gradcheck float64 atol=1e-5` & CUDA `rel_err < 1e-4` verified).

## 3. Abelian $S_4 / A_4 \cong \mathbb{Z}_2$ Ceiling vs $2O \subset SU(2) \to SO(3)$ Projective Holonomy
1. **Abelian Ceiling:** Any homomorphism into commutative $U(1)^k$ factors through $S_4 / [S_4, S_4] = S_4/A_4 \cong \mathbb{Z}_2$, collapsing `24`-element group state tracking to `~1/12..1/24` (`5.99%` observed).
2. **Double-Cover & Cayley Singularities:** Direct readout of $q \in SU(2)$ conflates antipodal fibers $\pm q \mapsto P \in SO(3)$ (`4.43%`), while the Cayley chart distorts $\theta=\pi/2$ involutions (`w=0`, decaying `36.46% -> 3.64%` over `L=32..512`).
3. **Lie-Exp $SO(3)$ Solution:** Pairing $\exp_{\mathfrak{su}(2)}(\omega) = \cos\|\omega\| + \frac{\sin\|\omega\|}{\|\omega\|}\omega$ with $R(q) \in \mathbb{R}^{3\times 3}$ ($R(-q)=R(q)$ and $R(q_2\star q_1)=R(q_2)R(q_1)$) achieves **`100.00% ± 0.00%` up to `16x` OOD (`L=512`)**.

## 4. Spectral Nullspace Orthogonal LoRA Projection
With frozen pretrained SVD top-$k$ right singular matrix $V_k \in \mathbb{R}^{d_{\text{in}}\times k}$ ($V_k^\top V_k = I_k$),
$$P_\perp = I - V_k V_k^\top, \qquad \Delta W = \frac{\alpha}{r} B A P_\perp \implies \Delta W V_k \equiv 0,$$
orthogonally isolating continual updates (`< 1e-5` verified leakage) from high-energy pretrained features.

## 5. Precision-Weighted Predictive Coding
$$c^{(k+1)} = c^{(k)} + \sigma(\log \pi)\, W_{\text{err}} \operatorname{RMSNorm}\big(z - W_{\text{dec}} c^{(k)}\big), \qquad y_{\text{PC}}(z) = W_{\text{dec}} c^{(K)}.$$

## 6. Deep Unitary/Episodic Split & $\mathbb{Z}_2$-Invariant $SO(3)$ Holonomy (`Gen4_Omega_SO3_PredCode`)
With register split $r_0 = \lfloor R/2 \rfloor$, unitary holonomy $g_{t,r} = \exp_{\mathfrak{su}(2)}(\omega_{t,r})$ ($s_{t,r}=0$ for $r < r_0$, $H_{0,r}=I_2$), projective readout $R(q) = R(-q) \in SO(3)$, and scalar residual gate $\gamma_0 = 1, \gamma_{\ell \ge 1} = 0$, a full multi-layer causal LM achieves `100.00%` exact $S_4$ state match through `4x` OOD (`L128`) and `93.06%` at `8x` OOD (`L256`) without auxiliary probes.

## 7. Exact Zero-Overhead Nullspace Weight Folding
$$\Delta W_\perp = \alpha\, B (A (I - V_k V_k^\top)), \qquad W_{\text{merged}} = W_0 + \Delta W_\perp, \qquad \|\Delta W_\perp V_k\|_F / \|\Delta W_\perp\|_F \le 7\times 10^{-7}.$$
Folding $\Delta W_\perp$ into `base.weight` matches unmerged outputs (`6/6` at `1.747B`) while lifting linear GEMMs `7.55x`, end-to-end HF decode `1.61x`, and static `CUDA Graph` HSPA decode `3.10x` (`1.808B`) / `2.35x` (`3.841B <= 4B`).
