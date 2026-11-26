# NANOLM Mathematical Foundations

This document states the exact mathematical formulations implemented in `nanolm_discovery/` (`operators.py`, `models.py`, `optim.py`, `config.py`), along with stability bounds, complexity scalings, and exact parallel–recurrent equivalence identities.

---

## 1. Invention 1 (`I1_PHMN`): Port-Hamiltonian Metriplectic Operator & Symplectic Synthesis

### 1.1 State Transition Decomposition (Conservative + Dissipative Ports)
Let each head $h \in \{1, \dots, H\}$ maintain a persistent matrix memory $S_t \in \mathbb{R}^{d_h \times d_h}$ ($d_h = D / H$). Rather than using unconstrained additive decay or strictly contractive delta updates, `PortHamiltonianOperator` decomposes each token step $t$ into an **asymmetric causal key-shift addressing signal** ($\tilde{x}_t$), a **Hamiltonian/group transport microstep**, and a **dissipative precision-weighted associative port**:

$$\tilde{x}_t = \operatorname{SiLU}\!\left(\sum_{j=0}^{3} c_j \odot x_{t-j}\right), \qquad u_t = \frac{W_u \tilde{x}_t}{\|W_u \tilde{x}_t\|_2}, \qquad k_t = \frac{W_k \tilde{x}_t}{\|W_k \tilde{x}_t\|_2}, \qquad v_t = W_v x_t, \qquad q_t = W_q \tilde{x}_t$$

Separating shifted addressing $\tilde{x}_t$ (`CausalKeyShift`, $4d$ params, $6d$-byte `bf16` ring state) from unshifted value $v_t$ enables single-layer induction binding (`(QUERY, key) -> val`) without mixing neighboring tokens' value payloads.

$$\beta_t = 1.98\,\sigma(g_{t,0}) \in (0, 1.98), \qquad \tau_t = \sigma(g_{t,1}) \in (0, 1), \qquad \gamma_t = \exp\!\big(-0.20\,\operatorname{softplus}(g_{t,2})\big) \in (0, 1]$$

The exact single-token state update is:

$$S_{t}^{\text{rot}} = (I - \beta_t u_t u_t^\top)\, S_{t-1}$$

$$S_t = \gamma_t\,(I - \tau_t k_t k_t^\top)\, S_{t}^{\text{rot}} + \tau_t k_t v_t^\top, \qquad y_t = S_t^\top q_t$$

### 1.2 Theorem 1 (Unconditional Spectral Stability)
Let the homogeneous state transition matrix at step $t$ be:

$$A_t = \gamma_t\,(I - \tau_t k_t k_t^\top)(I - \beta_t u_t u_t^\top)$$

Because $\|u_t\|_2 = 1$, $(I - \beta_t u_t u_t^\top)$ has eigenvalues $\{1 - \beta_t, 1, \dots, 1\}$ aligned with orthonormal eigenspaces, so for all $\beta_t \in (0, 2)$:

$$\|I - \beta_t u_t u_t^\top\|_2 = \max(|1 - \beta_t|, 1) = 1$$

Notably, when $\beta_t > 1$, $\det(I - \beta_t u_t u_t^\top) = 1 - \beta_t < 0$, allowing **orientation-reversing reflections** (required to represent parity $\mathbb{Z}_2$ and non-Abelian dihedral/symmetric groups such as $S_3$) while preserving unit spectral norm. Similarly, for $\tau_t \in (0, 1)$ and $\|k_t\|_2 = 1$, $\|I - \tau_t k_t k_t^\top\|_2 = 1$. By submultiplicativity:

$$\|A_t\|_2 \le \gamma_t\,\|I - \tau_t k_t k_t^\top\|_2\,\|I - \beta_t u_t u_t^\top\|_2 \le \gamma_t \le 1$$

unconditionally for every input token and parameter configuration (`test_unconditional_spectral_stability_of_metriplectic_operators`).

*Scope caveat:* As proven in the mechanism literature (`MATHEMATICAL_RESEARCH.md` OP02), this guarantees stability of the fixed-parameterized *linear homogeneous state transition* $S_{t-1} \mapsto S_t$; the full nonlinear token Jacobian $\partial S_t / \partial x$ can still have norm $> 1$ via input-dependent gate derivatives unless gradient-clipped during optimization.

### 1.3 Theorem 2 (Exact Nilpotent WY Chunk Factorization with Scalar Decay)
Computing $S_t$ step-by-step in Python over sequence length $T$ is hardware-inefficient. By interleaving the two microsteps per token into a virtual stream of length $M = 2T$:

- Even step $m = 2t$: $(w_{2t}, v_{2t}, q_{2t}, b_{2t}, \log g_{2t}) = (u_t, 0, 0, \beta_t, 0)$
- Odd step $m = 2t+1$: $(w_{2t+1}, v_{2t+1}, q_{2t+1}, b_{2t+1}, \log g_{2t+1}) = (k_t, v_t, q_t, \tau_t, \log \gamma_t)$

every microstep satisfies $S_m = g_m S_{m-1} + w_m z_m$ with rank-1 residual $z_m = b_m (v_m^\top - g_m w_m^\top S_{m-1})$. Partitioning $M$ into chunks of size $C$ ($C=32$, i.e. $16$ tokens/chunk under 2-microstep interleaving) with intra-chunk cumulative decay $\Gamma_{j \to m} = \exp(\sum_{r=j+1}^m \log g_r)$ ($\Gamma_{a\to a}=1$), unrolling from chunk entry $S_{\text{init}}$ yields the exact lower-triangular system:

$$(I - L)\,Z = b \odot V - (b \odot W \odot \Gamma_{-1 \to \cdot})\,S_{\text{init}}, \qquad L_{m, j} = \begin{cases} -b_m \Gamma_{j\to m}(w_m^\top w_j), & j < m \\ 0, & j \ge m \end{cases}$$

Because $L \in \mathbb{R}^{C \times C}$ is **strictly lower triangular**, $L^C = 0$ (nilpotent). Therefore $(I - L)^{-1}$ has an **exact closed-form binary-power product** in $\log_2 C$ matrix multiplications:

$$(I - L)^{-1} = \prod_{k=0}^{\lceil \log_2 C \rceil - 1} (I + L^{2^k})$$

Implemented with `autocast` disabled (`_no_autocast_ctx`, forcing exact `float32` matmul/cumsum inside mixed-precision `bfloat16` loops so MPS `bfloat16` downcasts never corrupt the nilpotent product) in `wy_metriplectic_chunk_scan`, this achieves machine-precision duality with token-by-token `forward_step()` (`max_abs_err < 1e-4` verified across all tests) while executing via batched GEMMs.

### 1.4 Theorem 3 (Finite-Precision WY Condition-Number Bound for Reflection Gates)
Although forward state evolution $\|A_t\|_2 \le 1$ is unconditionally unitary-or-contractive for $\beta_t \in (0, 2)$, the **intermediate chunk triangular solve** $(I - L)^{-1}$ can amplify entries exponentially when adjacent microsteps align negatively under large reflection step sizes $\beta_{\max} \approx 1.98$. In the worst case $|L_{m,j}| \le \beta_{\max}$, consecutive elimination steps yield $|[(I - L)^{-1}]_{C-1, 0}| \le \beta_{\max}(1 + \beta_{\max})^{C - 2}$. Because automatic differentiation backpropagates quadratic combinations of entries of $(I - L)^{-1}$, `float32` backward gradients exceed IEEE-754 dynamic range ($3.4 \times 10^{38}$) above:

$$C_{\text{crit}} \approx 1 + \frac{\ln(3.4 \times 10^{38})}{2\ln(1 + \beta_{\max})} \approx 41.6 \qquad (\beta_{\max} = 1.98)$$

At $C = 64$, aligned sequences overflow (`gnorm = 1,123.8 -> NaN` observed at step `11` on `seed=137` of `i1_phmn_discovery_18m`); enforcing **$C = 32 < 41.6$** keeps peak amplification below $2.98^{62} \approx 2.5 \times 10^{29} \ll 3.4 \times 10^{38}$, guaranteeing finite gradients (`max_gnorm = 6.06`) while also speeding up MPS backward execution by `28%`.

### 1.5 Symplectic Hamiltonian Phase-Space Synthesis
`SymplecticHamiltonianSynthesis` replaces the static SwiGLU channel MLP by partitioning $x = (q, p) \in \mathbb{R}^{d/2} \times \mathbb{R}^{d/2}$ and evolving one damped leapfrog integrator step:

$$q_{1/2} = q + \tfrac{1}{2} W_{\text{kin,out}}\operatorname{silu}(W_{\text{kin,in}}\,p), \qquad f(q_{1/2}) = -W_{\text{pot,out}}\operatorname{silu}(W_{\text{pot,in}}\,q_{1/2})$$

$$p' = (1 - \tfrac{1}{2} \sigma(W_{\text{damp}}\,x)) \odot p + f(q_{1/2}), \qquad q' = q_{1/2} + \tfrac{1}{2}\Delta q, \qquad y = W_{\text{out}}([q', p'] \odot m(u_t))$$

where $m(u_t)$ optionally modulates phase amplitude using epistemic uncertainty $u_t$.

---

## 2. Invention 2 (`I2_NEBN`): Natural-Parameter Epistemic Belief Network

### 2.1 Conjugate Exponential-Family Filtering Dual
`NEBNOperator` models sequence aggregation as Bayesian exponential-family evidence accumulation with closed-form predictive uncertainty. Given positive feature map $\phi(x) = \sqrt{d_h}\,\frac{\operatorname{elu}(x)+1}{\|\operatorname{elu}(x)+1\|_2}$, observation precision $\tau_t = \operatorname{softplus}(g_{t,1}) > 0$, retention $\gamma_t = \exp(-0.25\operatorname{softplus}(g_{t,0})) \in (0, 1]$, and learned prior precision $\lambda_0 = \operatorname{softplus}(\theta_0) + 10^{-2} > 0$:

**Recurrent $O(1)$-state inference form:**
$$\mathcal{H}_t = \gamma_t \mathcal{H}_{t-1} + (\tau_t \phi(k_t))\,v_t^\top \in \mathbb{R}^{d_h \times d_h}, \qquad \mathcal{P}_t = \gamma_t \mathcal{P}_{t-1} + \tau_t \phi(k_t) \in \mathbb{R}^{d_h}, \qquad \ell_t = \ell_{t-1} + \log \gamma_t$$

$$\Lambda_t(q_t) = \phi(q_t)^\top \mathcal{P}_t + \lambda_0 e^{\ell_t} + 10^{-4}, \qquad \mu_t = \frac{\phi(q_t)^\top \mathcal{H}_t}{\Lambda_t(q_t)}, \qquad u_t = \frac{1}{\Lambda_t(q_t)}$$

**Exact parallel chunk-GEMM training form** (`nebn_dual_filter`, chunk size $C=128$):
Within each chunk $c$, define intra-chunk log-retention $\Gamma_{t,j} = \exp(\min(0, \text{csum}_t - \text{csum}_j)) \mathbf{1}_{j \le t}$:

$$S^{\text{intra}}_{t,j} = \big(\phi(q_t)^\top (\tau_j \phi(k_j))\big)\,\Gamma_{t,j}$$

$$\mu_t = \frac{\sum_{j \in c, j \le t} S^{\text{intra}}_{t,j} v_j + \alpha_t \phi(q_t)^\top \mathcal{H}_{\text{prev}}}{\sum_{j \in c, j \le t} S^{\text{intra}}_{t,j} + \alpha_t \phi(q_t)^\top \mathcal{P}_{\text{prev}} + \lambda_0 e^{\ell_{\text{prev}} + \text{csum}_t} + 10^{-4}}$$

Since $\log \gamma_t \le 0$ everywhere, all exponents remain $\le 0$, guaranteeing overflow-free numerical stability and `< 1e-4` equivalence to `forward_step()`. Unlike softmax attention, $\Lambda_t(q_t)$ preserves unnormalized evidence counts, so **unseen or out-of-distribution queries drive $u_t \uparrow$ without post-hoc calibration**.

### 2.2 Variational Dendritic Synthesis
`VariationalDendriticSynthesis` uses $u_t$ directly via gating vector $g_{\text{unc}} = \sigma(W_u u_t + b_u)$:

$$y_t = W_{\text{down}}\Big(g_{\text{unc}} \odot \big(\operatorname{silu}(W_{\text{gate}} x_t) \odot W_{\text{up}} x_t\big) + (1 - g_{\text{unc}}) \odot W_{\text{up}} x_t\Big)$$

---

## 3. Invention 3 (`I3_HRWN`): Causal Dyadic Waves & `SCAM` Fixed-Slot Memory

### 3.1 Fundamental Capacity Motivation (Rate–Distortion Lower Bound)
Any fixed-size state matrix $S_t \in \mathbb{R}^{H \times d_h \times d_h}$ encoded with bounded precision spans at most $O(H d_h^2)$ real degrees of freedom; storing $N$ dense arbitrary verbatim key-value associations beyond this rank capacity suffers destructive interference when $N > \operatorname{rank}_{\text{eff}}(S)$. To circumvent catastrophic interference in constant asymptotic cache space ($O(K)$ bytes as $T \to \infty$), `SCAM` (Surprise-Addressed Content-Addressable Memory) maintains $K$ explicit content-addressable slots ($K=32$ by default).

### 3.2 Strictly Causal Top-$K$ Register Eviction & Wave Binding
- **Causal dyadic wave bank:** `HRWNOperator` applies four causal dilated depthwise 1D convolutions ($d \in \{1, 4, 16, 64\}$, kernel size $5$, max receptive history $256$) plus phase-conjugate demodulation $\bar{w}_t \odot \tanh(r_t) + i_t \odot \tanh(\phi_t)$.
- **Surprise scoring:** $s_t = \|x^{\text{re}}_t - \bar{w}_{t-1}\|_2 + \operatorname{softplus}(w_s^\top x_t + b_s)$.
- **Exact chunk-parallel / step duality (`scam_causal_topk_attention` $\leftrightarrow$ `scam_step_attention`):** At every causal position $t$, both parallel training and step-by-step decoding select the **exact top-$K$ indices from $\{0, \dots, t\}$ sorted by $s_j + 10^{-6} j$**, guaranteeing zero future leakage and `< 1e-6` parallel–recurrent equivalence.

---

## 4. Cycle 2 Synthesis (`G2_MPRN`): Metriplectic Precision-Resonant Network

`G2_MPRN` integrates four complementary mathematical mechanisms into one coherent block (`MPRNOperator` + uncertainty-modulated `SymplecticHamiltonianSynthesis`):
1. **Asymmetric 4-tap `CausalKeyShift` addressing:** Separates bigram/delimiter-aware addressing `(q, k, u, scam_q, scam_k)` computed on $\tilde{x}_t$ from clean payload values `(v, scam_v)` on $x_t$.
2. **Non-commutative + dissipative geometric state (`WY` Metriplectic Scan):** Preserves finite-group tracking ($S_3, \mathbb{Z}_2$) with unconditional $\|A_t\|_2 \le \gamma_t \le 1$.
3. **Subspace-energy epistemic calibration (`nebn_precision_trace`):** Propagates positive natural precision $(\mathcal{P}_t, \ell_t)$ in `float32` to obtain $u_t = 1/\Lambda_t(q_t)$, rescales state readouts via $\frac{2}{1 + \operatorname{clamp}(u_t, \max=10)}$, and normalizes the emitted uncertainty by retrieved subspace energy:

$$u_t^{\text{subspace}} = \frac{u_t}{\sqrt{\frac{1}{d_h}\|y_t^{\text{met}}\|_2^2} + 10^{-2}}$$

Unlike raw positive-orthant precision $u_t$ alone (which can drop spuriously if an unseen OOD query embedding aligns with the historical mean key vector $\bar{\mathcal{P}}_t$), $y_t^{\text{met}} = S_t^\top q_t$ vanishes on orthogonal/unseen queries, driving $u_t^{\text{subspace}}$ upward on out-of-distribution tokens.
4. **Epistemic-surprise `SCAM` register bank ($K=32$ slots):** Ranks admission by learned projection + epistemic uncertainty $\alpha_t + \bar{u}_t^{\text{subspace}}$, storing high-uncertainty / high-surprise associations verbatim in constant $O(K)$ memory.

---

## 5. Optimization & Training Objective (`MuonAdamW` + `JEP-CE`)

- **`MuonAdamW`:** Internal 2D linear operators/synthesis matrices ($M \in \mathbb{R}^{r \times c}$, $\min(r, c) \ge 8$, excluding embeddings) update via 5-step Newton–Schulz matrix orthogonalization ($a=3.4445, b=-4.7750, c=2.0315$) applied to velocity buffers, while embeddings, 1D RMSNorms, and bias vectors update via decoupled `AdamW`.
- **`JEP-CE` (`jep_weight > 0`):** Adds a training-only linear latent predictor $P_{\text{jep}} \in \mathbb{R}^{d \times d}$ (excluded from deployed parameter census and removed before inference) predicting the stop-gradient normalized multi-horizon future embedding target $\bar{e}_t = \operatorname{normalize}(\tfrac{1}{2}(e_{t+1} + e_{t+2}))$ weighted by precision certainty $w_t \propto (1 + \bar{u}_t)^{-1}$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{jep}}\, \mathbb{E}_t\!\left[w_t \Big(1 - \big\langle \widehat{P_{\text{jep}}(h_t)},\ \bar{e}_t \big\rangle\Big)\right]$$

---

## 6. Asymptotic Complexity & Persistent State Scaling

| Paradigm | Training Parallel Time (Length $T$) | Inference Step Time | Persistent State Bytes @ Context $T$ (`bf16` KV / `fp32` state) | Epistemic Signal $u_t$ |
|---|---|---|---|---|
| `B0_TRANSFORMER` | $O(T^2 d + T d^2)$ | $O(T d_{\text{kv}} + d^2)$ | $O(L \cdot T \cdot n_{\text{kv}} d_h)$ (grows linearly with $T$) | None native |
| `B1_GDN2_HYBRID` | $O(T d^2 + T C^2 H + \frac{1}{4} T^2 d)$ | $O(H d_h^2 + \frac{1}{4} T d_{\text{kv}})$ | $\tfrac{3}{4} O(1) + \tfrac{1}{4} O(T)$ | None native |
| `I1_PHMN` | $O(T d^2 + T C H d_h + T C^2 H)$ | $O(L H d_h^2 + L d^2)$ | **$O(L H d_h^2)$ constant in $T$** | None native |
| `I2_NEBN` | $O(T d^2 + T C H d_h)$ | $O(L H d_h^2 + L d^2)$ | **$O(L H (d_h^2 + d_h))$ constant in $T$** | Exact $\Lambda_t^{-1}$ |
| `I3_HRWN` | $O(T d^2 + T d K_{\text{scam}})$ | $O(L d^2 + L d K_{\text{scam}})$ | **$O(L d_{\text{win}} d + L K_{\text{scam}} d)$ constant in $T$** | Lagged wave surprise |
| `G2_MPRN` | $O(T d^2 + T C^2 H + T d K_{\text{scam}})$ | $O(L H d_h^2 + L d K_{\text{scam}})$ | **$O(L (H d_h^2 + K_{\text{scam}} d))$ constant in $T$** | Exact $\Lambda_t^{-1}$ + SCAM score |
| `G3_HMWR` | $O(T d^2 + T C^2 H_{\text{met}} + T d K_{\text{scam}})$ | $O(L H_{\text{met}} d_h^2 + L d_{\text{win}} + L d K_{\text{scam}})$ | **$O(L (H_{\text{met}} d_h^2 + d_{\text{win}} d + K_{\text{scam}} d))$ constant in $T$** | $s_t / (\|y_{\text{mem}}\|_{\text{rms}} + 10^{-3})^{1/2}$ |

---

## 7. Cycle 3 Evolution (`G3_HMWR`): Resolving Parameter Starvation via Tri-Scale Synthesis & Depth-Attention Residuals

### 7.1 Operator–Synthesis Allocation Bound (Cycle 2 Diagnosis)
In an iso-parameter regime matching `B0_TRANSFORMER` (whose per-layer parameter budget at width $d$ is $P_{\text{op}}^{(B_0)} + P_{\text{syn}}^{(B_0)} = (2 + 2 r_{\text{kv}}) d^2 + 3 d d_{\text{ffn}}$, e.g. $3 d^2 + 6 d^2 = 9 d^2$ at `discovery_18m`), stacking a full-width WY+SCAM `MPRNOperator` ($7.0 d^2$) with `SymplecticHamiltonianSynthesis` fixed projections ($1.5 d^2$ before FFN) yields:

$$P_{\text{fixed}}^{(G_2)} \approx 8.5 d^2 \implies d_{\text{ffn}}^{(G_2)} \approx \frac{9 d^2 - 8.5 d^2}{2 d} \approx 0.167 d \quad (d_{\text{ffn}} = 64 \text{ at } d=384;\ 32 \text{ at } d=576)$$

Channel synthesis width is starved by `12x` at `scale_42m` (`32` vs `1152`), crippling natural-language expressivity on real multi-domain corpora (`2.5161` vs `2.4520` BPB) despite winning algorithmic extrapolation (`0.751`).

### 7.2 `G3_HMWR` (`HMWROperator` + `DiscoveryBlock` Depth-Attention & Adaptive Synthesis)
`G3_HMWR` reduces leading operator complexity from $7.0 d^2$ to **$6.5 d^2$** while uniting all three verified architectural primitives under shared `CausalKeyShift` addressing $\tilde{x}_t$:

1. **Multi-scale causal dyadic wave field ($2 d^2 + 20 d$ params):** Computes $[x^{\text{re}}_t, r_t] = W_{\text{wave}} x_t$, filters $x^{\text{re}}$ across causal dilations $\delta \in \{1, 4, 16, 64\}$ with kernel size $5$ ($O(1)$ ring buffer of $340 d$ scalars), and emits $\mathbf{y}^{\text{wave}}_t = \bar{w}_t \odot \tanh(r_t)$ plus lagged causal surprise $s_t = \|x^{\text{re}}_t - \bar{w}_{t-1}\|_{\text{rms}} + \operatorname{softplus}(w_s^\top x_t + b_s)$.
2. **Half-subspace Metriplectic Householder WY port ($H_{\text{met}} = \max(1, \lfloor H/2 \rfloor)$, $d_{\text{met}} = d/2$):** Projects $(u_t, k_t, q_t)$ from $\tilde{x}_t$ and $(v_t, \beta_t, \tau_t, \gamma_t)$ from $x_t$ into the compressed $d/2$-dimensional head subspace (`wy_metriplectic_chunk_scan`, $C=32$, $\|A_t\|_2 \le \gamma_t \le 1$), returning $\mathbf{y}^{\text{met}}_t = W_{\text{met},o} y^{\text{sub}}_t$.
3. **Shifted surprise-evicted SCAM register bank ($d_{\text{scam}} = d/4$, $K=32$):** Uses asymmetric addressing $(q^{\text{scam}}_t, k^{\text{scam}}_t)$ on $\tilde{x}_t$ and unshifted $v^{\text{scam}}_t$ on $x_t$ ranked causally by $s_t + 10^{-6} t$, overcoming `I3_HRWN`'s associative-overwrite collapse (`0.146 -> 0.977`).
4. **Non-inverted causal epistemic uncertainty:** Replacing `I3_HRWN`'s inverted $1/(1+s_t)$, `HMWROperator` computes:

$$u_t^{(G_3)} = \frac{s_t}{\sqrt{\|y_t^{\text{met}}\|_{\text{rms}}^2 + \|y_t^{\text{scam}}\|_{\text{rms}}^2 + 10^{-3}}}$$

5. **Content-dependent Attention-over-Depth Residuals & Surprise-Adaptive Synthesis:** Instead of fixed identity addition across depth, each layer $\ell \ge 1$ routes over all prior block states $H_{0:\ell} = [h^{(0)}_t, \dots, h^{(\ell)}_t] \in \mathbb{R}^{(\ell+1)\times d}$ via a causal per-token softmax router (`bias[-1] = 2.5` initialization) and gates SwiGLU channel synthesis with surprise log-amplitude:

$$\alpha^{(\ell)}_t = \operatorname{softmax}\!\left(W_{\text{depth}}^{(\ell)} \operatorname{RMSNorm}(h^{(\ell)}_t) + b_{\text{depth}}^{(\ell)}\right), \qquad \hat{h}^{(\ell)}_t = \sum_{k=0}^{\ell} \alpha^{(\ell)}_{t, k}\, h^{(k)}_t$$

$$h^{(\ell)}_{\text{mid}} = \hat{h}^{(\ell)}_t + \operatorname{HMWR}(\operatorname{RMSNorm}(\hat{h}^{(\ell)}_t)), \qquad h^{(\ell+1)}_t = h^{(\ell)}_{\text{mid}} + \operatorname{SwiGLU}(\operatorname{RMSNorm}(h^{(\ell)}_{\text{mid}})) \odot \Big(0.5 + \sigma\big(w_a \ln(1 + u_t^{(G_3)}) + b_a\big)\Big)$$

This recovers full SwiGLU channel capacity ($d_{\text{ffn}} = 304$ at `20.5M`, $384$ at `47.4M`, $7,552$ at `3.88B`) while keeping tokenwise causality exact (`causality_err = 0.0`, `duality_err < 1e-6`).
