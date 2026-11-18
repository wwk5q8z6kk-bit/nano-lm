# NANOLM Ω — Exact Mathematical Formulations

This reference formalizes every mathematical mechanism implemented across `nanolm_discovery/` and `nanolm_omega/` (`backbone.py`, `cognition_efe.py`, `unified_system.py`), with stability invariants, complexity bounds, and equivalence identities.

---

## 1. Sequence Operator Foundations (`G3_HMWR` -> `G4_Omega`)

### 1.1 Asymmetric Four-Tap Addressing Shift (`CausalKeyShift`)
To separate contextual trigger pattern matching (e.g. `(QUERY, key)`) from unblurred payload storage, addressing vectors read a causal 4-tap depthwise-shifted representation $\tilde{x}_t$ while payload projections read clean token states $x_t$:

$$\tilde{x}_t = \operatorname{SiLU}\!\left(\sum_{j=0}^{3} c_j \odot x_{t-j}\right), \qquad c \in \mathbb{R}^{d \times 4}$$

with `0.7` lag-0 and `0.3` lag-1 structured initialization on the rightmost causal taps.

### 1.2 Half-Subspace Metriplectic WY Transport with Orientation-Reversing Reflections (`beta in (0, 2)`)
Partitioning $d_{\text{met}} = H_{\text{met}} d_h$ ($H_{\text{met}} = \max(1, \lfloor H/2 \rfloor)$) keeps fixed operator cost at $\le 6.5 d^2$ (preserving `ffn_hidden >= 240`). Each head maintains $S_t \in \mathbb{R}^{d_h \times d_h}$ driven by:

$$u_t = \frac{W_u \tilde{x}_t}{\|W_u \tilde{x}_t\|_2}, \qquad k_t = \frac{\operatorname{RMSNorm}(W_k \tilde{x}_t)}{\|\operatorname{RMSNorm}(W_k \tilde{x}_t)\|_2}, \qquad q_t = \operatorname{RMSNorm}(W_q \tilde{x}_t)$$

$$v_t^{(\ell)} = \begin{cases} W_v^{(0)} x_t, & \ell = 0 \\ \alpha_t^{(\ell)} \odot W_v^{(\ell)} x_t + (1 - \alpha_t^{(\ell)}) \odot v_t^{(0)}, & \ell \ge 1, \quad \alpha_t^{(\ell)} = \sigma(W_{\text{vg}}^{(\ell)} x_t + b_{\text{vg}}^{(\ell)}) \end{cases}$$

where $\beta_t = 1.98 \sigma(g_{t,0}) \in (0, 1.98)$, $\tau_t = \sigma(g_{t,1}) \in (0, 1)$, and $\gamma_t = \exp(-0.20 \operatorname{softplus}(g_{t,2})) \in (0, 1]$ define a two-microstep unitary reflection + dissipative write:

$$S_t^{\text{rot}} = (I - \beta_t u_t u_t^\top)\,S_{t-1}, \qquad S_t = \gamma_t (I - \tau_t k_t k_t^\top)\,S_t^{\text{rot}} + \tau_t k_t v_t^\top, \qquad y_t^{\text{met}} = S_t^\top q_t$$

**Proposition 1 (Unconditional Spectral Norm Bound):** Because $\|u_t\|_2 = \|k_t\|_2 = 1$, $\sigma_{\max}(I - \beta_t u_t u_t^\top) = \max(|1 - \beta_t|, 1) = 1$ for all $\beta_t \in (0, 2)$ (including negative determinants $1 - \beta_t < 0$ needed for non-Abelian permutations), and $\|I - \tau_t k_t k_t^\top\|_2 = 1$ for $\tau_t \in (0, 1)$. Thus every homogeneous transition satisfies $\|A_t\|_2 \le \gamma_t \le 1$.

**Proposition 2 (IEEE-754 WY Reflection Condition Bound):** Interleaving reflection (`w_{2t} = u_t, b_{2t} = beta_t, log g_{2t} = 0`) and write (`w_{2t+1} = k_t, b_{2t+1} = tau_t`) into chunk length $C$ solves $(I - L) Z = R$ identically in `float32` via $\prod_{k=0}^{\lceil \log_2 C \rceil - 1}(I + L^{2^k})$ (`_no_autocast_ctx`). Backpropagation squares entry bounds $|[(I-L)^{-1}]_{C-1, 0}| \le \beta_{\max}(1+\beta_{\max})^{C-2}$, overflowing `fp32` ($3.4 \times 10^{38}$) above $C_{\text{crit}} \approx 1 + \frac{\ln(3.4\times 10^{38})}{2 \ln(1 + 1.98)} \approx 41.6$. Selecting $C = 32$ enforces finite gradients unconditionally.

### 1.3 Multi-Scale Dyadic Wave Field + Shifted SCAM-`K` Episodic Registers
Alongside WY transport, four causal depthwise convolutions $\delta \in \{1, 4, 16, 64\}$ (kernel $5$, $340 d$-entry ring buffer) produce $\bar{w}_t$ and FP32-safe smoothed causal prediction surprise (preventing $0/0$ backward singularity on repeated tokens):

$$s_t = \sqrt{\|x_t^{\text{re}} - \bar{w}_{t-1}\|_{\text{rms}}^2 + 10^{-6}} + \operatorname{softplus}(w_s^\top x_t + b_s), \qquad y_t^{\text{wave}} = \bar{w}_t \odot \tanh(r_t)$$

To overcome finite rank interference in $S_t \in \mathbb{R}^{d_h \times d_h}$, `SCAM` ranks $\{0..t\}$ causally by $s_j + 10^{-6} j$ (`d_scam = d/4`, $K=32$) with asymmetric keys/queries $(q_s, k_s)$ on $\tilde{x}_t$ and clean values $v_s$ on $x_t$.

---

## 2. Tokenwise Depth-Attention, Uncertainty & Zero-Vocab Sparse Pass-2

Each layer $\ell \ge 1$ dynamically weights prior block representations $H_{0:\ell} = [h^{(0)}_t, \dots, h^{(\ell)}_t]$:

$$\alpha_t^{(\ell)} = \operatorname{softmax}\!\left(W_{\text{depth}}^{(\ell)} \operatorname{RMSNorm}(h_t^{(\ell)}) + b_{\text{depth}}^{(\ell)}\right), \qquad \hat{h}_t^{(\ell)} = \sum_{k=0}^{\ell} \alpha_{t, k}^{(\ell)} h_t^{(k)}$$

Native retrieval-normalized epistemic uncertainty gates channel SwiGLU output by amplitude `0.5 + sigmoid(adapt_gate(log1p(u_t)))`:

$$u_t^{\text{mem}} = \frac{s_t}{\sqrt{\|y_t^{\text{met}}\|_{\text{rms}}^2 + \|y_t^{\text{scam}}\|_{\text{rms}}^2 + 10^{-3}}}$$

When the optional Pass-2 `_LatentTokenRefiner` is enabled (`G4_Omega`), internal calibration supervises $z_t^{\text{lat}} = w_{\text{lat}}^\top \operatorname{RMSNorm}(\operatorname{sg}(h_t))$ against detached top-quartile token NLL via $\mathcal{L}_{\text{cal}} = 0.05\,\operatorname{BCE}(z_t^{\text{lat}}, \mathbb{I}[\text{CE}_t > q_{0.75}])$, drives detached Pass-2 routing $u_t^{\text{cal}} = \frac{1}{2}\sigma(\ln(1 + u_t^{\text{mem}}) - 0.5) + \frac{1}{2}\sigma(z_t^{\text{lat}})$, and emits a multiplicative readout $u_t^{\text{tok}} = u_t^{\text{mem}}\big(0.5 + \sigma(z_t^{\text{lat}})\big)$ to preserve the $1/\|y\|_{\text{rms}}$ OOD pole while bounded smooth anti-circularity uses denominator $(\|\Delta h^{(2)}\|_2 + 10^{-2})(\|\Delta h^{(1)}\|_2 + 10^{-2})$ (`<= 100` gradient bound).

**Honest Unembedding FLOP Analysis (`X8` vs. `G4_Omega` vs. Single-Pass `G4`):**  
`X8` computes top-2 margin $u_t^{\text{margin}} = 1 - (p_{t,(1)} - p_{t,(2)})$ by running the full $|V| \times d$ vocabulary projection **twice** ($78.56\text{M}$ FLOPs/tok at `20.54M` / `4k` ctx, $50.54\text{ MB}$ `32k` KV cache). Single-pass `G4_NoSparseRefiner` uses only **one** unembedding ($41.58\text{M}$ FLOPs/tok, $1.80\text{ MB}$ `32k` state), outperforming `G4_Omega` ($40.91\text{M}$ FLOPs/tok) by keeping all dense FFN width (`2.1112` vs `2.1175` BPB and `0.6185` vs `0.5260` `EpistAUROC`).

---

## 3. Open-Entity Relational Graph & Expected Free-Energy (`EFE`) Decision Objective

Given dynamic repository node set $V_N$ ($N \ge 4$ unbounded) encoded via character convolutions `Conv1d(E(chars))` (eliminating `M0..M11` vocabulary limits) and directed trace edges $(u \to v) \in E_t$:

$$m_u = \sum_{(u,v)\in E_t} \tanh\!\left(W_{\text{up}}[h_u; h_v]\right) + \sum_{(w,u)\in E_t} \tanh\!\left(W_{\text{down}}[h_u; h_w]\right), \qquad h_u \leftarrow \operatorname{GRU}(m_u, h_u)$$

A self-attention planner over `[z_global; h_{1:N}]` emits calibrated epistemic uncertainty $\hat{u}_t = \sigma(W_u z_t) \in [0, 1]^3$, expected information gain $\hat{I}_a(z_t) \ge 0$, pragmatic value $\hat{V}_a(z_t)$, and regression risk $\hat{R}_a(z_t) \in [0, 1]$. Action selection optimizes learned meta-weights $(\beta_{\text{epist}}(u_t), \lambda_{\text{risk}}(u_t)) = \operatorname{softplus}(W_{\text{efe}} \hat{u}_t + b_{\text{efe}})$:

$$\pi(a \mid Z_t) = \operatorname{softmax}_a\!\Big(\hat{V}_a(z_t) + \beta_{\text{epist}}(\hat{u}_t)\,\hat{I}_a(z_t) - \lambda_{\text{risk}}(\hat{u}_t)\,\hat{R}_a(z_t)\Big)$$

eliminating all hardcoded action-index thresholds (`suff_pen[4]`, `suff_pen[6]`).
