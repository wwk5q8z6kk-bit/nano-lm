# Mathematical Research Laboratory

**Discipline:** An equation is a hypothesis about a computational inductive bias, not evidence of cognitive superiority. Every mathematical mechanism below pairs an explicit domain of validity with empirical Jacobian/spectral diagnostics and a falsifier.

---

## 1. Constrained Multiobjective Pareto Formulation (Not Scalar IQ)

For architecture/training policy $a$, parameters $\theta$, workload distribution $W$, and resource profile $R$, seek the non-dominated Pareto set over:

$$
F(a, \theta; W) = \bigl(Q_{\text{gen}},\; Q_{\text{reason}},\; Q_{\text{grounded}},\; Q_{\text{OOD}},\; Q_{\text{calibration}},\; Q_{\text{memory}},\; -C_{\text{train}},\; -C_{\text{infer}},\; -L_{p95},\; -E_{\text{joule}},\; -M_{\text{peak}}\bigr)
$$

subject to $P_{\text{deployed}}(a, \theta) < 4{,}000{,}000{,}000$ and hard evidence-conservation minimums (zero silent mutation or fabricated assertions). System metrics (`Track 2`, including external evidence bytes and tool FLOPs) are tracked in a separate frontier from naked-weight models (`Track 1`).

---

## 2. Belief-State Filtering & Exact Source Pairing

For observations $o_t$, latent world state $z_t$, immutable evidence store $E_t$, learned memory $M_t$, computational action $a_t$, and outcome $y_t$:

$$
q_{\phi}(z_t \mid o_{\le t}, a_{<t}) \;\approx\; \mathcal{N}\!\left(p_\theta(o_t \mid z_t) \int p_\theta(z_t \mid z_{t-1}, a_{t-1})\, b_{t-1}(z_{t-1})\, dz_{t-1}\right),
$$

where $\mathcal{N}$ denotes normalization. This is an architectural framing, not a Bayesian guarantee unless densities and selective calibration are operationalized against held-out shifts. Crucially, $z_t$ is never relied upon as the sole record of verbatim facts; it operates alongside typed, offset-anchored records $E_t$ carrying speaker identity, valid/system time, negation, and supersession links (`nano/contracts.py`).

---

## 3. Geometrically Constrained Erase/Reflect/Write Fast State

For unit-normalized key vector $u_t \in \mathbb{R}^{d_h}$ ($\|u_t\|_2 = 1$), consider the generalized Householder transition operator:

$$
A_t = \rho_t (I - \beta_t u_t u_t^\top), \qquad 0 \le \rho_t \le 1,\quad 0 \le \beta_t \le 2.
$$

**Spectral properties of $A_t$:**
- For any vector $w \perp u_t$, $A_t w = \rho_t w$ (eigenvalue $\rho_t$, multiplicity $d_h - 1$).
- In direction $u_t$, $A_t u_t = \rho_t (1 - \beta_t) u_t$ (eigenvalue $\rho_t(1-\beta_t) \in [-\rho_t, \rho_t]$).
- Therefore $\|A_t\|_2 = \rho_t \le 1$ and $\det(A_t) = \rho_t^{d_h}(1 - \beta_t)$.

**Scope boundary & what $\|A_t\|_2 \le 1$ does NOT prove:**
1. Non-expansivity of the homogeneous operator $A_t$ alone does **not** prove non-exploding gradients or bounded trajectories for the full nonlinear write-gated associative recurrence:
   $$
   M_{t+1} = A_t M_t + \eta_t k_t (v_t - M_t^\top k_t)^\top,
   $$
   whose effective state-to-state Jacobian $\frac{\partial M_{t+1}}{\partial M_t}$ mixes $A_t$ with $-\eta_t k_t k_t^\top$ (and cross-terms when $k_t, u_t, \eta_t, \beta_t$ depend on $M_t$ through residual outputs).
2. A single rank-1 real Householder reflection ($\beta_t=2$) has eigenvalues $\{-1, +1, \dots, +1\}$ (order 2). It spans $Z_2$ parity directly and, across layers or with a 2-step composition of non-collinear reflections, generates orthogonal rotations $SO(d_h)$; however, a single-layer rank-1 reflection with fixed key direction cannot execute an arbitrary $SO(2)$ phase rotation of period $4$ ($Z_4$) in one step without an auxiliary layer or skew/complex rotation.
3. **Research test (`X03` / `X04`):** Decouple erase/decay $\rho_t$, reflection $\beta_t$, and write $\eta_t$ (`GeoEraseWrite`), measure both $\|A_t\|_2$ and empirical $\left\|\prod_{t} \frac{\partial M_{t+1}}{\partial M_t}\right\|_2$, and compare directly against `GDN-2` (channelwise independent erase/write) and `Mamba-3`-style complex/block-$SO(2)$ selective dynamics.

---

## 4. Energy-Dissipative Continuous-Time Dynamics

A dissipative latent flow obeys:

$$
\frac{dz}{dt} = (\Omega - D D^\top) z + B u, \qquad \Omega^\top = -\Omega.
$$

For $u = 0$, $\frac{d}{dt}\|z\|_2^2 = 2 z^\top \Omega z - 2 \|D^\top z\|_2^2 = -2 \|D^\top z\|_2^2 \le 0$, giving guaranteed autonomous energy stability under implicit/Cayley or structure-preserving discretizations. Pure symplectic/Hamiltonian flows ($D = 0$) conserve phase-space volume and are restricted strictly to physical simulator world models (`OP09`), never assumed to hold for open natural language (`OP08`).

---

## 5. Rate–Distortion Dual Memory with Hard Evidence Constraint

Classical Information Bottleneck $\min_{p(m\mid x)} I(M;X) + \lambda\,\mathbb{E}[d(Y,\hat Y(M))]$ discards rare exact tokens (drug names, doses, numeric IDs, negations) because their marginal entropy contribution is small while their task consequence is severe. Nano tests a dual-channel formulation:

$$
\min_{\phi, \psi, R}\; I(M_t; X_{\le t}) + \lambda\,\mathbb{E}\bigl[d\bigl(Y_t,\, \hat Y_\psi(M_t, R(q_t, E_t, M_t))\bigr)\bigr] + \gamma\,\text{Cost}(E_t, R)
\quad \text{s.t.} \quad \Pr\bigl[\text{corrupt}(E_{\text{req}})\bigr] \le \varepsilon,
$$

where $M_t$ is fixed-byte learned fast state, $E_t$ is an immutable content-addressed store of verbatim source spans (`ExactEvidenceStore`), and $R$ is an accountable retriever. External storage bytes, index overhead, and scan operations must be included explicitly in $\text{Cost}(E_t, R)$ (`X07`).

---

## 6. Adaptive Compute as Value of Information (VoI)

For computational action $a \in \{\text{think\_again}, \text{retrieve}, \text{ask}, \text{verify}, \text{use\_tool}, \text{output}, \text{abstain}\}$:

$$
U(a \mid s_t) = \mathbb{E}[Q_{t+1} - Q_t \mid s_t, a] - \lambda_F \operatorname{FLOPs}(a) - \lambda_L \operatorname{latency}(a) - \lambda_E \operatorname{joules}(a) - \lambda_R \operatorname{risk}(a).
$$

Evaluated against fixed-depth and uniform Best-of-$N$ baselines under a shared multi-item compute budget (`X08`).

---

## 7. Selective Risk, Calibration, and Epistemic States

Assertions require state $s \in \{\text{supported}, \text{denied/absent}, \text{missing}, \text{uncertain}, \text{conflicting}, \text{superseded}\}$ plus verbatim span binding. Report joint selective risk and coverage curves across confidence thresholds $\tau$:

$$
\operatorname{risk}(\tau) = \frac{\sum_i \mathbf{1}[c_i \ge \tau]\, \ell_i}{\sum_i \mathbf{1}[c_i \ge \tau]}, \qquad
\operatorname{coverage}(\tau) = \frac{1}{n} \sum_i \mathbf{1}[c_i \ge \tau].
$$

Never report low error at $\operatorname{coverage} \approx 0$ as success without disclosing coverage (`C12–C14` lesson). Distribution-free conformal guarantees require their stated exchangeability or bounded-shift calibration assumptions (`SY03`).

---

## 8. Learning Progress vs High-Loss Chasing

To avoid `Surprise-DoReMi` drifting toward noisy/unlearnable domains at the expense of `web`/`textbook` (`EXP-4` lesson), test improvement-per-unit-compute regularized by irreducible noise $n_i$ and measured forgetting $f_i$ on a frozen probe:

$$
s_i \leftarrow \frac{\max(0, \ell_i^{\text{past}} - \ell_i^{\text{now}})}{c_i + \varepsilon} - \lambda_n n_i - \lambda_f f_i, \qquad
p_i \propto \exp(\tau s_i), \qquad p_i \ge p_i^{\min},
$$

always benchmarked against a frozen stratified mixture control (`X12`).

---

## 9. Multi-Objective Training & Gradient Interference

For candidate staged/joint losses

$$
\mathcal{L} = \lambda_1 \mathcal{L}_{\text{LM}} + \lambda_2 \mathcal{L}_{\text{exact}} + \lambda_3 \mathcal{L}_{\text{state}} + \lambda_4 \mathcal{L}_{\text{retrieval}} + \lambda_5 \mathcal{L}_{\text{action}} + \lambda_6 \mathcal{L}_{\text{verify}} + \lambda_7 \mathcal{L}_{\text{calibration}},
$$

measure pairwise gradient cosines $\cos(\nabla_\theta \mathcal{L}_a, \nabla_\theta \mathcal{L}_b)$ and validate one auxiliary objective at a time (`X13`) before joint multi-task training.
