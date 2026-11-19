# Nano LM — Unified Portfolio of 54 Cross-Frontier Scientific Hypotheses

> **Rule:** Every hypothesis is a *falsifiable proposition*, not an approved architecture or claim of superiority. Names denote mathematical mechanisms, not branding commitments.

---

## Frontier A — Biological & Neuro-Inspired Computation (H-01..H-07)

| ID | Hypothesis Name | Mathematical Formulation / Mechanism | Best Simple Competitor | Cheapest Decisive Test & Kill Criterion |
|---|---|---|---|---|
| **H-01** | **Surprise-Triggered FFN Bypass (Predictive Coding)** | Bypass static SwiGLU FFN ($67\%$ of FLOPs) on token $t$ when Delta prediction error $\|v_t - S_{t-1}^\top k_t\|_2 < \epsilon_\tau$ | Uniform dense FFN or `ATRD` | Test on `15M` corpus BPB + wall-clock FLOPs; **KILL** if $>+0.5\%$ BPB at $30\%$ skip. |
| **H-02** | **Cholinergic / Neuromodulatory Gain Reset** | Global scalar surprise signal $m_t = \sigma(w^\top e_t)$ transiently amplifies learning rate $\beta_t$ and forget rate $(1 - a_t)$ at context boundaries | Unmodulated per-head $(a_t, \beta_t)$ | Boundary-shift MQAR (`T=192`); **KILL** if no gain over standard input-dependent gates. |
| **H-03** | **Sleep-Phase Offline Trajectory Consolidation** | Periodically distill high-reward verified episodic memories into slow weights via KL-anchored replay while pruning raw episodic cache | Online SFT / `WiSE-FT` | Continual multi-domain retention (`EXP-Omega2` benchmark); **KILL** if forgetting exceeds `NullspaceLoRA`. |
| **H-04** | **Dendritic Multi-Compartment Nonlinear Coincidence** | Replace single bilinear readout $q^\top S$ with $K$ compartment probes gated by rational/sigmoid interaction $\prod_c \sigma(q_c^\top S_c)$ (`BOBO` generalization) | Standard SwiGLU MLP + single-probe Delta | Matched-param `MQAR-Rev` + real corpus BPB (`EXP-U02`); **KILL** if matched by single-probe `SFOR` at equal FLOPs. |
| **H-05** | **Sparse Top-$k$ Lateral Inhibition (WTA Activation)** | Enforce top-$k$ winner-take-all sparsity across FFN/memory channels ($\le 15\%$ active) to cut superposition interference | Dense SwiGLU + L2 norm | Interference stress test (`N=16` concurrent keys); **KILL** if BPB regresses $>+1\%$ at equal active FLOPs. |
| **H-06** | **Theta-Gamma Phase-Multiplexed Working Slots** | Partition recurrent state $S_t$ into fast $\gamma$ oscillation slots updated per token and slow $\theta$ slots gated every $M$ tokens | Uniform single-timescale `GDN` | Hierarchical nested-scope brackets & multi-turn tasks; **KILL** if equaled by `3:1` depth hierarchy. |
| **H-07** | **Metaplastic Synaptic Tagging (`EWC`-in-Context)** | Protect high-precision slots in $S_t$ by scaling erase factor $\beta_t \propto 1 / (1 + P_t)$ using accumulated precision $P_t$ (`NEBN` + `Delta`) | Unprotected Gated Delta (`beta_t`) | Delayed rare-needle retention under noisy distractor stream; **KILL** if `SFOR` matches retention. |

---

## Frontier B — Fundamental Physics, Geometry & Mathematics (H-08..H-15)

| ID | Hypothesis Name | Mathematical Formulation / Mechanism | Best Simple Competitor | Cheapest Decisive Test & Kill Criterion |
|---|---|---|---|---|
| **H-08** | **Zero-Mean Full-Sphere Key Transport** | Center `Q, K` before $\ell_2$ normalization (no `SiLU` orthant skew): $\mathbb{E}[\langle k_i, k_j\rangle] \to 0$ prevents collateral Householder erasure | `ShortConv(F.silu)` (`NanoLM-v2/v3`) | Real `125M` autopsy (`EXP-U01`) + matched `SFOR` ablation (`EXP-U02`); **KILL** if `600M` training eliminates `SiLU` key correlation naturally. |
| **H-09** | **Rank-2 Householder `DeltaProduct` + Channelwise `KDA` Decay** | $S_t = \mathrm{Diag}(\alpha_t)(I-\beta_{t,2}u_{t,2}u_{t,2}^\top)(I-\beta_{t,1}u_{t,1}u_{t,1}^\top)S_{t-1} + \eta_t k_t e_t^\top$: single-step $SO(d_h)$ rotation + channelwise forgetting | Rank-1 `GeoEraseWrite` & scalar-decay `GDN` | $S_4$ (`L=192`) + `MQAR-Rev` + real BPB (`EXP-U02` `M3`); **KILL** if rank-1 `SFOR` or `HSPA` dominates Pareto front. |
| **H-10** | **Continuous-Time Skew-Symmetric Cayley Flow** | $\frac{dz}{dt} = (\Omega(x_t)-\Gamma(x_t))z + Bu_t$, $\Omega^\top = -\Omega$, discretized via Cayley step $(I-\frac{h}{2}A)^{-1}(I+\frac{h}{2}A)$, guaranteeing $\|M_t\|_2 \le 1$ | Discrete `Householder-Delta` / `Mamba-3` | $S_4$ `8×` extrapolation + `MQAR-Rev` (`EXP-U02` `M7`); **KILL** if unstable or inferior to `DeltaProduct-2` at matched FLOPs. |
| **H-11** | **Lie-Exp $SU(2) \to SO(3)$ Projective Adjoint Holonomy** | Parallel transport on unit quaternions via $\exp(\omega_t) \in SU(2)$ read out via $\mathbb{Z}_2$-invariant adjoint $R(q) \in SO(3)$ | `M3` (`DeltaProduct-2`) & `M0` (`Dense`) | Test `HSPA` NOT ONLY on $S_4$, but on `MQAR-Rev` & Unseen Copy (`EXP-U02` `M5`); **KILL** on general memory if rank-3 rotation lacks associative capacity. |
| **H-12** | **Sheaf-Cohomology Multi-Source Contradiction Localization** | Represent claims as local sections $s_v$ over entity-attribute overlap graph; nonzero coboundary $\delta^0 s \neq 0$ localizes minimal inconsistent subsets | Pairwise contradiction regex / LLM check | Multi-document contradiction localization benchmark; **KILL** if simple pairwise entity-key grouping achieves identical F1. |
| **H-13** | **Saddle-Free Open Output Gate (`SFOR`)** | Offset output gate $\mathrm{SiLU}(W_g x + \delta_g)$ ($\delta_g = +1.5$) eliminates step-0 $(W_o\approx 0, \mathrm{SiLU}(0)=0)$ gradient saddle | Zero-bias $\mathrm{SiLU}(W_g x)$ (`GDN`/`v3`) | Compare `closed` vs `open +1.5` at BOTH 55 steps AND 1,000+ Muon steps (`EXP-U02`); **KILL** if advantage disappears past step 500 under Muon. |
| **H-14** | **Spectral Nullspace Projection for Continual Learning** | Project adaptation updates $\Delta W$ onto $(I - V_k V_k^\top)$ orthogonal to top-$k$ right singular vectors of frozen weights (`SpectralNullspaceLoRA`) | L2 proximal penalty + `WiSE-FT` interpolation | Compare on matched `MATH-500` + tool-recovery split; **KILL** if `WiSE-FT + Self-KL` matches retention at lower compute. |
| **H-15** | **Gauge-Covariant Token-Graph Message Passing (No Positional IDs)** | Replace absolute positional IDs with relative causal-order De Bruijn gauge transformations on entity graphs (`C6`/`C7`) | RoPE + random $\mathrm{Sym}(N)$ augmentation (`C4b`) | Ablate `C7` *without* `SymbolicBeliefState` features (`EXP-U03`); **KILL** neural claim if Python feature generator accounts for $>80\%$ of gain. |

---

## Frontier C — Predictive Belief, Uncertainty & Causality (H-16..H-22)

| ID | Hypothesis Name | Mathematical Formulation / Mechanism | Best Simple Competitor | Cheapest Decisive Test & Kill Criterion |
|---|---|---|---|---|
| **H-16** | **Exponential-Family Natural-Parameter Filter (`NEBN`)** | Track natural sufficient stats $(H_t, P_t, \ell_t)$ so epistemic uncertainty $u_t = (\phi(q_t)^\top P_t + \lambda_0 e^{\ell_t})^{-1}$ scales inversely with observed evidence mass | Softmax attention entropy / `Gated-Delta` | `EXP-U02` `M6` + selective calibration AUROC on unseen probes; **KILL** if precision uncertainty does not beat logit margin. |
| **H-17** | **Frontier-Relative De Bruijn Causal Belief Workspace** | Dynamic index `0` = newest discovered causal node (`\0` innermost binder), removing `K=3` depth ceiling (`C7`) | Root-level De Bruijn (`C5`) & raw transcript (`C0`) | Pure-neural ablation without `SymbolicBeliefState` oracle features (`EXP-U03`); **KILL** if pure neural fails `N=14, 4-hop`. |
| **H-18** | **Interventional `do`-Calculus Graph Mutilation** | Compute $P(Y \mid do(X))$ via mutilated SCM inference rather than observational $P(Y \mid X)$ correlation | End-to-end transformer trained on observation traces | Simpson-confounder DAGs; **KILL** any pure-observational baseline claiming causal discovery without intervention tests. |
| **H-19** | **Bitemporal Truth Maintenance with Staleness Cascade** | Track `(valid_time, record_time)` separately and propagate `stale` along `DEPENDS_ON` edges (`nano/` + `belief_graph`) | Flat latest-timestamp overwrite | Delayed out-of-order correction benchmark; **KILL** if flat overwrite passes delayed-correction cases (it cannot). |
| **H-20** | **Typed Epistemic State Disentanglement (`S/A/M/U/C`)** | Explicitly classify `supported`, `denied/absent`, `missing`, `uncertain`, `conflicting` rather than collapsing `missing` & `denied` | Binary answer / abstain threshold | Grounded clinical + non-medical doc QA (`C12–C14`, `SLW-001`); **KILL** if binary confidence reaches equal risk-coverage. |
| **H-21** | **Split Epistemic vs Aleatoric Top-$K$ Entropy (`HEM-ATRD`)** | Route via $u_{\text{HEM}} = H(\mathrm{softmax}(\mathrm{top}_K(z)))/\ln K$ (vocab-size invariant, multi-tie aware) + Lyapunov non-degradation | Full-vocab Shannon (`X5`) & Top-2 Margin (`X8`) | Fix & audit `E24` paradox (`H1_x5` `1.000` vs `H4_hem` `0.8385`!) in `EXP-U03`; **KILL** if Lyapunov check blocks valid Pass-2 updates. |
| **H-22** | **Conformal Risk-Controlled Abstention Under Shift** | Calibrate per-slot nonconformity score $s_i$ to guarantee conditional error $\le \alpha$ via weighted conformal quantile | Uncalibrated raw threshold (`H6`) | Synthetic domain shift `C12->C14` risk-coverage curve; **KILL** if calibration collapses under shift without re-weighting. |

---

## Frontier D — Memory, Continual Learning & Rate-Distortion (H-23..H-29)

| ID | Hypothesis Name | Mathematical Formulation / Mechanism | Best Simple Competitor | Cheapest Decisive Test & Kill Criterion |
|---|---|---|---|---|
| **H-23** | **Rate-Distortion Dual-Channel Architecture (`S_t` + Verbatim Ledger `E_t`)** | By Shannon rate-distortion, fixed state $S_t$ of $B$ bits cannot preserve arbitrary unseen strings; pair $S_t$ with immutable exact-byte span store $E_t$ (`MM01`) | Pure neural recurrence / long KV cache | Unseen Open-Vocab Copy (`EXP-U02` + `X07` `16`-needle byte budget); **KILL** neural-only exactness if $S_t$ fails unseen IDs. |
| **H-24** | **Verification-Gated Procedural Skill Consolidation** | Admit solved trajectories to `ProceduralMemory` *only* if `independently_verified == True` + posterior reliability pruning | Unfiltered self-play / trajectory cache | Multi-episode skill transfer with adversarial trap specs; **KILL** if unfiltered cache stays clean (falsified in `EXP-OMEGA-04`). |
| **H-25** | **Commit-Gated Two-Tier KV Eviction** | Evict raw KV tokens only after verified atomic claims commit to immutable `Encounter` ledger (`CommitGatedKVCacheLedger`) | Sliding-window KV (`W=256`) / `H2O` attention sink | Long-context (`960+` tok) early-fact retention (`EXP-UMA`); **KILL** if sliding window matches early-span exact recall. |
| **H-26** | **Titans-Style Momentum Test-Time Memory with Session Isolation** | Online gradient-momentum MLP memory $M_t = \theta_t M_{t-1} - \eta_t \nabla \ell_{\text{assoc}}$ reset strictly per isolated context boundary (`E13`) | Linear Delta matrix state $S_t$ | Multi-key associative recall & non-linear key-value binding (`MQAR-NL`); **KILL** if linear `SFOR` matches non-linear recall at lower FLOPs. |
| **H-27** | **Surprise-Cohesive Tuple Crystallization (`ERN-Gen2`)** | Evict episodic KV cache keeping high-surprise *adjacent* `(key, val)` tuples (`pos[j+1]-pos[j]==1`) | Point-wise surprise eviction / recency FIFO | `1024->4096` needle retrieval under fixed cache budget $B=64$; **KILL** if uniform attention sink + Delta state wins. |
| **H-28** | **Zero-Copy Page-Aligned Memory-Mapped Multi-Tier Serving** | Align `.safetensors` headers to `16 KiB` VM pages + thermal-aware host routing (`compact_prefix_kv` vs `native_3m`) | Unaligned `mmap` / full recompute | M3 Max UMA bandwidth & latency audit (`info_density.py`); **RETAINED** if bit-exact (`20/20`) and `>=1.5x` faster. |
| **H-29** | **Cryptographically Verifiable Hard Deletion (`Right-to-Forget`)** | Evict both ledger records and invalidate dependent derived views via dependency graph with SHA deletion receipt (`ExactEvidenceStore`) | Approximate unlearning on weights | `nano/mechanisms/dual_memory.py` + `nano/dependency.py` deletion audit; **KILL** any weight-only deletion claim failing zero-residual check. |

---

## Frontier E — Adaptive Computation & Recursive Depth (H-30..H-35)

| ID | Hypothesis Name | Mathematical Formulation / Mechanism | Best Simple Competitor | Cheapest Decisive Test & Kill Criterion |
|---|---|---|---|---|
| **H-30** | **Tiny Recursive Latent+Draft Refinement (`TRM`)** | Compact shared core iterates $(z, y) \leftarrow \mathrm{Core}(x, z, y)$ for $K$ steps with deep supervision on $y_k$ | Fixed `K`-layer feedforward transformer / CoT | `EXP-U02` `M8` across `S4`, `MQAR-Rev`, `Parity-Dyck` at matched params; **KILL** if matched by single-pass at equal FLOPs. |
| **H-31** | **Monotonic Margin Non-Degradation Guard (`Lyapunov` Shield)** | Accept iterative refinement update $y^{(k+1)}$ iff task margin / verifier confidence strictly improves (eliminates `TaH` latent overthinking) | Unconditional last-step output (`X8` / `Huginn`) | `EXP-U02` `TRM + Guard` vs unguarded `TRM` across $K=1..8$ steps; **KILL** if unguarded $K$ never degrades early correct tokens. |
| **H-32** | **Learned Value-of-Computation (`VoC`) Cross-Query Allocator** | Allocate test-time compute $C_i$ across queries maximizing $\sum_i \mathbb{E}[\Delta Q_i \mid u_{\text{ep},i}, C_i] - \lambda C_i$ | Uniform Best-of-$N$ across all items | Shared-FLOP multi-item budget evaluation (`X08`); **KILL** if uniform Best-of-2 equals VoI allocator at matched total tokens. |
| **H-33** | **Prefix-KV Unary Trie Coalescing (`0`-Entropy Skipping)** | Skip forward passes whenever constrained grammar trie has out-degree 1 ($H(T_i \mid T_{<i}) = 0$) | Uncoalesced token-by-token constrained decoding | Already verified `20/20` exact identity (`450 -> 125` forward passes, `2.08x` speedup); generalize to JSON/SQL/code AST grammars. |
| **H-34** | **Depth Attention Residuals (`Block AttnRes`)** | Replace fixed $x + f(x)$ sum with softmax attention over preceding block outputs | Standard PreNorm residual & `mHC` | `EXP-U02` ablation (`KDA+DP2` ± `AttnRes`); **KILL** if plain PreNorm ties at `<50M` and adds overhead. |
| **H-35** | **Early-Late Contrastive Loop Decoding (`LoopCD`)** | Amplify reasoning signal via $\mathrm{logits} = (1+\gamma)z^{(K)} - \gamma z^{(1)}$ across recursive passes without retraining | Standard final-pass greedy decoding | Test zero-cost post-hoc on trained `M8_TRM` weights; **KILL** if increases calibration error or fails `pass@1`. |

---

## Frontier F — Causal & World-Model Intelligence (H-36..H-40)

| ID | Hypothesis Name | Mathematical Formulation / Mechanism | Best Simple Competitor | Cheapest Decisive Test & Kill Criterion |
|---|---|---|---|---|
| **H-36** | **Latent Action-Conditioned World-Model Planning vs Imitation** | Predict $z_{t+1} = f_\theta(z_t, a_t)$ + outcome value before emitting action $a_t$ (`C3/C6`) | Direct behavioral cloning (`C0`) | Multi-step hidden-hop SWE & physical grid transitions; **KILL** if `C0` + chain-of-thought matches at equal steps. |
| **H-37** | **Joint Embedding Predictive Abstraction (`JEPA`) + Exact Span Pointer** | Predict abstract future latent $\hat z_{t+\Delta}$ under variance-covariance regularization (`VICReg`) *while* leaving exact literals to pointer channel | Next-token cross-entropy only | Compare `CE` vs `CE + 0.05 JEPA` on `OOD-2x` planning & real BPB; **KILL** if `JEPA` adds zero OOD gain (as in `CRG-JEPA` `2.448`). |
| **H-38** | **Causal-Suffix Ordering (`[x_obs; Z]` vs `[Z; x_obs]`)** | In causal transformers/hybrids, place persistent belief tokens *after* new observations so `Z` reads `x_obs` in $O(1)$ depth | Prefix belief slots `[Z; x_obs]` | Direct controlled `swe_env` ablation at fixed `C5` canonicalization; **KILL** if bidirectional cross-attn matches identically. |
| **H-39** | **Counterfactual Trajectory Contrastive Learning** | Train on minimal intervention pairs $(e, do(v \to v'))$ to sharpen causal sensitivity of hidden states | Standard i.i.d. shuffled episodes | Held-out interventional queries (`nano/causal.py` + `swe_env`); **KILL** if counterfactual error remains unchanged. |
| **H-40** | **Multi-Domain Transfer Between Medical Scribing & SWE Debugging** | Train shared epistemic state/span/revision core on synthetic SWE + clinical encounters; test zero-shot transfer on unseen incident/legal QA | Domain-specific fine-tuning | Cross-domain held-out transfer matrix (`EXP-U05`); **KILL** if zero positive transfer across domains. |

---

## Frontier G — Representation, Optimization, Data & Post-Training (H-41..H-47)

| ID | Hypothesis Name | Mathematical Formulation / Mechanism | Best Simple Competitor | Cheapest Decisive Test & Kill Criterion |
|---|---|---|---|---|
| **H-41** | **Split Lag-0 / Lag-1 Key-Shift Induction Heads** | Half of heads use unshifted lag-0 keys (`exact match`), half use lag-1 Dirac key shift (`next-token induction`) without `SiLU` blur | Learned 4-tap `ShortConv(F.silu)` | `EXP-U02` + `15M` real corpus copy probe; **KILL** if learned 4-tap linear conv without `SiLU` matches at equal steps. |
| **H-42** | **Hybrid Byte-Anchored Open-Vocabulary Path (`BLT`-Lite)** | Represent rare/out-of-vocab literals (`len >= 6` digits/codes/drugs) via byte-level local encoder + exact pointer copy | Pure `32K` BPE subword tokenizer | Unseen drug/numeric/UUID copying under adversarial spelling jitter; **KILL** if byte-level encoding slows decode without copy gain. |
| **H-43** | **Muon Spectral Width-Depth Transfer Law** | Scale Muon LR as $\eta(d,L) = \eta_0 \sqrt{d_0/d}\sqrt{L_0/L}$ across `46M -> 125M -> 350M -> 2B` | Constant `lr=0.02` or AdamW | Check `S2_125m` (`0.9013` BPB!) & `350M` scaling curves; **KILL** if width-scaled LR diverges or trails tuned AdamW at `350M`. |
| **H-44** | **Marginal Learning Progress Curriculum vs Raw Surprise (`DoReMi`)** | Weight domains by smoothed $-\Delta \mathcal{L}_d / \Delta C_d$ minus irreducible noise estimate | Uniform stratified mixture or `Surprise-DoReMi` | `E11` / `15M` noisy-shard contamination test; **KILL** if uniform static mixture ties on clean OOD transfer. |
| **H-45** | **Self-Anchored Top-$K$+Tail KL Distillation + `WiSE-FT`** | $\mathcal{L} = (1-\alpha)\mathcal{L}_{\text{CE}} + \alpha T^2 D_{\text{KL}}(\pi_{\theta_0} \| \pi_\theta)$ + $\theta = 0.7\theta_{\text{post}} + 0.3\theta_0$ | Standard one-hot SFT + GRPO | Direct comparison against `SpectralNullspaceLoRA` on identical `MATH-500` + tool dataset; **KILL** if forgetting persists on unseen math. |
| **H-46** | **Eliminating Multi-Turn Shortcut (`ok:true => stop`)** | Include shield-verified `"ok":true -> next_tool` continuation trajectories in multi-turn SFT | Standard terminal-only tool traces | Measure 5-turn `Comp-2` recovery (`B8-3B` rose `5% -> 30%`, yet base `SmolLM3 + System-v4` hit `60%`!); isolate why `B8` still lagged base (`30%` vs `60%`). |
| **H-47** | **Multi-Token Prediction (`MTP-1`) + Lossless Speculative Verification** | Auxiliary $t+2$ prediction head enables self-speculative decoding with zero divergence (`0` false tokens) | Pure 1-token autoregressive decode | Measure wall-clock speedup on M3 Max MPS (`tok/s` vs acceptance rate); **KILL** if overhead exceeds acceptance gain at batch=1. |

---

## Frontier H — Verification & Autonomous Scientific Discovery (H-48..H-54)

| ID | Hypothesis Name | Mathematical Formulation / Mechanism | Best Simple Competitor | Cheapest Decisive Test & Kill Criterion |
|---|---|---|---|---|
| **H-48** | **Small-$N$ Brute-Force Specification Falsification (`IVE`)** | Cross-check candidate closed-form math/code solutions against exhaustive $n \in \{2..5\}$ brute-force enumerators to break correlated $<4B$ bugs | Majority vote / LLM self-arbitration (`System-v2`) | Test on real `frontier_math` failures from `B8-3B`; **KILL** on subproblems lacking computable small-$N$ reductions. |
| **H-49** | **Differential Slow-vs-Fast Fuzzing (`CEGAR` for `<4B` Coding)** | Synthesize $O(N^2)$ brute-force oracle + randomized edge-case fuzzer against $O(N\log N)$ solution to catch off-by-one/compression bugs | Single-pass LLM self-asserts | Test on `frontier_coding` tasks failed by `B8-3B` (`9/20` failed); **KILL** if oracle generation shares same bug. |
| **H-50** | **Reusable Verified Combinator Library (`DreamCoder` + `ProceduralMemory`)** | Compose verified CRT/sieve/union-find/DP combinators instead of synthesizing 80-line algorithms from scratch | Zero-library generation from raw prompt | Held-out unseen math/coding compositions; **KILL** if retrieval distraction harms accuracy. |
| **H-51** | **Co-Designed Grounded Verifier Recall Repair (NanoScribe `76/276` Gap)** | Diagnose and repair the `76/276` (`27.5%`) valid-gold rejections in `verify_grounded_span` WITHOUT introducing hazard false admissions (`0/7`) | Current `grounded.py` (`72.5%` gold retention) | Audit exact failure reasons on `C12–C14` + test `Qwen2.5-1.5B` live on `grounded` menu (`EXP-U04`); **KILL** any relaxation that admits `>0` hazards. |
| **H-52** | **BALD Mutual-Information Active Experiment Selection** | Select next probe maximizing $\mathbb{I}[\theta; y \mid x, \mathcal{D}] = H(\mathbb{E}[p(y\mid x,\theta)]) - \mathbb{E}[H(p(y\mid x,\theta))]$ over candidate laws | Uniform grid sweep | Symbolic law discovery (`Program C`); **KILL** if grid sweep achieves equal NLL at same query budget. |
| **H-53** | **Masked Discrete Diffusion (`LLaDA`-style) Span Repair Module** | Bidirectional masked refinement over structured JSON / encounter slots conditioned on immutable transcript spans | Left-to-right constrained trie decoding | Compare exact JSON + slot consistency at matched wall-clock ms (`EXP-U06`); **KILL** if trie-constrained AR dominates both accuracy & latency. |
| **H-54** | **Causal Mechanism-to-Corpus Transfer Gate** | Require every `115k` mechanism winner to preserve ranking across `>=1,000` steps (`>=4M` tok) on real `web/textbook/math/code` BPB before `125M+` promotion | `250`-step screening (`EXP-INV-02`, which reversed!) | Enforced automatically in `EXP-U02` Part A $\to$ Part B pipeline. |
