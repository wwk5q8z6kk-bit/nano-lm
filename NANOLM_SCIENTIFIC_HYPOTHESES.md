# NANOLM Ω — 24 Original Scientific Hypotheses (`TASK E`)

All hypotheses follow strict scientific discipline: concrete mechanism, why existing approaches fail, mathematically testable prediction, and falsification criterion. Hypotheses marked **(Tested)** include their verified status; **(Active Ω)** are directly tested in `EXP-Ω1..EXP-Ω3`.

---

## Group I — Algebra, Topology & State-Space Expressivity

1. **`H-Ω01` [Active Ω] $SU(2) \to SO(3)$ Projective Adjoint Lie-Exponential Holonomy:**
   Mapping transition increments via `su2_lie_exponential(w)` and reading out via $\mathbb{Z}_2$-invariant `so3_projective_adjoint(q)` ($R(-q) = R(q)$) covers pure $180^\circ$ involutions smoothly and resolves sign ambiguity, enabling exact $S_4$ composition (`~100%`) at `16x` OOD length where $U(1)$, $TC^0$ Transformers, linear-readout $SU(2)$, and Cayley-$SO(3)$ fail.
2. **`H-Ω02` [Tested Cycle-2 — SUPPORTED for Throughput] Reverse-Time Conjugate Spinor Adjoint ($O(1)$ Saved Trajectory):**
   Reversing gradients through $g_t \star \Omega_{t-1} + s_t$ via time-reversed conjugated rotors $\lambda_t = \bar{g}_{t+1}\star\lambda_{t+1}+\delta_t$ in a fused OpenAI Triton scan yields exact gradients while storing 1 trajectory tensor (`omega_all`), achieving `11.08x–35.00x` speedup and `2.33x` lower activation VRAM at `L=4096`.
3. **`H-Ω03` [Tested `Gen3_H4` — PARTIALLY FALSIFIED / MIXED] Bifurcated Unitary vs Episodic Connection Biases ($\beta_r$):**
   Splitting registers into persistent ($\sigma(-3.5)$) vs forgetting ($\sigma(-0.5)$) regimes improved exact state probes (`13.20% -> 15.28%` in `C2-X2`) but did not lower macro BPB (`0.3696` vs `0.3693`), rejected under strict primary-metric gate without adjoint upgrade.
4. **`H-Ω04` [Proposed] Non-Abelian Braid / Quaternion Block-Sparse Attention Hybrid (`3:1` HSPA-Ω + Diff-GQA):**
   Interleaving 3 `HSPA-Ω` holonomic layers with 1 `Diff-GQA` sparse retrieval anchor eliminates `32K`-context attention-sink interference while reducing KV cache `-75%`–`-100%`.
5. **`H-Ω05` [Proposed] Learnable Curvature Gauge Connection ($\mathcal{A}_t = \text{skew}(\omega_t) + [S_{t-1}, u_t]$):**
   Making the infinitesimal holonomy generator linearly dependent on accumulated state torsion allows dynamic state-conditional context switches inside a single recurrent pass.

## Group II — Continual Learning, Plasticity & Parameter Surgery

6. **`H-Ω06` [Active Ω] Spectral Nullspace LoRA Orthogonal Projection ($\Delta W V_k = 0$):**
   Restricting low-rank updates to $\text{span}(V_k)^\perp$ (top-$k$ right singular vectors of frozen pretrained $W_0$) mathematically guarantees zero first-order drift on dominant pretrained features (syntax, standard library idioms, Python algorithms), reducing catastrophic forgetting vs unconstrained LoRA and $L_2$-proximal LoRA in Domain $A \to B$ adaptation (`EXP-Ω2`) and preserving base coding in `Qwen3-1.7B` (`EXP-Ω3`).
7. **`H-Ω07` [Tested Cycle-1 — SUPPORTED] Zero-Init Step-0 KL-Invariant Holonomic Sidecar Graft:**
   Initializing residual output gates `sidecar_gate = 0` produces `0.00e+00` max logit discrepancy before fine-tuning, avoiding the `+1.10` BPB collapse seen in `I0_naive_swap`.
8. **`H-Ω08` [Tested Frontier `E1` — FALSIFIED / REJECTED at small scale] Asymmetric Differential Delta Weights (`C2_DiffNSDelta`):**
   Dual signed value deltas failed to outperform standard gated DeltaNet (`0.6335` vs `0.6331` BPB) at slower throughput.
9. **`H-Ω09` [Proposed] Dynamic Layerwise Singular-Rank Allocation via Spectral Entropy:**
   Assigning larger nullspace protection `k_l` to upper code/reasoning MLP layers and smaller `k_l` to formatting layers.
10. **`H-Ω10` [Proposed] Second-Order K-FAC / Gauss-Newton Preconditioned Nullspace Adapters:**
    Protecting directions weighted by diagonal activation covariance $V_k \Sigma_x V_k^\top$ rather than isotropic weights $W_0$ alone.

## Group III — Predictive Coding, Free Energy & Precision Routing

11. **`H-Ω11` [Tested Cycles 1–2 — PROMOTED `BEST`] Iterative Precision-Weighted Predictive-Coding FFN (`PredictiveCodingCell`):**
    Replacing static single-pass `SwiGLU` with 2-step free-energy error correction ($\epsilon = z - W_{\text{dec}} c$, $c \leftarrow c + \pi W_{\text{err}}\text{Norm}(\epsilon)$) dropped LM BPB from `0.3800` (`Gen1_H1`) to `0.3723` (C1) and `0.3693 ± 0.00041` (C2, `3` seeds), lifting state tracking `13x–22x`.
12. **`H-Ω12` [Tested `nanolm-lab` — SUPPORTED] Vocabulary-Invariant Top-2 Hypothesis-Margin Compute Routing (`u_t = 1 - (p_(1) - p_(2))`):**
    Beats Shannon entropy routing under large vocabularies (`32k+`) by avoiding uniform-tail inflation, saving `2.15 GB` VRAM and cutting copy NLL `-52.9%`.
13. **`H-Ω13` [Tested `E1` — SUPPORTED] Damped Jacobi Equilibria with Entropy Early Exit (`C4_JCLE`):**
    Weight-tied equilibrium loops improved math BPB (`0.8000 -> 0.7884`) while exiting `60.5%` of easy tokens early (`mean_loops=2.18`).
14. **`H-Ω14` [Proposed] Surprise-Gated Holonomic Write Allocation (`w_t \propto u_t^{\text{margin}}`):**
    Driving `SpinorRegisterField` write probability explicitly from Top-2 prediction margin so predictable boilerplate skips memory writes cleanly.

## Group IV — Zero-Oracle Cognitive Control, Verification & Formal Reasoning

15. **`H-Ω15` [Active Ω] Value-of-Information (`VoI`) Zero-Oracle Sandbox & Symbolic Execution (`OmegaCognitiveController`):**
    Routing strictly on surface prompt schema (`CODE_PYTHON`, `DIAGRAM_MERMAID`, `SYMBOLIC_COMPUTE`, `STRUCTURED_JSON`) to run type-driven `python3 -I` smoke falsification (`NameError`/`IndexError` repair), self-synthesized exact Python solvers (for DAG/math/state tracking), and Mermaid edge-count repair closes runtime & arithmetic failure modes unreachable by `SyntaxTTC`.
16. **`H-Ω16` [Proposed] Counterfactual AST Operator Rebinding for Safe Symbolic Verification:**
    Automatically compiling custom symbol definitions from problem prose into an isolated operator table before invoking symbolic verification, avoiding standard-precedence hallucinations.
17. **`H-Ω17` [Proposed] Small-$N$ Brute-Force Induction Oracle for Closed-Form Math Discovery:**
    Checking model-proposed closed-form integer formulas against $O(2^n)$ brute force for $n=3, 4$ inside `python3 -I` before evaluating $n=20$.
18. **`H-Ω18` [Proposed] Dual-Process Epistemic Abstention Head Calibrated via Brier / Proper Scoring:**
    Explicitly separating *copy confidence* from *absence/conflict probability* directly inside logits (resolving `H6` failures).

## Group V — Hardware Optimization, Roofline Economics & Autonomous Discovery

19. **`H-Ω19` [Active Ω] Analytical Roofline Compute-VRAM-Bandwidth Pre-Flight Optimization (`ComputePlanner`):**
    Modeling weights + FP32 optimizer states + `eff_seq` chunked activations + decode bandwidth overhead predicts peak VRAM within `~12.9%` mean error (`3.16%` on `8.19B` GPU inference) and selects minimum-dollar feasible GPUs prior to pod allocation.
20. **`H-Ω20` [Proposed] Persistent Shared-Memory CUDA-Graph Decode Fusion for `HSPA_4B`:**
    Fusing RMSNorm + linear projections + `step_holonomy` + readout into 1 CUDA-Graph per layer to drop `nanolm-hspa-4b` decode latency from `~74 ms/tok` to `< 12 ms/tok` at `batch=1` across `512..32,768` context lengths.
21. **`H-Ω21` [Proposed] Int4 Quantized Frozen Backbone + BF16 Holonomic Registers (`<= 4B` Edge Deployment):**
    4-bit NF4 base weights (`~1.05 GB` at `2B`) plus full-precision `SU(2)->SO(3)` registers (`< 2 MB` at `32K`) enable sub-2GB workstation/mobile inference with unbounded horizon.
22. **`H-Ω22` [Active Ω] High-KL (`kl_beta >= 0.10`) Hard-Curriculum Selective GRPO:**
    Combining early-stopped 2-epoch SFT (preventing reward saturation) with stronger KL anchoring preserves coding syntax while unlocking nonzero RLVR updates (`19+` groups vs `1/72` in `E2`).
23. **`H-Ω23` [Proposed] Multi-Teacher `8B + 32B` Top-K+Tail Logit Distillation into `HSPA-Ω-2B`:**
    Transferring dark-knowledge tail calibration alongside verified tool traces without cross-architecture tokenizer mismatch.
24. **`H-Ω24` [Proposed] Closed-Loop Autonomous Experiment Registry with Pareto Non-Regression Gates:**
    Requiring strict multi-seed improvement over prior `BEST` across capability, OOD retention, and speed/memory efficiency before architecture promotion.
