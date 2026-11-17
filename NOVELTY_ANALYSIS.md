# Scientific Novelty Analysis & Prior-Art Comparison

This analysis evaluates `HSPA` honestly against published literature, explicitly distinguishing genuinely novel synthesis from prior components and identifying what experimental evidence did and did not validate.

---

## 1. Prior Literature Context

1. **Quaternion Neural Networks & Quaternionic Transformers (Parcollet et al. 2019; Tay et al. 2019; Zhu et al. 2018):** Prior work parameterizes linear weight matrices $W \in \mathbb{H}^{m \times n}$ inside dense MLP/Conv/Attention layers to reduce weight parameters by $4\times$ on spatial/audio tasks. **Gap:** They do *not* formulate sequence-time recurrence as parallel-transport holonomy across associative registers, and standard Quaternion Transformers still incur $O(T^2)$ self-attention and linear $O(T)$ KV caches.
2. **State-Space Models & Linear Attention with Non-Diagonal/Householder Transitions (Gu & Dao 2024 Mamba-2; Yang et al. 2024 Gated DeltaNet; Peng et al. 2025 RWKV-7; Slagle et al. 2024; Zhang et al. TTT; Behrouz et al. Titans):** Modern linear sequence models replace diagonal scalar/complex decays ($U(1)$ rotary phases in LRU / S5 / RoPE-SSM) with rank-1 Householder reflections $I - \beta_t k_t k_t^\top$ (DeltaNet) or dense gradient steps on MLP states (TTT / Titans).
   - **Conceptual Difference of `HSPA`:** Instead of $d_h \times d_h$ outer-product rank-1 matrix updates ($\text{O}(d_h^2)$ state per head) or expensive inner-loop autodiff backprop through MLPs per chunk (TTT/Titans), `HSPA` evolves **competitive $R$-slot $SU(2)$ spinor vector registers** ($\text{O}(R \cdot d_{\text{fiber}})$ linear state size) under exact associative left-acting quaternionic Cayley rotors $u(\boldsymbol{\omega})$ + conjugate unbinding $\bar{q} \star \Omega$.
   - **Causal Control Test (`B1_Abelian_SSM` vs `Gen1_H1`):** Keeping parameters (`14.01M` vs `14.02M`), register count ($R=8$), fiber dimensionality (`192`), and Hillis-Steele scan depth identical, swapping $U(1)$ complex rotors (`B1`) for non-abelian $SU(2)$ Cayley rotors (`Gen1_H1`) reduced 2-seed validation BPB from `0.3826` (`B1`) to `0.3800` (`Gen1_H1`), establishing an algebraic gain from non-commutativity alone.
3. **Predictive Coding Networks (Rao & Ballard 1999; Whittington & Bogacz 2017; Millidge et al. 2022):** Classical predictive coding trains local Hebbian energy minimization or bidirectional recurrent conv nets for vision, rarely integrated into causal autoregressive sequence models as an iso-parameter drop-in replacement for SwiGLU.
   - **Empirical Novelty Demonstrated in X2:** Integrating `PredictiveCodingCell` on top of the non-abelian $SU(2)$ spinor stream (`Gen1_H2`) produced the lowest 2-seed BPB (`0.3723` vs `0.3794` Transformer baseline and `0.3800` SwiGLU-HSPA), concentrated sharply in `state` modeling (`0.4438` vs `0.4745` Transformer) and exact state-transition probe accuracy (`13.54%` vs `1.04%` Transformer).

---

## 2. Explicit Falsification Boundaries (What Is *Not* Proven Yet)

To meet top-venue scientific standards, we record three clear negative / incomplete results:
1. **No Compositional $S_4$ Proof Yet:** In Experiment X1 (`400` training steps at `2.46M` parameters), neither Transformer nor `HSPA` solved 24-state unbroken `S_4` permutation products above low single-digit percentages (`2.1%–7.3%`). Claims of practical $S_4$ group tracking generalization remain **unproven** at that tiny training budget and require deeper/longer curriculum training.
2. **Gen-2 Wavepacket + Kuramoto Coupling Failed to Beat `Gen1_H2`:** Adding depthwise conv (`k=4`) and cross-register Kuramoto coupling *without* `PredictiveCodingCell` (`Gen2_H3`) regressed BPB from `0.3723` back to `0.3803`. `PredCode` — not Kuramoto coupling — drove the X2 improvement.
3. **CUDA Throughput Deficit Without Fused Kernels:** Because PyTorch executes `hamilton_product` scans via ~`4` elementwise kernels per $\log_2 T$ tree level without warp-level shared memory fusion, pure-PyTorch `HSPA` ran **3.2x slower** on 4B incremental decode (`135 ms/tok` vs `41.8 ms/tok`) and **8.2x slower** on 14M training throughput (`13.2k` vs `107.4k tok/s`).
4. **No Flagship `>4B` Comparative Outperformance Claim:** While X3 demonstrated flat `1.781 MB` persistent state and `5.51 GB` VRAM savings at `3.845B` (`16K` context), `HSPA-4B` has **not** yet been pre-trained on a multi-hundred-billion-token corpus to compare downstream benchmarks against `>7B` Transformers.
