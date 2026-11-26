# NanoLM Architecture — `HSPA` & `HSPA-Ω` (`<= 4B` Cap)

> Autonomous architectural discovery target: maximize reasoning, non-abelian state tracking, coding, continual plasticity, and memory efficiency within a hard `4,000,000,000` (`4B`) total parameter ceiling (`~2B` preferred target, `~3.85B` flagship configuration). See also [NANOLM_ARCHITECTURE_VISION.md](NANOLM_ARCHITECTURE_VISION.md).

---

## 1. Tracked Architecture Lineage (`ExperimentRegistry`)

| Role | Identifier | Parameters | State / Property | Multi-Seed Metric | Registry Verdict |
|---|---|---:|---|---|---|
| **`BASELINE`** | `b0-transformer-micro` (`C2-X2`) | `13,966,464` | `24,576 KB` KV @ `8K` | `0.3798 ± 0.00093` BPB (`3` seeds) | Baseline RoPE GQA + SwiGLU |
| **`BEST` (LM)** | **`gen1-h2-hspa-predcode-micro`** | `14,022,536` | **`24.0 KB`** ($O(1)$, `1,024x` smaller) | **`0.3693 ± 0.00041` BPB** (`49,633 tok/s`) | **Promoted `BEST`** (`19.1x` higher `state` probe `13.20%`) |
| `EXPERIMENTAL` | `gen3-h4-hspa-wavepred-micro` | `14,033,944` | `30.0 KB` ($O(1)$) | `0.3696 ± 0.00082` BPB (`15.28%` state = `22.1x`) | **Rejected vs `Gen1_H2` on BPB** (`0.3696 > 0.3693`) |
| `REJECTED` | `gen2-h3-hspa-evolved-micro` (`C1`) | `14,027,648` | `33.0 KB` | `0.3803 ± 0.00180` BPB | Rejected (ablating PredCode hurt) |
| **`BREAKTHROUGH` (`S_4`)** | **`HSPA-Ω` (`su2_lie_exp + so3_adjoint`)** | **`5,784`** (probe) | **$SO(3)$ Projective Gauge** | **`100.0% ± 0.0%` at `16x` OOD (`L=512`)** | **Verified (`EXP-Ω1`)** vs `1.8%–6.0%` controls |
| **`BREAKTHROUGH` (`S_4` LM)** | **`gen4-omega-so3-predcode-small_x1`** | **`2,466,732`** | **Full LM `[:half_r]` Unitary + `res_mix_gate`** | **`100.00%` (`L32..128`), `93.06%` (`8x` OOD `L256`)** | **Verified (`EXP-Ω4`)** vs `2.77%–6.25%` controls |
| `EXPERIMENTAL` (`14M`) | `gen4-omega-so3-predcode-micro` (`Ω5`) | `13,948,944` | `-14.3%` VRAM vs `Gen1_H2` (`3,263.9 MB`) | `0.3775 ± 0.00437` BPB (`math 9.03%`, `state 3.47%`) | **Rejected vs `Gen1_H2` (`0.3701` BPB)** (`res_mix_gate=0` in 7/8 layers) |
| **`BREAKTHROUGH` (`1.75B`)** | **`NanoLM-Ω-1.7B` (`Naked_Greedy`)** | **`1,747,444,768`** | `Nullspace k=48 + HSPA-Ω Sidecar` | **`40/56 (71.43%)`, `coding=3/4`** | **Outperforms `8.19B` `Qwen3-8B` (`39/56 = 69.64%`)** |
| **`BREAKTHROUGH` (`Ω-v2`)** | **`NanoLM-Ω-v2-1.7B` (`Cognitive_v2`)** | **`1,747,444,768`** | `Merged Nullspace + Controller-v2` | **`41/56 (73.21%)`, `diagrams 4/4`, `tools 4/4`** | **Lifts T4 FP16 (`38/56`) `+3` (`6/6` merge parity, `1.61x` speed)** |
| `BASELINE` (`C4`) | `b2-gated-deltanet-micro` (`FP2`) | `14,018,688` | Chunkwise WY DeltaNet (`1.91e-6` parity) | `0.3785 ± 0.00008` BPB (`bit=0.0%`) | Modern recurrent baseline (`EXP-FP2`) |
| `EXPERIMENTAL` (`C4`) | `gen5-fp-ph-holonomy-micro` (`FP2`) | `14,008,545` | `PortHamiltonianFFN` + `CausalSourceCopyGate` | `0.3756` BPB, **`96.67%` bitemporal recall (`201x`)** | **`201x` copy-gate breakthrough; `Gen1_H2` retains `BEST` BPB (`0.3751`)** |
| **`BREAKTHROUGH` (`Ω-v3`)** | **`NanoLM-FP-v3-1.7B` (`WorldCtrl_v3`)** | **`1,747,444,768`** | `AppendOnlySourceLedger + Controller-v3` | **`FP-5Axis-30: 15/30 (50.0%)` vs `SC3 9/30`; `15D: 41/56`** | **Beats `3x` Equal-Compute `SC3` (`+66.7%` rel, `overlaps=0`)** |

---

## 2. Core Components Across [architecture.py](nanolm_holonomy/architecture.py), [triton_holonomy.py](nanolm_holonomy/triton_holonomy.py), [omega_architecture.py](nanolm_holonomy/omega_architecture.py) & [cuda_graph_decode.py](nanolm_holonomy/cuda_graph_decode.py)

1. **Fused Non-Abelian Spinor Recurrence (`SpinorRegisterField` + `holonomy_scan_fast`):**
   Parallel associative quaternion scan $\Omega_t = g_t \star \Omega_{t-1} + s_t$ executed via OpenAI Triton (`_triton_holonomy_fwd_kernel`) with an analytical time-reversed conjugate adjoint storing only `omega_all` (`11.08x–35.00x` speedup over Cycle-1 PyTorch Blelloch scan), plus exact $O(1)$-memory step decode (`step_holonomy`, diff `< 2.2e-6`).
2. **Lie-Exp $SU(2) \to SO(3)$ Projective Adjoint (`HSPA-Ω`):**
   Replaces the singular Cayley chart at $\theta = \pi/2$ involutions with smooth `su2_lie_exponential(omega)` and resolves $\pm q$ double-cover interference using $\mathbb{Z}_2$-invariant `so3_projective_adjoint(q)` ($R(-q) = R(q)$), lifting unbroken $S_4$ generalization from `3.64%` (`Cayley`) to **`100.00%` (`EXP-Ω1`)**.
3. **Iterative Precision-Weighted Free-Energy FFN (`PredictiveCodingCell`):**
   Two-step prediction-error correction outperforming static SwiGLU across Cycle-1 and 3-seed Cycle-2 LM evolution (`0.3693` vs `0.3798` BPB).
4. **Zero-Overhead Spectral Nullspace Adaptation (`SpectralNullspaceLoRALinear.merge()`) + Static CUDA Graph + Controller-v2 ([omega_system.py](nanolm_holonomy/omega_system.py)):**
   Constrains fine-tuning updates via $\Delta W = \frac{\alpha}{r} B A (I - V_k V_k^\top)$ so $\Delta W V_k \equiv 0$, cutting continual Domain-$A$ forgetting across `3/3` seeds (`+4.47 -> +3.83`) and recovering Python `coding` (`0/4..1/4 -> 3/4`) in `1.747B` fine-tuning (`EXP-Ω2` & `EXP-Ω3`), then folds $\Delta W_\perp$ into `base.weight` (`7.55x` GEMM, `1.61x` end-to-end HF decode, `6/6` match across `196` layers). `StaticGraphHSPADecoder` yields **`3.10x`** (`1.808B`) and **`2.35x`** (`3.841B <= 4B`) decode speedup; `OmegaCognitiveController-v2` adds Top-2 logit margin gating, schema-complete JSON protection, chained Mermaid normalization, and multi-probe Python smoke verification (`41/56 = 73.21%`).
5. **`Gen4_Omega_SO3_PredCode` (`HSPAFamily.GEN4_OMEGA_SO3_PREDCODE`):**
   Splits `[:half_r]` unitary `su2_lie_exponential` registers ($s=0$) from `[half_r:]` episodic registers with identity-anchored state, `so3_projective_adjoint` $\mathbb{Z}_2$-invariant readout, and scalar `res_mix_gate` (`1.0` layer 0, `0.0` deeper), solving full-LM unbroken $S_4$ composition (`100.00%` `L32..128`, `93.06%` `8x` OOD `L256` in `EXP-Ω4`).

---

## 3. Flagship 2B & <= 4B Configurations

| Config | Parameters | `<= 4B` Valid | Sequence Horizon (`max_seq_len`) | Persistent State (`32K`) |
|---|---:|---:|---:|---:|
| `HSPA_2B_CONFIG` (`nanolm-hspa-2b`) | `1,808,239,648` | Pass | `65,536` | `1.19 MB` ($O(1)$) |
| `HSPA_OMEGA_2B_CONFIG` (`nanolm-hspa-omega-2b`) | `1,808,108,608` | Pass | `65,536` | `1.00 MB` ($O(1)$, `51.023 ms/tok` graph) |
| `HSPA_4B_CONFIG` (`nanolm-hspa-4b`) | `3,845,419,056` | Pass | `65,536` | **`1.781 MB`** (`1,725x` smaller than KV) |
| `HSPA_OMEGA_4B_CONFIG` (`nanolm-hspa-omega-4b`) | `3,840,503,904` | Pass | `65,536` | **`1.50 MB`** ($O(1)$, `100.291 ms/tok` graph) |
| `HSPA_GEN5_2B_CONFIG` (`nanolm-hspa-gen5-2b`) | `1,809,521,793` | Pass | `65,536` | `PortHamiltonianFFN` + `CausalSourceCopyGate` |
| `HSPA_GEN5_4B_CONFIG` (`nanolm-hspa-gen5-4b`) | `3,842,826,561` | Pass | `65,536` | `PortHamiltonianFFN` + `CausalSourceCopyGate` |
| `TRANSFORMER_4B_CONFIG` | `3,843,291,648` | Pass | `65,536` | `3,072.0 MB` KV at `32,768` |
| `NanoLM-Ω-1.7B` / `Ω-v2` / `FP-v3` (Grafted `Qwen3-1.7B`) | `1,747,444,768` | Pass | `32,768` | `448 MB` KV @ `4K` + `HSPA-Ω` sidecar |
