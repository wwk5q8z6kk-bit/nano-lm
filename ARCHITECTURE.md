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
| **`BREAKTHROUGH` (`1.75B`)** | **`NanoLM-Ω-1.7B` (`Naked_Greedy`)** | **`1,747,444,768`** | `Nullspace k=48 + HSPA-Ω Sidecar` | **`40/56 (71.43%)`, `coding=3/4`** | **Outperforms `8.19B` `Qwen3-8B` (`39/56 = 69.64%`)** |

---

## 2. Core Components Across [architecture.py](nanolm_holonomy/architecture.py), [triton_holonomy.py](nanolm_holonomy/triton_holonomy.py) & [omega_architecture.py](nanolm_holonomy/omega_architecture.py)

1. **Fused Non-Abelian Spinor Recurrence (`SpinorRegisterField` + `holonomy_scan_fast`):**
   Parallel associative quaternion scan $\Omega_t = g_t \star \Omega_{t-1} + s_t$ executed via OpenAI Triton (`_triton_holonomy_fwd_kernel`) with an analytical time-reversed conjugate adjoint storing only `omega_all` (`11.08x–35.00x` speedup over Cycle-1 PyTorch Blelloch scan), plus exact $O(1)$-memory step decode (`step_holonomy`, diff `< 2.2e-6`).
2. **Lie-Exp $SU(2) \to SO(3)$ Projective Adjoint (`HSPA-Ω`):**
   Replaces the singular Cayley chart at $\theta = \pi/2$ involutions with smooth `su2_lie_exponential(omega)` and resolves $\pm q$ double-cover interference using $\mathbb{Z}_2$-invariant `so3_projective_adjoint(q)` ($R(-q) = R(q)$), lifting unbroken $S_4$ generalization from `3.64%` (`Cayley`) to **`100.00%` (`EXP-Ω1`)**.
3. **Iterative Precision-Weighted Free-Energy FFN (`PredictiveCodingCell`):**
   Two-step prediction-error correction outperforming static SwiGLU across Cycle-1 and 3-seed Cycle-2 LM evolution (`0.3693` vs `0.3798` BPB).
4. **Spectral Nullspace Continual Adaptation (`SpectralNullspaceLoRALinear`) + Zero-Oracle Controller ([omega_system.py](nanolm_holonomy/omega_system.py)):**
   Constrains fine-tuning updates via $\Delta W = \frac{\alpha}{r} B A (I - V_k V_k^\top)$ so $\Delta W V_k \equiv 0$, cutting continual Domain-$A$ forgetting across `3/3` seeds (`+4.47 -> +3.83`) and recovering Python `coding` (`0/4..1/4 -> 3/4`) in `1.747B` fine-tuning (`EXP-Ω2` & `EXP-Ω3`).

---

## 3. Flagship 2B & <= 4B Configurations

| Config | Parameters | `<= 4B` Valid | Sequence Horizon (`max_seq_len`) | Persistent State (`32K`) |
|---|---:|---:|---:|---:|
| `HSPA_2B_CONFIG` (`nanolm-hspa-2b`) | `1,808,239,648` | Pass | `65,536` | `1.19 MB` ($O(1)$) |
| `HSPA_4B_CONFIG` (`nanolm-hspa-4b`) | `3,845,419,056` | Pass | `65,536` | **`1.781 MB`** (`1,725x` smaller than KV) |
| `TRANSFORMER_4B_CONFIG` | `3,843,291,648` | Pass | `65,536` | `3,072.0 MB` KV at `32,768` |
| `NanoLM-Ω-1.7B` (Grafted `Qwen3-1.7B`) | `1,747,444,768` | Pass | `32,768` | `448 MB` KV @ `4K` + `HSPA-Ω` sidecar |
