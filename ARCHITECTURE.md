# NanoLM Architecture — Holonomic Spinor Predictive Architecture (`HSPA`)

> Autonomous architectural discovery target: maximize reasoning, state tracking, coding, mathematics, and memory efficiency per parameter and per context token within a hard `4,000,000,000` (`4B`) total parameter ceiling (`~2B` preferred development scale, `~3.85B` flagship configuration).

---

## 1. Architectural Versions Tracked (`BASELINE`, `BEST`, `EXPERIMENTAL`)

Recorded verbatim in [x2_lm_evolution.json](artifacts/nanolm_holonomy_v1/x2_lm_evolution.json) via `ExperimentRegistry`:

| Role | Identifier | Parameters | State @ 8K Context | Mean Val BPB (2 Seeds) | Status |
|---|---|---:|---:|---:|---|
| **`BASELINE`** | `b0-transformer-micro` (`B0_Transformer`) | 13,966,464 | 24,576.0 KB (linear $O(T)$ KV) | `0.3794 ± 0.00090` | Canonical GQA + QK-Norm + RoPE + SwiGLU baseline |
| **`BEST`** | `gen1-h2-hspa-predcode-micro` (`Gen1_H2_HSPA_PredCode`) | 14,022,536 | **24.0 KB** ($O(1)$, `1,024x` smaller) | **`0.3723 ± 0.00235`** | **Promoted** (`beat_best: 0.3794 -> 0.3723`; 13x higher state-probe accuracy) |
| **`EXPERIMENTAL`** | `gen2-h3-hspa-evolved-micro` (`Gen2_H3_HSPA_Evolved`) | 14,027,648 | 33.0 KB ($O(1)$) | `0.3803 ± 0.00180` | Rejected vs `BEST` (removing PredCode reverted BPB to SwiGLU parity) |
| **Control (Abelian)** | `b1-abelian-ssm-micro` (`B1_Abelian_SSM`) | 14,010,240 | 24.0 KB ($O(1)$) | `0.3826 ± 0.00016` | Rejected (`0.3826 > 0.3794`), isolating SU(2) > U(1) |
| **Gen-1 Control** | `gen1-h1-hspa-spinor-micro` (`Gen1_H1_HSPA_Spinor`) | 14,019,456 | 24.0 KB ($O(1)$) | `0.3800 ± 0.00040` | Rejected vs `B0` (beat `B1` U(1), isolated SwiGLU vs PredCode gap) |

---

## 2. Core Computational Paradigm (`HSPAUnifiedLM` in [architecture.py](nanolm_holonomy/architecture.py))

Instead of quadratic token-pair softmax routing (`O(T^2)`) or commutative diagonal SSM recurrences, **`HSPA`** unifies two mechanisms per block:

1. **Sequence Mixing — Non-Abelian $SU(2)$ Spinor Register Field (`SpinorRegisterField`):**
   - Each sequence layer maintains $R$ competitive persistent memory registers, where every register stores $G = d_{\text{fiber}} / 4$ **quaternionic spinor fibers** ($\Omega_{t,r} \in \mathbb{H}^G$).
   - Each incoming hidden vector $h_t \in \mathbb{R}^d$ computes competitive write attention $w_t \in \Delta^{R-1}$, retention/overwrite strength $\beta_{t,r} \in (0,1)$, transport mobility $m_{t,r} \in (0,1)$, Lie generator $\boldsymbol{\omega}_{t,r} \in \mathfrak{su}(2)^G$, binding rotor $\rho_{t,r} \in SU(2)^G$, query unbinding rotor $q_{t,r} \in SU(2)^G$, and payload spinor $\phi_t \in \mathbb{H}^G$.
   - Training and prompt prefill execute in **$O(\log T)$ parallel depth** via Hillis-Steele associative scan (`parallel_holonomy_scan`), while incremental generation uses an **exact $O(1)$-memory single-token update** (`step_holonomy`, verified `inc_diff < 2e-6` against full parallel prefill across all architectures).
2. **Channel / Inference Dynamics — Precision-Weighted Predictive-Coding Cell (`PredictiveCodingCell`):**
   - Experimentally promoted component replacing static SwiGLU FFNs at exact iso-parameter budget (`3` projection matrices `W_enc`, `W_dec`, `W_err`): initializes latent hypothesis $c^{(0)} = \text{SiLU}(W_{\text{enc}} z)$, measures top-down reconstruction error $\varepsilon^{(k)} = z - W_{\text{dec}} c^{(k)}$, and applies surprise-precision-gated error corrections over `K=2` internal steps.
   - Controlled X2 ablations established that **removing `PredictiveCodingCell` caused `Gen1_H1` (`0.3800`) and `Gen2_H3` (`0.3803`) to fall behind `Gen1_H2` (`0.3723`)**.

---

## 3. Flagship 2B & <= 4B Configurations

Verified exact integer match between analytical `HSPAArchConfig.parameter_count` and instantiated PyTorch `num_parameters`:

| Config | Parameters | `<= 4B` Check | Persistent State @ 16K Context (BF16) |
|---|---:|---:|---:|
| `HSPA_2B_CONFIG` (`nanolm-hspa-2b`) | `1,808,214,016` | Pass (`~1.81B`) | `1.19 MB` (flat across context length) |
| `HSPA_4B_CONFIG` (`nanolm-hspa-4b`) | `3,845,356,032` | Pass (`< 4.00B`) | **`1.781 MB`** (flat across `512`–`16,384` tokens) |
| `TRANSFORMER_4B_CONFIG` (`nanolm-transformer-4b`) | `3,843,291,648` | Pass (`< 4.00B`) | `1,536.0 MB` KV cache at `16,384` (`862x` larger) |

For adapting existing `<= 4B` Transformer checkpoints, [sidecar_evolution.py](nanolm_holonomy/sidecar_evolution.py) additionally defines `HolonomicRegisterSidecar`, a zero-initialized (`sidecar_gate = 0` -> exact `0.00e+00` step-0 logit preservation) upper-layer $SU(2)$ register sidecar trained with early-stopped SFT and Verifier-Guided GRPO.
