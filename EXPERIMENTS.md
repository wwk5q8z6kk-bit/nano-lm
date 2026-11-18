# Experiments — Complete Telemetry Across Cycles 1, 2 & `Ω`

> Source directories: [artifacts/nanolm_holonomy_v1](artifacts/nanolm_holonomy_v1) (`$1.3026` L40S), [artifacts/nanolm_holonomy_v2](artifacts/nanolm_holonomy_v2) (`$0.7021` L40S), [artifacts/nanolm_omega_v1](artifacts/nanolm_omega_v1) (`$0.3201` A40).

---

## 1. `C2-X0` — Fused OpenAI Triton Kernel vs PyTorch Blelloch Scan (`L40S`, [c2_x0_kernel_microbench.json](artifacts/nanolm_holonomy_v2/c2_x0_kernel_microbench.json))

Shape `[B=4, R=8, G=48, seq_len, 4]`, mean over `6` `fwd+bwd` iterations:

| Sequence Length | PyTorch Blelloch (`ms`) | PyTorch VRAM (`MB`) | Fused Triton + $O(1)$ Adjoint (`ms`) | Triton VRAM (`MB`) | Speedup (`x`) |
|---:|---:|---:|---:|---:|---:|
| `256` | `12.158` | `153.73` | **`1.097`** | **`85.07`** | **`11.08x`** |
| `512` | `16.603` | `319.55` | **`1.468`** | **`168.07`** | **`11.31x`** |
| `1,024` | `31.652` | `685.55` | **`1.348`** | **`336.07`** | **`23.48x`** |
| `2,048` | `168.952` | `1,467.05` | **`4.986`** | **`672.07`** | **`33.89x`** |
| `4,096` | `529.037` | `3,126.05` | **`15.116`** | **`1,344.07`** (`2.33x` lower) | **`35.00x`** |

---

## 2. `EXP-Ω1` — Non-Abelian $S_4$ Holonomy & `16x` OOD Extrapolation (`3` Seeds, [exp_omega1_s4_holonomy.json](artifacts/nanolm_omega_v1/exp_omega1_s4_holonomy.json))

Trained strictly on `L=32` (`600` steps), tested zero-shot up to `L=512`:

| Controlled Arm (`n=3` Seeds) | Params | `L=32` | `L=64` (`2x`) | `L=128` (`4x`) | `L=256` (`8x`) | `L=512` (`16x`) |
|---|---:|---:|---:|---:|---:|---:|
| `b0_transformer_rope` | `646,208` | `0.0521` | `0.0417` | `0.0417` | `0.0182` | `0.0391 ± 0.0064` |
| `b1_abelian_u1` | `1,368` | `0.0312` | `0.0417` | `0.0365` | `0.0417` | `0.0599 ± 0.0133` |
| `h1_su2_lie_linear` | `2,904` | `0.0417` | `0.0313` | `0.0443` | `0.0391` | `0.0443 ± 0.0037` |
| `h2_su2_cayley_adjoint` | `5,784` | `0.3646` | `0.1536` | `0.0703` | `0.0364` | `0.0364 ± 0.0133` |
| **`h3_omega_so3_lie_adjoint` (`HSPA-Ω`)** | **`5,784`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000 ± 0.0000`** |

*(Cycle-1 `X1` & Cycle-2 `C2-X1` full-LM probes prior to Lie-Exp + $SO(3)$ adjoint upgrade remained at `2.8%–7.3%`, directly motivating `EXP-Ω1`.)*

---

## 3. `C2-X2` & `EXP-Ω2` — 3-Seed LM Evolution & Spectral Nullspace Continual Plasticity

### `C2-X2`: Triton 3-Seed `~14M` Multi-Domain LM (`L40S`, [c2_x2_triton_lm.json](artifacts/nanolm_holonomy_v2/c2_x2_triton_lm.json))

| Architecture | Params | Mean Val BPB (`± std`) | State Probe Acc | Math Probe Acc | Train `tok/s` | Decode `ms/tok` | `ExperimentRegistry` |
|---|---:|---:|---:|---:|---:|---:|---|
| `b0-transformer-micro` | `13,966,464` | `0.3798 ± 0.00093` | `0.69%` | `4.86%` | **`139,238`** | **`5.508`** | `BASELINE` |
| **`gen1-h2-hspa-predcode-micro`** | `14,022,536` | **`0.3693 ± 0.00041`** | `13.20%` (`19.1x`) | **`6.94%`** | `49,633` (`3.77x` C1) | `12.198` (`2.05x` C1) | **`BEST` Promoted** |
| `gen3-h4-hspa-wavepred-micro` | `14,033,944` | `0.3696 ± 0.00082` | **`15.28%` (`22.1x`)** | `6.25%` | `49,143` | `13.131` | Rejected vs `0.3693` |

### `EXP-Ω2`: Domain $A \to B$ Continual Adaptation (`3` Seeds, [exp_omega2_nullspace_continual.json](artifacts/nanolm_omega_v1/exp_omega2_nullspace_continual.json))

| Arm (`pre_dom_a_bpb = 0.3897 ± 0.0016`) | Post Dom-A BPB | Forgetting ($\Delta\text{BPB}$) | Dom-B Plasticity BPB | Joint Mean BPB |
|---|---:|---:|---:|---:|
| `Standard_LoRA` | `4.8599 ± 1.0755` | `+4.4703 ± 1.0759` | **`1.4245 ± 0.0307`** | `3.1422 ± 0.5322` |
| `Proximal_L2_LoRA` (`wd=0.25`) | `4.7760 ± 1.0701` | `+4.3863 ± 1.0705` | `1.4245 ± 0.0309` | `3.1002 ± 0.5287` |
| **`Spectral_Nullspace_LoRA` (`k=32`)** | **`4.2190 ± 0.4647`** | **`+3.8293 ± 0.4658`** | `1.4261 ± 0.0305` | **`2.8225 ± 0.2209`** |

---

## 4. `C2-X3` — `3.845B <= 4B` Scaling up to `32,768` Tokens ([c2_x3_32k_scaling.json](artifacts/nanolm_holonomy_v2/c2_x3_32k_scaling.json))

| Context `T` | `Transformer-4B` VRAM (`MB`) | `Transformer-4B` Decode (`ms`) | `HSPA-4B` VRAM (`MB`) | `HSPA-4B` Decode (`ms`) | State Ratio (`KV / HSPA`) |
|---:|---:|---:|---:|---:|---:|
| `512` | **`7,524.65`** | **`18.741`** | `8,408.57` | `74.452` | `27.0x` |
| `2,048` | **`7,831.24`** | **`17.714`** | `9,261.52` | `74.944` | `107.8x` |
| `8,192` | `9,039.79` | **`23.990`** | `9,261.57` | `75.294` | `431.2x` |
| `16,384` | `10,655.84` | **`42.057`** | **`9,261.63`** | `74.911` | `862.4x` |
| **`32,768`** | `13,887.97` | `75.026` | **`9,261.76` (`-4,626 MB`)** | **`74.264` (Faster)** | **`1,724.8x` (`1.781 vs 3,072 MB`)** |

---

## 5. `C2-X4` & `EXP-Ω3` — `1.747B` vs `8.191B` 15-Dimension Objective Benchmark

*(Note: `NanoLM-Ω-1.7B` evaluated on `NVIDIA A40`; `Qwen3-8B` and `NanoLM-v2` evaluated on `NVIDIA L40S` under identical deterministic greedy decoding and zero-contamination `56`-task verifier harness.)*

| Arm | Params | Overall | Accuracy `[95% CI]` | `cod` | `math` | `sum` | `json` | `diag` | `tol` | `plan` | `mem` | Peak VRAM (`MB`) | Cap / B |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-1.7B` Base (`E2`) | `1.721B` | `34/56` | `60.71%` `[48.2, 71.4]` | `3/4` | `1/4` | `3/4` | `3/4` | `3/4` | `2/4` | `0/4` | `1/4` | `3,899.1` | `0.3528` |
| `Holonomic-v2_Greedy` (`C2-X4`) | `1.747B` | `35/56` | `62.50%` | `1/4` | `2/4` | **`4/4`** | `2/4` | `2/4` | **`4/4`** | `0/4` | `2/4` | `4,296.8` | `0.3577` |
| `External Qwen/Qwen3-8B` | **`8.191B`** | `39/56` | `69.64%` | **`4/4`** | `1/4` | `3/4` | **`3/4`** | **`4/4`** | `2/4` | **`1/4`** | **`3/4`** | `16,541.5` | `0.0850` |
| **`NanoLM-Ω-1.7B` (`Naked_Greedy`)** | **`1.747B`** | **`40/56`** | **`71.43%` `[60.7, 82.1]`** | `3/4` | **`4/4`** | `3/4` | `2/4` | `2/4` | **`4/4`** | **`1/4`** | `2/4` | **`4,403.0`** | **`0.4088`** |
| `NanoLM-Ω-1.7B` (`Cognitive_Sys`) | `1.747B` | `39/56` | `69.64%` `[57.1, 80.4]` | `3/4` | **`4/4`** | `3/4` | `2/4` | `3/4` | `3/4` | **`1/4`** | `1/4` | `4,417.3` | `0.3985` |
