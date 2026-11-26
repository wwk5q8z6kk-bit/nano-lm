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
| `NanoLM-Ω-v2` (`Merged_Greedy`, T4 FP16) | `1.747B` | `38/56` | `67.86%` `[55.4, 78.6]` | `3/4` | `3/4` | `3/4` | `2/4` | `1/4` | **`4/4`** | **`1/4`** | `2/4` | `6,164.5` | `0.3883` |
| **`NanoLM-Ω-v2` (`Cognitive_v2`, T4 FP16)** | **`1.747B`** | **`41/56`** | **`73.21%` `[60.7, 83.9]`** | `3/4` | `3/4` | `3/4` | `2/4` | **`4/4`** | **`4/4`** | **`1/4`** | `2/4` | `6,501.4` | **`0.4190`** |

*(Cycle 3 `EXP-Ω7` executed on Free Kaggle `2x Tesla T4` FP16 at `$0.00` with `196` merged nullspace layers, `6/6` exact greedy match, and `1.61x` end-to-end speedup `8.73 -> 14.09 tok/s`.)*

---

## 6. `EXP-Ω4` — Full-LM Non-Abelian $S_4$ Composition (`3` Seeds `101/202/303`, `steps=1200`, `train_len=32`, T4 `$0.00`)

| Model Arm | Params | `L32` (ID) | `L64` (`2x`) | `L128` (`4x` OOD) | `L256` (`8x` OOD) |
|---|---:|---:|---:|---:|---:|
| `b0-transformer-small_x1` | `2,460,480` | `0.0416 ± 0.0148` | `0.0208 ± 0.0148` | `0.0625 ± 0.0225` | `0.0277 ± 0.0049` |
| `b1-abelian-ssm-small_x1` | `2,460,864` | `0.0382 ± 0.0177` | `0.0452 ± 0.0049` | `0.0347 ± 0.0130` | `0.0556 ± 0.0196` |
| `gen1-h2-hspa-predcode-small_x1` | `2,466,630` | `0.0416 ± 0.0148` | `0.0416 ± 0.0148` | `0.0382 ± 0.0130` | `0.0625 ± 0.0085` |
| **`gen4-omega-so3-predcode-small_x1`** | **`2,466,732`** | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | **`0.9306 ± 0.0354`** |

---

## 7. `EXP-Ω5` — 3-Seed `14M` `Gen4` LM Evolution (`seeds=42/137/271`, `train_steps=800`, T4 `$0.00`)

| Model Arm | Params | Val BPB (`mean ± std`) | Throughput (`tok/s`) | Peak VRAM (`MB`) | Math Probe | State Probe | Registry Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `b0-transformer-micro` | `13,966,464` | `0.3824 ± 0.00242` | `58,669.9` | `1,173.2` | `2.78%` | `1.39%` | Baseline |
| **`gen1-h2-hspa-predcode-micro`** | `14,022,536` | **`0.3701 ± 0.00470`** | `6,945.4` | `3,810.0` | `8.33%` | **`13.19%`** | **Retains `BEST`** |
| `gen4-omega-so3-predcode-micro` | `13,948,944` | `0.3775 ± 0.00437` | `7,775.8` | `3,263.9` (`-14.3%`) | **`9.03%`** | `3.47%` | **`did_not_beat_best` vs `Gen1_H2`** |

---

## 8. `EXP-Ω6` — Static CUDA-Graph HSPA Decode & Nullspace Merge Microbench (`Tesla T4` FP16, `$0.00`)

| Benchmark | Config / Shape | Baseline Eager | Optimized (`CUDAGraph` / `Merge`) | Speedup | Parity / Error |
|---|---|---:|---:|---:|---|
| HSPA-Ω Decode (`seq=512, dec=32`) | `nanolm-hspa-omega-2b` (`1.808B`) | `158.413 ms/tok` | **`51.023 ms/tok`** (`1.00 MB` state) | **`3.10x`** | `greedy_parity=True`, `4,221.25 MB` VRAM |
| HSPA-Ω Decode (`seq=512, dec=32`) | `nanolm-hspa-omega-4b` (`3.841B <= 4B`) | `236.105 ms/tok` | **`100.291 ms/tok`** (`1.50 MB` state) | **`2.35x`** | `greedy_parity=True`, `8,409.83 MB` VRAM |
| Nullspace LoRA Merge | `[2048, 2048]` Linear (`k=48, r=16`) | `252.24 us` | **`33.42 us`** | **`7.55x`** | `max_diff=0.001953`, ortho err `5e-7` |
| Nullspace LoRA Merge | `[2048, 6144]` Linear (`k=48, r=16`) | `267.62 us` | **`100.47 us`** | **`2.66x`** | `max_diff=0.001953`, ortho err `7e-7` |

---

## 9. Cycle 4 (`NANOLM Ω-v3`, First-Principles Interaction Intelligence — [artifacts/nanolm_fp_v3](artifacts/nanolm_fp_v3), `$0.00` Compute)

### `EXP-FP1` — Typed Relational/Bitemporal Microcosm (`3` Seeds `101/202/303`, `~49.7K` Iso-Param, Kaggle `2x T4`)

| Arm | Params | Canonical | Paraphrase | `6x` OOD (`N=160`) | Lyapunov $\max\Delta\mathcal{H}$ |
|---|---:|---:|---:|---:|---:|
| `D0_Deterministic_Rules` | `0` | **`1.0000`** | `0.6143` | `0.6571` | N/A |
| `flat_transformer` | `49,779` | `0.8238 ± 0.1592` | `0.8524 ± 0.1174` | `0.4762 ± 0.1197` | N/A |
| `unsigned_gat` | `49,734` | **`1.0000`** | **`1.0000`** | **`0.8381 ± 0.0356`** | N/A |
| `signed_mesh` | `49,783` | `1.0000` | `1.0000` | `0.3809 ± 0.2763` | N/A |
| `sheaf_port_hamiltonian` | `49,688` | `1.0000` | `1.0000` | `0.5809 ± 0.0713` | **`0.0`** |
| `dynamic_rewire_ph` | `49,688` | `1.0000` | `1.0000` | `0.4381 ± 0.0883` | **`0.0`** |

### `EXP-FP2` — `Gen5` Sequence LM (`2A` $S_4$, `2B` 3-Seed `14M` AdamW + Copy-Gate Ablation, `2C` 2-Seed Muon, Kaggle `2x T4`)

| Arm (`14M` `EXP-FP2B` AdamW) | Params | Val BPB (`AdamW`) | Val BPB (`Muon`) | Exact Copy | Bitemporal Recall | $S_4$ `L256` (`2A`) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `b0-transformer-micro` | `13,966,464` | `0.3784 ± 0.00106` | `0.3849` | `0.00%` | `0.00%` | `3.91%` | Baseline |
| `b2-gated-deltanet-micro` | `14,018,688` | `0.3785 ± 0.00008` | `0.3865` | `0.00%` | `0.00%` | `6.25%` | Modern Recurrent Baseline |
| **`gen1-h2-hspa-predcode-micro`** | `14,022,536` | **`0.3751 ± 0.00217`** | **`0.3749`** | `0.00%` | `0.00%` | `3.12%` | **Retains `BEST` BPB** |
| `gen4-omega-so3-predcode-micro` | `13,948,944` | `0.3775 ± 0.00243` | — | `0.00%` | `0.00%` | **`97.66%`** | **Retains $S_4$ Dominance** |
| **`gen5-fp-ph-holonomy-micro`** | `14,008,545` | `0.3756 ± 0.00000` | `0.3793` | **`4.29%`** | **`96.67%`** | `3.12%` | **`201x` Copy-Gate Breakthrough (`did_not_beat_best` BPB)** |
| `gen5-fp-nocopy-micro` (ablation) | `13,971,296` | `0.3789 ± 0.00010` | — | `0.00%` | `0.48%` | — | Proves `CausalSourceCopyGate` causality |

### `EXP-FP3` — 8-Primitive Ablation & `VoI` Halting Pareto (`3` Seeds, Kaggle `2x T4`)
- **Full $\mathcal{U}_t$:** **`1.0000 ± 0.0000`**; all `8` single-primitive knockouts collapsed (`no_adaptation_weights=0.1198`, `no_identity_p_ij=0.1771`, `no_transition_clock_q_ij=0.2031`, `no_relation_edges=0.2448`, `no_info_source_ledger=0.2500`, `no_state=0.3125`, `no_objective_voi=0.4271`, `no_constraint_dissipation=0.6146`).
- **VoI Halting:** `voi_lambda=0.28` achieved **`1.0000 @ 3.375` mean steps**, strictly Pareto-dominating `fixed_k=4` (`1.0000 @ 4.00`) and `fixed_k=8` (`0.8229 @ 8.00`).

### `EXP-FP4` — `1.747B` `WorldController_v3` vs `EqualCompute_SC3` on `FP-5Axis-30` & `15D-56` (Local M3 Max MPS, `overlaps=0`)

| Arm (`FP-5Axis-30`) | Params | Overall | Accuracy `[95% CI]` | Calls | `rel` | `bitemp` | `exact` | `causal` | `xdom` |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `D0_Deterministic_Symbolic` | `0` | `11/30` | `36.67%` `[16.7, 53.3]` | `0` | `2/6` | `4/6` | `3/6` | `0/6` | `2/6` |
| `NanoLM_1.7B_Naked_Merged_Greedy` | `1.747B` | `9/30` | `30.00%` `[13.3, 46.7]` | `30` | `1/6` | `1/6` | `3/6` | **`2/6`** | `2/6` |
| `NanoLM_1.7B_EqualCompute_SC3` | `1.747B` | `9/30` | `30.00%` `[13.3, 46.7]` | `90` | `1/6` | `1/6` | `3/6` | **`2/6`** | `2/6` |
| **`NanoLM_1.7B_WorldController_v3`** | **`1.747B`** | **`15/30`** | **`50.00%` `[33.3, 66.7]`** | **`64`** | **`4/6`** | `3/6` | **`4/6`** | `1/6` | **`3/6`** |

- **`benchmark_15d` (`56` tasks):** `NanoLM_FP_v3_1.7B_Controller_v3` scored **`41/56 (73.21% [62.5%, 83.9%])`** (`coding=3/4`, `math_logic=4/4`, `summarization=4/4`, `general_knowledge=4/4`, `tool_use=4/4`, `hallucination_resistance=4/4`, `diagrams_charts=3/4`, `planning_decomposition=1/4`, `21` repairs, `cap/B=0.4190`, adapter SHA `66f2ce64e47ff701`).
