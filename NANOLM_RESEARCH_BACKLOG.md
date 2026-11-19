# NANOLM Ω — Ranked Autonomous Research Backlog (`TASK F`)

> Ranking weights: Expected Scientific Value (`35%`), Capability Lift (`30%`), Feasibility / Compute Efficiency (`25%`), Reproducibility & Low Risk (`10%`). Scores `1–10`.

| Rank | ID | Hypothesis / Initiative | Sci | Lift | Feas | RiskOk | Composite | Target Cost | Current Status |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| **1** | **`H-Ω01`** | Lie-Exp $SU(2) \to SO(3)$ Adjoint Holonomy ($S_4$ `16x` OOD) | `9.8` | `9.6` | `9.8` | `9.8` | **`9.74`** | `\$0.01` | **Executing (`EXP-Ω1`)** |
| **2** | **`H-Ω06`** | `SpectralNullspaceLoRA` Continual Anti-Forgetting | `9.3` | `9.5` | `9.7` | `9.6` | **`9.48`** | `\$0.02` | **Executing (`EXP-Ω2` & `Ω3`)** |
| **3** | **`H-Ω15`** | Zero-Oracle `OmegaCognitiveController` (`v2`: Top-2 margin + JSON + Mermaid + Multi-Probe) | `9.0` | `9.8` | `9.6` | `9.5` | **`9.45`** | `\$0.00` | **Completed (`EXP-Ω7`, `41/56`)** |
| **4** | **`H-Ω02`** | Fused OpenAI Triton Holonomy Scan + $O(1)$ Conjugate Adjoint | `9.5` | `9.2` | `9.2` | `9.5` | **`9.34`** | `\$0.70` | **Completed (`C2-X0`, `35x`)** |
| **5** | **`H-Ω19`** | Roofline `ComputePlanner` Pre-Flight Estimator | `8.5` | `8.8` | `9.9` | `9.9` | **`9.08`** | `\$0.00` | **Implemented & Calibrated (`12.9%`)** |
| **6** | **`H-Ω11`** | Precision-Weighted `PredictiveCodingCell` FFN | `9.2` | `9.0` | `9.0` | `9.2` | **`9.10`** | `\$1.30` | **Promoted (`BEST` in `C1/C2`)** |
| **7** | **`H-Ω22`** | Early-Stop 2-Ep SFT + High-KL (`0.10`) Selective GRPO | `8.2` | `8.9` | `9.3` | `9.0` | **`8.77`** | `\$0.10` | **Executing (`EXP-Ω3`)** |
| **8** | **`H-Ω12`** | Top-2 Hypothesis-Margin Routing ($1 - (p_{(1)}-p_{(2)})$) | `9.1` | `8.7` | `8.4` | `8.8` | **`8.79`** | Sibling Ref | **Verified (`nanolm-lab` `#1` BPB)** |
| **9** | **`H-Ω20`** | CUDA-Graph + Projection-Fused Decode Kernel (`1.81B` & `3.84B <= 4B`) | `8.4` | `9.1` | `8.1` | `8.8` | **`8.58`** | `\$0.00` | **Completed (`EXP-Ω6`: `3.10x` / `2.35x`)** |
| **10** | **`H-Ω04`** | `3:1` Hybrid `HSPA-Ω` + `Diff-GQA` Pretraining (`125M–350M`) | `9.0` | `9.0` | `7.2` | `8.2` | **`8.48`** | `\$4.50` | **Next Cycle Priority #2** |
| **11** | **`H-Ω17`** | Small-$N$ Brute-Force Induction Oracle for Math Discovery | `8.6` | `8.9` | `8.0` | `8.4` | **`8.53`** | `\$0.15` | **Queued** |
| **12** | **`H-Ω14`** | Margin-Driven Holonomic Register Write Allocation | `8.7` | `8.4` | `8.2` | `8.2` | **`8.44`** | `\$0.80` | **Queued** |
| **13** | **`H-Ω13`** | Damped Jacobi Equilibrium + Entropy Early Exit (`C4_JCLE`) | `8.5` | `8.2` | `8.0` | `8.5` | **`8.29`** | `\$0.37` | **Supported (`E1`, `60.5%` exit)** |
| **14** | **`H-Ω23`** | Dual-Teacher (`8B + 32B`) Top-K+Tail Logit KD into `HSPA-2B` | `8.1` | `8.8` | `7.6` | `8.3` | **`8.21`** | `\$3.50` | **Queued** |
| **15** | **`H-Ω18`** | Dual-Process Epistemic Abstention Head (Brier-Calibrated) | `8.8` | `8.0` | `7.8` | `8.0` | **`8.23`** | `\$0.60` | **Queued** |
| **16** | **`H-Ω24`** | Multi-Axis Pareto Non-Regression Registry Gate | `7.9` | `8.1` | `9.0` | `9.5` | **`8.40`** | `\$0.00` | **Active (`ExperimentRegistry`)** |
| **17** | **`H-Ω16`** | Counterfactual AST Operator Rebinding | `8.0` | `8.1` | `8.3` | `8.2` | **`8.12`** | `\$0.10` | **Queued** |
| **18** | **`H-Ω07`** | Zero-Init Step-0 KL-Invariant Graft Verification | `8.0` | `8.2` | `9.2` | `9.5` | **`8.51`** | `\$0.00` | **Active Gate (`diff == 0.0`)** |
| **19** | **`H-Ω21`** | NF4 Base + FP32 Holonomic Sidecar Edge Serving (`< 2 GB`) | `7.5` | `8.3` | `7.9` | `8.0` | **`7.90`** | `\$0.15` | **Queued** |
| **20** | **`H-Ω05`** | State-Torsion Dependent Curvature Connection | `8.6` | `7.4` | `7.0` | `7.2` | **`7.70`** | `\$0.50` | **Exploratory** |
| **21** | **`H-Ω09`** | Spectral Entropy Layerwise Nullspace Rank Allocation | `7.7` | `7.6` | `8.0` | `7.8` | **`7.76`** | `\$0.25` | **Queued** |
| **22** | **`H-Ω10`** | K-FAC Activation-Weighted Nullspace Projection | `8.1` | `7.5` | `6.8` | `7.4` | **`7.54`** | `\$0.40` | **Exploratory** |
| **23** | **`H-Ω03`** | Bifurcated Connection Bias Alone (`Gen3_H4`) | `7.0` | `6.1` | `8.8` | `8.5` | **`7.18`** | `\$0.25` | **Superseded by `H-Ω01`** |
| **24** | **`H-Ω08`** | Asymmetric Differential Delta (`C2_DiffNSDelta`) | `6.2` | `5.0` | `7.2` | `8.5` | **`6.32`** | `\$0.10` | **Rejected (`E1`)** |
