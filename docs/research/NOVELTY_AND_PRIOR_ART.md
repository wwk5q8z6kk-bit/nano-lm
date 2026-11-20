# Novelty Audit & Contemporary Literature Registry (Through Nov 2026)

This is part of the [November 17 research proposal](FOUNDATION_INTELLIGENCE_PROGRAM.md). Its size limits, experiments and reported scores belong to that proposal. The [current charter](../PROJECT_CHARTER.md) defines NanoLM’s scope.

**Policy:** Existing published mechanisms are **prior art**, never Nano inventions. A Nano novelty claim is permitted only after prior-art review, controlled ablation isolating the exact mechanism, analytic/Jacobian verification, multi-seed replication, and matched modern baselines.

---

## 1. Primary 2024–2026 Prior Art Registry

| Ref | Paper / Model (Year) | Core Mechanism | Relationship & Overlap with Nano |
|---|---|---|---|
| 1 | Dao & Gu, *Mamba-2 / SSD* (ICML 2024) | Structured State Space Duality; chunked matrix forms | Baseline recurrent duality paradigm (`P1_gla` family) |
| 2 | Yang et al., *Gated Delta Networks* (ICLR 2025) | $S_t = \alpha_t(I - \beta_t k_t k_t^\top)S_{t-1} + \beta_t k_t v_t^\top$ with $\beta_t \in (0,1)$ | Direct ancestor of Nano `H2_delta`/`Sharp-Delta` (`beta in (0,1)` contraction baseline) |
| 3 | Kimi Team, *Kimi Linear* (2025) | Per-channel Kimi Delta Attention (`KDA`) + 3:1 global anchors | Direct prior art for `3:1` hybrid layout and per-channel forget gates (`AX04`) |
| 4 | Lahoti et al., *Mamba-3* (ICLR 2026) | Complex/rotational state transitions, stronger discretization, MIMO updates | Key 2026 comparator for `Householder-Delta` non-commutative state tracking (`AX03`) |
| 5 | Hatamizadeh et al., *Gated DeltaNet-2* (2026) | Independently gated channelwise erase and write controls + chunked kernels | Direct 2026 competitor addressing single-gate erasure/write coupling (`AX02`) |
| 6 | Kimi Team, *Attention Residuals (`AttnRes`)* (2026) | Per-token softmax attention over prior layer/block depth states | Direct prior art for Nano `Depth-AttnRes` (`M3` / `AX09`) — must benchmark side by side |
| 7 | Xie et al., *mHC: Manifold-Constrained Hyper-Connections* (ICML 2026) | Doubly-stochastic / manifold residual mixing for stability at scale | Alternative depth-routing stability control (`AX10`) |
| 8 | DeepSeek Team, *Engram: Conditional Memory via Scalable Lookup* (2026) | $O(1)$ hash-addressable knowledge lookup table decoupled from FFN FLOPs | Prior art for `X6_pkm_muon` / `MM05` conditional lookup |
| 9 | Behrouz et al., *Titans* (2025) | Surprise-gated neural long-term test-time memory | Direct prior art for `X3_sgrd` / `MM06` surprise-driven memory |
| 10 | Behrouz & Mirrokni, *Nested Learning* (NeurIPS 2025) | Multi-timescale nested optimization & update frequencies | Direct prior art for `MM07` stability–plasticity |
| 11 | *In-Place Test-Time Training* (2026) | Fast-weight adaptation inside deployed checkpoints without global rewrites | Prior art for `MM08` test-time adaptation |
| 12 | Meta, *Byte Latent Transformer (`BLT`)* (2024) | Entropy-driven dynamic byte patching without fixed subword tokenizer | Direct prior art for `RP02` rare drug/identifier exactness |
| 13 | Wang et al., *AdaRoPE* (ICML 2026) | Adaptive rotary position encodings | Context/length-extrapolation prior art |
| 14 | Liu et al., *Muon is Scalable for LLM Training* (2025) | Newton-Schulz orthogonalized matrix momentum | **Accounts for `-6.33%` of Track A's `-7.01%` BPB gain** (`OP01`) |
| 15 | Li et al., *DataComp-LM (`DCLM`)* (NeurIPS 2024) | Model-based quality/reasoning corpus filtering | Prior art for `DT04` |
| 16 | Sardana et al., *Beyond Chinchilla-Optimal* (ICML 2024) | Inference-compute-aware overtraining horizons for compact LMs | Governs `L3` token-budget decisions (`DT` / `L3`) |
| 17 | DeepSeek-V3 (2024) & DeepSeek-R1 (Nature 2025) | MLA (`AX06`), Aux-Loss-Free MoE (`AX14`), MTP (`DT07`), pure RLVR (`GRPO`) | Prior art for `nanolm` MLA/MoE/MTP/RLVR building blocks |
| 18 | Setlur et al. (ICML 2025); *Thinking Hard, Not Smart* (2026) | Verifier-needed test-time scaling; shared-budget VoI limits | Motivates `RE02` / `X08` |
| 19 | *Less is More: Recursive Reasoning with Tiny Networks* (2025); Juršs et al. (2024) | Compact shared-core recursive refinement | Direct prior art for `D` / `RE03` |
| 20 | Meta, *V-JEPA 2* (2025) | Latent action-conditioned predictive world modeling | Prior art for `SY04` / Sec. 4.9 |
| 21 | Nie et al., *LLaDA* (NeurIPS 2025) | Large language masked diffusion | Prior art for challenger track `F` / `AX15` |
| 22 | Lin et al., *Domain-Shift-Aware Conformal Prediction* (ICML 2026) | Selective risk bounds under covariate shift | Prior art for `SY03` |
| 23 | *Symplectic Neural Flows* (SIAM 2026) | Structure-preserving continuous dynamics | Prior art for `OP09` (physics sims only) |
| 24 | Open weights: `Qwen3.5-2B`, `MiniCPM5-2B` (`2.52B`), `Granite 4.2 3B`, `Phi-4-mini` (`3.84B`), `Gemma-4-E2B` (`5.1B` out-of-cap) | Reference 2026 compact model baselines | Required `Track 1` comparators |

---

## 2. Classification: Prior Art vs Candidate Nano Hypotheses

| Mechanism | Classification | Required Evidence Before Any Novelty Claim |
|---|---|---|
| `3:1` DeltaNet + GQA hybrid (`DDDA`) | **Prior-art architecture class** (`Kimi Linear`, `Qwen3.5-2B`) | Matched comparison against `Qwen3.5-2B` and `KDA` |
| `Muon` optimizer (`-6.33%` BPB) | **Prior art** (Liu et al., 2025) | Never cite `Muon` gain as an architecture gain |
| `Depth-AttnRes` (`M3`) & `Canon-SwiGLU` (`M4`) | **Closely related to 2026 `AttnRes` / `Canon` convs** | Multi-seed `X06` vs `AttnRes` + `mHC` at matched params & >100M tokens |
| `Diff-GQA` anchor | **Prior art** (Differential Attention adapted to sparse anchors) | Multi-needle `X05` (`1..32` needles, `8k+`) beyond `79k` toy net |
| $\beta \in (0,2)$ Householder-Delta + `Sharp-V` + `Gated-VRes` | **Candidate Nano architectural composition** | `X03`/`X04` vs `GDN-2` & `Mamba-3` + `X10` iso-param/iso-FLOP multi-seed transfer |
| `GeoEraseWrite` + `ExactEvidenceStore` (`Track 2`) | **Candidate Nano co-design hypothesis** | `X07` + `X15`/`X16` showing Pareto gain on grounding, deletion & energy |
