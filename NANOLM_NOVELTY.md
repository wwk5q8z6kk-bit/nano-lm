# NANOLM Novelty Audit & Original Contributions

To distinguish authentic architectural invention from rebranding, every mechanism implemented in [`nanolm_discovery/`](nanolm_discovery/) is classified below into **adapted prior art**, **non-trivial mathematical extensions**, and **original architectural syntheses**.

---

## 1. Three-Bucket Epistemic Classification

| Classification | Mechanism | Exact Provenance / Formulation | Empirical Evidence & Verification Status |
|---|---|---|---|
| **Bucket A: Adapted Prior Art** *(Credited explicitly)* | `B0_TRANSFORMER` (GQA + RoPE + SwiGLU + Pre-RMSNorm) | Standard autoregressive baseline ([`operators.py`](nanolm_discovery/operators.py), [`models.py`](nanolm_discovery/models.py)) | Immutable baseline control across `115k`, `8.65M–47.64M`, `222M`, `3.89B`. |
| **Bucket A: Adapted Prior Art** | `B1_GDN2_HYBRID` (3:1 Gated DeltaNet-2 + GQA) | Contractive delta rule ($\beta_t \in (0, 1)$) + `CausalKeyShift` (`EXP-INV-01/02`) | Strong overwrite (`0.965`), weak $S_3$ (`0.188`), `2.4818` BPB (`250` steps). |
| **Bucket A: Adapted Prior Art** | `MuonAdamW` + `JEP-CE` | 5-step Newton–Schulz matrix orthogonalization (`optim.py`) + cosine target (`models.py`) | Short-horizon penalty (`+0.034` BPB with `0.10 JEP`); pure-CE `Muon` caught up steeply (`-0.359` BPB across steps `250->1000`). |
| **Bucket B: Mathematical Extension** | Interleaved $M=2T$ scalar-decayed nilpotent **WY Metriplectic Chunk Scan** (`wy_metriplectic_chunk_scan`) | Extends delta-rule WY factorization to **reflection gates $\beta_t \in (0, 1.98)$** via 2-microstep interleaving while proving exponential condition bound **$C_{\text{crit}} \approx 41.6$** (`chunk_size=32`) and isolating `float32` via `_no_autocast_ctx` | Unconditional $\|A_t\|_2 \le 1$; `< 2e-6` CUDA `float16`/CPU `fp32` parallel–step equivalence; removing $\beta>1$ collapses `Needle_OOD` (`0.999 -> 0.364`). |
| **Bucket B: Mathematical Extension** | Asymmetric **4-tap `CausalKeyShift`** on addressing `(q, k, u, scam_q, scam_k)` | Separates bigram/delimiter-shifted addressing from clean payload `(v, scam_v)` using `4d` params and `6d`-byte `bf16` ring state | Required for induction binding (`OW_4x` jumped `0.146` unshifted `I3` -> `0.977` shifted `G3`). |
| **Bucket C: Original Synthesis** | Strictly Causal Top-$K$ Surprise-Evicted **SCAM Register Bank** (`scam_causal_topk_attention` $\leftrightarrow$ `scam_step_attention`) | Ranks causal history $\{0..t\}$ by continuous predictive surprise $s_j + 10^{-6} j$ with deterministic top-$K$ (`K=32`) heap parity between batched `topk` and single-token eviction | Exact `< 1e-6` train–inference match; ablating SCAM dropped `Needle_OOD` (`0.999 -> 0.729`). |
| **Bucket C: Original Synthesis** | Retrieval-Normalized Epistemic Uncertainty $u_t^{(G_3)}$ | Replaces inverted raw $1/(1+s_t)$ and spurious `1/prec` with surprise divided by retrieved subspace RMS: $s_t / \sqrt{\|y^{\text{met}}\|_{\text{rms}}^2 + \|y^{\text{scam}}\|_{\text{rms}}^2 + 10^{-3}}$ | Eliminates Cycle 1–2 epistemic inversion (`0.204`/`0.362` -> `0.686` internal / `0.875` entropy AUROC). |
| **Bucket C: Original Synthesis** | **`G3_HMWR`** Tri-Scale Operator + Tokenwise **Depth-Attention Residuals** + **Surprise-Adaptive Synthesis** | Rebalances sequence operator cost (`8.5 d^2 -> 6.5 d^2`) via $H/2$ WY + Wave-4 + shifted `d/4` SCAM so `ffn_hidden` stays `304–384` (vs `32–64`), while routing across depth $H_{0:\ell}$ and gating SwiGLU by $\log(1+u_t)$ | Depth/adaptive ablation proved **`+0.0125 BPB`** (`>30 sigma`) and **`+0.095 S3_4x`**; `4x` Kaggle horizon beat `I3_HRWN` by **`+0.0312` BPB** and beat **`B0` on `2x` OOD context (`2.1322` vs `2.1526`)**. |

---

## 2. Dynamic Five-Axis Innovation Scores (`discovery_results.json`)

Computed automatically by [`scripts/run_autonomous_discovery_cycles.py`](scripts/run_autonomous_discovery_cycles.py#L774-L814) using observed Cycle 1–3 OOD gains, multi-domain BPB deltas relative to `B0_TRANSFORMER`, and `log10` state-compression ratios at `T = 32,768`:

| Candidate | Originality | Mathematical Coherence | Experimental Feasibility | Computational Efficiency | Capability Gain | Mean Score (`1–10`) |
|---|---:|---:|---:|---:|---:|---:|
| **`g3_hmwr`** *(Cycle 3)* | `9.20` | `9.40` | **`9.30`** | `7.31` | **`10.00`** | **`9.042`** |
| `g2_mprn` *(Cycle 2)* | `8.80` | `9.40` | `9.00` | `8.01` | `10.00` | `9.042` |
| `g2_no_precision` *(Cycle 2 abl.)* | `8.70` | `9.30` | `9.00` | `8.02` | `10.00` | `9.004` |
| `i2_nebn` *(Cycle 1)* | `8.40` | `9.00` | `9.20` | `8.21` | `9.61` | `8.884` |
| `g3_no_depth_adapt` *(Cycle 3 abl.)* | `8.80` | `9.10` | `9.30` | `7.31` | `9.86` | `8.874` |
| `i1_phmn` *(Cycle 1)* | `8.00` | `9.20` | `8.80` | **`8.23`** | `10.00` | `8.846` |
| `i3_hrwn` *(Cycle 1 best @ 250)* | `7.80` | `8.00` | `8.50` | `7.33` | `5.04` | `7.334` |

---

## 3. Genuine Scientific Boundaries (What Is NOT Claimed)

1. **Not yet pretraining-horizon Pareto across in-distribution BPB:** At `1,000` steps (`4.10M` tokens, `20.55M` params), `G3_HMWR` (`2.1342`) still trails `B0_Transformer` (`2.1188`) by `-0.0154` BPB in-distribution, while already beating `B0` at `2x` OOD context (`2.1322` vs `2.1526`).
2. **Un-fused Python kernel dispatch overhead:** Without custom Metal/Triton WY+Wave kernels, `G3_HMWR` runs at `~0.47x` `B0` training throughput (`12,843` vs `27,324 tok/s` on T4) and `2.46 tok/s` vs `5.16–22.59 tok/s` at `3.88B` (`512–2k`), despite needing fewer FLOPs/tok (`7.82G` vs `9.25G` at `4k`).
3. **Clinically unvalidated:** NanoScribe P1 clinical extraction still uses `B0`-lineage trunks (`H6` rejected); `G3_HMWR` has not yet been fine-tuned on clinical encounters.
