# NANOLM Failures, Falsifications & Engineering Post-Mortems

Every failure, instability, falsified hypothesis, and empirical regression encountered across Cycles 1–3 and `EXP-INV-06` is documented verbatim below with exact telemetry, root-cause taxonomy, and resolution status.

---

## Failure Ledger Summary

| ID | Category | Phenomenon Observed | Root Cause | Fix / Scientific Outcome |
|---|---|---|---|---|
| **F1** | *Numerical Instability* | Immediate `NaN` in `wy_metriplectic_chunk_scan` under `torch.autocast("mps", bfloat16)` | `bfloat16` GEMM downcast corrupted nilpotent $(I-L)^{-1}$ inversion and cumulative decay | Isolated scan/filter math in `float32` via `_no_autocast_ctx` (`operators.py`). **Fixed.** |
| **F2** | *Finite-Precision Overflow* | `i1_phmn` `seed=137` exploded at step `11` (`gnorm=1,123.8 -> NaN`) under `chunk_size=64` | Exceeded IEEE-754 `float32` critical chunk bound $C_{\text{crit}} \approx 41.6$ for reflection $\beta_{\max}=1.98$ | Enforced `chunk_size=32 < C_crit` (`max_gnorm=6.06`, `+28%` MPS speedup). **Fixed.** |
| **F3** | *Initialization Starvation* | Pre-fix mechanistic `d=64, L=2` runs scored `OOD_Comp = 0.094–0.159` (chance) | Uniform `std=0.02` + `gate_proj.bias=0` ($\gamma_0 \approx 0.87$) starved gradients across chunks | Scale-adaptive `min(0.06, 0.8/sqrt(fan_in))`, residual `1/sqrt(2L)`, decay bias `-3.0` ($\gamma_0 \approx 0.99$). **Fixed.** |
| **F4** | *Addressing Defect* | Symmetric unshifted associative addressing collapsed on `Overwrite 4x OOD` (`0.146–0.171`) | `q_t` and `k_t` projected from unshifted $x_t$ could not bind bigram `(QUERY, key) -> val` without interfering with `v_t` | `CausalKeyShift` (`[0.7, 0.3]` init on `(q, k, u, scam_q, scam_k)` with clean `v`) -> `OW_4x = 0.977–0.986`. **Fixed.** |
| **F5** | *Epistemic Inversion & Falsification* | `i2_nebn` internal `1/prec` `AUROC = 0.362`; `g2_mprn` underperformed `g2_no_precision` | Positive-orthant $\phi(x)$ aligned with cumulative key buffer $\mathcal{P}_t$ even on unseen keys; `nebn_precision_trace` hurt **both** OOD (`0.737 vs 0.751`) and BPB (`2.5161 vs 2.5047`) | **Hypothesis falsified.** Removed precision trace from `G3_HMWR`; replaced with $s_t / (\|y^{\text{met}}\|_{\text{rms}}^2 + \|y^{\text{scam}}\|_{\text{rms}}^2 + 10^{-3})^{1/2}$. |
| **F6** | *Architectural Blind Spot* | `i3_hrwn` failed `EXP-INV-01` (`0.176`), inverted uncertainty (`AUROC = 0.204`), and stagnated at `1,000` steps (`2.1654`) | No shifted SCAM addressing, inverted $1/(1+s_t)$ uncertainty proxy, no $\beta>1$ non-commutative transport, finite `256`-tok conv window | Replaced with `G3_HMWR` (`OOD_Comp = 0.739`, `AUROC = 0.686`, `2.1342` @ 1k steps). **Resolved.** |
| **F7** | *Parameter Starvation* | Cycle 2 `g2_*` regressed on real corpus (`2.5161` vs `2.4520`) despite winning OOD (`0.751`) | Full-width `MPRNOperator` (`7.0 d^2`) + fixed `SymplecticHamiltonianSynthesis` (`1.5 d^2`) left `ffn_hidden = 64` at `20.5M` and `32` at `47.6M` | Compressing WY to $H/2$ + SCAM to $d/4$ in `G3_HMWR` (`6.5 d^2`) restored `ffn_hidden = 304 / 384`. **Resolved.** |
| **F8** | *Horizon & Objective Mismatch* | Short `250`-step gate crowned `i3_hrwn` (`2.4520`) and rejected `g3_hmwr` (`2.5085`); `Muon+0.10 JEP` lost `+0.034` to `+0.047` BPB | Local convolutions align faster than gated recurrences in `<= 250` steps; auxiliary `0.10 JEP` diverted early gradient budget away from CE | Proved in `EXP-INV-06A`: `g3_hmwr` crossed over before step `500` (`2.2535 vs 2.2908`) and pure-CE `Muon` fell `-0.3593` BPB (`250->1000`). **Diagnosed.** |
| **F9** | *Systems & Stride Failures* | Non-contiguous `targets.view(-1)` crashed CPU run; unchunked `32k` prefill used `7,876.7 MiB` vs `B0` `3,646.6 MiB` (`EXP-INV-06B`) | Memmap `buf[:, 1:]` stride incompatible with `.view()`; Python chunked full-`32k` WY+SCAM activations retained transient intermediates | Replaced `.view(-1)` -> `.reshape(-1)` (`models.py`, **Fixed**); streaming `1k`-chunk prefill + Triton/Metal fusion scheduled for Cycle 4 (**Open**). |

---

## Detailed Forensic Highlights

### F2: Analytic Proof & Reproduction of WY Reflection Instability
During Cycle 1 initial bring-up at `chunk_size=64`, `i1_phmn_discovery_18m` crashed cleanly on `seed=137` at step `11` while `i1_phmn_seed42` completed cleanly: rare token alignments induced negative inner products across consecutive reflection microsteps ($\beta_t \approx 1.98$), causing backward sensitivity gradients across the nilpotent inversion $(I-L)^{-1}$ (`(1 + 1.98)^124 ≈ 10^58 > 3.4e38`) to overflow `float32`. Bounding $C \le 32 < C_{\text{crit}} \approx 41.6$ restored strict finite gradients permanently without clamping $\beta_t < 1$ (preserving $S_3$ parity expressivity).

### F7 & F8: Why Multi-Scale, Multi-Horizon Testing Was Essential
Had the program stopped after Cycle 2's `250`-step M3 Max runs, the empirical verdict would have claimed: *"wave+SCAM (`I3_HRWN`) beats Transformer on language (`2.4520` vs `2.4652`) while hybrid `G2`/`G3` lag permanently."* Extending to `1,000` steps on Kaggle Tesla T4 GPUs exposed **both** half-truths:
1. `I3_HRWN`'s early speed was an initialization/local-filter mirage that hit a structural ceiling by step `500` (`2.1654` final).
2. `G3_HMWR`'s deeper depth-attention + half-subspace Householder + SCAM machinery converged slightly slower over `0–250` steps, crossed `I3_HRWN` decisively before step `500`, finished `+0.0312 BPB` better in-distribution (`2.1342` vs `2.1654`), and beat `B0_Transformer` out-of-distribution at `2x` length (`2.1322` vs `2.1526`).
