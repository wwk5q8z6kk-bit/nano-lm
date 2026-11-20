# NANOLM Ω — Comprehensive Failure, Falsification & Post-Mortem Analysis

This forensic ledger consolidates **every failure, instability, falsified architectural hypothesis, and hardware anomaly** observed across Discovery Cycles 1–3 (`F1–F9`) and Program Ω (`F-Ω01` to `F-Ω08`), ensuring every negative result is preserved with root cause and verified mitigation.

---

## 1. Foundational Discovery Failures (`Cycles 1–3`, `F1–F9`)

| ID | Category | Phenomenon Observed | Root Cause | Verified Resolution / Outcome |
|---|---|---|---|---|
| **F1** | *Precision Downcast* | Immediate `NaN` in `wy_metriplectic_chunk_scan` under `mps` `bfloat16` autocast | `bfloat16` GEMMs corrupted nilpotent $(I-L)^{-1}$ inversion & cumulative log decay | Isolated scan/filter math via `_no_autocast_ctx` in `float32` (`nanolm_discovery/operators.py`). **Fixed.** |
| **F2** | *Condition Bound Overflow* | `i1_phmn` `seed=137` exploded at step `11` (`gnorm=1,123.8 -> NaN`) under `chunk_size=64` | Exceeded IEEE-754 `fp32` bound $C_{\text{crit}} \approx 41.6$ for $\beta_{\max}=1.98$ (`(1+1.98)^124 ≈ 10^58`) | Bounded chunk size $C \le 32 < 41.6$ (`max_gnorm=6.06`). **Fixed.** |
| **F3** | *Init Decay Starvation* | Unscaled small models stalled at chance `OOD_Comp=0.094–0.159` | Uniform `std=0.02` + `gate_proj.bias=0` ($\gamma_0 \approx 0.87$) attenuated signals | Applied `min(0.06, 0.8/sqrt(fan_in))`, residual `1/sqrt(2L)`, decay bias `-3.0` ($\gamma_0 \approx 0.99$). **Fixed.** |
| **F4** | *Addressing Interference* | Symmetric unshifted keys collapsed on `Overwrite 4x OOD` (`0.146–0.171`) | `q, k` and payload `v` competed on identical token index | Asymmetric 4-tap `CausalKeyShift` (`[0.7, 0.3]` init on `(q, k, u, scam_q, scam_k)`). **Fixed (`0.977`).** |
| **F5** | *Epistemic Inversion* | Positive-orthant `1/prec` gave `AUROC=0.362` (`i2_nebn`) & hurt `g2_mprn` (`2.5161` vs `2.5047`) | $\phi(q) \ge 0$ aligned with historical buffer $\mathcal{P}_t$ on unseen keys | **Falsified `1/prec`.** Replaced with retrieval-normalized surprise $u_t^{\text{mem}}$ (`0.686–0.934`). |
| **F6** | *Conv-Only Blind Spot* | `i3_hrwn` failed $S_3$ / Overwrite (`0.176` OOD), inverted AUROC (`0.204`), plateaued @ 1k (`2.1654`) | No shifted addressing, no $\beta>1$ orientation flips, finite `256`-tok receptive window | Replaced with `G3_HMWR` (`OOD_Comp=0.739`, `2.1342` BPB). **Resolved.** |
| **F7** | *Operator Starvation* | Full-width `MPRNOperator` (`7.0 d^2`) + `1.5 d^2` synthesis starved `ffn_hidden=32..64` | Exceeded fixed iso-parameter budget -> `2.5161` vs `2.4520` BPB | Compressed WY (`H/2`) + `SCAM-32` (`d/4`) -> `ffn_hidden=304/384`. **Resolved.** |
| **F8** | *Short-Horizon Mirage* | `250`-step screen ranked `i3_hrwn` #1 and `g3_hmwr` last; `Muon+0.10 JEP` lagged `+0.034..0.047` | Dense convs align faster at step `<=250` while aux JEP diverts early CE gradient | Proved crossover by step `500` in `EXP-INV-06A` (`G3` `-0.0312` BPB over `I3`, `Muon` slope `-0.359`). |
| **F9** | *Stride & Prefill Memory* | Memmap `buf[:, 1:]` `.view(-1)` crash; `222M` unchunked `32k` prefill allocated `7,876.7 MiB` vs `B0` `3,646.6 MiB` | Strided views break `.view()`; full-`32k` unchunked chunk/SCAM intermediates retained simultaneously | Fixed `.reshape(-1)` in `models.py`; `C_stream=1024` chunked prefill scheduled in Cycle 4A (**Open**). |

---

## 2. Program Ω Scientific Falsifications & Engineering Post-Mortems (`F-Ω01` to `F-Ω08`)

1. **`F-Ω01` — `X8` Dual-Vocabulary Unembedding FLOP & Wall-Clock Tax:**  
   Audit of `.worktrees/nanolm-lab/lab/arch/arch04.py` lines `317 & 331` revealed two full `32,768`-vocab unembeddings (`norm(x) @ tok_emb.weight.T`) per pass, inflating `4k` FLOPs at `20.5M` from `40.91M–41.58M` (`G4`) to `78.56M` (`1.89x–1.92x`), slowing `47M` RTX 4090 throughput `1.81x` (`6,569.5s` vs `3,632.4s`), slowing `125M` L40S throughput `2.65x` (`41.5k` vs `110.0k tok/s`), ballooning `32k` persistent KV state `28.0x` (`50.54 MB` vs `1.80 MB`), and failing to track `S_3` parity (`S3_4x = 0.1723`). *Note on Trade-off:* Because `X8` includes two exact `GQA` layers per 8 blocks, it retained a `2.0797 BPB` short/mid-context (`256–512`) copy advantage over pure-$O(1)$ `G4` (`2.1112`), cleanly framing the exact accuracy-vs-memory Pareto frontier.

2. **`F-Ω02` — Closed-Entity `M0..M11` `Sym(12)` Cognitive Ceiling (`OMEGA-EXP-02`):**  
   Testing `C4b` (`5.28%` Super-OOD) and `C5` (`4.17%–7.78%` Super-OOD) on `HardCausalSWEEnv` (`N = 16..24` repositories with distractor-first unordered goals) exposed their catastrophic entity ceiling: fixed slot embeddings mod-clip/alias distinct dependencies beyond index `11`, driving contradiction revision accuracy to **`0.00%–3.92%`**. Replaced cleanly by `OmegaEFEPlanner` (`Conv1d` char-spans + 3-hop directed message passing -> **`83.33%` Super-OOD**).

3. **`F-Ω03` — Unshielded Hardcoded Penalty Dependency (`OMEGA-EXP-02`):**  
   Removing hardcoded `suff_pen[4]`/`suff_pen[6]` from `C5` (`c5_first_mention_unshielded`) spiked `Super-OOD` premature patch/regression errors from `0.28% -> 9.17%`, proving legacy cognition safety lived in brittle index rules rather than learned state uncertainty. `OmegaEFEPlanner` achieved **`0.00%` premature rate across all 3 regimes (`ID`, `OOD`, `Super-OOD`)** with zero hardcoded action penalties, whereas removing its risk term (`omega_no_efe_risk`) degraded OOD resolution `-20.00pp` (`95.56% -> 75.56%`) and Super-OOD `-15.27pp` (`83.33% -> 68.06%`).

4. **`F-Ω04` — Falsification of `H-Ω04` (Iso-Param Conditional Pass-2 `LatentTokenRefiner`):**  
   Controlled Kaggle Dual-T4 ablations cleanly **falsified `H-Ω04` under fixed `20.56M` parameter parity**: `g4_omega` (with Pass-2 `_LatentTokenRefiner`) scored `2.1175 ± 0.0032` ID / `2.1148 ± 0.0032` `2x` OOD BPB, **losing `-0.0063` ID and `-0.0069` `2x` OOD BPB** against single-pass `g4_no_sparse_refiner` (`2.1112 ± 0.0081` ID / `2.1079 ± 0.0082` OOD) while also losing mechanistic epistemic AUROC (`0.5260` vs `0.6185`). Meanwhile, removing `Gated-VRes` (`g4_no_vres`) crashed corpus BPB to **`2.1508 ± 0.0146`** (`-0.0396 BPB` worse) and `OW_4x` by `-15.1pp` (`0.7339` vs `0.8850`). Thus, investing the budget into dense trunk width + Layer-0 Per-Head `Gated-VRes` + `QK-RMSNorm` beats sparse Pass-2 latent recycling at `20.5M`.

5. **`F-Ω05` — Turing (`sm_75`) CUDA `float16` `step=113` NaN & Multi-Thread GIL Contention (`Kaggle v2/v3`):**  
   Kaggle v2 & v3 both aborted deterministically at `x8_dual_unembed_atrd seed=42 step=113` with `loss=nan` under `torch.autocast("cuda", torch.float16)`, whereas an identical local run forcing every `Linear`/`Conv1d` through IEEE `float16` + `4096` scaling stayed completely finite (`max_logit=15.27`, `loss=7.085`), isolating a Turing `sm_75` CUDA `float16` SDPA/kernel numerical bug. Simultaneously, v3 `ThreadPoolExecutor` multi-threading suffered `3.1x` PyTorch autograd/GIL mutex slowdown (`299s` for 113 steps vs `96s` in v2). **Resolved permanently:** Turing (`major < 8`) executes native `float32` (`<= 5.53 GiB / 14.56 GiB` VRAM) across independent OS subprocesses (`subprocess.Popen`), achieving `10,451 tok/s` (`X8`) and `8,884 tok/s` (`G4`) cleanly.

6. **`F-Ω06` — Discriminative Latent Router OOD Dilution Paradox (`OMEGA-EXP-01-MECH`):**  
   Three controlled mechanistic experiments (`115k`, 3 seeds) mapped how internal gating interacts with unseen-key epistemic uncertainty (`evaluate_epistemic_calibration`, unseen token `45`):
   - `g4_no_sparse_refiner` (pure structural $u_t^{\text{mem}} = s_t / \text{rms}$): **`EpistAUROC = 0.6185`** (OOD denominator shrinks when unseen key fails to retrieve -> spikes cleanly).
   - `g4_uncalibrated_router` (unsupervised $z_t^{\text{lat}}$ corrupted by Pass-2 anti-circularity gradients): **`0.3854` AUROC** (active sign inversion!), `Entropy AUROC = 0.6836`.
   - ID-only additive self-calibration (`0.5 * sigmoid(...)`): **`0.3958` AUROC** (`Entropy` lifted to `0.7852`, `OW_4x` `0.9086`), because linear hidden probe `score_proj(norm(h))` trained solely on ID queries still anti-correlates on unvisited token embeddings while `sigmoid(log1p(u_mem)-0.5)` saturates the `1/retr_rms` spike.
   - Multiplicative OOD-preserving readout $u_t^{\text{mem}} (0.5 + \sigma(z_t^{\text{lat}}))$: **recovered `+0.1406 AUROC` (`0.3854 -> 0.5260`)**, `Entropy AUROC = 0.7852`, `OW_4x = 0.9086`, yet still confirmed that **pure un-blended structural memory telemetry (`0.6185`) remains superior on OOD covariate shifts.**

7. **`F-Ω07` — `pred_err = sqrt(0)` Singularity in Wave Surprise (`Kaggle v4` -> `v5`):**  
   In Kaggle v4, `g4_no_vres seed=137` (`cuda:1`) hit `step=125: nan` in pure `float32` (while `seed=42` finished 1,000 steps cleanly). Root cause: `(re_in - lagged).float().pow(2).mean(dim=-1).sqrt()` omitted epsilon inside `.sqrt()`, yielding `0.5 / 0.0 = Inf` -> `NaN` backward gradient on consecutive identical corpus tokens. Fixing `.add(1e-6).sqrt()`, tightening `_OmegaHMWRBlock` WY chunks to `min(chunk_size, 16)`, and adding non-finite gradient skipping achieved **`skipped_grad_steps = 0` across 100% of Kaggle v5 (`g4_no_vres` & `g4_no_sparse_refiner`, 4,000 steps total)**.

8. **`F-Ω08` — Legacy System `task_hint` / Keyword Brittle Routing (`OMEGA-EXP-03`):**  
   On 60 unlabelled multi-pillar tasks, `legacy_keyword_system_v2` achieved `28.33%` (`4.588` cost/win, `0.00%` abstention/conflict) vs `omega_unified_system` **`100.0%`** (`2.225` cost/win, `100.0%` abstention/conflict), with removing `nano/` provenance (`omega_no_slw_provenance`) collapsing overall accuracy to `78.33%` (`30%` SLW, `40%` continual consolidation, `50%` abstention).
