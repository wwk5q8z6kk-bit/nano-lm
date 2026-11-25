# Empirical Results: Gate 1 Bottleneck Decomposition & Pre-Registered 3-Seed Depth-Routing / Coda-Refinement Tournament (`H-13` & `H-10`)

**Completed:** `2026-11-23` · **Branch:** `research/gate1-and-depth-routing-v1` (Draft PR #203)  
**Compute & Spend:** `3 × Tesla T4` sessions on Kaggle free tier (`45,000` Muon optimizer steps, `1,474,560,000` trained tokens across `15` independent runs) · **`$0.00` cloud spend**.  
**Raw artifacts:** `results/tournament/seed_{1000,1001,1002}.json`, `results/tournament/summary.json`, `results/gate1_bottleneck_results.json`, `results/sub4b_roofline_and_scaling.txt`, `results/kaggle_dense_40m_baseline.json`.

---

## 1. Executive Summary & Verdicts

We executed the pre-registered 5-arm, 3-seed tournament (`PROTOCOL.md`) on Corpus `v1` (`FineWeb-Edu 62%`, `Cosmopedia-v2 13%`, `FineMath-4+ 13%`, `GitHub Code Clean 12%`, `32,768` BPE) using `OptaxParityMuon` (`lr = 4e-3, wd = 0.1, warmup = 300, cosine floor = 1%`, `bs = 64, seq = 512, steps = 3,000 = 98.30M tokens/arm`) with zero per-arm hyperparameter tuning and a Latin-square execution order on the same physical GPU within each seed.

### Mechanical Verdicts (`score_tournament.py` on `seed_{1000,1001,1002}.json`, Unaltered)

| Decision Rule | Mechanical Output (`summary.json`) | Key Empirical Evidence (`3`-Seed Mean ± SD, `df = 2`, $t_{0.95} = 4.303$) |
|---|---|---|
| **`D1_depth_routing`** | **`CLOSE_DEPTH_ROUTING_AT_14M`** *(see Predicate Audit below: `block_attnres` rejected on wall-clock; `value_res` `+7` scalars beats `dense` by `+0.0166` BPB at zero speed/VRAM cost and beats `10L` `dense10`, queued for dedicated `40M` gate)* | `block_attnres` gains `+0.0192 ± 0.0014` BPB equal-token ($t = 24.72$) but **loses `-0.0383` raw / `-0.0281` steady-state BPB at equal wall-clock** (`-38%` tok/s, `+1.09 GB` VRAM), with `86.6%` of its gain already captured by `+7` scalars in `value_res` (`+0.0166 ± 0.0009` equal-token, $t = 31.75$; `+0.0164 ± 0.0007` equal-wall-clock, $t = 38.40$). |
| **`D2_coda_refinement`** | **`REJECT_UNCONDITIONAL_CODA_REFINE`** | `coda_refine` (`10` layer passes, zero-init gated shared top-2-layer pass) yields **`-0.0002 ± 0.0006` BPB** equal-token ($t = -0.67$, `-1.7%` of `dense10`'s `+0.0136` gain) and **`-0.0272` raw / `-0.0082` steady-state BPB** at equal wall-clock despite learning nonzero gates (`rms = 0.095–0.139`). |

### Transparent Predicate Audit of `D1` (Preserving Registered Rules Without Retroactive Edit)

In `PROTOCOL.md` / `score_tournament.py`, `D1` was written as a three-way `if / elif / else`:
1. `ADOPT_BLOCK_ATTNRES` required `ba_tok >= 0.004` (`+0.0192`, **PASS**), `ba_tok` significant ($t = 24.72 > 4.303$, **PASS**), `ba_vr >= 0.002` (`+0.00257`, **PASS** on point mean, though **non-significant** at $t = 2.627 < 4.303$), **and** `ba_time >= 0.004` (**`-0.03831`, HARD FAIL**).
2. `ADOPT_VALUE_RES_REJECT_BLOCK_ATTNRES` required `vr_tok >= 0.004` (`+0.01663`, **PASS by `4.2×`**), `vr_time >= 0.004` (`+0.01642`, **PASS by `4.1×`**), `vr_tok` significant ($t = 31.75 > 4.303$, **PASS**), **and** `ba_vr < 0.002` (`+0.00257 >= 0.002`, **tripped boolean guard**).
3. Consequently, `score_tournament.py` fell through to `else -> CLOSE_DEPTH_ROUTING_AT_14M` because `D1`'s boolean partition omitted the quadrant `(ba_time < 0.004) ∧ (ba_vr >= 0.002)` where `block_attnres` fails wall-clock catastrophically while clearing `+0.002` equal-token point margin (non-significantly) over `value_res`.

Following Nano LM epistemic governance, we **preserve `CLOSE_DEPTH_ROUTING_AT_14M` unaltered** in `score_tournament.py` and `summary.json`:
- **Full softmax depth routing (`H-13 Block AttnRes`) is closed** for training efficiency (`-38%` steady-state throughput, `+24.5%` peak VRAM, negative equal-wall-clock return).
- **Layer-0 Value Residual (`value_res`, `+7` scalars)** is **not** silently auto-promoted past the `D1` fall-through; instead, its unexpected `+0.0166` BPB (`+0.0349` on `code`, beating `10`-layer untied `dense10` with `1.48M` fewer parameters) qualifies it for an explicit, pre-registered **`40.38M` (`M`-scale, `800M`-token) confirmation gate**.

---

## 2. Main 3-Seed Tournament Results (`seeds = 1000, 1001, 1002`)

### 2.1 Equal-Token & Equal-Wall-Clock Comparison

> **Note on first-arm `/kaggle/input` `mmap` warmup:** The first arm in each Kaggle session faults `train.bin` (`1.5 GB`) from network storage into OS RAM during steps `1..500`, adding `~650s` one-time I/O overhead to whichever arm runs first. Our Latin-square design rotated the first slot across `dense` (`s1000`), `block_attnres` (`s1001`), and `coda_refine` (`s1002`). Below we report **both** the exact pre-registered raw `train_sec` metrics from `summary.json` and the **steady-state cached GPU** (`steps 500..3000`) throughput and equal-wall-clock BPB. Every qualitative and quantitative conclusion is identical under both measures.

| Arm | Params | Equal-Token `mix` BPB (Mean ± SD) | Per-Seed `mix` (`1000, 1001, 1002`) | $\bar{\Delta}_{\text{tok}}$ vs `dense` ($t$-stat) | Raw Equal-Time `mix` BPB | Steady-State Tok/s (`500..3000`) | Steady-State Equal-Time `mix` BPB | Peak VRAM |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `dense` (`8L` baseline) | `14,292,224` | `1.2221 ± 0.0030` | `1.2236, 1.2187, 1.2241` | `0.0000` (ref) | `1.2221` | `47,931` *(49,786 unconf.)* | `1.2221` | `4.457 GB` |
| **`value_res` (`+7` $\lambda_\ell$)** | **`14,292,231`** | **`1.2055 ± 0.0021`** | **`1.2066, 1.2031, 1.2068`** | **`+0.0166` ($t = 31.75$)** | **`1.2057`** | **`49,253`** | **`1.2057` (`+0.0164`)** | **`4.457 GB`** |
| `block_attnres` (`H-13`) | `14,300,928` | `1.2029 ± 0.0035` | `1.2059, 1.1991, 1.2038` | `+0.0192` ($t = 24.72$) | `1.2604` | `30,567` | `1.2502` (`-0.0281`) | `5.551 GB` |
| `coda_refine` (`H-10`, `10` pass) | `14,292,480` | `1.2224 ± 0.0036` | `1.2239, 1.2183, 1.2249` | `-0.0002` ($t = -0.67$) | `1.2493` | `42,532` | `1.2303` (`-0.0082`) | `4.555 GB` |
| `dense10` (`10L` untied) | `15,768,064` | `1.2085 ± 0.0021` | `1.2065, 1.2084, 1.2107` | `+0.0136` ($t = 6.92$) | `1.2133` | `43,641` | `1.2139` (`+0.0082`) | `4.566 GB` |

### 2.2 Per-Domain Held-Out BPB (`655,360` Held-Out Tokens: `163,840` per Domain)

| Arm | `web` (`62%` mix wt) | `textbook` (`13%`) | `math` (`13%`) | `code` (`12%`) | $\bar{\Delta}$ `code` vs `dense` ($t$-stat) | $\bar{\Delta}$ `math` vs `dense` ($t$-stat) |
|---|---:|---:|---:|---:|---:|---:|
| `dense` | `1.2629` | `0.9792` | `1.3873` | `1.0962` | `0.0000` | `0.0000` |
| **`value_res`** | **`1.2503`** | **`0.9673`** | **`1.3631`** | **`1.0613`** | **`+0.0349` ($t = 14.54$)** | **`+0.0242` ($t = 14.49$)** |
| `block_attnres` | `1.2469` | `0.9636` | `1.3623` | `1.0627` | `+0.0335` ($t = 30.41$) | `+0.0250` ($t = 14.19$) |
| `coda_refine` | `1.2630` | `0.9795` | `1.3877` | `1.0963` | `-0.0002` ($t = -0.13$) | `-0.0004` ($t = -0.49$) |
| `dense10` | `1.2519` | `0.9659` | `1.3666` | `1.0760` | `+0.0202` ($t = 4.24$) | `+0.0207` ($t = 6.80$) |

**Domain insight:** `value_res` (`+7` scalars) **beats `block_attnres` (`+8,704` scalars) outright on `code`** (`1.0613` vs `1.0627`, `-0.0014` BPB for `block_attnres` vs `value_res`) and ties it within `0.0008` BPB on `math` (`1.3631` vs `1.3623`), while beating `10`-layer untied `dense10` on `web`, `math`, and `code` (`+0.0147` BPB ahead of `dense10` on `code`). `block_attnres`'s remaining `+0.0026` BPB edge over `value_res` comes almost entirely from prose (`web +0.0034`, `textbook +0.0037`).

---

## 3. Evaluation of Pre-Registered Predictions (`P1–P5`) & Mechanistic Diagnostics

1. **`P1` (Baseline repeatability `< 0.004` BPB SD) — CONFIRMED:**  
   `dense` achieved `SD = 0.00298` BPB (`value_res` `0.00208`, `dense10` `0.00210`).
2. **`P2` (`Block AttnRes` `+0.004..0.015` equal-token, `< 5%` runtime cost, `≥ +0.002` over `value_res`) — PARTIALLY CONFIRMED ON TOKENS, FALSIFIED ON RUNTIME:**  
   Equal-token gain exceeded expectation (`+0.0192` BPB), but training step time increased by **`+61.8%`** (`30.6k` vs `49.3k` tok/s) and VRAM by **`+1.094 GB`** because all `5` partial-block states (`[B, T, d]`) must be retained in autograd memory and reduced through `17` depth-softmax routers (`RMSNorm + einsum + softmax`). Even with a fused Triton inference kernel, training-time activation storage and depth-router backward passes break wall-clock parity at `14M` scale, whereas `value_res` gets `86.6%` of the gain (`104%` on `code`, `97%` on `math`) for `0%` overhead.
3. **`P3` (`Value Residual` `+0.001..0.006` BPB) — EXCEEDED BY `~3×` (`+0.0166` BPB):**  
   Inspecting the learned $\lambda_1..\lambda_7$ weights (`v_l = \lambda_l v_l + (1 - \lambda_l) v_0`) across all three seeds reveals a striking, seed-invariant depth profile:
   - `seed 1000`: `[0.4148, 0.5351, 0.3734, 0.8515, 0.6954, 0.1590, 0.0989]`
   - `seed 1001`: `[0.5564, 0.3808, 0.6413, 0.8479, 0.2459, 0.3605, 0.0556]`
   - `seed 1002`: `[0.4131, 0.3697, 0.8272, 0.7062, 0.4177, 0.1642, 0.1108]`
   In **every seed**, middle layer `l = 4` keeps a `0.706–0.852` nominal weight on its local computed value $v_4$, whereas the **final layer `l = 7` drops to $\lambda_7 \in [0.0556, 0.1108]$ (`mean 0.0884`), placing an `88.9–94.4%` (`mean 91.2%`) nominal affine mixing coefficient ($1 - \lambda_7$) on Layer 0 ($v_0$)**. *(Scope note per P0 audit PR #211: because $\lambda_\ell$ is unconstrained and $\|v_\ell\|_2$ vs $\|v_0\|_2$ norms were not logged separately at `14M`, $1 - \lambda_7 = 0.912$ measures the nominal mixing scalar rather than normalized vector-energy share; normalized vector norms, $v_0$ knockout ablations, and prior-art comparators from ResFormer ACL 2025 / SVRA, BoV, and generic $h_0$ residual shortcuts are tested directly in the `42M` confirmation gate.)* In a tied-embedding PreNorm Transformer, late residual states $h_6, h_7$ rotate toward next-token prediction logits; direct $v_0$ skip connections allow late attention heads to access Layer-0 value projections from attended positions (`+0.0349` BPB on `code` identifier/symbol prediction) without burdening the residual stream.
4. **`P4` (`Coda Refinement` recovers `≥ 40%` of `dense10` equal-token gain) — FALSIFIED (`-1.7%`):**  
   `coda_refine` learned nonzero channel gates (`rms = 0.0953, 0.1385, 0.1030`; `abs_max = 0.3941, 0.4826, 0.4485`), ruling out dead-gradient failure. Nevertheless, re-applying `blocks[6..7]` without step-conditioned parameters or middle-layer recurrence yielded `0.0000` gain across every domain, falsifying naive shared coda looping.
5. **`P5` (Winning candidates shrink `Ranks 2..64` BPB) — FALSIFIED (MIGRATION INVARIANCE DISCOVERED):**

| Arm | `Rank 1` Calib BPB | `Ranks 2..8` BPB | `Ranks 9..64` BPB | `Rank ≥ 65` Miss BPB | `Top-1` Acc | `Top-64` Acc |
|---|---:|---:|---:|---:|---:|---:|
| `dense` | `0.0913` | `0.2580` | `0.3156` | `0.5573` | `37.12%` | `80.90%` |
| **`value_res`** | **`0.0901`** | **`0.2602`** | **`0.3153`** | **`0.5399` (`-0.0174`)** | **`37.46%`** | **`81.45%`** |
| `block_attnres` | `0.0902` | `0.2592` | `0.3140` | `0.5395` (`-0.0178`) | `37.61%` | `81.47%` |
| `coda_refine` | `0.0914` | `0.2578` | `0.3148` | `0.5584` (`+0.0011`) | `37.14%` | `80.86%` |
| `dense10` | `0.0912` | `0.2570` | `0.3111` | `0.5493` (`-0.0080`) | `37.64%` | `81.14%` |

   Just as `Ranks 2..8` remained invariant across `14M → 40M → 101M` scale in Gate 1 (`0.2593 → 0.2596 → 0.2579` BPB), it stays invariant at `0.2570–0.2602` BPB across all 5 architectures because `Ranks 2..8` acts as a **steady-state flow queue**: `value_res` promotes `+0.55%` of tokens out of `Rank ≥ 65` into `Top-64` (`-0.0174` BPB on `Rank ≥ 65`) while promoting `+0.34%` of former `Ranks 2..8` tokens into `Rank 1`, leaving the occupancy and cross-entropy mass of `Ranks 2..8` conserved (`~0.26` BPB).

---

## 4. Summary of Gate 1 & Sub-4B Roofline Audits (Completed in This Package)

1. **`cache-exchange-L` (`100.68M` Dense, `results/gate1_bottleneck_results.json`):**  
   Held-out prefix n-gram cache gain shrinks monotonically with scale: `14M (+0.0036) → 40M (+0.0018) → 101M (+0.0012` mix BPB; `code +0.0043`, foreign-window control `0.0000`), closing explicit `512`-window copy caches (`< 0.01` BPB hurdle). Non-copyable tokens account for `93.89%` of `101M` mix loss and `91.04%` of the `14M → 101M` scaling gain.
2. **Sub-4B Exact Parameter & `batch=1` DRAM Roofline (`results/sub4b_roofline_and_scaling.txt`):**  
   Corrected the Blueprint's Gated DeltaNet parameter accounting (`52.65M` params/block at `d = 2560`, `1.264B` across `24` blocks rather than `0.48–0.84B`). At `V = 65,536`, matched `32`-layer `Dense D0` (`2.936B`) is `945M` parameters smaller than `3:1` Hybrid `H1` (`3.882B`) and moves fewer DRAM bytes/token at `batch = 1` (`8.02` vs `8.46 GB/tok` at `32k` BF16) below a **`44,032`-token** crossover.

---

## 5. Next Pre-Registered Step Recommended for Nano LM

1. **Register a dedicated `40.38M` (`M`-scale, `800M` tokens, `3` seeds) confirmation gate for `value_res` (`Dense-M + Muon` vs `Dense-M + ValueRes + Muon`):**  
   Test whether the `+0.0166` mix BPB (`+0.0349` `code`, `+0.0242` `math`) zero-cost gain at `14.29M` persists at `40.38M` (`12` layers, `+11` scalars) over full `800M`-token horizon (`~20 tok/param`) against the `1.0478` BPB TPU / `1.0506` Kaggle `Dense-M` reference, and verify whether Layers `9..11` reproduce the $\lambda_\ell \to 0.05..0.16$ Layer-0 value-skip signature.
2. **Keep `H-13 Block AttnRes` and unconditional `H-10 Coda Refine` closed** unless a future study targets a specific regime not tested here (e.g., `≥ 24`-layer deep stacks with custom fused Triton depth-routing kernels for `H-13`, or learned step-specific adapters + sparse entropy gating for `H-10`).
