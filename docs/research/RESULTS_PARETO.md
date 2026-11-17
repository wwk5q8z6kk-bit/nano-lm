# Results & Pareto Frontier Ledger

**Strict honesty rule:** Every table below separates verified measurements from confounded/preliminary runs, discloses parameter counts and denominators, and avoids collapsing capability, calibration, latency, and memory into one synthetic score.

---

## 1. Track A Controlled 46.2M Pretraining Grid (`600M` Tokens, `n=2` Seeds, `LAB-DATA-01`)

Source (`LOCAL_WORKTREE_UNMERGED`): `.worktrees/nanolm-lab/lab/reports/LAB-ARCH-02_results.md` (`34` runs).

| Arm | Optimizer | Total / Non-Emb Params | Train TFLOP·s | Global KV (`B/tok`) | Composite BPB (`mean ± half-range`) | $\Delta\%$ vs `T0/AdamW` | Copy Loss (nats) | Reading |
|---|---|---|---|---|---|---|---|---|
| `X4_nanolm_v2_muon` | `muon` | `46.15M` / `29.38M` | `1.06e5` (`-42.7%`) | `3,072` (`-70%`) | `0.9575 ± 0.0008` | `-7.01%` | `0.763` | **`-0.73%` (`-0.0070` BPB) over `T0+Muon` at `-70%` KV & `-42.7%` FLOPs** |
| `T0_dense` | `muon` | `46.28M` / `29.50M` | `1.85e5` | `10,240` | `0.9645 ± 0.0004` | `-6.33%` | `1.177` | Proves `Muon` accounts for `~90%` of raw `X4` vs `T0/AdamW` gap |
| `X2_sharp_delta_vres` | `adamw` | `46.15M` / `29.38M` | `1.06e5` | `3,072` | `0.9689 ± 0.0009` | `-5.90%` | **`0.536`** | Strongest matched-AdamW architecture; rescues `T1_vres` copy collapse |
| `X3_sgrd_delta` | `adamw` | `46.15M` / `29.38M` | `1.06e5` | `3,072` | `0.9815 ± 0.0005` | `-4.67%` | `0.802` | **Falsified halting:** `expected_loops -> 1.00`; worse than `X2` |
| `X1_gated_vres` | `adamw` | `46.15M` / `29.38M` | `1.06e5` | `3,072` | `0.9869 ± 0.0013` | `-4.15%` | `0.746` | Rescues scalar `T1_vres` copy collapse |
| `T1_vres` (scalar) | `adamw` | `46.28M` / `29.50M` | `1.85e5` | `10,240` | `0.9919 ± 0.0002` | `-3.66%` | **`9.207`** | **REJECTED** — catastrophic OOD copy collapse (`8.3%` acc) |
| `T0_dense` | `adamw` | `46.28M` / `29.50M` | `1.85e5` | `10,240` | `1.0296 ± 0.0048` | `+0.00%` | `3.314` | Baseline reference |
| `R1_loop` (flops match) | `adamw` | `46.28M` | `1.85e5` (`432M` tok) | `10,240` | `1.1147 ± 0.0003` | `+8.26%` | `2.208` | **REJECTED** at `46M` iso-FLOP |
| `P2_delta` (no `A`) | `adamw` | `~46M` | `1.72e5` | `0` | `1.0466 ± 0.0023` | `+1.65%` | `11.651` | **REJECTED** — pure recurrence fails exact copy without sparse `A` |
| `P1_gla` (no `A`) | `adamw` | `~46M` | `1.70e5` | `0` | `1.0623 ± 0.0033` | `+3.17%` | `12.027` | **REJECTED** |

At `121M` (`LAB-SCALE-01`, `n=1` seed, `600M` tokens, `Muon`): `S1_nanolm_v2_125m` (`0.9106` BPB, `4,096 B/tok` KV) tied `S0_dense_125m` (`0.9105` BPB, `16,384 B/tok` KV) while beating code BPB (`0.7377` vs `0.7420`).

---

## 2. Generation-3 Mechanistic Suite (`EXP-1`–`EXP-5`) & Identified Confounds

Source (`LOCAL_WORKTREE_UNMERGED`): `.worktrees/nanolm-v3/lab/reports/gen3_empirical_results.json`.

| Experiment | Reported Headline | Critical Confound / Scope Limit |
|---|---|---|
| `EXP-1` (`E-STATE-01`) | Householder $\beta\in(0,2)$ `100%` $Z_2$ 4x-OOD & `99.37%` $S_3$ 3x-OOD vs Contraction `50.67%` / `18.98%` | Single seed (`seed=42`); toy synthetic automaton (`64`–`144` tokens). Needs multi-seed + `GDN-2`/`Mamba-3` comparators (`X03`). |
| `EXP-2` (`E-DIFF-01`) | `Diff-GQA` `11.33%` (`3.225` nats) vs `Standard GQA` `3.12%` (`3.778` nats) on 8-needle OOD (`224` tok) | `79.5k` params, `300` steps, `n=1` seed; `3.12%` equals uniform random value chance (`1/32`). Absolute recall (`11.3%`) remains very low. |
| `EXP-3` (`LAB-ARCH-04`) | `G5_nanolm_v3_all` `2.35192` vs `Dense_GQA` `2.37813` (`-1.10%`) | **Confounded:** `G0..G5` used **`6.73M` non-emb params vs `5.90M` (`+14.1%` larger backbone)**, trained only `220` steps (`1.80M` tok, `n=1`), with `0.0%` copy accuracy on all arms. |
| `EXP-4` (`LAB-CURR-01`) | `Surprise-DoReMi` Math+Code BPB `2.748 -> 2.508` (`-8.73%`) | `1.80M` tok (`n=1`); degraded `web` (`1.933 -> 2.019`, `+4.4%`) and `textbook` (`1.776 -> 1.821`, `+2.5%`). |
| `EXP-5` (`LAB-INT-02`) | `I2` graft `0.000000` step-0 KL; `I1` RAD-KD recovered `3.505 -> 1.201` BPB | `122,880` adaptation tokens only; base remained `0.9494` BPB (`97.06%` copy vs `I1` `0.17%`). |

---

## 3. Live Apple M3 Max (`64 GB UMA`, `bfloat16`, Untrained) Decode Crossover

Source (`LOCAL_WORKTREE_UNMERGED`): `.worktrees/nanolm-v3/artifacts/nanolm_v3/benchmark_15pillar_and_mps_profile.json`.

| Model | Deployed / Active | Prefill `512 -> 4096` (`tok/s`) | Decode `512` (`tok/s`) | Decode `2048` (`tok/s`) | Decode `4096` (`tok/s`) | Measured Cache @ `4096` | Analytical Cache @ `32k` |
|---|---|---|---|---|---|---|---|
| `nanolm-2b-dense` | `2.04B` / `2.04B` | `1800.7 -> 1282.2` | **`48.36`** | **`29.64`** | `19.21` | `288.00 MB` | `2.2500 GB` |
| `nanolm-v3-2b` | `2.07B` / `2.07B` | `895.1 -> 740.2` | `25.17` | `24.00` | **`22.58`** | `100.05 MB` | `0.5899 GB` |
| `nanolm-v3-2b-sgrd` | `2.08B` / up to `2.73B` | `693.0 -> 553.8` | `16.76` | `17.38` | `16.37` | `133.39 MB` | `0.7865 GB` |
| `nanolm-v3-3.9b-moe` | `3.93B` / `1.14B` | `368.9 -> 646.7` | `5.37` | `5.22` | `4.90` | **`57.64 MB`** | **`0.3297 GB`** |

*Key insight:* `nanolm-v3-2b` is **slower** than dense at `ctx=512` (`-48%` decode, `-50%` prefill) and crosses over **only around `ctx=2.5k–4k`** (`+17.5%` at `4096`), demonstrating a context-dependent Pareto trade-off rather than universal hardware superiority. Note also that eager PyTorch `MoE` dispatch bottlenecks `nanolm-v3-3.9b-moe` at `~5 tok/s` (`batch=1`), and `gen3.py` triggers `BFloat16` vs `float32` `RMSNorm` fallback warnings on MPS.

---

## 4. External Compact Reference Models & Track B Post-Training (`Qwen3-1.7B`)

Source (`LOCAL_WORKTREE_UNMERGED`): `.worktrees/nanolm-lab/lab/reports/TRACK_B_LEADERBOARDS_AND_ABLATIONS.md` (`vLLM`, A40 48GB, zero-temp, `seed=1234`).

| Model | Total / Active Params | GSM8K | MATH-500 | HumanEval | IFEval | ARC-C | 8-Pillar (`t=0/t=1`) | LB-B Reliability (`t=0/t=1`) |
|---|---|---|---|---|---|---|---|---|
| `Qwen2.5-1.5B-Instruct` | `1.54B` / `1.54B` | `67.5%` | `48.6%` | `52.4%` | `40.5%` | `53.9%` | `65.1% / —` | `60.9% / —` |
| `Qwen3-1.7B` (Base) | `1.72B` (`2.03B` untied) | `75.0%` | `70.0%` | `66.5%` | `66.9%` | `52.1%` | `70.5% / 83.3%` | `76.0% / 97.9%` |
| `Qwen3.5-2B` (3:1 GDN+GQA) | `2.27B` / `1.88B` | `70.1%` | `63.6%` | `53.7%` | `66.9%` | `52.5%` | `76.5% / 37.9%` | `64.1% / 25.0%` |
| `SmolLM3-3B` | `3.08B` / `3.08B` | `82.5%` | `63.2%` | `72.0%` | `67.5%` | `59.8%` | `77.3% / 87.1%` | `74.5% / 91.7%` |
| `Granite-4.0-H-Micro` | `3.19B` / `3.19B` | `80.7%` | `49.2%` | `77.4%` | `79.7%` | `65.5%` | `77.3% / —` | `88.5% / —` |
| `Phi-4-mini-instruct` | `3.84B` / `3.84B` | `82.3%` | `34.0%` | `70.7%` | `65.2%` | `57.6%` | `74.2% / —` | `69.8% / —` |
| `NanoLM-B1-SFT` (on Qwen3-1.7B) | `1.72B` | `55.6%` | `41.0%` | `50.6%` | `60.4%` | `55.0%` | `73.5% / 32.6%` | `91.7% / 62.5%` |
| `NanoLM-B2-KD8B` (off-policy) | `1.72B` | `68.0%` | `56.0%` | `59.1%` | `56.6%` | `52.6%` | `83.3% / 57.6%` | `94.8% / 96.4%` |
| `NanoLM-B3-OnPolicy-KD8B` | `1.72B` | `73.2%` | `62.6%` | `63.4%` | `65.1%` | `52.6%` | `78.0% / 75.0%` | `90.1% / 88.5%` |

---

## 5. P1 Grounding & Source-Conservation Scorecard (`C12`–`C15` + `X02`)

| Cohort / Policy | Grounded Exact `/120` | Exact Assertions Emitted `/80` | Recovered Free Assertions Lost | Key Mechanism Diagnosis |
|---|---|---|---|---|
| `C14` Free Raw (`context`) | `0` (strict parse failed `24/24`) | `0` | `13/13` locally recovered lost | Checker rejected valid free slots via `independent_binder_state_mismatch` |
| `C14` Constrained (`context`) | `21` (`16 U + 5 M`) | `0` | N/A | Menu offered concrete `S/A/C` choices in only `4/120` slots (`0/80` exact gold) |
| `C14` Free Replay under `grounded.py` (`X02` trace) | Diagnostic replay only | `12/13` survive in extraction | `1/13` (`retain-08 duration`) | `retain-08`: `"yesterday"` inside explicit med clause `"I took ibuprofen ... yesterday"` leaked into `FieldName.DURATION`, creating false conflict |
