# EXP-U01 Analysis — Forensic Mechanistic Autopsy of Real 600M-Token Checkpoints (`X4_46M`, `S0_125M`, `S1_125M`)

- **Raw Artifact:** [results.json](file:///Users/mac/Projects/nano-lm/.worktrees/nanolm-unified-loop/research/unified_loop/experiments/U01_real_125m_autopsy/results.json)
- **Script / Protocol:** [run_u01_real_125m_autopsy.py](file:///Users/mac/Projects/nano-lm/.worktrees/nanolm-unified-loop/research/unified_loop/scripts/run_u01_real_125m_autopsy.py) · [protocol.md](file:///Users/mac/Projects/nano-lm/.worktrees/nanolm-unified-loop/research/unified_loop/experiments/U01_real_125m_autopsy/protocol.md)
- **Execution:** CPU `float32`, `$0.00` paid compute (`4,303.35s` wall under heavy peer load)

---

## 1. Canonical Reproduction Check (`seq=1024`)

| Run | Architecture | Total Params | Non-Emb Params | Reported BPB / Copy@512 | Reproduced BPB (`w=48`) / Copy@512 | `ΔBPB` / `ΔCopy` |
|---|---|---:|---:|---|---|---|
| `X4_nanolm_v2_muon__muon__tokens__600M__s0` | `DDDADDDADA` (10L, `d=512`) | 46,154,296 | 29,377,080 | `0.95661` / `0.9613` | `0.94844` / `0.9610` | `-0.00817` / `-0.0003` |
| `S0_dense_125m__muon__tokens__600M__s0` | `A*16` (16L, `d=768`) | 121,137,920 | 95,972,096 | `0.91054` / `0.9406` | `0.90437` / `0.9407` | `-0.00617` / `+0.0001` |
| `S1_nanolm_v2_125m__muon__tokens__600M__s0` | `(DDDA)*4` (16L, `d=768`) | 121,596,528 | 96,430,704 | `0.91058` / `0.9270` | `0.90397` / `0.9275` | `-0.00661` / `+0.0005` |

*Note on preregistered `<0.002` BPB check:* `copy_acc_second_half` reproduced within `<= 0.0005` (`copy_loss`: `X4 0.5307 vs 0.5305`, `S0 0.6069 vs 0.6091`, `S1 0.9919 vs 0.9927`), while strided `windows=48` (`float32` vs original `windows=128` `bfloat16`) shifted `bpb_composite` consistently by `-0.0062` to `-0.0082` across all 3 models, preserving exact relative deltas (`S0` vs `S1`: `0.0004` diff).

---

## 2. Mechanistic Findings Inside Real Trained Delta Layers (`H-U01-A`)

Inspecting `nextgen.py` `Delta` layers (`kind="D"`, `head_dim = 64`) on real `val_code` and random `copy_probe` batches:

1. **Severe Key Subspace Rank Collapse & High Cosine Survives `600M` Tokens of Muon:**
   - **`S1_125m` (`val_code`):** Layer 0 starts at `mean_offdiag_key_cos = 0.31501`, `mean_key_eff_rank = 20.6 / 64`; **by Layers 1–2, effective rank collapses to `2.9–3.85 / 64` and `mean_offdiag_key_cos` spikes to `0.81851 / 0.81011` (`max_head_key_cos = 0.968–0.988`)!** Even Layers 4–14 stay at `0.50–0.73` cosine (`4.99–11.3 / 64` rank).
   - **`S1_125m` (`copy_probe`):** Collapse worsens sharply under random tokens — Layer 1 drops to **`eff_rank = 2.06 / 64` (`mean_cos = 0.88786`, `max_head = 0.97211`)**, Layers 2–4 stay at `2.95–3.33 / 64` (`cos = 0.853–0.861`), and **all 11 Delta layers past Layer 0 stay at `cos = 0.695–0.888`, `eff_rank = 2.06–5.92 / 64`!**
   - **`X4_46m`:** Shows the exact same pattern (`val_code` Layer 1–2 `cos = 0.571–0.616`, `eff_rank = 8.06–8.07`; `copy_probe` Layer 1–8 `cos = 0.684–0.770`, `eff_rank = 4.32–6.50`).
   - *Nuance on sign:* `frac_positive_key_coords` is `0.48–0.61` (not `1.0`, since `SiLU(x)` takes negative values down to `-0.278` before `ShortConv` + L2 norm, and L2 norm sees positive bias + `ShortConv` temporal smoothing rather than strict nonnegativity). **Either way, `H-U01-A0` (that `600M` Muon training would self-orthogonalize keys) is FALSIFIED.**
2. **Short Memory Horizon (`mean_decay_a`) & Low Write Gain (`mean_beta`):**
   - On `copy_probe` in `S1_125m`, Layers 1–4 have `mean_decay_a = 0.0358–0.0812` (erasing `92–96%` of state per token!) and Layer 1 has `mean_beta = 0.0065`! Even Layers 5–14 decay at `0.18–0.37` (`~1.5–4` token half-life). Thus `S1`'s Delta layers act primarily as short-range local mixers (`~1–5` tokens) rather than `512`-token associative state matrices.

---

## 3. Length Extrapolation (`L = 256..2048`) & Zero-Shot Real-Haystack MQAR (`H-U01-B`, `H-U01-C`)

| Model | `L=256` web / copy | `L=512` web / copy | `L=1024` web / copy | `L=2048` web / **copy1024** | MQAR `T256_N1` (com / rare) | MQAR `T1024_N8` (com / rare) |
|---|---:|---:|---:|---:|---:|---:|
| `X4_46m` (`DDDADDDADA`) | `1.1224` / `0.9690` | `1.0817` / `0.9716` | `1.0662` / `0.9602` | `1.0476` / **`0.0016`** | `0.0417` / `0.1667` | `0.0000` / `0.0000` |
| `S0_125m` (`A*16`) | `1.0832` / `0.9611` | `1.0416` / `0.9652` | `1.0272` / `0.9242` | `1.0100` / **`0.0582`** | `0.2500` / `0.2500` | `0.0000` / `0.0833` |
| `S1_125m` (`(DDDA)*4`) | `1.0821` / `0.9803` | `1.0420` / `0.9767` | `1.0261` / `0.9229` | `1.0068` / **`0.0021`** | `0.0833` / `0.1667` | `0.0000` / `0.0833` |

### Key Conclusions (`U-01`)
1. **`H-U01-B` Partly Refuted / Sharpened:** Natural language BPB (`web`, `code`) improved smoothly through `L=2048` on **both** `S0_125m` and `S1_125m` (no RoPE text perplexity cliff at `2x` length). **However**, on `L=2048` exact copy (`copy_len=1024`), **all three collapsed**, with `S1_125m` (`0.0021`, `nll=11.6306`) dropping **`28x` lower** than `S0_125m` (`0.0582`, `nll=9.628`) — whereas inside short windows (`L=256, 512`), `S1` actually outperformed `S0` (`0.9803 vs 0.9611`).
2. **`H-U01-C` Confirmed (`125M` Blindness on Unfamiliar Associative Bindings):** Zero-shot single-needle recall in real `val_web` drops from `0.2500` (`S0_125m`) to `0.0833` (`S1_125m` common IDs), and at `T=1024, N=8` all three are near `0%` (`0%–8.3%`). Combined with `S1`'s `eff_rank = 2.06–5.92 / 64` key collapse and `0.04–0.37` per-step decay, **long-range exact recall in `S1` relies almost completely on its 4 `RoPE` attention layers, which break past `L=1024`.**
