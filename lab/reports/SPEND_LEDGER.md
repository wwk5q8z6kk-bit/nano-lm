# RunPod Spend Ledger (Phases 1–6 — Final Closeout)

Envelope: \$180.00 autonomous budget (\$200.00 hard cap, \$20.00 owner reserve untouched).

| Pod ID | GPU | \$/h | Start (UTC) | Stop (UTC) | Purpose | Realized Cost |
|---|---|---:|---|---|---|---:|
| `l89h71yq2bd58j` | H100 NVL (secure, US) | 3.19 | 2026-11-17 04:46 | 2026-11-17 04:59 | `LAB-ARCH-02` Pilot P0 throughput calibration | \$0.70 |
| `e71ekqgldujy96` | RTX 4090 (secure, IE) | 0.89 | 2026-11-17 04:46 | 2026-11-17 09:32 | Pilot P0 + `LAB-ARCH-02` grid worker `q0` (6 runs) | \$4.24 |
| `q44zayfv4tkgfx` | RTX 4090 (secure, US) | 0.89 | 2026-11-17 05:07 | 2026-11-17 08:55 | `LAB-ARCH-02` grid worker `q1` (5 runs) | \$3.38 |
| `jyg0e9oyx8lnt6` | RTX 4090 (secure, IE) | 0.89 | 2026-11-17 05:07 | 2026-11-17 09:03 | `LAB-ARCH-02` grid worker `q3` (5 runs) | \$3.50 |
| `fx6y6jctxinfkx` | RTX 4090 (secure, IE) | 0.89 | 2026-11-17 05:07 | 2026-11-17 09:05 | `LAB-ARCH-02` grid worker `q2` (5 runs) | \$3.53 |
| `pahl3151c37dlu` | RTX 4090 (secure, US) | 0.89 | 2026-11-17 05:07 | 2026-11-17 09:58 | `LAB-ARCH-02` grid worker `q4` (5 runs) | \$4.32 |
| `2wjeaobs3gnszi` | A40 48GB (secure, CA) | 0.59 | 2026-11-17 05:02 | 2026-11-17 10:56 | Track B 6-baseline matrix + `B1` + `B2` + `B3` KD + evals | \$3.48 |
| `0kakkj7pm077hz` | RTX 4090 (secure) | 0.89 | 2026-11-17 10:53 | 2026-11-17 13:29 | `LAB-ARCH-03` `X4_nanolm_v2_muon` (`s0, s1`) | \$2.31 |
| `x6qlkebreuukep` | RTX 5090 32GB (secure) | 1.39 | 2026-11-17 10:54 | 2026-11-17 14:11 | `LAB-ARCH-03` `X1 s0`, `X2 s0`, `X3 s0` | \$4.56 |
| `rxh6fj13xxjxbp` | RTX 5090 32GB (secure) | 1.39 | 2026-11-17 10:54 | 2026-11-17 14:20 | `LAB-ARCH-03` `X1 s1`, `X2 s1`, `X3 s1` | \$4.77 |
| `w91r5t0zq1pr62` | L40S 48GB (secure) | 1.09 | 2026-11-17 10:53 | 2026-11-17 14:49 | `LAB-SCALE-01` `125M` `S0_dense_125m` + `S1_nanolm_v2_125m` (`Muon`) | \$4.28 |
| `5yw0z6ebyic7qu` | H200 SXM 141GB (secure) | 5.29 | 2026-11-17 17:08 | 2026-11-17 20:59 | Phase 2: `Qwen3-32B` probe + `B4a/B4b/B4c` KD/FFT + `B5` RLVR + evals | \$20.37 |
| `d6a3l21u5tkrlc` | RTX 4090 (secure) | 0.89 | 2026-11-17 17:08 | 2026-11-17 20:43 | `LAB-ARCH-04` `X5_atrd_muon s0` + `X6_pkm_muon s0` (`600M` tok each) | \$3.19 |
| `84fdprlrg4uo8b` | RTX 4090 (secure) | 0.89 | 2026-11-17 17:08 | 2026-11-17 19:28 | `LAB-ARCH-04` `X5_atrd_muon s1` (OOM diag) + `X6_pkm_muon s1` (`600M` tok) | \$2.07 |
| `6duh0m3bwqwt19` | H200 SXM 141GB (secure) | 5.29 | 2026-11-17 21:35 | 2026-11-17 23:20 | Phase 2 Step 4: `Qwen3-1.7B`, `8B`, `32B`, `gpt-oss-120b` Comp 1 & 2 (v1) | \$9.26 |
| `d8fq11bqgf89h9` | RTX 4090 (secure) | 0.89 | 2026-11-18 01:39 | 2026-11-18 03:32 | `LAB-ARCH-05` `X8_margin_atrd_muon s0` (`600M` tok — `0.9548` BPB) | \$1.67 |
| `3tgu2te6rwti8s` | RTX 4090 (secure) | 0.89 | 2026-11-18 01:39 | 2026-11-18 03:35 | `LAB-ARCH-05` `X8_margin_atrd_muon s1` (`600M` tok — `0.9538` BPB) | \$1.72 |
| `fedq4pvk225egm` | H200 SXM 141GB (secure) | 5.29 | 2026-11-18 03:49 | 2026-11-18 05:14 | Phase 3: `32B+120B` Stratified KD + `NanoLM-B6-2B` + `NanoLM-B6-3B` + `System-v2` | \$7.51 |
| `mqmxyjlqeevye8` | H200 SXM 141GB (secure) | 5.29 | 2026-11-18 05:55 | 2026-11-18 07:17 | Phase 4: `NanoLM-B6-3B-Fixed` (`473/480` SFT + GRPO) + `System-v3` `1.7B..120B` | \$7.25 |
| `y05unbw7rthg69` | L40S 48GB (secure) | 1.09 | 2026-11-18 10:04 | 2026-11-18 14:12 | `LAB-SCALE-02` `S2_nanolm_v3_125m s0` (`600M` tok — `0.9030` BPB `#1` `125M`) | \$4.50 |
| `g01l7zm6coe9d0` | H200 SXM 141GB (secure) | 5.29 | 2026-11-18 10:04 | 2026-11-18 11:56 | Phase 5A: `NanoLM-B7-3B` + `NanoLMSystem-v4` `1.7B..120B` (`3B=69%`, `120B=100%`) | \$9.89 |
| `jd8l9kbrtzdo7u` | H100 NVL 94GB (secure) | 3.19 | 2026-11-18 12:06 | 2026-11-18 12:32 | Phase 5B: Causally-shifted `NanoLM-B7-3B-Fixed` (`81.1%` P8, `65%` `#1` `<4B` Math) | \$1.40 |
| `012up3gv65insi` | H200 SXM 141GB (secure) | 5.29 | 2026-11-18 12:45 | 2026-11-18 13:32 | Phase 5C: Causal-Completed + `WiSE-FT` **`NanoLM-B8-3B`** (`85.6%` P8, `38%` C1, `66%` C2) | \$4.13 |
| `kd9qw8xwczgx1j` | H200 SXM 141GB (secure) | 5.29 | 2026-11-18 16:17 | 2026-11-18 17:50 | Phase 6: **`NanoLM-B9-2B`** (`1.72B`, `83.3%` P8, **`72.0% > Naked 120B`**) + `System-v5` `1.7B..120B` | \$8.21 |
| `y2w6enx8kkw2fx` | RTX 5090 32GB (secure) | 1.39 | 2026-11-18 16:17 | 2026-11-18 18:47 | `LAB-SCALE-03` `S2_nanolm_v3_125m s1` (`600M` tok — `0.9013` BPB; `n=2` `0.9021 ± 0.0009`) | \$3.48 |
| `7lgtpru2bcyczr` | RTX 4090 (secure) | 0.89 | 2026-11-18 16:54 | 2026-11-18 20:27 | `LAB-SCALE-03` `S0_dense_350m s0` (`600M` tok — `0.8677` BPB, `74.7%` OOD copy) | \$3.16 |
| `ydlvfq3siahi4d` | RTX 4090 (secure) | 0.89 | 2026-11-18 16:47 | 2026-11-19 00:09 | `LAB-SCALE-03` **`S2_nanolm_v3_350m s0`** (`600M` tok — **`0.8634` BPB `#1` `350M`**, `98.6%` copy) | \$6.55 |

- **Phases 1–4 realized spend (19 pods, all removed):** \$92.11
- **Phase 5 realized spend (4 pods, all removed):** \$19.92
- **Phase 6 realized spend (4 pods, all removed):** \$21.40
- **Total realized campaign spend (27 pods, 0 Nano LM pods active):** **\$133.43 / \$180.00** (`74.1%` of autonomous budget; **\$46.57** remaining; \$20.00 owner reserve untouched)
