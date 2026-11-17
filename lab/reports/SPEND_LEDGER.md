# RunPod Spend Ledger (Track A + Track B — Final Closeout)

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

- **Active pods remaining:** `0` (`runpodctl get pod` verified empty)
- **Final realized campaign spend:** **\$39.07 / \$180.00** (**21.7%** of autonomous budget; **\$140.93** remaining)
