# Track A Empirical Architecture, Optimizer & Scaling Results (Auto-Generated)

## 1. Iso-Parameter (~29.5M Non-Emb / ~46.2M Total) Architecture & Optimizer Grid

Runs: 39. Values = mean over seeds; ± = half-range across seeds. Δ% is relative to `T0_dense/adamw/tokens` (negative = better).

| Arm | Optim | Match | n | Tokens | Train TFLOP·s | KV B/tok | Composite BPB | Δ% vs Ref | web | textbook | math | code | copy loss | ctx gain (nats) | tok/s | Decision / Notes |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| X8_margin_atrd_muon | muon | tokens | 2 | 600M | 1.05e+05 | 3,072 | 0.9543 ± 0.0005 | -7.32% | 1.0674 | 0.7576 | 1.1051 | 0.7738 | 1.445 | +0.533 | 91,050 | **WIN** (prereg rule met) |
| X5_atrd_muon | muon | tokens | 1 | 600M | 1.05e+05 | 3,072 | 0.9555 | -7.20% | 1.0689 | 0.7587 | 1.1065 | 0.7746 | 3.065 | +0.519 | 86,227 | promising (< -1%, 1 seed) |
| X4_nanolm_v2_muon | muon | tokens | 2 | 600M | 1.06e+05 | 3,072 | 0.9575 ± 0.0008 | -7.01% | 1.0691 | 0.7580 | 1.1089 | 0.7821 | 0.763 | +0.523 | 166,225 | **WIN** (prereg rule met) |
| X6_pkm_muon | muon | tokens | 2 | 600M | 1.06e+05 | 3,072 | 0.9575 ± 0.0003 | -7.01% | 1.0688 | 0.7581 | 1.1086 | 0.7831 | 1.649 | +0.522 | 150,917 | **WIN** (prereg rule met) |
| T0_dense | muon | tokens | 2 | 600M | 1.85e+05 | 10,240 | 0.9645 ± 0.0004 | -6.33% | 1.0751 | 0.7589 | 1.1183 | 0.7951 | 1.177 | +0.525 | 248,599 | **WIN** (prereg rule met) |
| X2_sharp_delta_vres | adamw | tokens | 2 | 600M | 1.06e+05 | 3,072 | 0.9689 ± 0.0009 | -5.90% | 1.0795 | 0.7725 | 1.1218 | 0.7910 | 0.536 | +0.517 | 238,809 | **WIN** (prereg rule met) |
| X3_sgrd_delta | adamw | tokens | 2 | 600M | 1.06e+05 | 3,072 | 0.9815 ± 0.0005 | -4.67% | 1.0891 | 0.7832 | 1.1383 | 0.8079 | 0.802 | +0.509 | 90,977 | **WIN** (prereg rule met); SGRD loops=1.00, γ=0.852 |
| X1_gated_vres | adamw | tokens | 2 | 600M | 1.06e+05 | 3,072 | 0.9869 ± 0.0013 | -4.15% | 1.0926 | 0.7860 | 1.1445 | 0.8188 | 0.746 | +0.505 | 234,504 | **WIN** (prereg rule met) |
| T1_vres | adamw | tokens | 2 | 600M | 1.85e+05 | 10,240 | 0.9919 ± 0.0002 | -3.66% | 1.0978 | 0.7889 | 1.1505 | 0.8245 | 9.207 | +0.527 | 275,706 | **WIN** (prereg rule met) |
| H3_delta_vres | adamw | tokens | 2 | 600M | 1.76e+05 | 3,072 | 0.9942 ± 0.0000 | -3.44% | 1.0971 | 0.7891 | 1.1546 | 0.8331 | 0.879 | +0.507 | 202,310 | **WIN** (prereg rule met) |
| R2_delta_loop | adamw | tokens | 2 | 600M | 2.42e+05 | 3,072 | 0.9976 ± 0.0023 | -3.11% | 1.1002 | 0.7937 | 1.1549 | 0.8392 | 0.589 | +0.502 | 150,464 | **WIN** (prereg rule met) |
| H2_delta | adamw | tokens | 2 | 600M | 1.76e+05 | 3,072 | 1.0035 ± 0.0005 | -2.54% | 1.1047 | 0.7975 | 1.1654 | 0.8451 | 1.063 | +0.502 | 207,105 | **WIN** (prereg rule met) |
| H1_gla | adamw | tokens | 2 | 600M | 1.74e+05 | 3,072 | 1.0056 ± 0.0030 | -2.33% | 1.1067 | 0.7988 | 1.1680 | 0.8479 | 1.093 | +0.505 | 217,183 | **WIN** (prereg rule met) |
| T2_swa | adamw | tokens | 2 | 600M | 1.78e+05 | 3,072 | 1.0260 ± 0.0010 | -0.35% | 1.1251 | 0.8186 | 1.1886 | 0.8726 | 2.752 | +0.504 | 250,829 | no detectable difference (<1%) |
| R2_delta_loop | adamw | flops | 2 | 460M | 1.85e+05 | 3,072 | 1.0288 ± 0.0006 | -0.08% | 1.1254 | 0.8241 | 1.1909 | 0.8783 | 0.513 | +0.505 | 151,379 | no detectable difference (<1%) |
| T0_dense | adamw | tokens | 2 | 600M | 1.85e+05 | 10,240 | 1.0296 ± 0.0048 | +0.00% | 1.1286 | 0.8228 | 1.1921 | 0.8761 | 3.314 | +0.509 | 267,371 |  |
| R1_loop | adamw | tokens | 2 | 600M | 2.58e+05 | 10,240 | 1.0415 ± 0.0001 | +1.15% | 1.1387 | 0.8345 | 1.2026 | 0.8929 | 0.964 | +0.486 | 201,628 | worse |
| P2_delta | adamw | tokens | 2 | 600M | 1.72e+05 | 0 | 1.0466 ± 0.0023 | +1.65% | 1.1218 | 0.8116 | 1.2353 | 0.9423 | 11.651 | +0.434 | 184,370 | worse |
| P1_gla | adamw | tokens | 2 | 600M | 1.7e+05 | 0 | 1.0623 ± 0.0033 | +3.17% | 1.1292 | 0.8161 | 1.2561 | 0.9809 | 12.027 | +0.423 | 202,188 | worse |
| R1_loop | adamw | flops | 2 | 432M | 1.85e+05 | 10,240 | 1.1147 ± 0.0003 | +8.26% | 1.1991 | 0.8992 | 1.2865 | 0.9897 | 2.208 | +0.483 | 198,638 | worse |

Reference `T0_dense` seed half-range on composite BPB: 0.0048 (0.46%).

## 2. Multi-Scale Transfer (`LAB-SCALE-01`, ~96.3M Non-Emb / ~121M Total)

Runs: 2. Values = mean over seeds; ± = half-range across seeds. Δ% is relative to `S0_dense_125m/muon/tokens` (negative = better).

| Arm | Optim | Match | n | Tokens | Train TFLOP·s | KV B/tok | Composite BPB | Δ% vs Ref | web | textbook | math | code | copy loss | ctx gain (nats) | tok/s | Decision / Notes |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S0_dense_125m | muon | tokens | 1 | 600M | 3.45e+05 | 16,384 | 0.9105 | +0.00% | 1.0269 | 0.7025 | 1.0544 | 0.7420 | 0.609 | +0.530 | 110,029 |  |
| S1_nanolm_v2_125m | muon | tokens | 1 | 600M | 3.47e+05 | 4,096 | 0.9106 | +0.00% | 1.0272 | 0.7062 | 1.0546 | 0.7377 | 0.993 | +0.520 | 71,860 | 1-seed check |
`ctx gain` = loss(positions 0–64) − loss(last 256 positions) on web val (larger = better long-context utilization).
