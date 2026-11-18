# EXP-U03 Analysis — Scaffold Attribution (`E23 C7-Omega`) & Routing Paradox (`E24 HEM-ATRD`)

- **Raw Artifact:** [results.json](file:///Users/mac/Projects/nano-lm/.worktrees/nanolm-unified-loop/research/unified_loop/experiments/U03_scaffold_routing_audit/results.json)
- **Script / Protocol:** [run_u03_scaffold_routing_audit.py](file:///Users/mac/Projects/nano-lm/.worktrees/nanolm-unified-loop/research/unified_loop/scripts/run_u03_scaffold_routing_audit.py) · [protocol.md](file:///Users/mac/Projects/nano-lm/.worktrees/nanolm-unified-loop/research/unified_loop/experiments/U03_scaffold_routing_audit/protocol.md)
- **Elapsed / Cost:** `1,315.53s` wall on contended CPU (`$0.00` paid compute)

---

## 1. Part A — `E23 C7-Omega` Factorial Attribution (`3` Seeds)

Our hypothesis (`H-15`/`H-17` audit) suspected Python-computed `SymbolicBeliefState` binary oracle features (`sym_feats`) drove most of `C7-Omega`'s `1.000` Deep-OOD (`N=14`, 4-hop, unseen high entity IDs `6..13`) generalization. **The controlled factorial experiment refuted that suspicion:**

| Arm | Params | `canonical_mode` | `sym_feats` | Shield | ID Verified | Deep-OOD Verified | Deep-OOD Value Custody | Contradiction Rev | Stress Verified (`noise=3.5`) | Stress Unsafe |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|
| `A0_c0_raw_sym0` | 41,959 | `none` | OFF | OFF | 0.8708 | 0.0000 ± 0.0000 | 0.4533 | 0.0000 | 0.0000 | 0.7733 |
| `A1_c0_raw_symON` | 41,959 | `none` | ON | OFF | 0.8708 | 0.0000 ± 0.0000 | 0.4533 | 0.0000 | 0.0000 | 0.7733 |
| `A2_root_levels_sym0` | 41,959 | `root_levels` | OFF | OFF | 1.0000 | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7367 |
| `A3_root_levels_symON` | 41,959 | `root_levels` | ON | OFF | 1.0000 | 0.0000 ± 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.8667 |
| `A4_frontier_sym0` | 41,959 | `frontier_debruijn` | **OFF** | **OFF** | **1.0000** | **1.0000 ± 0.0000** | **1.0000** | **1.0000** | **0.9967** | 0.0033 |
| `A5_frontier_symON` | 41,959 | `frontier_debruijn` | ON | OFF | 1.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 0.9933 | 0.0067 |
| `A6_frontier_sym0_shield` | 41,959 | `frontier_debruijn` | OFF | ON | 1.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `A7_frontier_symON_shield` | 41,959 | `frontier_debruijn` | ON | ON | 1.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `A8_classical_rule_0param` | **0** | deterministic | ON | ON | **1.0000** | **1.0000 ± 0.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0000** |

### Scientific Conclusions (Part A)
1. **Binary `sym_feats` Contributed Zero Verification Gain Over Raw Representations:** Turning `sym_feats=ON` on `raw` (`A1`: `0.000`) or `root_levels` (`A3`: `0.000`) yielded **0.000** Deep-OOD verified resolution (`A3` even hit `0.8667` unsafe rate).
2. **Pure Neural GRU+Attn + Frontier De Bruijn Gauge (`A4_frontier_sym0`) Survives Complete Removal of `sym_feats` and Shield:** `A4` gets **`1.0000 ± 0.0000`** on `N=14`, 4-hop Deep-OOD (`0.9967` under `noise=3.5`). Because frontier-relative De Bruijn numbering assigns `0` to the most recently encountered causal node, any depth `D` fault trajectory maps onto an identical relative index pattern for neural weights.
3. **Important Boundary Disclosed (`A8_classical_rule_0param`):** First, chronological De Bruijn canonicalization (`ws.map_entities`) is still a deterministic external gauge preprocessor outside the neural network; second, on this synthetic finite-state environment, a **`0`-parameter deterministic rule (`A8`) also achieves `1.0000`** with zero training. Thus **C7 proves the mathematical power of causal-frontier gauge canonicalization**, rather than proving neural reasoning exceeds symbolic solvers where state transitions are fully observable (`Contract Principle #5` & `#7`).

---

## 2. Part B — `E24 HEM-ATRD` vs `X5 Shannon` Routing Paradox (`5` Seeds)

| Arm (`V=128`) | Params | Final CF 8-Way Acc | Pass-1 CF Acc | Unconditional Pass-2 CF Acc (`rawP2`) | Eval Query Routed | Copy Waste | OOD-Shift Copy Retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| `H0_pass1_unrouted` | 13,424 | 0.0984 ± 0.0238 | 0.0984 | 0.0984 | 0.0000 | 0.0000 | 1.0000 |
| `H1_x5_full_vocab_shannon` | 23,760 | **1.0000 ± 0.0000** | 0.1063 | **1.0000** | 1.0000 | 0.2792 | 0.8874 ± 0.0160 |
| `H2_x8_top2_margin_quantile` | 23,760 | 0.5766 ± 0.4281 | 0.1110 | **0.6141 ± 0.4275** | 1.0000 | 0.2978 | 0.8788 ± 0.0383 |
| `H3_mm_atrd_top2_shield` | 23,760 | 0.6922 ± 0.1851 | 0.1094 | **0.8641 ± 0.0853** | 0.9453 | 0.0000 | **1.0000** |
| `H4_hem_atrd_topk_shield` | 23,760 | 0.7469 ± 0.3621 | 0.1031 | **0.7750 ± 0.3333** | 0.9469 | 0.0000 | **1.0000** |
| `H5_hem_train_warmup` | 23,760 | 0.7766 ± 0.1995 | 0.1219 | 0.8328 ± 0.2263 | 0.9422 | 0.0000 | 1.0000 |
| `H6_hem_warmup_decoupled_eval` | 23,760 | 0.7828 ± 0.1990 | 0.1219 | 0.8328 ± 0.2263 | 0.9422 | 0.0000 | 1.0000 |

### Three Root Causes Exposed by Per-Seed Forensics
1. **`H2_x8` Gradient Scaling Bug:** `margin_shield_refiner.py:173` multiplies Pass-2 hidden deltas by `u_sel` (`1 - margin`) **only** in `x8_margin_quantile` mode, causing `rawP2` failure on `seed 0` (`0.125`) and `seed 2` (`0.1953`) despite `eval_qry_routed = 1.000`.
2. **Lyapunov Shield Over-Rejection on Shortcut-Confident Pass-1 (`H3` & `H4`):** On `H3_mm_atrd`, unconditional `rawP2 = 0.8641`, yet final shielded accuracy dropped **`-17.2 pp` to `0.6922`** because Pass-1 confidently backed a wrong local hint (`m1 > m2`) and the margin shield rejected Pass-2's correction!
3. **Hard `min_ambiguity_floor = 0.46` Blocking ~15% of CF Queries + Early-Training Pass-2 Variance:** On `H5/H6` `seed 1` (`rawP2 = 0.9219 -> final = 0.8047, ev_rt = 0.8516`) and `seed 3` (`rawP2 = 1.000 -> final = 0.8672, ev_rt = 0.8594`), the fixed `0.46` floor dropped `14%–15%` of CF queries completely; meanwhile on `H4` `seed 2` (`rawP2 = 0.6172`) and `seed 4` (`rawP2 = 0.2578`), sparse quantile routing (`target_route_frac=0.30` using narrow top-8 support vs `X5`'s broad tail entropy) starved Pass-2 early in training before `st=150`. **Verdict:** Keep `HEM` epistemic metric ONLY for vocab-invariance, **do not** use naive Lyapunov margin comparison when Pass-1 can be confidently wrong (`reconcile H-21/H-31`), and stabilize Pass-2 training with non-gated / softly-weighted auxiliary supervision!
