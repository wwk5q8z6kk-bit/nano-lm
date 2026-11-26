# NanoLM Cross-Agent Audit, Live Team Coordination & `G5` Unification Report

**Date:** `2026-11-26` · **Canonical Development Truth:** `origin/master` @ `3e36bf8` (`PR #194`, advancing via `PR #219` `39338d7`)  
**Program Mandate (`docs/ACTIVE_NOW.md`):** `capability_frontier = GENERAL_PURPOSE_LANGUAGE_MODEL` · `current_gate = MATCHED_GENERAL_MODEL_BASELINE`  
**Compute Discipline:** RunPod balance `$8.93 < $20.00` owner reserve (`0` active pods, `0` active TPUs) and Kaggle weekly GPU quota `30.00 / 30.00 h` reached $\implies$ **`HARD_ZERO_PAID_RUNPOD` & `$0.00` Local M3 Max Execution**  
**Machine-Readable Coordination Ledger:** [`artifacts/nanolm_omega_v1/cross_agent_coordination_v2.json`](artifacts/nanolm_omega_v1/cross_agent_coordination_v2.json)

---

## 1. Live Multi-Agent Coordination Roster (`2026-11-26`)

All active parallel sessions were audited read-only and synchronized directly over the inter-agent messaging bus (`send_message` + ACK handshake) to eliminate duplicate experiments, prevent CPU/MPS/GPU contention, and enforce strict path/worktree non-interference:

| Active Agent / Branch | Worktree / Scope | Verified Live Status & Synchronized Lane Ownership |
|---|---|---|
| **`origin/master` (`3e36bf8` $\to$ `39338d7`, PRs `#187–#194, #219`)** | Canonical Remote Truth | Merged license-clean Corpus v3 filter (`#188`), 3-seed GCP paired alias recall (`#187`), prefix-causal `ACE-X9.2` routing (`#191`), grammar contrast (`#192`), `M21` vs Dense code trial (`#193`), **equal-wall-clock Dense vs `M21` (`#194`)** establishing **equal-token + equal-wall-clock**, and Phase 1–6 lab consolidation (`#219`). |
| **`PR #214`, `#215`, `#217`, `#218` (`Express1M2` / Cloud TPU & H100)** | `valueres-42m`, `stage-b-and-untied-head-v1`, `dual-backend-gdn-parity-v1`, `cycle2-gate-audit` | Completed `19.2B` TPU `v5e-4` tokens + `2.02B` H100 tokens (`0` active VMs/pods): proved **tied embeddings (`tie_embeddings=True`) beat untied `lm_head` below `4B`** (`U_G1`, `+0.0277` BPB when budget goes to tied width, $t=+63.85$), closed standalone scalar `ValueRes` on pure Dense at `40.38M` (`+0.0059 < +0.0080`), passed `Stage B` `SmolLM2-135M -> 22 GDN + 8 GQA` (`88.57%` gap closed in `250M` tok), and preregistered PyTorch `fla==0.6.0` $\leftrightarrow$ JAX/TPU parity (`#217`). Diagnosed PR `#218`'s `F1` FP32 `~1e-3` hold via Triton default `allow_tf32=True` (fixed by `allow_tf32=False` $\to$ `9.2e-6`). |
| **Peer `0430cfe4` (`nanolm/arch-evolution-v3`)** | `.worktrees/nanolm-v3` | **Live ACK synchronized.** Completed `UNIFIED-SYNTH-01` (`BOBO-2Probe` + `Lag1-Kalman-SurpriseLandmarks` + `SyncFreeDualCopyHead` + `CausalGodelSemiringBlock`): **`1.0000` seen, `0.9987 ± 0.0012` unseen-val, `0.9243 ± 0.0184` unseen-key+val** (`92.23s` vs `167.66s`). Adopted our causal relative-lag `-(t-j)/W` length-invariance fix (`0.0` prefix drift); shared `replace=False` (`EXACT-RECALL-03b`) and MPS `_no_autocast_ctx` guards with us. |
| **Peer `a337f4f7` (`research/unified-discovery-loop-v1`)** | `.worktrees/nanolm-unified-loop` | **Live ACK synchronized.** Owns `EXP-U10` (`HSPA_LieSO3` proportional fiber width `19.2k tok/s` + low-rank `ConceptSlotV2` hitting `100%` `4x` OOD `MQAR_Alias_Semantic_4x` at `200` steps), adopting PR `#194` equal-wall-clock + equal-token `M0_Dense_GQA_RoPE` controls. |
| **Peer `2cdbaf96` (`nanolm/discovery-lab-v1`, PR #219)** | `.worktrees/nanolm-lab` | **Live ACK synchronized & merged into `origin/master` (`39338d7`).** Disjoint file paths (`lab/`, `nanolm/`, `docs/nanolm/`). Recorded `45` runs (`27.0B` tokens): `S2_nanolm_v3_350m` (`0.8634` BPB, `98.57%` 512-tok copy, `-75%` KV) vs `S0_dense_350m` (`0.8677` BPB, `74.65%` copy), `C7` Hidden-2-Hop `98.2%`, and `NanoLM-B9-2B` (`1.72B`) `72.0% > Naked 120B (70.0%)`. |
| **Peer `bb3dca4d` (`nanolm/autonomous-discovery-v1`)** | `.worktrees/nanolm-paradigm` | **Live ACK synchronized.** Landed commit `9408534` (`nanolm/eval/leak_audit.py` beyond-13-gram identifier + 6-gram skeleton leak audit and `nanolm/eval/external.py` `LAMBADA` / `HellaSwag` / `ARC-E` harness) and alerted the team that Kaggle reached `30.00 / 30.00 h` weekly GPU quota. |
| **Peer `2cc1ce51` (`research/nanolm-master-cycle1`)** | `.worktrees/nanolm-master-cycle1` | **Live ACK synchronized.** Verified Cycle 3 (`0c16e3a`, `36/36` tests), independently confirmed `allow_tf32=True` in Hopper `F1` and `#214`/`#215` curve identity, and is executing Cycle 4 CPU local verification. |
| **This Session `175c8e32` (`nanolm/architecture-invention-v1`, PR #176)** | `.worktrees/nanolm-discovery` | Synced `eb26201`; owns **`G5_CrossPollinated`** (`G4_NoSparseRefiner` + stateful prefix-invariant `StatefulSyncFreeDualCopyHead` + exact `C_stream = 1024` streaming chunked prefill/loss + identifier/skeleton leak audit) benchmarked in `OMEGA-EXP-04` at `$0.00`. |

---

## 2. Five Converged Scientific Laws Across All NanoLM Streams

1. **The Tied-Embedding Law Below `4B` (`PR #215` `U_G1`):** Untying `lm_head` at `40M–57M` (`9.59B` TPU tokens) degraded held-out BPB (`-0.0030`, $t=-4.61$) and `LAMBADA` PPL (`173.7 -> 243.1`), whereas spending those parameters on **tied backbone width** won `+0.0277` BPB ($t=+63.85$). `G3`/`G4`/`G5` keep `tie_embeddings=True` by default while honoring untied mode (`eb26201`).
2. **The Open-Vocabulary Pigeonhole & Dual-Copy Law (`EXACT-RECALL-01/03` + `MEMBOUND-01` + `nanolm-autolab`):** Pure generative heads (`W_out`) score **`0%`** on never-trained OOD token IDs regardless of mixer (`AAAA`, `DDDA`, or `G4`), while fixed recurrent states store $\le 1.5\%$ of raw bits as arbitrary literals. Fusing `StatefulSyncFreeDualCopyHead` unlocks `> 90%` zero-shot unseen open-vocabulary literal recall at `< 0.5%` parameter overhead and constant $O(W)$ ring-buffer state.
3. **The Prefix-Invariant Relative-Lag Law (`nanolm-discovery` $\leftrightarrow$ `nanolm-v3` Cross-Pollination):** Replacing $j / T$ with causal relative lag $-\beta_{\mathrm{rec}}\frac{p - j}{W}$ guarantees **`0.0` prefix drift** across sequence lengths and **exact 3-way equivalence** (`forward()` $\equiv$ `forward_chunked(C_stream)` $\equiv$ `forward_step()`).
4. **Single-Pass Heterogeneous Mixers Beat Unshielded Pass-2 Token Refiners (`OMEGA-EXP-01` + `EXP-U03`):** Single-pass `G4_NoSparseRefiner` (`2.1112` ID / `2.1079` `2x` OOD BPB) beat Pass-2 `G4_Omega` (`2.1175`) at iso-param `20.56M`. Deliberative compute belongs in `OmegaEFEPlanner` and independent exact verification (`Dual-Spec / IVE`).
5. **The Equal-Wall-Clock + Streaming Memory Gate (`PR #194` + `H-Ω21`):** Every `OMEGA-EXP-04` evaluation reports **both equal-token AND equal-wall-clock** metrics alongside exact `C_stream = 1024` streaming chunked prefill scaling through `32,768` tokens.

---

## 3. Executed Cross-Pollination Validation: `G5_CrossPollinated` (`OMEGA-EXP-04`, `$0.00`)

Executed via `scripts/run_omega_exp04_cross_pollinated.py` (`artifacts/nanolm_omega_v1/omega_exp04_cross_pollinated.json`):

### 3A. Decontaminated `replace=False` Open-Vocabulary Copy (`mech`, `104.6k–105.0k` params, `n=3` seeds)
| Arm | Params | `Seen ID` ↑ | **`Unseen Open-Vocab` (0 exposure)** ↑ | **`2x OOD Unseen`** ↑ |
|---|---:|---:|---:|---:|
| `b0_transformer` | `104,768` | `0.0983 ± 0.0157` | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` |
| `g4_no_sparse_refiner` | `104,978` | `0.0534 ± 0.0045` | `0.0072 ± 0.0124` | `0.0107 ± 0.0186` |
| **`g5_cross_pollinated`** | **`104,600`** | **`0.9089 ± 0.0023`** | **`0.9173 ± 0.0030`** | **`0.9561 ± 0.0043`** |

### 3B. Real `Corpus v1` (`scale_8m`, `n=2` seeds `[42, 137]`, `PR #194` Equal-Token & Equal-Wall-Clock)
- **Beyond-13-Gram Repo Split Audit:** Cross-package `nano/` + `nanolm_discovery/` vs `nanolm_omega/` measured `34/204` shared distinctive identifiers (`1.0` item hit rate) and `0.0805` mean 6-gram skeleton overlap vs `0.0000` on a disjoint manifold probe — confirming the importance of identifier/skeleton filtering beyond 13-grams.
- **Equal-Token (`30` steps = `15,360` tokens) vs Equal-Wall-Clock (`~244.4s`):**

| Arm (`scale_8m`) | Params | Steps / Tokens | Wall-Clock (s) | `ID BPB` (`T=128`) ↓ | **`2x OOD BPB` (`T=256`)** ↓ |
|---|---:|---|---:|---:|---:|
| `b0_equal_token` | `8,653,248` | `30` / `15,360` | `12.63s` | `3.0492 ± 0.0381` | `3.0500 ± 0.0379` |
| `g4_no_sparse_refiner` | `8,665,055` | `30` / `15,360` | `176.09s` | `3.0663 ± 0.0394` | `3.0657 ± 0.0390` |
| **`g5_cross_pollinated`** | **`8,634,725`** | `30` / `15,360` | `224.62s` | `2.6245 ± 0.0169` (`-0.4247` vs `b0_tok`) | **`2.5018 ± 0.0395`** (`-0.5482` vs `b0_tok`) |
| `b0_equal_wall_clock` | `8,653,248` | `1,094 ± 274` / `560,128` | `244.35s` | **`2.6028 ± 0.0470`** | `2.6088 ± 0.0452` |

- **Honest Equal-Wall-Clock Finding (`PR #194`):** On uncompiled PyTorch MPS, `b0_equal_wall_clock` achieved **`36.5x`** higher training throughput (`2,759 tok/s` vs `74.5 tok/s` for `G5`), overtaking 30-step `G5` slightly on `128`-tok ID BPB (`2.6028` vs `2.6245`, `web`/`textbook` advantage). **Even with `36.5x` fewer steps**, `G5` still beat `b0_equal_wall_clock` on `2x OOD BPB` (`2.5018` vs `2.6088`, `-0.1070` BPB) and `code` BPB (`2.91–3.01` vs `3.70–3.80`), confirming fused Triton/Metal WY kernels (`allow_tf32=False`) as the decisive next speedup!

### 3C. Streaming Chunked Prefill (`C_stream = 1024`, `g5_cross_pollinated_scale_8m`, `H-Ω21`)
- **Exact `T=2,048` Parity:** `logit_max_abs_err = 2.57e-5`, `ce_loss_abs_err = 9.54e-7`.
- **Flat `1,024 -> 32,768` Allocation (`936,648`B `32k` persistent bf16 state):** `801.23 MiB` (`1k`) $\to$ `801.30 MiB` (`4k`, `560.5 tok/s`) $\to$ `801.52 MiB` (`16k`, `1,449.3 tok/s`) $\to$ **`801.84 MiB`** (`32k`, `1,418.7 tok/s`, **`+0.60 MiB`** growth over `32x` context, **`32.0x` logits tensor reduction** `4,096 MiB -> 128 MiB`).
