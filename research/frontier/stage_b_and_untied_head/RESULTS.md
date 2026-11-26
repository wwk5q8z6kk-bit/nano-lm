# Results: Stage B (`SmolLM2-135M -> 22 GDN + 8 GQA`, `F2` Gate) & `40.38M / 57.2M` Untied-Head x ValueRes Factorial

**Pre-registration commit (prior to execution):** `e7b69e54024b392ab52d2bdc6b337d2bacbb763e` (`PROTOCOL.md`, `score_stage_b.py`, and `score_untied_head.py` untouched after execution).  
**Total experimental scale:** `2.024B` Hopper `H100` student distillation tokens across `10` Stage-B receipts + `9.594B` Cloud TPU `v5e-4` `Corpus v2` pretraining tokens across `12` Untied-Head factorial receipts (`22` verified receipts, `0` non-finite steps, `0` orphaned cloud nodes).

---

## Part I — Stage B (`RESEARCH-PLAN §20.5a`): `SmolLM2-135M -> 22 Gated DeltaNet + 8 GQA` (`F2` Gate)

### 1. Held-Out Mix Cross-Entropy (`nats/token`) & Causal Decomposition (`250,347,520` Student Tokens/Run)

All 10 Stage-B receipts share identical provenance (`code_sha256 = db73d7d764fb9772f3ba0ad0d1b754e328d16a48aeeee47b315aa600c4c0a513`, `train_head64mb_sha256 = 1333f2504a379b8528f6eb1d381784c5dda777f7aa4cb8ca40937ed8767b57e4`, `tasks_sha256 = 9a281bb487ceb72207874d7bd47f1a8fe4a91c8058a928a867472fdc18658aac`, `NVIDIA H100 80GB HBM3` + `triton==3.7.1`):

| Arm | Seeds | Student Tokens | Post-Stage-1 (`20.97M`) Mix | Final (`250.35M`) Mix (`nats`) | `A_all30 -> P` Gap Closed | Domain Nats (`web / textbook / math / code`) |
|---|---:|---:|---:|---:|---:|---|
| **`P` (Unconverted Parent)** | — | `0` | — | **`2.32551`** | `100.0%` (ceiling) | `2.7345 / 1.8296 / 1.9384 / 1.1688` |
| **`T` (Inherited Q/K/V/O + Stage 1 + Stage 2)** | `3` | `250,347,520` | `2.82265 ± 0.0068` | **`2.43563 ± 0.00078`** | **`88.57%` (`B2 PASS`)** | `2.8427 / 1.9437 / 2.0365 / 1.2981` |
| **`C1` (Random GDN + Stage 1 + Stage 2)** | `3` | `250,347,520` | `2.97202 ± 0.0072` | **`2.47168 ± 0.00104`** | **`84.82%`** | `2.8811 / 1.9863 / 2.0617 / 1.3264` |
| **`C2` (Random GDN, No Stage 1, Stage 2 Only)** | `1` | `250,347,520` | — | **`2.51465`** | `80.36%` | `2.9197 / 2.0273 / 2.1044 / 1.3663` |
| **`R` (All-Random From Scratch + KD)** | `1` | `250,347,520` | — | **`3.19368`** | `9.85%` | `3.6644 / 2.8690 / 2.5757 / 1.7830` |
| **`A_all30` (All-30 Converted, Stage 1 Only)** | `1` | `20,971,520` | `3.28855` | **`3.28855`** | `0.0%` (`50%` target: `2.80703`) | `3.4397 / 2.7241 / 3.1795 / 3.2372` |

#### Three-Factor Causal Decomposition over From-Scratch Hybrid KD (`R -> T`: `+0.75805` nats total)
1. **Backbone + Embedding Inheritance (`R -> C2`):** `3.19368 -> 2.51465` = **`+0.67903` nats** (`89.58%` of total improvement).
2. **Stage-1 Sublayer MSE Alignment (`C2 -> C1`):** `2.51465 -> 2.47168` = **`+0.04297` nats** (`5.67%` of total improvement).
3. **Attention Q/K/V/O Projection Inheritance (`C1 -> T`):** `2.47168 -> 2.43563` = **`+0.03604 ± 0.00154` nats** (`4.75%` of total improvement; per-seed `+0.03427, +0.03686, +0.03700`, **$t = +40.608$, $p = 0.0006$**). Notice that immediately after Stage 1 (`20.97M` tokens), `T` leads `C1` by **`+0.1494` nats** (`2.8227` vs `2.9720`), and `C1` narrows ~`76%` of that mixer-init gap during `229M`-token Stage-2 KD while `T` retains a highly consistent `+0.0360` nats edge.

### 2. Secondary 7-Task Zero-Shot Retention (`B3`)

| Benchmark (Metric) | Parent `P` | `T` Mean (`3` seeds) | `C1` Mean (`3` seeds) | `C2` (No Align) | `R` (All Random) | `T - C1` ($t$-stat) | `T` Chance-Adj Retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HellaSwag** (`acc_norm`) | `0.4322` | **`0.4016`** | `0.3892` | `0.3758` | `0.2675` | **`+0.0124` ($t=+9.73^*$)** | **`83.2%`** |
| **ARC-Easy** (`acc_norm`) | `0.5968` | **`0.5610`** | `0.5529` | `0.5412` | `0.3784` | **`+0.0081` ($t=+4.98^*$)** | **`89.7%`** |
| **ARC-Challenge** (`acc_norm`) | `0.2952` | **`0.2830`** | `0.2693` | `0.2747` | `0.2295` | **`+0.0136` ($t=+9.26^*$)** | **`72.9%`** |
| **PIQA** (`acc_norm`) | `0.6877` | **`0.6645`** | `0.6592` | `0.6502` | `0.5838` | `+0.0052` ($t=+2.92$) | **`87.6%`** |
| **OpenBookQA** (`acc_norm`) | `0.3300` | **`0.3253`** | `0.3153` | `0.3180` | `0.2520` | `+0.0100` ($t=+2.50$) | **`94.2%`** |
| **LAMBADA** (`acc`) | `0.4258` | **`0.3668`** | `0.3553` | `0.3282` | `0.1306` | `+0.0115` ($t=+2.22$) | **`86.1%`** |
| **WinoGrande** (`acc`) | `0.5312` | `0.5188` | **`0.5301`** | `0.5225` | `0.5114` | `-0.0113` ($t=-1.27$) | `60.4%` (`0.031` above chance) |

### 3. Registered Stage-B Verdicts (`results/summary_stage_b.json`)
- **`B1 = FAIL`:** `T` beats `C1` in `3/3` seeds (`t = +40.608`) and on `6/7` downstream tasks (statistically significant at `df=2` on HellaSwag, ARC-Challenge, and ARC-Easy), but `mean(C1 - T) = +0.03604` nats is below the pre-registered `0.10` nats threshold because `C1` *also* converges strongly under staged alignment + KD.
- **`B2 = PASS`:** `T` closes **`88.57%`** (`>> 50%`) of the `A_all30 -> P` gap (`2.43563 <= 2.80703`), and `C1` independently closes **`84.82%`**.
- **`F2_gate = HYBRID_DISTILLATION_SUCCEEDS_WEIGHT_INHERITANCE_NOT_NEEDED`:**
  1. **Architectural unblocking for Stage C (`0.5B` -> sub-4B):** Converting `73%` (`22/30`) of a pretrained Transformer's attention layers into `GatedDeltaNet` via staged alignment + KD recovers `85%–89%` of parent capability in just `250M` tokens (`+0.72` to `+0.76` nats ahead of from-scratch KD `R`), even when attention head dimensions prevent projection weight copying (`C1`).
  2. **Zero-cost free lunch when dimensions match:** Copying Q/K/V/O projections (`T`) costs `0` FLOPs and `0` extra parameters yet delivers a reliable `+0.0360` nats ($t=40.6$) and `+0.8%` to `+1.4%` reasoning boost on ARC/HellaSwag, so Q/K/V/O inheritance should be retained whenever teacher/student head dimensions align, without restricting teacher choice when they do not.

---

## Part II — `40.38M / 57.2M` Untied-Head x ValueRes x Equal-Parameter Tied Factorial (`9.59B` `Corpus v2` Tokens)

### 1. Three-Seed Results (`12 × 799,539,200` Unique Tokens on Cloud TPU `v5e-4`, `optax.contrib.muon`)

All 12 receipts share `code_sha256 = 280d4443373d2d3669edacc7da1f0f3fd0c068b2a4d0edf174d526346e8d0c4c` and identical `Corpus v2` validation shard SHA256s (`results/summary_untied_head.json`):

| Arm | Total Params | Non-Embed Block Params | Equal-Token Mix BPB (`mean ± sd`) | Equal-Wall-Clock Mix BPB | Gain vs `dense_tied` (Eq-Tok `$t$`) | Tok/s/chip | LAMBADA (`acc / ppl`) | ARC-E / HellaSwag / PIQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`dense_tied`** | `40,379,904` | `23,602,688` | **`1.0163 ± 0.0006`** | `1.0163` | baseline (matches PR #214 `1.0163`!) | `261,412` | `0.2331 / 173.65` | `0.3977 / 0.2823 / 0.5885` |
| **`dense_untied`** | `57,157,120` | `23,602,688` | **`1.0193 ± 0.0005`** | `1.0198` | **`-0.0030` ($t = -4.611^*$)** | `257,800` | `0.2032 / 243.08` | `0.4081 / 0.2802 / 0.5830` |
| **`dense_untied_vr`** | `57,157,127` | `23,602,695` | **`1.0099 ± 0.0010`** | `1.0122` | **`+0.0063` ($t = +16.116^*$)** | `254,310` | `0.2062 / 218.66` | `0.4125 / 0.2848 / 0.5854` |
| **`dense_tied_57m`** (`d=640`) | `57,355,904` | `36,384,384` | **`0.9916 ± 0.0010`** | `1.0453`† | **`+0.0247` ($t = +31.640^*$)** | `189,619` | **`0.2340 / 155.63`** | **`0.4119 / 0.2869 / 0.5941`** |

*†Methodological note on `dense_tied_57m` equal-wall-clock interpolation:* Because `dense_tied_57m` runs `24,400` steps in `4,216.5s`, at `dense_tied`'s `3,058.5s` finish time it is only `72.5%` through its cosine LR schedule (before terminal cosine annealing). A dedicated shorter-step schedule would be required for an annealed equal-time comparison across widths; the primary width control here is equal-parameter / equal-token (`0.9916` vs `dense_untied` `1.0193`, **`+0.0277` BPB, $t = +63.846$**).

### 2. Falsification of Both Untied-Head Hypotheses (`U_G1` & `U_G2`)
1. **`U_G1_untied_head_gate = REJECT_UNTIED_HEAD_FOR_SUB4B_TRUNK`:**
   - At Chinchilla `20` tok/param (`800M` tokens, `V = 32,768`), untying `lm_head` **significantly hurts** held-out BPB (`-0.0030` BPB, $t = -4.611$) and **severely degrades rare/exact-word prediction** (`LAMBADA ppl` worsens from `173.65` to `243.08`), while `emb` and `lm_head` diverge to near-orthogonality (`mean_cos_emb_vs_lm_head = 0.00235`).
   - Allocating the identical `+17M` parameters to tied-head backbone width (`dense_tied_57m`) beats `dense_untied` by **`+0.0277` BPB ($t = +63.846$)** and breaks `1.00` BPB (`0.9916`). **Keep tied input/output embeddings on the Nano LM trunk** (saving `168M–390M` dead output-table parameters at `d=2560` for active backbone layers/width!).
2. **`U_G2_valueres_interaction = ADDITIVE_OR_INDEPENDENT_OF_HEAD_TYING`:**
   - Falsifies the hypothesis from PR #214 that Layer-7 $v_0$ skip concentration ($\lambda_7 = 0.1230$) was caused by weight-tying `W_out = W_emb^T`: even with a completely untied `lm_head`, Layer 7 still places **`83.5%` nominal weight (`~79%` normalized vector-norm share)** on $v_0$ ($\lambda_7 = 0.1646$; `ablate_late_6_7` `+0.7765` BPB vs `ablate_mid_3_4` `+0.0136` BPB; `perm_v0` `+0.7050` BPB), and `ValueRes`'s marginal gain **increases by `58%`** when untied (`+0.00933` BPB, $t = +11.094$, vs `+0.00590` BPB when tied).
   - Mechanistically, late-layer Layer-0 transport counteracts deep PreNorm residual norm growth (`v_cur_rms` grows `6.4x` from `0.0608` at Layer 1 to `0.3895` at Layer 7) and helps align untied input/output token spaces (`cos(emb, lm_head)` rises `7.6x` from `0.00235` to `0.01797`).

---

## Resource & Teardown Ledger

- **RunPod (`Stage B`):** `7` short-lived Hopper `H100 80GB HBM3` pods across Waves 1–4, all confirmed deleted (`runpodctl pod list == []`, `currentSpendPerHr = 0`, `$59.91` total spend, `$8.93` remaining credit).
- **Google Cloud TPU (`Untied-Head Factorial`):** `1` spot `v5litepod-4` (`nanolm-untied57m-v1` in `us-east1-c`), self-deleted immediately on `EXIT` trap (`gcloud compute tpus tpu-vm list --zone=us-east1-c == []`, `~$5.59` GCP spot credit).
