# Pre-Registered Protocol: Stage B `SmolLM2-135M → Gated DeltaNet` Inheritance Gate (`F2`) & `40.38M / 57.2M` Untied-Head × ValueRes Factorial

**Registered before any execution of `stage_b_runner.py` or `untied_head_jax.py` (`2026-11-26`).**  
**Base lineage:** `origin/master` (`3e36bf8`) · Implements `RESEARCH-PLAN.md §20.5a` (`Stage B`, `F2` gate) and the structural follow-up from Draft PR #214 (`RESULTS.md §4.1`).

---

## Part I — Stage B: `SmolLM2-135M → 22 Gated DeltaNet + 8 GQA` Inheritance & Alignment Gate (`RESEARCH-PLAN §20.5a`)

### 1. Scientific Question & Exact Alignment with `RESEARCH-PLAN §20.5a`

Having verified Hopper `GatedDeltaNet` correctness and utilization in `R0-v2` (`triton==3.7.1`, `184.2 TFLOPS`, `F1 = ADVANCE_STAGE_C_ON_HOPPER`, PR #214) and confirmatory layer placement (`KEEP_ATTN = [1, 2, 3, 4, 21, 22, 23, 24]`, `20/20` random draws beaten by `0.211` nats), **Stage B** tests whether **inheriting attention Q/K/V/O projections into Gated DeltaNet mixers + staged sublayer alignment (`T`)** recovers language-model capability significantly faster per student token than **random mixer initialization under the identical staged alignment and end-to-end distillation pipeline (`C1`)**.

- **Pinned Parent / Teacher (`P`):** `HuggingFaceTB/SmolLM2-135M` @ immutable commit `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (`Apache-2.0`, `L=30, d=576, H=9, KV=3, hd=64, ffn=1536, V=49,152`).
- **Hybrid Student Architecture:** `8` kept GQA layers `[1, 2, 3, 4, 21, 22, 23, 24]` + `22` `fla.layers.GatedDeltaNet(hidden_size=576, num_heads=9, head_dim=64, expand_v=1, mode="chunk")` layers.
- **Decontaminated Pinned Corpus (`SmolLM2` native `49,152` BPE):** Built from the four pinned public `Corpus v2` repositories (`FineWeb-Edu` `87f0914`, `Cosmopedia-v2` `3ba9d60`, `FineMath-4+` `e92b25a`, `GitHub Code` `c48d40f`; `62% / 13% / 13% / 12%` mix):
  - **Held-out validation (`262,144` tokens):** `64 × 1024` sequences per domain from shard `0`, never used in training or placement selection.
  - **Training corpus (`≥ 270M` tokens):** Drawn exclusively from shards `≥ 1` after filtering out any document sharing a 13-word window with the held-out validation sequences (exact SHA256 hashes recorded in output receipts).

### 2. Five Arms & Strict Equal Student-Token Budget (`250,347,520` Tokens per Trained Run)

Every trained arm sees **`7,640` steps × `32,768` tok/step (`bs=32, seq=1024`) = `250,347,520` (`250.35M`) student tokens**:

| Arm | Seeds | Weight Initialization | Stage 1: Per-Layer MSE Sublayer Alignment | Stage 2: End-to-End Distillation (`0.5 CE + 0.5 KL(P ∥ S)`) | Total Student Tokens |
|---|---:|---|---|---|---:|
| **`T` (Treatment)** | `1000, 1001, 1002` | Parent backbone + **inherited Q/K/V/O** into 22 GDN mixers | `640` steps (`20,971,520` tok), `lr=1e-3 → 1e-4` | `7,000` steps (`229,376,000` tok), `lr_gdn=5e-4, lr_base=1e-4` | `250,347,520` |
| **`C1` (Fair Control)** | `1000, 1001, 1002` | Parent backbone + **random init** on 22 GDN mixers | **Identical** `640` steps (`20,971,520` tok), same batches per seed | **Identical** `7,000` steps (`229,376,000` tok), same batches per seed | `250,347,520` |
| **`C2` (No-Align Ablation)** | `1000` | Parent backbone + **random init** on 22 GDN mixers | None (`0` tok) | `7,640` steps (`250,347,520` tok), `lr_gdn=5e-4, lr_base=1e-4` | `250,347,520` |
| **`R` (All-Random Ref)** | `1000` | **All weights random** (backbone + 22 GDN mixers) | None (`0` tok) | `7,640` steps (`250,347,520` tok), `lr=5e-4` across all params | `250,347,520` |
| **`P` (Parent Ceiling)** | — | Unconverted `SmolLM2-135M` (`0` training) | — | — | `0` |
| **`A_all30` (Stage-A Ref)** | `1000` | All `30` layers converted (inherited Q/K/V/O), Stage-1 aligned (`640` steps, `21.0M` tok), `0` e2e steps | `640` steps (`20,971,520` tok) | None (`0` tok) — defines live held-out Stage-A baseline for `B2` | `20,971,520` |

### 3. Registered Stage-B Criteria (`score_stage_b.py` — Unchanged from `RESEARCH-PLAN §20.5a`)

Let $L(M)$ denote held-out language-model cross-entropy loss in **nats/token** on the weighted held-out mix (`0.62 web + 0.13 textbook + 0.13 math + 0.12 code`):
- **`B1` (Inheritance advantage over fair control `C1`):** `PASS` iff $L(T_s) < L(C1_s)$ in **all 3 seeds (`1000, 1001, 1002`)** **and** $\frac{1}{3}\sum_s [L(C1_s) - L(T_s)] \ge 0.10\text{ nats}$ (~`10.5%` lower perplexity).
- **`B2` (Gap closure toward unconverted parent `P`):** `PASS` iff $\frac{1}{3}\sum_s L(T_s) \le L(P) + 0.50 \cdot [L(A_{\text{all30}}) - L(P)]$ on the identical held-out mix (closing $\ge 50\%$ of the gap between the Stage-1-aligned all-converted model and the unconverted parent).
- **`B3` (Secondary 7-task zero-shot retention, reported with seed SEs & paired bootstrap CIs, no pass/fail):** Score `P`, `T` (all 3 seeds), `C1` (all 3 seeds), `C2`, and `R` on `hellaswag, arc_easy, arc_challenge, piqa, winogrande, openbookqa, lambada`, reporting raw accuracy and chance-adjusted retention $\frac{\text{acc} - \text{chance}}{\text{acc}(P) - \text{chance}}$.
- **`F2` Gate Decision (`RESEARCH-PLAN §20.6`):** `ADVANCE_TO_STAGE_C_0_5B` iff **`B1` AND `B2`** both pass; if neither `T` nor `C1` closes `50%` of the gap at `250M` tokens, `EXTEND_ONCE_TO_500M_OR_INCONCLUSIVE`; if `C1` closes `≥ 50%` of the gap while `B1` fails (`mean[L(C1) - L(T)] < 0.10`), `ADOPT_HYBRID_DISTILLATION_WITHOUT_WEIGHT_INHERITANCE` (establishing that backbone inheritance + KD suffices without attention-projection weight copying!).

---

## Part II — `40.38M / 57.2M` Untied-Head × ValueRes × Equal-Parameter Tied Factorial (Cloud TPU `v5e-4`)

### 1. Hypothesis & Experimental Matrix (`untied_head_jax.py`)

In Draft PR #214, both `value_res` ($\lambda_7 = 0.123$) and `resid_x0_skip` ($\alpha_7 = 0.073$) independently routed `88–93%` nominal weight from Layer 0 into **Layer 7** right before the tied `emb.T` output projection. We test whether **untying the output LM head (`W_out` $\ne$ `W_emb`)** resolves this representation bottleneck directly, improves both held-out BPB and 7-task accuracy at near-zero FLOP cost, and renders Layer-7 `ValueRes` largely redundant.

All 4 arms train on `Corpus v2` (`799,539,200` unique tokens = `24,400` steps × `32,768` tok/step, `3` seeds `1000, 1001, 1002` = `9.59B` tokens) using `optax.contrib.muon` (`lr=4e-3, wd=0.1, warmup=500, floor=1%`):

| Arm | Architecture (`L=8, V=32,768`) | Total Params | Non-Embed Block Params | Role |
|---|---|---:|---:|---|
| `dense_tied` | `d=512, ffn=1408, H=8, KV=4`, tied `emb` | `40,379,904` | `23,602,176` | Baseline reference |
| `dense_untied` | `d=512, ffn=1408, H=8, KV=4`, separate `lm_head` `[512, 32768]` | `57,157,120` | `23,602,176` (`0%` extra block FLOPs) | Untied output head candidate |
| `dense_untied_vr` | `dense_untied` + `+7` learned $\lambda_1..\lambda_7$ `ValueRes` skips | `57,157,127` | `23,602,183` | Sub-additivity / $\lambda_7$ interaction probe |
| `dense_tied_57m` | `d=640, ffn=1728, H=10, KV=5`, tied `emb` | `57,355,904` | `36,383,744` (`+54.2%` block FLOPs) | Equal-total-parameter tied width control |

### 2. Pre-Registered Predictions (`U1–U3`) & Mechanical Decision Rules (`U_G1`, `U_G2`)

- **U1 (Untied head equal-wall-clock efficiency):** `dense_untied` beats `dense_tied` by $\ge 0.010$ mix BPB at **both** equal tokens and equal wall-clock ($|t| > 4.303$, `< 5%` step-time cost) and recovers $\ge 45\%$ of `dense_tied_57m`'s equal-token gain while beating `dense_tied_57m` at **equal inference block FLOPs** (and staying competitive or ahead at equal training wall-clock).
- **U2 (Sub-additivity with `ValueRes`):** Once `lm_head` is untied, `ValueRes`'s marginal gain shrinks (`mean[BPB(dense_untied) - BPB(dense_untied_vr)] < 0.0059` BPB) and Layer 7's learned $\lambda_7$ rises above its tied-embedding mean (`0.1230`), confirming that tied-embedding output rotation drove part of the Layer-7 Layer-0 skip demand.
- **`U_G1` (Untied Head Adoption Gate):** `ADOPT_UNTIED_HEAD_FOR_SUB4B_TRUNK` iff `dense_untied` beats `dense_tied` by $\ge 0.008$ mix BPB (`|t| > 4.303`, `all_positive`) at **both** equal tokens and equal wall-clock (note: at `d=2560, V=65,536`, untying `Dense D0` raises total params from `2.936B` to `3.104B`, preserving `+896M` headroom under the `< 4.0B` cap).
- **`U_G2` (Interaction Verdict):** `CONFIRMED_TIED_HEAD_SUBADDITIVITY` iff `mean_s[BPB(dense_untied) - BPB(dense_untied_vr)] < mean_s[BPB(dense_tied) - BPB_PR214(value_res)]` (`0.00590`) **and** `mean_s[lambda_7(dense_untied_vr)] > 0.150`.
