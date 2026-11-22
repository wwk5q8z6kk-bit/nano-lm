# Gate 1 Bottleneck Decomposition & Depth-Routing / Coda-Refinement Tournament (`H-13` & `H-10`)

**Registered before any training execution of `kaggle_tournament.py` (`2026-11-22`).**  
**Base lineage:** `origin/master` (`3e36bf8`) · Corpus `hassaneljesr/nanolm-corpus-v1-800m` (`32,768` BPE, `FineWeb-Edu 62%`, `Cosmopedia-v2 13%`, `FineMath-4+ 13%`, `GitHub Code Clean 12%`).

---

## 1. Empirical Premise (Measured in Gate 1 Before This Protocol)

Prior to registering this tournament, we executed three foundational audits preserved in `results/`:

1. **Cross-Stack `40.38M` Dense Baseline Parity (`results/kaggle_dense_40m_baseline.json`):**  
   Completed PyTorch FP16 training on Tesla T4 (`799,997,952` tokens, `8.361 h`, `\$0` free tier) reached **`1.0506` mix BPB** (`web 1.1095, textbook 0.7936, math 1.1859, code 0.8787`), matching JAX BF16 on Cloud TPU v5e (`1.0466`) and TPU v6e (`1.0478` 3-seed mean) within `0.0028` BPB, with `-0.0643` BPB improvement over the final doubling of tokens (`393M → 800M`).
2. **Gate 1 Held-Out Token-Level Bottleneck Decomposition (`results/gate1_bottleneck_results.json`):**  
   Across `655,360` held-out tokens (`20 × 16 × 512` crops per domain, `np.random.default_rng(0)`) on authenticated `14.29M` (`S`), `40.38M` (`M`), and `100.68M` (`L`) dense checkpoints:
   - **In-window (`512`-token) explicit copying is saturated:** `cache-exchange-L` (`100.68M`) yields only **`+0.0012` mix BPB** (`+0.0043` code; `0.0000` foreign-window control), completing `14M (+0.0036) → 40M (+0.0018) → 101M (+0.0012)`. Non-copyable tokens carry **`93.89%`** of `101M` mix loss (`88.38%` on code) and **`91.04%`** of the `14M → 101M` scaling gain.
   - **Scale-invariant `Ranks 2..8` near-miss frontier:** While deep parametric misses (`Rank ≥ 65`) fall sharply (`0.4894 → 0.4093 → 0.3638` BPB, `79.3%` of scaling gain), **`Ranks 2..8` remains invariant at `0.2593 → 0.2596 → 0.2579` BPB** (`26.2%` of total `101M` loss; `y_t ∈ Top-64` on `87.54%` of tokens).
3. **Sub-4B Exact Parameter & DRAM Roofline (`results/sub4b_roofline_audit.json`):**  
   Qwen-style Gated DeltaNet mixers at `d=2560` contain `52.65M` parameters/block (`1.264B` across 24 blocks), making `32`-layer `Dense D0` (`2.936B`) `945M` parameters smaller and lower in `batch=1` DRAM traffic (`8.02` vs `8.46 GB/tok` at `32k` BF16) up to `~44k` context. By contrast, **Block Attention Residuals (`H-13`, `arXiv:2603.15031`)** adds `< 0.06%` parameters and `< 0.01%` DRAM traffic, while **Coda Latent Refinement (`H-10`, inspired by `LOTUS`)** tests whether shared top-layer refinement reduces `Ranks 2..64` ambiguity without sacrificing unique lower-layer FFN capacity.

---

## 2. Experimental Design (`kaggle_tournament.py`)

Every arm is trained with **identical hyperparameters and no per-arm tuning**, using the Question 11 champion optimizer recipe (`OptaxParityMuon`, verified to `< 1e-5` numerical agreement with `optax.contrib.muon` in `test_contracts.py`):
- **Optimizer:** `OptaxParityMuon` (`ns_steps=5`, `consistent_rms=0.2`, `beta=0.95`, bias-corrected Nesterov on 2D block weights; bias-corrected Nesterov AdamW `b1=0.9, b2=0.95, eps=1e-8` on embeddings, 1D norms/scalars, and 1D depth-query vectors), `lr = 4e-3`, `wd = 0.1`, global grad L2 clip `1.0`.
- **Schedule & exposure:** `bs = 64`, `seq = 512` (`32,768` tok/step), `warmup = 300` steps, cosine decay to `lr_floor = 0.01` (`1%`), **`3,000` steps = `98,304,000` tokens/arm** (`6.88` tok/param; `3×` the earlier `33M` store pilot).
- **Seeds & hardware/thermal control:** **3 independent seeds (`1000, 1001, 1002`)** (`15` full training runs = `1.475B` total trained tokens). Within each seed, **all 5 arms execute sequentially on the same physical GPU** after a 15-second steady-state GPU burn-in, in a **Latin-square cyclic rotation across seeds** so neither GPU-model differences nor within-session thermal drift bias any arm:
  - Seed `1000`: `dense → value_res → block_attnres → coda_refine → dense10`
  - Seed `1001`: `block_attnres → coda_refine → dense10 → dense → value_res`
  - Seed `1002`: `coda_refine → dense10 → dense → value_res → block_attnres`
  Each arm resets `torch.manual_seed(seed)` and `np.random.default_rng(seed + 1)` before initialization so shared initial weights and training crop sequences are identical across arms within each seed.
- **Evaluation:** Held-out `val_{web,textbook,math,code}.bin` (`np.random.default_rng(0)`, `20 × 16 × 512 = 655,360` tokens) every `500` steps (`500, 1000, 1500, 2000, 2500, 3000`), with evaluation wall time excluded from `train_sec`, plus full rank-bucket (`Rank 1`, `Ranks 2..8`, `Ranks 9..64`, `Rank ≥ 65`) decomposition at step `3,000`.

### Competing Arms

| Arm | Architecture (`d=256, heads=4, kv=2, ffn=704, V=32,768`) | Exact Params | Layer Passes | Role |
|---|---|---:|---:|---|
| `dense` | `8`-layer PreNorm GQA Transformer (tied embeddings) | `14,292,224` | `8` | Champion reference (`D0 + Muon`) |
| `value_res` | `8`-layer PreNorm GQA + per-layer Layer-0 value skip $\lambda_\ell$ (`layers 1..7`) | `14,292,231` (`+7`) | `8` | Simpler cross-depth baseline |
| `block_attnres` | `8`-layer GQA + **Block Attention Residuals** ($B=4$ blocks × `2` layers + final router, zero-init $q_{\text{depth}} \in \mathbb{R}^{256}$) | `14,300,928` (`+8,704`) | `8` | **H-13** learned depth routing |
| `coda_refine` | `8`-layer unique GQA + zero-init channel-gated `1`-step latent pass of `blocks[6..7]` | `14,292,480` (`+256`) | `10` | **H-10** shared coda refinement |
| `dense10` | `10`-layer untied PreNorm GQA Transformer | `15,768,064` (`+1.48M`) | `10` | Equal-FLOP untied control for `coda_refine` |

---

## 3. Pre-Registered Predictions & Decision Rules (`score_tournament.py`)

### Predictions
- **P1 (Baseline repeatability):** `dense` 3-seed sample SD on `mix` is `< 0.004` BPB.
- **P2 (`H-13` Block AttnRes):** `block_attnres` beats `dense` on 3-seed mean `mix` by `0.004–0.015` BPB at equal tokens, remains positive at equal wall-clock (`< 5%` step-time overhead), and beats `value_res` by `≥ 0.002` BPB.
- **P3 (`Value Residual` control):** `value_res` (`+7` scalars) improves over `dense` by `0.001–0.006` BPB with `< 1%` overhead.
- **P4 (`H-10` Coda Refinement):** `coda_refine` beats `dense` (`8L`) at equal tokens across all 3 seeds and recovers `≥ 40%` of `dense10`'s equal-token gain over `dense` (`8L`), while losing part or all of its lead at equal wall-clock due to `10` vs `8` layer passes (`+25%` compute).
- **P5 (Rank-bucket signature):** Any winning candidate reduces `Ranks 2..64` BPB in addition to `Rank ≥ 65` BPB.

### Mechanical Decision Rules (Fixed Before Any Run)
Let $\bar{\Delta}(A, B) = \text{mean}_s[\text{BPB}_s(B) - \text{BPB}_s(A)]$ (positive = $A$ beats $B$), with two-sided $95\%$ paired $t$-threshold $|t| > 4.303$ (`df=2`). Equal-wall-clock BPB linearly interpolates each arm's `mix` curve at `dense`'s final `train_sec` within the same seed.
- **`D1` (Depth routing):**
  - `ADOPT_BLOCK_ATTNRES` iff $\bar{\Delta}(\text{block\_attnres}, \text{dense}) \ge 0.004$ at **both** equal tokens and equal wall-clock, $|t| > 4.303$, **and** $\bar{\Delta}(\text{block\_attnres}, \text{value\_res}) \ge 0.002$.
  - `ADOPT_VALUE_RES_REJECT_BLOCK_ATTNRES` iff `block_attnres` fails the `0.002` margin over `value_res`, while $\bar{\Delta}(\text{value\_res}, \text{dense}) \ge 0.004$ at both equal tokens and equal wall-clock (`|t| > 4.303`).
  - `CLOSE_DEPTH_ROUTING_AT_14M` otherwise.
- **`D2` (Coda latent refinement):**
  - If $\bar{\Delta}(\text{coda\_refine}, \text{dense}) \ge 0.004$ at equal tokens (`|t| > 4.303`) **and** recovers $\ge 50\%$ of $\bar{\Delta}(\text{dense10}, \text{dense})$:
    - `ADOPT_UNCONDITIONAL_CODA_REFINE` if equal-wall-clock $\bar{\Delta}(\text{coda\_refine}, \text{dense}) > 0$;
    - `ADVANCE_CODA_TO_SPARSE_GATED_40M` if equal-wall-clock $\bar{\Delta} \le 0$ (restricting follow-up strictly to sparse high-entropy token gating).
  - `REJECT_UNCONDITIONAL_CODA_REFINE` otherwise.
