# EXP-U02 Part B Protocol — Real Subword Corpus Cross-Paradigm Transfer

- **Experiment ID:** `EXP-U02-Part-B`
- **Status:** Preregistered BEFORE executing `scripts/run_u02_part_b_real_corpus.py`
- **Hardware / Cost:** Local Apple Silicon M3 Max (`mps` + `cpu`), `$0.00` paid compute
- **Corpus:** `/Volumes/Express1M2/nanolm-lab/data/v1/` (`vocab = 32,768`, `train.bin` = 800M domain-mixed tokens + held-out `val_web.bin`, `val_textbook.bin`, `val_math.bin`, `val_code.bin`)
- **Bytes per Token (`manifest.json`):** `web = 4.5163`, `textbook = 5.0293`, `math = 3.0994`, `code = 3.3728`

## 1. Contender Selection Rule (Strictly from Part A Results)
1. **Mandatory Controls:**
   - `M0_Dense_GQA_RoPE` (Standard RoPE Causal Attention + SwiGLU baseline)
   - `M1_GDN1_Cone_Closed` (Legacy `S1_125m` DeltaNet control audited in `EXP-U01`)
2. **Top `3` Distinct Part A Architectural Winners:**
   - `M9_DualChannel_Pointer` (`FullSphere + SFOR + Causal Copy Pointer`: `#1` on `Unseen_OpenVocab_Copy = 0.962` and `MQAR_Selective_Overwrite_4x = 0.576`)
   - `M5_HSPA_LieSO3` (`SU(2) -> SO(3)` quaternion spinor transport: `#1` on `Hierarchy_Denom_Rollup_4x = 0.259`, `close = 0.163`, `#1` ID on `S4 = 0.103`)
   - `M7_SkewCayley_Flow` (Continuous rank-1 skew + dissipative Woodbury Cayley ODE: `#1` on `Parity_Dyck_4x = 0.928`, `#2` on `Hierarchy_Denom_Rollup_4x = 0.256`, `close = 0.153`)

## 2. Scale & Iso-Parameter Parity (`d_model = 128`, `n_heads = 4`, `n_layers = 4`)
Matched on **non-embedding backbone trainable parameters** (`~853.1K ± 0.15%`, ~`5.05M` total incl. `4.19M` tied token embeddings):
- `M0_Dense_GQA_RoPE`: `ffn_hidden = 384` (`853,121` non-emb) — trained on `mps`
- `M1_GDN1_Cone_Closed`: `ffn_hidden = 334` (`852,241` non-emb, `-0.10%`) — trained on `mps` (exact `C=32` chunked WY scan)
- `M5_HSPA_LieSO3`: `ffn_hidden = 496` (`853,313` non-emb, `+0.02%`) — trained on `cpu` (`threads=1` worker per seed)
- `M7_SkewCayley_Flow`: `ffn_hidden = 254` (`854,273` non-emb, `+0.14%`) — trained on `cpu` (`threads=1` worker per seed)
- `M9_DualChannel_Pointer`: `ffn_hidden = 312` (`851,858` non-emb, `-0.15%`) — trained on `mps` (`C=32` chunked WY scan)

## 3. Training & Evaluation Protocol (`n = 2` Seeds: `[0, 1]`)
- **Training:** `1,000` steps/seed, `B = 8`, `T = 128` (`1,024` tokens/step, `1.024M` training tokens/seed), AdamW (`lr = 1.5e-3`, `weight_decay = 0.01`, `100`-step linear warmup + cosine decay, grad clip `1.0`).
- **Evaluation:**
  1. Validation BPB (`loss_nats / (ln(2) * bytes_per_token)`) on all `4` held-out domains (`web`, `textbook`, `math`, `code`) at **`1×` (`T = 128`)** and **`2×` (`T = 256`)** context (`24` evenly spaced windows per split).
  2. Zero-shot exact sequence copy probe (`L_half = 32, 64, 128` across `L = 64, 128, 256`).
  3. Zero-shot Real-Haystack MQAR inside `val_web` distractors (`(T=128, N=4)` and `(T=256, N=4)`) across both `common_ids (200..2000)` and `rare_tail_ids (20000..32700)`.
  4. Per-layer key effective rank (`exp(entropy(sigma^2))` out of `hd=32`) & mean off-diagonal key cosine on `M0, M1, M7, M9`.
