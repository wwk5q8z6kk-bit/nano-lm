# NANOLM Baseline Technical Assessment

**Date:** 2026-11-17  
**Canonical Base:** `origin/master` @ `acc9b7e896c3b8189d29761d8401e6833c222a28`  
**Discovery Branch:** `nanolm/architecture-invention-v1` (isolated worktree)

---

## 1. Purpose of This Baseline Audit

Before inventing and testing fundamentally different computational architectures under `nanolm_discovery/`, this document establishes an exact, code-verified baseline of what exists in `nano-lm` on canonical `origin/master` alongside the unmerged sibling research worktrees.

---

## 2. Canonical Integrated Neural Architecture (`origin/master`)

### 2.1 Original Nano `3.15M` Trunk (`pretrain/`, `nano_ai/training/model.py`)
The integrated original Nano architecture is a standard pre-RMSNorm causal grouped-query Transformer with rotary embeddings and SwiGLU FFN:

| Property | Exact Verified Specification | Source |
|---|---:|---|
| Class / Config | `NanoGPT` / `NANO_MODEL_CONFIG` | [`nano_ai/training/model.py`](nano_ai/training/model.py) |
| Total deployed parameters | **`3,148,608`** (`3.15M`, tied embeddings) | [`nano_ai/training/pointer_model.py`](nano_ai/training/pointer_model.py) |
| Vocabulary size | `4,098` (`4,096` byte-level BPE + specials in `pretrain/tokenizer.json`) | [`pretrain/train.py`](pretrain/train.py) |
| Model width / Layers | `d = 192`, `L = 6` (`ffn = 512`, `~2.67x`) | [`nano_ai/training/model.py`](nano_ai/training/model.py) |
| Attention geometry | `6` query heads, `2` KV heads (`3:1` GQA), `head_dim = 32` | [`nano_ai/training/model.py`](nano_ai/training/model.py) |
| Positional encoding | `RoPE` (`theta = 10,000`), `sequence_length = 512` | [`nano_ai/training/model.py`](nano_ai/training/model.py) |
| Normalization | `nn.RMSNorm` pre-norm (`n1`, `n2`, final `nf`), `eps` default | [`nano_ai/training/model.py`](nano_ai/training/model.py) |
| Output head | Tied linear projection `emb.weight` (`0` extra parameters) | [`nano_ai/training/model.py`](nano_ai/training/model.py) |
| Training recipe | `AdamW(0.9, 0.95)`, `wd=0.1`, `lr=3e-3` cosine to `10%`, `z_loss=1e-4`, `4000` steps (`32.8M` tok on FineWeb sample-10BT) | [`pretrain/train.py`](pretrain/train.py), [`pretrain/train.log`](pretrain/train.log) |
| Recorded pretrain outcome | Loss `8.349 -> 3.702` train / `3.962` val (`20.4 min` on M3 Max MPS `fp32`, `~25–27k tok/s`) | [`pretrain/train.log`](pretrain/train.log) |

### 2.2 Supervised Pointer & Evidence-Query Extensions (`nano_ai/training/`)
Integrated through PRs `#107`–`#109` (`H6` terminal decision):
- `NanoPointerModel` (`3,157,273` params = `3,148,608` trunk + `8,665` pointer heads): attaches five-state field classifiers and token boundary heads on the frozen v0.1 geometry ([`nano_ai/training/pointer_model.py`](nano_ai/training/pointer_model.py)).
- `NanoEvidenceQueryPointerModel` (`H3..H6`, `3,286,469` params = `3,148,608` trunk + `137,861` evidence-query head params): adds global non-causal slot reads over final causal hidden states plus one shared 5-state classifier ([`nano_ai/training/evidence_query_model.py`](nano_ai/training/evidence_query_model.py)).
- **Recorded H6 Empirical Outcome:** Evaluated once on Kaggle (`CPU`, `5,000` fields, SHA-256 `cb612410...`) and **REJECTED** (`3901/5000 = 78.02%` joint). Notably, **held-value copying passed** (`2277/2987 = 76.23% >= 2167`), whereas all three failed gates were epistemic state discrimination: `absence` (`199/413 = 48.18%`), `conflict` (`143/250 = 57.20%`), and `uncertainty` (`190/250 = 76.00%`) ([`artifacts/nano_h6/H6_TERMINAL_DECISION.json`](artifacts/nano_h6/H6_TERMINAL_DECISION.json)).

### 2.3 Scale Test Stage S (`~10M`, `scale/AUDIT.md`)
- Scaled standard GQA+SwiGLU+RoPE to `~10M` (`d=320, L=8, H=8, KV=2, hd=40, ff=864`), trained on `200M` FineWeb tokens on Kaggle T4 (`val loss = 3.284`) plus scribe finetuning (`12,000` examples).
- Passed average-case scribe faithfulness gates (`parse 100%`, `recall 88%`, `halluc 7.5%`), yet retained the **exact same `23-pt` OOD held-out-value gap** (`77%` held-out vs `100%` seen; `3M` baseline gap was `22 pts`) — establishing that naive parameter/data scaling of a standard GQA Transformer did not close the OOD copy/epistemic calibration gap ([`scale/AUDIT.md`](scale/AUDIT.md)).

### 2.4 Symbolic Cognitive Substrate (`nano/`) & Controlled Baseline (`B0`)
- [`nano/architecture.py`](nano/architecture.py) and [`nano/kernel.py`](nano/kernel.py) separate the deterministic 14-layer cognitive substrate (`Layer I..XIV` across ontology, identity isolation, provenance, temporal validity, dependency invalidation, and verification) from neural mechanisms (`Spec 3`) and learning (`Spec 4`).
- In `nanolm_discovery/`, our formal **`BASELINE` (`B0_TRANSFORMER`)** generalizes `NanoGPT` cleanly across all iso-parameter experimental tiers (`115k`, `8.65M`, `20.55M`, `47.64M`, and `3,887,254,272` parameters `< 4.0B`) with Pre-RMSNorm, causal RoPE (`theta=500,000`), Grouped-Query Attention, SwiGLU, tied embeddings, and precision-safe `float32` RMS variance accumulation.

---

## 3. Custom vs. Conventional Classification & Known Root Bottlenecks

| Component | Status | Classification | Root Limitation |
|---|---|---|---|
| `NanoGPT` / `B0_Transformer` | Integrated | **100% Conventional Transformer** (GQA + RoPE + SwiGLU + tied head) | Quadratic/linear-growing KV cache (`O(T)` memory/tok during decode), weak non-commutative finite-group state tracking at finite depth ($\text{TC}^0$ limit), zero native epistemic variance signal (`uncertainty` only post-hoc softmax entropy). |
| `NanoEvidenceQueryPointerModel` | Integrated (`H6` Rejected) | Domain-specific multi-slot span/state read head | Single static forward representation conflates high-confidence direct extraction with absent/conflicting/uncertain epistemic states. |
| `nano/` cognitive substrate | Integrated | **Custom Deterministic Substrate** | Strong symbolic contracts (`Identity`, `Entity`,provenance), previously disconnected from a compatible differentiable neural state/belief representation. |
| Cross-branch `nanolm`, `nanolm-v3`, `nanolm-lab`, `nanolm-holonomy`, `nanolm-paradigm` | Local unmerged research worktrees | Concurrent sibling explorations (VRes/MLA, Householder-Delta, SU(2) spinors, ERN) | Several earlier runs suffered from single-seed comparisons, parameter-count confounds, or slow unchunked Python/MPS scans; all remain isolated from `origin/master`. |

---

## 4. Datasets & Local Execution Environment

- **Real multi-domain validation & pretraining corpus (`nanolm.lab.corpus.v1`):** Cached read-only (`vocab = 32,768`, `train_sha256 = 867beb0d...`, `tokenizer_sha256 = 2eb9b402...`) containing `800M` decontaminated `uint16` training tokens (`web` 62%, `textbook` 13%, `math` 13%, `code` 12%) plus four disjoint `1.0M`-token held-out validation shards (`val_web.bin`, `val_textbook.bin`, `val_math.bin`, `val_code.bin`) with verified exact manifest `bytes_per_token` (`4.5163`, `5.0293`, `3.0994`, `3.3728`).
- **Local compute host:** Apple M3 Max (`16` CPU cores, `64 GB` unified memory, `torch 2.11.0`, `mps` backend available, `$0.00` paid compute budget).
- **Kaggle free CUDA backend (`EXP-INV-06`):** `2x NVIDIA Tesla T4` (`CUDA 13.0`, `torch 2.11.0+cu130`, `14.56 GiB` VRAM per GPU, `$0.00` paid compute budget) used for `4x`-horizon training (`1,000` steps) and discrete VRAM long-context profiling.
- **CI self-containment rule:** All unit/invariant tests in `nanolm_discovery/tests/test_discovery.py` execute strictly in-memory with synthetic tensors and never touch external storage volumes.
