# NanoLM Phase 1 Technical Audit — Ground Truth

## 1. Repository, Worktree, and Lineage State

Verified directly against Git refs, filesystem state, checkpoints, and source code:

- **Canonical remote mainline**: `origin/master` at `acc9b7e` (`Harden native Qwen loading and assess research execution isolation (#118)`).
- **Pre-existing main checkout**: The primary working tree was on feature branch `codex/p1-h7-scribing-path` (`2d613d7`) with 9 modified tracked files and 2 untracked files in `nanoscribe/`, `scripts/`, and `artifacts/info_density_uma_v1/`.
- **Clean isolation action**: All uncommitted work on `codex/p1-h7-scribing-path` and all historical freeze tags (`paper-alpha-v1`, `post-alpha-evidence-freeze-2026-07-31`) were preserved untouched. Independent NanoLM foundation development is isolated on branch `nanolm/foundation-v1` rooted cleanly at `origin/master`.

## 2. Verified Inventory of Pre-Existing Models, Tokenizers, Data, and Checkpoints

**Finding: Zero 2B–4B NanoLM foundation models existed in the repository prior to this initiative.**

| Path / Artifact | Actual Verified Spec | Scale | Technical Assessment |
|---|---|---|---|
| `pretrain/train.py`, `sft/model_nano.py`, `nano_ai/training/model.py` | Hardcoded 6-layer GQA (6Q:2KV) + SwiGLU + RoPE + RMSNorm, `d=192`, `S=512`, `V=4096/4098`, no KV cache | **3.15M** | Toy mechanism prototype; obsolete for 2B–4B training/inference |
| `nano_ai/training/evidence_query_model.py` | 3.15M frozen trunk + 5-slot clinical state/span pointer head | **3.29M** | Domain-specific Scribe extraction head (H6 terminal/rejected); not reusable as general LM |
| `scale/kaggle_scale_test.py` | Hardcoded 8-layer GQA (`d=320, S=512, V=4098`) | **10.0M** | Historical Scribe scaling rung |
| `artifacts/native_checkpoints_bpe/` (on `/Volumes/Express1M2`) | 16-layer decoder (`d=384, H=6, S=512`) Scribe screening runs | **30.0M** | Domain-specific synthetic Scribe screen |
| `trajectory/kaggle_ownstack_160m.py` & `checkpoints/chinchilla-160m/` | 14-layer GQA (`d=1024, H=16, KV=4, ff=2752, S=512, V=4098`) | **159.0M** | Largest historical from-scratch rung; severely bottlenecked by `V=4098` and `S=512` |
| `pretrain/tokenizer.json`, `sft/tokenizer.json` | 4,096 / 4,098 vocab Byte-Level BPE trained on 12k FineWeb docs | `V=4,098` | Severe sequence inflation on code (`2.05 B/tok`), zero reserved special-token headroom |
| `.local-data/nano-training-data/` | Local cached `fineweb-edu` and `tinystories` uint16 shards (1M train + 100k val tokens each) | ~2.2M tok | Clean local educational prose suitable for offline smoke/ablation validation |
| Local HF Cache (`~/.cache/huggingface/hub`) | `Qwen/Qwen2.5-1.5B-Instruct` (1.54B) and `Llama-3.2-3B-Instruct-4bit` (3.21B) | 1.54B–3.21B | Useful external reference baselines only; not NanoLM |

## 3. Component Disposition Matrix

| Category | Components | Disposition & Rationale |
|---|---|---|
| **Valuable & Reusable Principles** | Z-loss (`1e-4`), depth-scaled residual init (`0.02/sqrt(2L)`), SHA-256 artifact manifests (`benchmarks/`), RLVR verifiable-reward gating (`sft/grpo.py`), content-addressed copy curricula (`stage_m/`) | Carried forward as clean first-principles engineering practices inside `nanolm/` |
| **Worth Rebuilding Cleanly** | Model architecture, KV cache, BPE tokenizer, intra-document sequence packing, mixed-precision WSD trainer, programmatic capability benchmarks | Built cleanly from scratch in `nanolm/` without legacy globals or `S=512` limits |
| **Domain-Specific (Historical Only)** | `nanoscribe/`, `nano_ai/`, `scribe/`, `wedge_v1/`, `nano/` clinical schemas and 5-slot state/span contracts | Decoupled completely from NanoLM; retained strictly as historical repository record |
| **Protected Historical Evidence** | `papers/`, `trajectory/results_*`, `POST_ALPHA_EVIDENCE_FREEZE*` | Read-only historical archive; never mutated |

## 4. Available Compute & Storage Verified

- **Local Hardware**: Apple M3 Max, 64 GB unified memory, PyTorch MPS (`bfloat16` & `float16` verified), `102 GiB` internal free disk + `1.3 TiB` free on mounted `/Volumes/Express1M2`. Capable of live instantiation and `bfloat16` inference profiling of full 2.0B–3.9B NanoLM models, plus rapid 25M–125M architecture/data ablations.
- **Cloud Compute**: Kaggle CLI (`hassaneljesr`) and `runpodctl` configured (`0` active pods verified — zero idle spend). Reserved for multi-billion-token pretraining and large-scale distillation after local preflight gates pass.
