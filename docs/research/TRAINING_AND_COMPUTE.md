# Training Ladder, Data Provenance & Compute Governance

---

## 1. Data Provenance, Deduplication & Safety Gates

1. **Source & License Registry (`DT01`):** Every pretraining, distillation, and benchmark source must carry content SHA-256, origin URL, license class (distinguishing permissive Apache/MIT code from non-commercial or restricted weights/datasets), cleaning transform, and split ID.
2. **Patient / Clinical Data Boundary:** Zero PHI, zero private owner clinical cases, and zero unauthorized uploads on local or cloud storage (`ACTIVE_NOW.md` standing rule). All current P1 cohorts are synthetic and rights-cleared; external clinical validation remains gated on formal ethics/permission approval and dual-clinician IAA.
3. **Diagnostic 4+ Domain Mixture:** Evaluate web, textbook/scientific, code, mathematics, procedural world-state trajectories, and grounded evidence tasks separately. A scalar 4-domain BPB average must never conceal domain regressions (as seen in `EXP-4`).

---

## 2. Controlled Training Ladder (`L0`–`L5`)

| Rung | Scale | Purpose | Cost & Authorization Rule |
|---|---|---|---|
| **L0** | `3M–30M` | Mathematical/mechanism probes (`Z_2`, `Z_4`, `S_3`, multi-key erase/write, multi-needle, dual-memory parity) | `$0` local CPU / MPS (`ALLOWED`) |
| **L1** | `125M–350M` | Shared-corpus iso-FLOP & optimizer factorial (`1B–5B` tokens as budgeted, `>=3` seeds where feasible) | Free Kaggle or active RunPod budget |
| **L2** | `~800M` | Conditional scaling bridge for `L1` Pareto winners only | Experiment-scoped budget check |
| **L3** | `~2B` | Conditional flagship pretrain; e.g. $6 N D \approx 4.8 \times 10^{20}$ FLOPs at $D=40\text{B}$ tok, $1.2 \times 10^{21}$ at $D=100\text{B}$ tok; horizon set by empirical scaling curves (`Sardana et al.`), not a rigid `20x` rule | **Hard-gated by `G0`–`G6` + explicit owner authorization (`G7`)** |
| **L4** | `2B–<4B` | Grounded SFT, verified Top-$K$+Tail logit KD, executable RLVR (`GRPO`), forgetting audits across all 15 pillars | Experiment-scoped authorization |
| **L5** | System | Governed dual memory, permissioned continual updates, causal interactive worlds, non-medical DomainPacks | Separate `Track 1` vs `Track 2` validation |

---

## 3. Mandatory Checkpoint Card Schema

Every saved checkpoint beyond `L0` smoke tests must record:
1. Total deployed parameter census + active/token breakdown (including embeddings/unembedding/adapters/fast weights);
2. Model/optimizer/RNG state digests, git SHA, worktree cleanliness, PyTorch/CUDA/MLX versions;
3. Immutable dataset manifest SHA, contamination scan receipt, tokens per domain;
4. Training FLOPs, wall-clock seconds, watt-hours/GPU-hours, realized `$`, peak VRAM/UMA;
5. Multi-seed 15-pillar + failure-strata + negative-control metrics, coverage, calibration, and teacher provenance (if distilled).

---

## 4. Compute Backend Discipline

- **Local M3 Max (64 GB UMA):** Rapid code iteration, `L0` mechanism probes, MLX/MPS kernel profiling, quantization (`q4`/`q8`), and `nanoscribe` verification (`$0`). Never extrapolate `L0`/`ctx=512` MPS throughput to `2B` full pretraining economics; watch for `BFloat16`/`float32` `RMSNorm` fallback warnings on MPS (`gen3.py`).
- **Kaggle (Free GPU/CPU):** Bounded reproducible experiments fitting runtime/memory quotas; always enforce `X01` evaluator negative controls (`H7` warning: clean job completion $\neq$ valid evaluator).
- **RunPod:** Exact GPU/runtime (`4090`/`5090`/`L40S`/`A40`/`H100`/`H200`), persistent storage, or managed inference within the standing `$180` autonomous spend envelope (`$200` hard cap, `$20` reserve untouched). All active background jobs on sibling worktrees (`.worktrees/nanolm-lab`, `.worktrees/nanolm-posttrain`) must be audited in spend ledgers and torn down promptly when complete.
