# Experiment Register & Forensic Reconciliation (`X00`–`X20`)

This is part of the [November 17 research proposal](FOUNDATION_INTELLIGENCE_PROGRAM.md). Its size limits, experiments and reported scores belong to that proposal. The [current charter](../PROJECT_CHARTER.md) defines NanoLM’s scope.

This ledger enforces strict provenance separation between **VERIFIED** (in accessible commits/artifacts with passing integrity checks), **REPORTED / CONFOUNDED**, **INVALID**, **CROSS_BRANCH** (on `origin/codex/p1-h7-scribing-path` or `origin/nanolm/posttrain-v1`), and **LOCAL_WORKTREE_UNMERGED** (only in `.worktrees/*` locally).

---

## 1. Forensic Inventory (`X00` — Reconciled 2026-11-17)

| Component | Branch / Worktree / SHA | Forensic Status | Verified Findings & Confounds |
|---|---|---|---|
| **Canonical Trunk** | `origin/master` @ `acc9b7e` (PR #118) | **VERIFIED** | Contains `nano/` (Core), `nano_ai/` (H1–H6), `nanoscribe/` v0, `fabric/`, `wedge_v1/`, `benchmarks/`. `docs/ACTIVE_NOW.md` on `master` is dated `2026-10-12`. |
| **H6 Clinical Extraction** | `origin/master`: `artifacts/nano_h6/H6_TERMINAL_DECISION.json` | **VERIFIED / REJECTED** | `3901/5000` joint (`78.02%`), held-value `2277/2987` (`76.23%` passed), absent `199/413` (`48.18%` failed), conflict `143/250` (`57.2%` failed), uncertain `190/250` (`76.0%` failed). Binding errors `858` (714 supported). Terminal; never rerun. |
| **H7 Boundary Screen** | `CROSS_BRANCH` `origin/codex/p1-h7-scribing-path` @ `f82d82a` (PR #117) | **VERIFIED / INVALID** | `h7_screen_evaluation.json` (`SHA-256 9d1b8035…`) ran on Kaggle (`$0`, `4017.5s`) and recorded `REJECT`, but `evaluate_evidence_query_h7.py:684` called `_fixture_cases`, replacing 4,750 non-missing gold targets with `5,000` `MISSING` targets (`INVALID_GOLD_ANNOTATIONS`). Mechanism effect unestablished. |
| **C12–C14 Fresh Cohorts** | `CROSS_BRANCH` `origin/codex/p1-h7-scribing-path` @ `f82d82a` | **VERIFIED / FAILED TRANSFER** | `p1_retention_validation_20261026` (`24` fresh cases, `120` slots): `Qwen2.5-1.5B` abstained `24/24` free and `23/24` constrained. Under `context` policy, `13/13` locally recovered exact-correct free assertions were lost in checking; constrained menus covered only `4/120` slots (`0/80` exact gold); `19` NL denials failed legacy transport. Opt-in `nanoscribe/grounded.py` (`evidence-grounded-v1`) recovers `12/13` lost free assertions into extraction (`X02` localizes `retain-08` cross-field duration leak). |
| **C15 UMA / Info-Density** | `CROSS_BRANCH` `origin/codex/p1-h7-scribing-path` @ `f82d82a` | **VERIFIED (RUNTIME)** | M3 Max FP16 live benchmark on 20 pilot dialogues: `20/20` slot parity, `450 -> 125` forward passes, `2.08x` speedup (`43.55s -> 20.98s`), `72.22%` DRAM read drop (`$0`). Native H6 MI density `0.1297` vs Qwen pilot `0.000464` bits/field/MiB. Runtime optimization, not cognitive validation. |
| **NanoLM Track A (36 runs)** | `LOCAL_WORKTREE_UNMERGED` `.worktrees/nanolm-lab` (`nanolm/discovery-lab-v1` @ `68291f0`) + `/Volumes/Express1M2/nanolm-lab/runs/` | **REPORTED / PARTIALLY CONFOUNDED** | `46.2M` `n=2` `600M`-token grid on `LAB-DATA-01`: `X4_nanolm_v2_muon` (`0.9575±0.0008` BPB) beats `T0_dense/adamw` (`1.0296±0.0048`) by `-7.01%`, but **`-6.33%` is `Muon`** (`T0_dense/muon` = `0.9645±0.0004`); hybrid architecture gain over `T0+Muon` is **`-0.73%` (`-0.0070` BPB)** at `-70%`..`-80%` KV & `-42.7%` FLOPs. `121M` (`n=1`) tied dense (`0.9106` vs `0.9105`) at `-75%` KV. Active Phase 2 RunPod scripts running in that worktree. |
| **NanoLM-v3 Gen-3 (`EXP-1..5`, 2B profiles)** | `LOCAL_WORKTREE_UNMERGED` `.worktrees/nanolm-v3` (`nanolm/arch-evolution-v3` @ `86371cb`) | **LOCALLY TESTED / CONFOUNDED** | `10/10` unit & WY/recurrent parity tests pass locally (MPS emits `BFloat16`/`float` `RMSNorm` mismatch warning). **Critical confounds:** `EXP-3` (`2.37813->2.35192` BPB) ran only `220` steps (`1.80M` tok, `n=1`, `0%` copy acc) and gave `G0..G5` **`+14.1%` more non-embedding params** (`6.73M` vs `5.90M`) than `Dense_GQA`; `EXP-2` (`3.12%->11.33%`) was a `79.5k`-param toy net (`3.12%` $\approx$ random `1/32`); `EXP-4` traded web/textbook regressions for math/code; `2.07B`/`3.93B` are **untrained** MPS shape profiles (`25.17` vs `48.36 tok/s` @ `ctx=512`; `22.58` vs `19.21 tok/s` @ `ctx=4096`); `suite_15pillar.py` (`42` items) is a smoke suite, not comprehensive eval. |
| **Track B Post-Training (`B1`–`B3`)** | `LOCAL_WORKTREE_UNMERGED` `.worktrees/nanolm-lab` (`68291f0`) + `TRACK_B_LEADERBOARDS_AND_ABLATIONS.md` | **REPORTED (EXTERNAL BASE)** | Trained **`Qwen/Qwen3-1.7B`** (`1.72B` tied / `2.03B` untied), NOT a from-scratch Nano-v3 model. Unmodified `Qwen3-1.7B` scored `74.98%` GSM8K / `70.0%` MATH-500 / `66.46%` HumanEval; `B1` SFT dropped to `55.57%` / `41.0%` / `50.61%`; `B3` KD-8B recovered to `73.16%` / `62.6%` / `63.41%` while improving custom 8-pillar coding (`56.3->68.8%`) and reliability (`76.0->90.1%`). |
| **NanoLM Posttrain-v1** | `CROSS_BRANCH` `origin/nanolm/posttrain-v1` @ `ee3df6b` (PR #171) | **VERIFIED (SMALL PILOT)** | `nanolm_posttrain/`: `<4B` looped GQA spec + `Qwen2.5-1.5B` 6-epoch LoRA (`$0` MPS) lifting `6/12 -> 8/12` on 12 unseen synthetic tasks. Active background RunPod script running in `.worktrees/nanolm-posttrain`. |

---

## 2. Ordered Research Experiment Queue (`X00`–`X20`)

| ID / Stage | Experiment & Controlled Treatment | Controls | Decision Criterion |
|---|---|---|---|
| **`X00` / now** | Forensic reconciliation of local worktrees, SHAs, raw logs, and confounds | `origin/master` & frozen hashes | Complete provenance ledger; zero unverified claims |
| **`X01` / now** | Evaluator integrity & negative controls (`annotated_evaluation.py`) | Oracle, all-missing, shuffled, wrong-patient, offset-corrupt, abstain | All 6 controls pass before any new eval run |
| **`X02` / now** | `C14` source-conservation trace & cross-field attribution fix (`nanoscribe/grounded.py`) | `context` vs `grounded` policy + 7 mandatory probes | Zero silent assertion drops; `retain-08` cross-field duration leak fixed |
| **`X03` / cheap** | Pure geometric state tracking ($Z_2$, $Z_4$, $S_3$, `1x`/`4x`/`16x` OOD) + full Jacobian SVD | Contraction $\beta\in(0,1)$, Householder $\beta\in(0,2)$, `GDN-2`, `ComplexRotoSSM` ($SO(2)$), `GeoEraseWrite` | Distinguish transition norm $\|A_t\|_2$ from nonlinear Jacobian & group expressivity (`3` seeds) |
| **`X04` / cheap** | Joint erase/reflect/write stability under multi-key edits & selective deletion | `ContractionDelta`, `HouseholderDelta`, `GDN-2`, `ComplexRotoSSM`, `GeoEraseWrite` | Matched-param multi-key edit/retention accuracy (`3` seeds) |
| **`X05` / cheap** | Multi-needle distractor & contradiction retrieval (`1/4/8/16` needles, `T=64..384`) | `DenseGQA`, `DiffGQA`, pure recurrent, `3:1` Hybrid | Absolute retrieval accuracy & KV/state byte cost (`3` seeds) |
| **`X06` / cheap** | Depth routing ablation: `Laminar Depth-AttnRes` vs `AttnRes` (2026) vs `mHC` vs residual | Identical params, width, depth, optimizer | Validation BPB, gradient norms, throughput overhead |
| **`X07` / cheap** | Dual-memory vs pure fast state (`ExactEvidenceStore` + recurrent state) | No store, scan-budgeted retrieval, dense attention | Exact factual conservation, deletion fidelity, explicit byte & scan accounting (`3` seeds) |
| **`X08` / cheap** | Shared-budget VoI adaptive compute controller vs fixed `1/2/4/8` & Best-of-$N$ | Shared multi-item FLOP budget | Verified tasks solved per total FLOP/joule |
| **`X09` / cheap** | Tokenizer / byte-patching / hybrid value channel audit | Subword vs `BLT`-style dynamic byte patches | Exact literal/drug/ID copying at matched FLOPs |
| **`X10` / L1** | Core architecture iso-FLOP (`125M`–`350M`, `3` seeds): `DenseGQA` vs `GDN-2` Hybrid vs `Mamba-3` Hybrid vs `GeoEraseWrite` Hybrid | Exact matched non-emb + total params, tokenizer, corpus, FLOPs | Pareto across 4+ domains, copy, retrieval, speed |
| **`X11` / L1** | Optimizer factorial (`AdamW` vs `Muon` $\times$ dense vs hybrid) | Tuned LRs on separate dev set | Exact separation of optimizer vs architecture gain |
| **`X12` / L1** | Curriculum & clean data (`Uniform` vs `Surprise-DoReMi` vs `LP-Curriculum`) | Frozen mixture & FLOPs | Per-domain BPB + protected-probe forgetting |
| **`X13` / L1** | Auxiliary representational objectives (LM vs MTP vs exact-span/state loss) | One objective addition at a time + gradient cosines | Downstream transfer vs gradient interference |
| **`X14` / L1** | Continual adaptation (`Frozen+Retrieval` vs `LoRA` vs `Titans/TTT` vs `Nested`) | Fixed update tokens & memory ceiling | Retention/plasticity/deletion/adversarial-write |
| **`X15` / sys** | Repaired P1 grounded pilot on fresh unexposed cohort + external clinician review | Deterministic reference + compact `Qwen` + native `Nano` | Risk-coverage curve & human acceptability |
| **`X16` / sys** | Non-medical transfer (scientific source QA, incident diagnosis, causal traces) | Identical `nano/` & verification APIs | Transfer delta outside medicine |
| **`X17` / gate** | Scaling ladder (`3M -> 30M -> 125M -> 350M`) | Identical data & eval protocol | Power-law trend gating `2B` proposal |
| **`X18` / hw** | Decode kernel matrix (`CPU`/`MPS`/`CUDA`, `512/4k/32k/128k`, `bs=1/4/16`, `fp16/bf16/q4`) | Reference kernels + parity check | Real latency/peak-memory/joule Pareto |
| **`X19` / L3** | Conditional `2B` flagship pretrain (ONLY after `G0`–`G7` approval) | Strongest matched `<4B` baseline & open `Qwen3.5-2B` | Independent 15-pillar & OOD holdout |
| **`X20` / L4** | Grounded post-training, Top-$K$+Tail KD, executable RLVR, external validation | In-cap open checkpoints (`MiniCPM5-2B`, `Qwen3.5-2B`, `Granite 3B`) | Verified task improvement without overclaiming |
