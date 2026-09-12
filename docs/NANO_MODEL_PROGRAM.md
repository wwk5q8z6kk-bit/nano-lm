# Nano Model Program

Machine-readable companion: [`NANO_MODEL_PROGRAM.json`](NANO_MODEL_PROGRAM.json).

Authority chain: `docs/PROJECT_AUTHORITY.md` → this document → `docs/EXECUTION_PLAN.md` → campaign manifests.

## Mission

Build **Native Nano** (owned architecture, random init, our weights) for faithful P1 scribing while running coordinated parallel tracks for student adaptation, open teacher collective, verifier stack, and product ladder P1→P9.

## Five coordinated tracks

| Track | Role |
|-------|------|
| **N** Native Nano | Architecture research; pretrain → post-train → deploy |
| **S** Pretrained specialist | Controls, students, production candidates (not Nano itself) |
| **T** Open teacher collective | Per-capability donors; no wholesale distillation |
| **V** Verifier system | Deterministic → lexical → learned → reference → human |
| **P** Product ladder | P1 Scribing → P2 Summarization → P3 Charting → P4–P9 |

## Three learning eras

1. **Pretraining** — representations from large curated corpora
2. **Post-training** — SFT, preference optimization, RL/RLVR, tools, safety
3. **Continuous improvement** — evaluate → failures → targeted data → retrain → deploy

SFT is one stage, not a substitute for pretraining.

## Critical corpus correction

`artifacts/campaign/p1_distill_train_v1.json` (96 templated examples) is **`NATIVE_UNIT_OVERFIT_FIXTURE`**.

Valid uses: forward/backward smoke, checkpoint/resume, loss plumbing, overfit regression.

**Invalid uses:** architecture ranking, 30M vs 100M promotion claims, open-vocabulary generalization, P1 mastery, scientific winner selection.

Campaign Round 1 30M rankings remain preserved as **`EXPLORATORY_SCREENING_RANKING`** (provisional leaders: `evidence_bottleneck`, `span_port`). Revalidate on a real architecture-screen corpus (Wave 1) before confirmatory 100M promotion.

## Master gates (summary)

| Gate | Requirement |
|------|-------------|
| G0 | Measurement integrity (denominators, CI includes nanoscribe) |
| G1 | Corpus validity (registry, license, partition, hash) |
| G2 | Evaluation power (screening v2, frozen splits) |
| G3 | Tokenizer frozen per architecture line |
| G4 | 30M revalidation on real corpus × 3 seeds |
| G5 | 100M promotion (only after G4) |
| G6–G7 | 300M / credible base pretrain |
| G8–G11 | Continued pretrain, SFT, preference/RLVR, tools |
| G12–G16 | Verifier, safety, quantization, shadow, P1 human eval |
| G17–G20 | P1 exit, P2/P3, production release, flywheel |

**Do not scale an invalid experiment.** Do not post-train without a base. Do not train on evaluation sets.

## Immediate waves

### Wave 0 (current)

Parallel lanes: measurement/CI · corpus factory · eval v2 · student gap (done) · verifier hardening · agent canary · P2/P3 probes.

### Wave 1

30M decoder control + `evidence_bottleneck` + `span_port` × 3 seeds × **real architecture-screen corpus**.

### Wave 2

100M promotion of top two mechanisms only if Wave 1 survives.

### Wave 3–5

Student adaptation (QLoRA after canary pass) · 300M Native · credible base pretrain budget.

## RunPod surfaces

Managed reference → Public Endpoint → vLLM/SGLang Serverless → Axolotl Serverless (student) → PyTorch Pod (Native custom arch). No paid resource without experiment manifest. Terminate after verified artifact pull.

See `artifacts/campaign/gpu_training_topology_v1.json` for screening vs real-training GPU economics.
