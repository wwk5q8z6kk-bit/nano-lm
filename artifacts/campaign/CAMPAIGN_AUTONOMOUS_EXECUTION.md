# Campaign autonomous execution (single source of truth)

**Authority:** This document supersedes all prior wave-specific launch prompts, ad-hoc RunPod chat instructions, and stale hub ID notes. Paid compute follows this file only.

**Program:** P1 faithful scribing · NanoScribe · RunPod multi-surface research OS (not “rent GPUs”).

**Last updated:** 2026-08-23

---

## 0. Mental model — RunPod as research OS

RunPod is a **multi-surface execution OS**: public/managed APIs, Hub Serverless workers, Flash boot, Pod Templates, raw Pods, and Clusters. Surfaces are interchangeable **only** through manifests, wallet gates, and artifact contracts—not by habit or stale IDs.

### Execution hierarchy (mandatory — try in order)

1. **Public / managed endpoint** (GPT-OSS, Qwen AWQ, Cogito, Kimi canary)
2. **Hub Serverless** (vLLM, SGLang, Axolotl serverless listings)
3. **Flash** (when available for the listing)
4. **Pod Template** (official PyTorch `runpod-torch-v240`, autoresearch hub pods)
5. **Raw Pod** (image + docker-args — last resort, health-gated)
6. **Cluster** (only when single-node infeasible and budget authorized)

**Default forbidden:** raw pods for baseline inference; Axolotl for **Native** training; `workersMin=1` idle burn; training on `p1_screening_eval_v1`; Kimi blocking the campaign; stale hard-coded Hub listing IDs without live resolve.

---

## 1. Control plane (~10 min, **no paid compute**)

Before **every** wave or relaunch, run inventory (human or `scripts/campaign_control_plane.py inventory`):

| Check | Command / module |
|-------|------------------|
| Live wallet | `runpodctl user` · `nanoscribe.runpod_wallet` |
| Spend rate | `currentSpendPerHr` must trend → 0 when idle |
| GPUs | `runpodctl gpu list` |
| Hub | `runpodctl hub search/get` · `nanoscribe.runpod_hub.discover_hub_catalog` |
| Templates | `runpodctl template search/get` |
| Pods | `runpodctl pod list` |
| Serverless | `runpodctl serverless list` |

**Maintain in `artifacts/campaign/campaign_status.json`:**

- `live_runpod_balance`, `current_spend_per_hr`, `spend_limit`
- `campaign_remaining = min(authorized_remaining, clientBalance - $10 floor)` — **never** trust ledger alone when wallet is lower
- Hub entries: `stable_locator` (e.g. `runpod-workers/worker-vllm`) + `resolved_listing_id` + `resolved_at`
- Active pods, serverless endpoints, worker counts, queue depth

### Experiment manifest (required before paid job)

Schema: `artifacts/campaign/experiment_manifest.v1.schema.json`

Every paid job MUST have a manifest JSON with:

- `experiment_id`, `git_sha`, `dataset_revision`
- `command`, `surface`, `gpu_or_model`, `hub_listing_id` (if Hub)
- `max_runtime_min`, `max_cost_usd`, `artifact_dest`
- `termination_condition` (success, fail-fast GPU, budget, manual)

**Rules:**

- **No manifest → no compute.**
- **Queue empty → scale to zero / delete ephemeral endpoint** (`workersMin=0`; delete if min=0 won’t stick).
- **Done → persist artifacts → terminate** pods/endpoints.

---

## 2. Wallet gates (live ~$163, $10 floor → ~$153 usable)

Use **`runpodctl user` `clientBalance`** as physical ceiling. Ledger `authorized_remaining` is policy only.

| Elapsed (campaign session) | Cumulative **new** spend target |
|--------------------------|----------------------------------|
| 0–20 min | < $5 (canaries, hub resolve, smoke) |
| 20–45 min | ~$20–35 |
| ~100 min | ~$50–60 |
| ~2.5 h | ~$85–95 |
| ~3.25 h | ~$115–125 |
| Final cap | ≤ ~$153 usable (balance − floor) |

**Adaptive:** stop early when the experiment question is answered; extend only for crucial remaining uncertainty (owner-level tradeoff).

---

## 3. Priority order (autonomous agent)

1. Control-plane inventory + terminate idle/broken pods (zero `$`/hr).
2. Wallet + hub discovery → refresh `campaign_status.json`.
3. Leakage / partition gates (`p1_screening_eval_v1` frozen; train on `p1_distill_train_v1` only).
4. **Managed reference** — GPT-OSS-120B C1/C2 (winner if operational); Qwen3-32B-AWQ challenger; Cogito if responds; Kimi **occasional canary only** (never blocks wave).
5. **Inference plane** — deploy ephemeral Qwen3.8-27B-FP8 vLLM Hub worker on **48GB PRO** tier (not 180GB B200 serverless); fair benchmark vs SGLang once; pick engine on TTFT, p95, tok/s, structured validity, worker-seconds, $/eval.
6. **Student plane** — C1/C2 structured eval via **vLLM/SGLang serverless** or managed endpoints; **NOT** raw A100 baseline inference pods.
7. **Gap gate** — if student vs managed ref gap justifies: Axolotl Serverless **50-step canary** → short QLoRA → re-eval gain/$.
8. **Native plane** — central custom training: `train_native_nano.py`; 30M factorial A/B/C/D × 2 seeds = 8 runs; split 4+4 across GPUs; 20–30 min successive halving → 100M winners → maybe 300M.
9. **Native surface** — PyTorch template or **autoresearch** pod on H100/A100; **B200 only** with sm_100-compatible PyTorch (not `runpod/pytorch:2.4.0` sm_90 image).
10. **Verifier** — expand hard set (500–2000 deterministic examples) on CPU; 4090/A40 only if discriminative; **no learned training** if deterministic baseline ≥ ~0.95 (48-case run DONE at 1.0).
11. **Disagreement → teacher data** — continuous TEACHER_VERIFIED side-effect; no bulk synthetic replacement.
12. **Parameter Golf** — training-systems **donor only** (not 16MB objective).
13. **autoresearch** — bounded config mutation with fixed eval boundaries.
14. **P2/P3 probes** — architecture reconnaissance only; not integration frontier advance.

Execute as **DAG parallelism** where lanes are independent (managed ref ∥ hub deploy ∥ native manifest prep ∥ verifier expand).

---

## 4. Inference plane (parallel)

| Lane | Surface | Notes |
|------|---------|-------|
| Managed public | GPT-OSS-120B, Qwen3-32B-AWQ | **BEST_OPERATIONAL_MANAGED_REFERENCE** when C1/C2 pass |
| Kimi K3 | Managed | **BLOCKED** — no retries blocking Wave |
| Qwen serverless (legacy) | Deleted endpoint | Recreate via Hub vLLM, ephemeral |
| Student serving | Hub vLLM `runpod-workers/worker-vllm` | 48GB PRO ~$1.75/h class; `MODEL_NAME=Qwen/Qwen2.5-32B-Instruct` or campaign model |
| Challenger | Hub SGLang | One fair benchmark vs vLLM |

**Lifecycle:** `workersMin=0` → batch jobs → collect metrics → **delete endpoint** if idle.

---

## 5. Native plane (central)

- **NOT Axolotl** for Native Nano training.
- **DO** use official PyTorch template (`runpod-torch-v240`) or **autoresearch** hub on H100/A100.
- **2× B200** as experiment factories **only** after on-pod `nanoscribe.runpod_gpu_preflight` passes (sm_100 ≤ max arch in wheel).
- Entrypoint: `python3 scripts/train_native_nano.py --run-id <canonical_run_id>` after `pip install -r requirements.txt` (`numpy<2.5` on Py3.11).
- **5-minute GPU util gate:** if GPU util < 5% after 5 min → terminate pod, log failure (`scripts/runpod_pod_health_gate.sh`).

---

## 6. Student plane

1. Serve baseline via **vLLM/SGLang Serverless** (Hub-first).
2. Run C1/C2 structured eval from **local orchestrator** against served OpenAI-compatible URL.
3. Compare to managed reference (GPT-OSS-120B winner).
4. QLoRA via **Axolotl Serverless** (`axolotl-ai-cloud`) only after gap gate — 50-step canary first.

---

## 7. Verifier plane

- Expand hard set locally (deterministic generation; target 500–2000 when data allows).
- Run `scripts/verifier_lane.py` on CPU.
- Learned verifier **NOT justified** when deterministic `baseline_accuracy` ≈ 1.0 on easy 48.
- 4090 pod only when expanded set is discriminative (< ~0.95 baseline).

---

## 8. Disagreement → teacher data

Continuous pipeline from eval disagreements; TEACHER_VERIFIED labeling; never train on `p1_screening_eval_v1`.

---

## 9. P2/P3 concurrent probes

Lightweight architecture reconnaissance (summarization/charting sketches) **in parallel** with P1; does not advance P1 integration frontier.

---

## 10. Preflight & hygiene

- `nanoscribe.runpod_gpu_preflight` — `torch.cuda.get_device_capability` vs `get_arch_list()`; block B200 + sm_90-only images.
- `numpy>=2.0,<2.5` on Python 3.11 RunPod images (2.5.0 unavailable).
- Network volume `04himzqxbm` must mount (`volumeInGb` > 0 in pod get); if 0, fix before relaunch.

---

## 11. Operator commands (quick reference)

```bash
# Control plane (no spend)
python3 scripts/campaign_control_plane.py inventory

# Wallet
python3 -c "from nanoscribe.runpod_wallet import effective_campaign_budget; import json; print(json.dumps(effective_campaign_budget(), indent=2))"

# Hub catalog
python3 -c "from nanoscribe.runpod_hub import discover_hub_catalog; import json; print(json.dumps(discover_hub_catalog(), indent=2))"

# Wave orchestrator (Hub-first status write)
python3 scripts/campaign_wave_v2.py --no-managed-ref   # status only
python3 scripts/campaign_wave_v2.py --native           # native pods (gated)

# Ephemeral serverless (example)
scripts/p1_serverless_launch.sh deploy   # then batch, then teardown

# Native CPU smoke (required before pod)
python3 scripts/train_native_nano.py --cpu-smoke

# Pod health gate (background after launch)
scripts/runpod_pod_health_gate.sh <pod_id> [rate_hr]
```

---

## 12. Success criteria (Wave checkpoint)

- Zero idle pod `$`/hr unless manifest-active job.
- Hub IDs resolved at launch with stable locators recorded.
- Student eval served via Serverless or managed — not idle A100.
- Native runs produce checkpoints under `artifacts/native_checkpoints/` with loss trending.
- Verifier expanded hard set metrics recorded; learned training gated.
- `campaign_status.json` + `checkpoint_v1.json` reflect live wallet and surfaces.

---

*End of autonomous execution prompt.*
