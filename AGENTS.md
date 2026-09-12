# Agent operating instructions

## CURRENT PROGRAM

```text
authority = docs/PROJECT_AUTHORITY.md
mission = docs/PROJECT_CHARTER.md
current_state = docs/ACTIVE_NOW.json + docs/ACTIVE_NOW.md
execution = docs/EXECUTION_PLAN.md

capability_frontier = P1_SCRIBING
product_frontier = NanoScribe
nano_core = nano/   (contracts, kernel, ontology, capability registry)

macro_sequence =
P1 faithful scribing
→ P2 summarization
→ P3 longitudinal charting
→ P4–P9 intelligence expansion

Nano Core is domain-general. NanoScribe is P1 + Medical DomainPack,
not the architecture. DomainPack-0 (NANO-SLW-001) proves the substrate.

training_backend = RUNPOD
training_status = ACTIVE

routine_runpod_training =
ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET

materially_costly_run =
EXPERIMENT_SCOPED_AUTHORIZATION

confirmatory_evidential_run =
PREREG_PLUS_EXPERIMENT_SCOPED_AUTHORIZATION

phi_or_private_data =
NOT_AUTHORIZED

Wedge = supporting verified-information subsystem

NanoScribe (P1) = nanoscribe/ encounter v0 + harness + campaign

Nano Core (built) = nano/ — NANO-CLIN-001 ledger slice + NANO-SLW-001
synthetic world. Model research stays a separate lane.

Accelerated campaign v2 = tool calling + agent platform + multi-track (frontier branch)

H6 / Nano AI / span-port =
CROSS_BRANCH_NOT_YET_INTEGRATED

E1 / E4 =
preserved scoped empirical verdicts,
not current-program STOP instructions

July-31 IDLE / NanoScribe STOP =
HISTORICAL_PROGRAM_STATE
```

Canonical index: [`docs/README.md`](docs/README.md) · **Agent knowledge pack:** [`docs/knowledge/AGENT_PROGRAM_KNOWLEDGE.md`](docs/knowledge/AGENT_PROGRAM_KNOWLEDGE.md)

## Typed authority

[`docs/PROJECT_AUTHORITY.md`](docs/PROJECT_AUTHORITY.md) — empirical artifacts win over narrative; program charter wins over stale planning stubs. Superseded `papers/*` stubs → [`docs/archive/legacy/`](docs/archive/legacy/).

## Durable scoped facts

- E1 KILL = scoped to the old closed task under frozen \(U\) — not a full-program kill.
- E4 KILL = scoped to the tested R★ regime — not a full-program kill.
- Paper α = protected empirical foundation (`paper-alpha-v1`); do not reopen the old substrate claim.
- Agent-applied rubric audit ≠ clinician / human dual-IAA evaluation.
- `OLD_TASK_U` forbidden.
- Doc-reset PRs must not modify Evidence Core / ledger / freeze artifacts.
- No PHI / private owner material in current Nano experiments.

## Hard gates

- No freeze tag create/move/push; no B17 freeze-recipe execution (historical only).
- No evidence-protected path edits in documentation-reset PRs.
- Materially costly and confirmatory runs follow experiment-scoped authorization in the active plan. Routine RunPod training is in-workflow within the active experiment budget.

## Verify

```bash
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
bash scripts/ci_nanoscribe.sh
```

Full agent knowledge: [`docs/knowledge/AGENT_PROGRAM_KNOWLEDGE.md`](docs/knowledge/AGENT_PROGRAM_KNOWLEDGE.md) · Gates: [`docs/knowledge/PROGRAM_CHECKPOINTS.json`](docs/knowledge/PROGRAM_CHECKPOINTS.json)

## Learned User Preferences

- Operate autonomously on Nano P1: use `docs/knowledge/AGENT_PROGRAM_KNOWLEDGE.md` as the operational A→Z map; build, test, measure, and continue without routine owner prompts or stopping after a single PR or experiment.
- Maximize intelligence gained per dollar and wall-clock hour; do not default to self-hosting the largest open-weight checkpoint available.
- Treat B300 availability as a feasibility gate, not proof that arbitrary frontier checkpoints fit on one or two GPUs.
- Treat RunPod as a multi-surface research OS: prefer Public Endpoints and Hub Serverless over raw GPU Pods; use raw Pods only when Hub cannot express the experiment; query live wallet/pods/serverless/Hub catalog (`discover_hub_catalog`) before each paid wave—never stale Hub IDs.
- Prefer hosted API/inference or RunPod Public Endpoints for frontier teachers when cheaper; reserve B200/B300 for student training, distillation, Native Nano, and verifier work.
- Enforce RunPod cost discipline: $180 autonomous spend envelope, $200 hard campaign cap, $20 owner reserve; live wallet balance is the physical spend ceiling (min with authorized remaining)—query via `runpodctl user` before each paid wave.
- Use ephemeral Serverless for inference bursts (`workersMin=0`, create→batch→delete); terminate pods only after verified artifact pull (torch.load/adapter verify); delete endpoints between batches—no idle burn.
- No experiment manifest (git SHA, dataset revision, termination condition) → no paid compute.
- Use `origin/master` as development truth; never trust or push from a stale local `master`.
- Do not reopen the documentation-reset program or turn work back into governance exercises when implementation can proceed.
- Doc-reset branches must branch from clean `origin/master` with zero Evidence Core diff—never merge branches whose ancestry includes evidence reconciliation commits (e.g. `9fe5b6b6`).

## Learned Workspace Facts

- Current capability frontier is P1 faithful scribing; product frontier is NanoScribe; RunPod is the active GPU training backend within the active experiment budget.
- P1 foundation PRs landed on master: #37 Encounter Representation v0, #38 constrained evidence transport/evaluation, #40 minimal model adapter and baseline bridge, #41 Qwen inference + three-track harness (`origin/master` ~c4822b9).
- P1 model research runs four parallel tracks: frontier teacher (capability ceiling), large student, strong control (Qwen3.8-27B Serverless), and native Nano vNext screening (~30M–100M); Qwen2.5-1.5B is historical continuity only.
- Teacher collective (Capability Registry → per-capability open-weight competition) is the strategic teacher direction; Nano is the broader faithful-representation program—not a Qwen wrapper or LoRA baseline; Qwen3.8-27B stays P1 control/student, distinct from agent-policy teachers (e.g. Qwen3-Coder-Next).
- Large frontier checkpoints may exceed practical self-host limits (e.g. 300B+ models often need multi-node serving); choose teacher modality by economics and fit, not checkpoint size alone.
- RunPod GPU/runtime: B200 (sm_100/Blackwell) requires PyTorch with sm_100 support (`runpod/pytorch:2.4.0` sm_90-only is incompatible); `Qwen/Qwen3.8-27B-FP8` serves on 48GB PRO Serverless with vLLM/SGLang—not the 180GB B200 Serverless tier.
- Large-student path: vLLM/SGLang Serverless C1/C2 structured baseline → Axolotl Hub Serverless for QLoRA/SFT; on Serverless output-capture failure fall back to Pod QLoRA (`student_qlora_pod_canary.py`)—not raw A100 inference or training Pods by default.
- Native Nano trains on official PyTorch or autoresearch Pod templates—not Axolotl; `p1_distill_train_v1` is a unit/overfit fixture only (`NATIVE_UNIT_OVERFIT_FIXTURE`)—build architecture-screen corpus via `nanoscribe/native/corpus/` before GPU ranking; extended training is a separate pod lane, not screening-scale.
- Qwen3-32B-AWQ is the managed-ref winner for C1/C2 baselines; Kimi K3 public endpoint may return HTTP 500—use GPT-OSS-120B or Qwen3-32B-AWQ managed references so Kimi outage must not block the campaign.
- `p1_screening_eval_v1` is frozen forever; distillation data must be disjoint (`p1_distill_train_v1`), generated from failure/disagreement patterns—not screening evaluation artifacts.
- P1 primary model interface is structured CandidateAtom JSON (tool-call path)—not raw span-port text; measured binding bottleneck is exact gold span transport (~11% vs ~85% campaign target)—do not confuse strong assertion metrics with span fidelity.
- Pod SSH uses `ssh.runpod.io` with `~/.runpod/ssh/runpodctl-ssh-key` (`{podId}-{hostId}@ssh.runpod.io` via `scripts/runpod_pod_ssh.sh`); native/QLoRA weight pulls prefer direct TCP SCP (`pull_native_weights`/`pull_qlora_adapter`); verifier learned training is SKIP when deterministic hard-set baseline_accuracy ≥ ~0.95.
