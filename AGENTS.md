# Agent operating instructions

## CURRENT PROGRAM

```text
authority = docs/PROJECT_AUTHORITY.md
mission = docs/PROJECT_CHARTER.md
current_state = docs/ACTIVE_NOW.json + docs/ACTIVE_NOW.md
execution = docs/EXECUTION_PLAN.md

capability_frontier = P1_SCRIBING
proving_application = NanoScribe
nano_core = nano/                  # domain-general symbolic substrate
nano_learned_core = nano_ai/       # learned scribing path through frozen H6

macro_sequence =
P1 faithful scribing
→ P2 summarization
→ P3 longitudinal charting
→ P4–P9 intelligence expansion

training_backends = KAGGLE + RUNPOD
training_status = ACTIVE

backend_policy =
KAGGLE_FREE_WHEN_RUNTIME_FITS;
RUNPOD_FOR_EXACT_GPU_RUNTIME_OR_MANAGED_INFERENCE

routine_runpod_training =
ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET

materially_costly_run =
EXPERIMENT_SCOPED_AUTHORIZATION

confirmatory_evidential_run =
PREREG_PLUS_EXPERIMENT_SCOPED_AUTHORIZATION

phi_or_private_data =
NOT_AUTHORIZED

Wedge = supporting verified-information subsystem

H6 / Nano AI = INTEGRATED_ON_MASTER_AND_H6_REJECTED
span-port / P4 / CUAD = CROSS_BRANCH_NOT_YET_INTEGRATED

E1 / E4 =
preserved scoped empirical verdicts,
not current-program STOP instructions

July-31 IDLE / NanoScribe STOP =
HISTORICAL_PROGRAM_STATE
```

Canonical index: [`docs/README.md`](docs/README.md) · operational map:
[`docs/knowledge/AGENT_PROGRAM_KNOWLEDGE.md`](docs/knowledge/AGENT_PROGRAM_KNOWLEDGE.md).

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
- H6's one-shot evaluation is consumed and terminal. Do not rerun or relabel it.

## Hard gates

- No freeze tag create/move/push; no B17 freeze-recipe execution (historical only).
- No evidence-protected path edits in documentation-reset PRs.
- Materially costly and confirmatory runs follow experiment-scoped authorization in the active plan. Routine RunPod training is in-workflow within the active experiment budget.

## Verify

```bash
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
python3 fabric/test_fabric.py
python3 trajectory/test_recompute_c3.py
python3 -m pytest -q nanoscribe
python3 nanoscribe/benchmark.py
python3 -m pytest -q nano
python3 -m pytest -q nano_ai/tests
python3 -m pytest -q benchmarks/
```

## Learned User Preferences

- Operate autonomously on Nano P1: build, test, measure, and continue without routine owner prompts or stopping after a single PR or experiment.
- Maximize intelligence gained per dollar and wall-clock hour; do not default to self-hosting the largest open-weight checkpoint available.
- Treat B300 availability as a feasibility gate, not proof that arbitrary frontier checkpoints fit on one or two GPUs.
- Prefer hosted API/inference for the frontier teacher when cheaper; reserve self-hosted B200/B300 for student training, distillation, Native Nano, and verifier work.
- Prefer Kaggle for free, reproducible experiments when its runtime fits; use RunPod only when the exact GPU/runtime, persistence, or managed inference surface is required.
- Enforce RunPod cost discipline: $180 autonomous spend envelope, $200 hard campaign cap, $20 owner reserve not for routine use; tear down idle pods promptly.
- Use `origin/master` as development truth; never trust or push from a stale local `master`.
- Do not reopen the documentation-reset program or turn work back into governance exercises when implementation can proceed.

## Learned Workspace Facts

- Current capability frontier is P1 faithful scribing; NanoScribe is the proving application for the AI, not a separate product scope. Kaggle and RunPod are complementary training backends selected by experiment fit and cost.
- Nano Core (`nano/`), its reproducible clinical/synthetic-world evidence, and the coherent learned Nano AI package through H6 (`nano_ai/`) are integrated on `master` via PRs #107–#109.
- The agentic benchmark instrument and v1.1 action-log contracts are integrated on `master` via PR #110.
- P1 foundation PRs landed on master: #37 Encounter Representation v0, #38 constrained evidence transport/evaluation, #40 minimal model adapter and baseline bridge, #41 Qwen inference + three-track harness (`origin/master` ~c4822b9).
- P1 model research runs four parallel tracks: frontier teacher (capability ceiling), large student, compact control (e.g. Qwen2.5-1.5B baseline adapter), and native Nano vNext screening (~30M–100M).
- Qwen is a compact control / baseline adapter path, not Nano itself; Nano is the broader faithful-representation program, not merely a tiny LM, Qwen wrapper, or LoRA baseline.
- Large frontier checkpoints may exceed practical self-host limits (e.g. 300B+ models often need multi-node serving); choose teacher modality by economics and fit, not checkpoint size alone.
- Historical local `master` at `9fe5b6b6` diverged from canonical remote and must not be treated as push target.
