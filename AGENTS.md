# Agent operating instructions

## CURRENT PROGRAM

```text
authority = docs/PROJECT_AUTHORITY.md
mission = docs/PROJECT_CHARTER.md
current_state = docs/ACTIVE_NOW.json + docs/ACTIVE_NOW.md
execution = docs/EXECUTION_PLAN.md

capability_frontier = GENERAL_PURPOSE_LANGUAGE_MODEL
model_objective = FRONTIER_CAPABILITY
current_gate = MATCHED_GENERAL_MODEL_BASELINE
historical_application = NanoScribe (outside active scope)
training_backends = KAGGLE + GOOGLE_CLOUD + RUNPOD
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

## Execution principles

- Build and measure the native model. Report teacher, adapted-checkpoint and tool-assisted results under their own identities.
- Prefer controlled comparisons on public or licensed data with recorded code, tokenizer, seeds, compute and held-out sources. Check data overlap before claiming generalization.
- Choose local, Kaggle, credit-backed Google Cloud or RunPod according to runtime fit and useful evidence per dollar and wall-clock hour. Bind each paid job to an experiment budget, termination condition and durable result destination. Google credits and RunPod funds are separate.
- The existing RunPod campaign allows $180 autonomous spend, $200 hard cap and a $20 owner reserve. Shut down idle resources promptly.
- Use `origin/master` as development truth. Preserve unmerged experiments and distinguish a measured branch result from code integrated on master.
- NanoScribe, including its P1 backlog, is historical. Maintain its reproduction path. Future work on a separate application requires its own scope.
- Multi-agent/worktree coordination: follow [`docs/AGENT_CONTRACT.md`](docs/AGENT_CONTRACT.md); start each session with `bash scripts/agent_bootstrap.sh <agent> "<task>" [paths]`.
