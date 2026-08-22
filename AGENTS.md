# Agent operating instructions

## CURRENT PROGRAM

```text
authority = docs/PROJECT_AUTHORITY.md
mission = docs/PROJECT_CHARTER.md
current_state = docs/ACTIVE_NOW.json + docs/ACTIVE_NOW.md
execution = docs/EXECUTION_PLAN.md

capability_frontier = P1_SCRIBING
product_frontier = NanoScribe

macro_sequence =
P1 faithful scribing
→ P2 summarization
→ P3 longitudinal charting
→ P4–P9 intelligence expansion

local_zero_cost_exploratory_training = ALLOWED
paid_training = OWNER_GATED
frozen_confirmatory_execution = PREREG_PLUS_OWNER_GATED

Wedge = supporting verified-information subsystem

H6 / Nano AI / span-port =
CROSS_BRANCH_NOT_YET_INTEGRATED

E1 / E4 =
preserved scoped empirical verdicts,
not current-program STOP instructions

July-31 IDLE / NanoScribe STOP =
HISTORICAL_PROGRAM_STATE
```

Canonical index: [`docs/README.md`](docs/README.md).

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
- Paid / confirmatory runs follow experiment-scoped authorization in the active plan.

## Verify

```bash
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
python3 -m pytest fabric/test_fabric.py trajectory/test_recompute_c3.py -q
```
