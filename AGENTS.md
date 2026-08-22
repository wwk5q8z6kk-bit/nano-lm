# Agent operating instructions

**Canonical project truth:** [`docs/README.md`](docs/README.md)

Read in authority order when instructions conflict:

```text
docs/PROJECT_CHARTER.md
docs/CAPABILITY_LADDER.md
docs/SYSTEM_ARCHITECTURE.md
docs/ROADMAP.md
docs/ACTIVE_NOW.md  (+ ACTIVE_NOW.json)
docs/EXECUTION_PLAN.md
```

## What Nano is

Nano is a research and engineering program for **compact, reliable intelligence** — not a single 3.15M LM, not Wedge alone, not a verification wrapper.

- **Current frontier:** P1 Master Scribing (medical DomainPack)
- **Nano Core + DomainPack:** domain-general primitives + medical-first proving ground
- **Governing rule:** smallest sufficient solver — software, retrieval, schemas, compact models, or hybrids

## Directory roles

| Path | Role |
|------|------|
| `docs/` | Current program truth — **start here** |
| `papers/` | Science — preregistrations, results, manuscripts (do not treat as master plan) |
| `trajectory/` | Experimental records |
| `wedge_v1/` | Supporting verified-information subsystem |
| `nano_ai/` | Model / intelligence core |
| `frontier/` | Branch-local notes only — **not canonical** |

## Hard gates (never bypass)

- Do **not** move or rewrite: `papers/EVIDENCE_LEDGER.*`, `papers/EMPIRICAL_FOUNDATION.md`, `PREREG_*`, `RESULT_*`, freeze tags, SHA manifests, tagged result JSON
- Do **not** launch paid compute without explicit owner authorization
- Do **not** use PHI or private owner corpus in git
- Do **not** make clinical capability claims without external + human validation
- Do **not** cite E1/E4 KILL as killing the full Nano program — scope to tested regime/utility
- `OLD_TASK_U` runs are forbidden

## Claim discipline

- Separate science (Layer 1), systems, and product claims
- No mechanism claims beyond measured evidence
- Agent-applied rubric audits ≠ human/clinician evaluation
- Benchmark result ≠ evidence-ledger claim ≠ product authorization

## Execution

1. Check [`docs/ACTIVE_NOW.md`](docs/ACTIVE_NOW.md) for current gate and bounded work
2. Follow [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md) for tasks
3. Run `python3 scripts/check_active_now.py` after editing ACTIVE_NOW files
4. Owner speech acts: [`papers/OWNER_SPEECH_ACTS.md`](papers/OWNER_SPEECH_ACTS.md) — `continue` ≠ commit/push/tag/execute

## Verification commands

```bash
python3 scripts/check_active_now.py
python3 -m pytest fabric/test_fabric.py -q
```

## Historical docs

Legacy planning paths (`papers/STRATEGIC_RESET.md`, `papers/AZ_EXECUTION_PLAN.md`, etc.) are **superseded stubs**. See [`docs/archive/LEGACY_STRATEGY_INDEX.md`](docs/archive/LEGACY_STRATEGY_INDEX.md).
