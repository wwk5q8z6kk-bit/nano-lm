# Project Authority

When documents conflict, resolve in this order:

```text
1. docs/PROJECT_CHARTER.md
2. docs/CAPABILITY_LADDER.md
3. docs/SYSTEM_ARCHITECTURE.md
4. docs/ROADMAP.md
5. docs/ACTIVE_NOW.md  (+ ACTIVE_NOW.json)
6. docs/EXECUTION_PLAN.md
```

Lower layers must not silently supersede higher layers.

## Directory roles

| Path | Authority |
|------|-----------|
| `README.md` | Public overview — points here |
| `docs/` | **Current** program truth, architecture, execution |
| `papers/` | **Science** — preregistrations, results, manuscripts, empirical history |
| `trajectory/` | Experimental records, reproducibility artifacts |
| `artifacts/` | Machine evidence bundles |
| `nano_ai/` | Model / intelligence core implementation |
| `wedge_v1/` | Supporting verified-information subsystem |
| `frontier/` | Branch-local development notes only — **not** canonical |

## Superseded planning locations

These may remain for historical links but must not be read as master plans:

| Legacy path | Current authority |
|-------------|-------------------|
| `papers/STRATEGIC_RESET.md` | [PROJECT_CHARTER.md](PROJECT_CHARTER.md) + [ROADMAP.md](ROADMAP.md) |
| `papers/AZ_EXECUTION_PLAN.md` | [archive/AZ_EXECUTION_PLAN_POST_E1_20260731.md](archive/AZ_EXECUTION_PLAN_POST_E1_20260731.md) (historical) |
| `papers/AMBITION.md` | [PROJECT_CHARTER.md](PROJECT_CHARTER.md) |
| `papers/WEDGE_V1.md` | [subsystems/WEDGE.md](subsystems/WEDGE.md) |
| `papers/EXECUTION_QUEUE.md` | [EXECUTION_PLAN.md](EXECUTION_PLAN.md) |
| `papers/TECHNOLOGY_ROADMAP.md` | [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) + [research/SYSTEM_RESEARCH_PROGRAM.md](research/SYSTEM_RESEARCH_PROGRAM.md) |

## Evidence-protected (do not relocate casually)

- `papers/EMPIRICAL_FOUNDATION.md`
- `papers/EVIDENCE_LEDGER.md` and ledger JSON
- `papers/PREREG_*`, `papers/RESULT_*`, manuscripts
- Freeze tags, SHA manifests, raw result JSON at tagged commits
- `trajectory/results_*.json` as primary scientific record

## Owner gates (unchanged)

Paid compute, PHI/private data, protected tag moves, publication claims, and clinical capability claims require explicit owner authorization. See [ACTIVE_NOW.md](ACTIVE_NOW.md).
