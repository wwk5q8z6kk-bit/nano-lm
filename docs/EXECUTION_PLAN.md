# Execution Plan

Executable tasks under [ACTIVE_NOW.md](ACTIVE_NOW.md). Historical queue preserved at `papers/EXECUTION_QUEUE.md` (superseded stub).

## Phase A — Documentation reset (current)

| ID | Task | Done when |
|----|------|-----------|
| A1 | Create `docs/` canonical set | All core files present; index in `docs/README.md` |
| A2 | Rewrite root `README.md` | Opens with Nano mission, not 3.15M headline |
| A3 | Supersede stubs at legacy planning paths | Stubs point to `docs/` |
| A4 | Archive historical strategy copies | `docs/archive/` with explicit SUPERSEDED banners |
| A5 | Rewrite `AGENTS.md` | Points agents to `docs/` authority |
| A6 | `ACTIVE_NOW` JSON/Markdown consistency | `python scripts/check_active_now.py` passes |
| A7 | Owner review | File map + README approved before `master` merge |

## Phase B — P1 scribe (next, after A7)

| ID | Task | Done when |
|----|------|-----------|
| B1 | Encounter representation schema v0 | JSON schema + docs; supports entity/event/evidence refs |
| B2 | Span/evidence bottleneck | Measured on held-out medical dialogue subset (no PHI in repo) |
| B3 | Verified record → note rendering | Note is view of record, not primary truth |
| B4 | External eval protocol draft | [domains/medical/EVALUATION_PROTOCOL.md](domains/medical/EVALUATION_PROTOCOL.md) instantiated |
| B5 | P1 exit gate checklist | Metrics + human review plan — not claimed passed |

## Explicitly out of scope (until authorized)

- P2/P3 implementation beyond interface contract
- Paid RunPod jobs
- LM training runs
- Evidence Core / ledger edits
- Clinical deployment claims

## Verification commands

```bash
python scripts/check_active_now.py
pytest fabric/test_fabric.py -q
# Full wedge pins when on frontier branch:
# python -m wedge_v1 smoke
```
