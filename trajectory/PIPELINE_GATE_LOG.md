# Pipeline gate log

*Append-only. Owner/agent records Gate N results here.
Constants: see `papers/SEQUENTIAL_PIPELINE.md`.*

| Gate | Stage | Result | Date | Notes |
|------|-------|--------|------|-------|
| 0 | Decision lock | **PASS** | 2026-07-31 | C1–C9, E1 KILL, ledger locked |
| 1 | E3 construct | **PASS — EXACT_SURVIVES** | 2026-07-31 | Bounded agent-rubric audit n=100 (`agent-rubric-pass-1`); faithful-rate 0.00; not skipped. `results_e3_human.json`, `STAGE1_E3_CONSTRUCT_FIRST_PRINCIPLES.md` |
| 2 | Regime R★ | **PASS** | 2026-07-31 | R★ hardened — `REGIME_P1_where_classical_fails.md` |
| 3 | P2 protocol | **PASS** | 2026-07-31 | `trajectory/PREREG_E4_Rstar_killgate.md` freezes \(U_{R★}\), baselines, KILL/SURVIVE/GRADED, data definition, limitations. **No E4 run.** |
| 4 | E4 kill gate | **BLOCKED** | — | Until owner authorizes Stage 4 execution against frozen P2 **without** mid-stream edits |

### Correction note (Gate 1)
Earlier “PASS (skipped)” was wrong; superseded by EXACT_SURVIVES after bounded agent-rubric audit.
