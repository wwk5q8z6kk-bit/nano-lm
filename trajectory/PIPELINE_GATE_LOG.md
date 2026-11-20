# Pipeline gate log

*Append-only. Owner/agent records Gate N results here.
Constants: see `papers/SEQUENTIAL_PIPELINE.md`.*

| Gate | Stage | Result | Date | Notes |
|------|-------|--------|------|-------|
| 0 | Decision lock | **PASS** | 2026-07-31 | C1–C9, E1 KILL, ledger locked |
| 1 | E3 construct | **PASS — EXACT_SURVIVES** | 2026-07-31 | Bounded agent-rubric audit n=100 (`agent-rubric-pass-1`); faithful-rate 0.00; not skipped. `results_e3_human.json`, `STAGE1_E3_CONSTRUCT_FIRST_PRINCIPLES.md` |
| 2 | Regime R★ | **PASS** | 2026-07-31 | R★ hardened — `REGIME_P1_where_classical_fails.md` |
| 3 | P2 protocol | **PASS** | 2026-07-31 | `trajectory/PREREG_E4_Rstar_killgate.md` freezes \(U_{R★}\), baselines, KILL/SURVIVE/GRADED, data definition, limitations. **No E4 run.** |
| 4 | E4 kill gate | **KILL** | 2026-07-31 | Owner `AUTHORIZE_E4_BUILDER_AND_EXECUTE`. R★ world frozen; probe `in_Rstar=true` (B1/B3/B4). Under frozen \(U_{R★}\): \(U^\star_{\mathrm{class}}=0.638\) (C-M2) ≥ \(U^\star_{\mathrm{gen}}=-1.623\) (G-ref verify-on) − δ. Sensitivity flip false. Artifacts: `results_e4_utility.json`, `results_e4_classical_probe.json`. Per §3: stop generative-substrate product track for tested R★; ≤1 revision budget; no NanoScribe/fabric expansion. |

### Correction note (Gate 1)
Earlier “PASS (skipped)” was wrong; superseded by EXACT_SURVIVES after bounded agent-rubric audit.

### Design-track note (2026-07-31) — superseded
Owner authorized `AUTHORIZE_E4_DESIGN_ONLY` (docs). Superseded same day by
`AUTHORIZE_E4_BUILDER_AND_EXECUTE` → Gate 4 **KILL** (see execution-track note).

### Execution-track note (2026-07-31)
Owner authorized `AUTHORIZE_E4_BUILDER_AND_EXECUTE` (message: "authorized").
Auth record: `trajectory/e4/AUTH_RECORD.md`. Gate 4 **KILL**.
Ambition path: consequences table §3 — product track stop for this R★; revision budget 1.
