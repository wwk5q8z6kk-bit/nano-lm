# Canonical Status Table

**Authority:** this file + `CANONICAL_STATUS_TABLE.json`
**Updated:** 2026-07-31 (`AUTHORIZE_E4_DESIGN_ONLY`)
**Rule:** every status-bearing doc must derive from this table. Do not invent parallel statuses.

| Object | Canonical status | Notes |
|--------|------------------|-------|
| Paper α core | `PUBLIC_FROZEN_CORRECTED` | Tag `paper-alpha-v1` plus claim-sync corrections (nano **32.8M** tokens; schedule-aware scale language; E3 agent-rubric limitation). |
| E1 | `PUBLIC_EVIDENCE_ARCHIVED` | KILL under frozen U; M1 dominates official M0; M2 within δ=0.05. |
| E2 | `GATED_STOP` / `NO_RESULT` | Prereg frozen; no results JSON; not "in flight." |
| E3 normalize | `PUBLIC_EVIDENCE_ARCHIVED` | Auto arm: 0/486 rescues on M0 exact failures. |
| E3 agent audit | `AGENT_SINGLE_PASS` / `NO_IAA` | Rater `agent-rubric-pass-1`; 0/100 faithful; not clinician evaluation. |
| E3 human arm | `NOT_RUN` | Dual-clinician IAA + synonym ontology open. |
| Fabric | `PUBLIC_SCOPED_SLICE` / `NOT_PRODUCT` | Thin synthetic regression harness ≠ NanoScribe. |
| R★ | `DESIGN_HARDENED` | Anti-circular I*/X*/B*; no E4 data world. |
| E4 | `DESIGN_IN_PROGRESS` / `EXECUTION_BLOCKED` / `NO_BUILDER` / `NO_DATA` / `NO_RESULT` | Design track authorized; execution not authorized. |
| Program posture | `IDLE_AFTER_FREEZE` + design carve-out | Freeze complete; ambition continues via E4 design only (`papers/AMBITION.md`). |

## Derived-doc checklist

Status language in README, `EMPIRICAL_FOUNDATION`, `RESEARCH_PROGRAM`, `AMBITION`,
`EVIDENCE_LEDGER`, `EVIDENCE_MANIFEST`, `CLAIM_GLOSSARY`, `REGIME_P1`, `DECISION_P1`,
`PIPELINE_GATE_LOG`, and freeze reports must match the statuses above.

## Transition log

| When | Change |
|------|--------|
| Pre-freeze discovery | Paper α `PUBLIC_FROZEN_WITH_CORRECTIONS_PENDING`; E1/E3 `LOCAL_RESULT_REPORTED` |
| After F0–F5 (freeze) | Paper α `PUBLIC_FROZEN_CORRECTED`; E1/E3 normalize `PUBLIC_EVIDENCE_ARCHIVED`; E4 `BLOCKED` |
| AUTHORIZE_E4_DESIGN_ONLY | E4 → `DESIGN_IN_PROGRESS` / `EXECUTION_BLOCKED`; R★ → `DESIGN_HARDENED`; ambition note added |
