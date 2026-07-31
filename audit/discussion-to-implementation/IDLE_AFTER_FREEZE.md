# IDLE After Freeze — Governance Note

**Decision gate:** `IDLE_AFTER_FREEZE`  
**Date:** 2026-07-31  
**Tag (local):** `post-alpha-evidence-freeze-2026-07-31`  
**E4:** remains **BLOCKED** — this note is **not** authorization to run E4.

## Default posture

After the evidence + claim-sync commits and annotated tag:

1. No new experiments.
2. No paid compute.
3. No fabric / NanoScribe expansion.
4. No E2 runs.
5. No E4 builder/data/result work until a separate explicit owner authorization.

Canonical statuses: `CANONICAL_STATUS_TABLE.md`.

## E4 precommitted consequence table (governance only)

If E4 is ever authorized later under a frozen \(U_{R★}\) and locked classical/generative set, interpret outcomes as:

| Outcome | Meaning | Program consequence |
|---------|---------|---------------------|
| **KILL** | Best classical under \(U_{R★}\) beats or ties (within \(\delta\)) best generative | Generative still not preferred in R★; product thesis fails again; do not expand generative substrate |
| **GRADED** | Generative wins on some locked slices / axes but not the primary \(U_{R★}\) decision | Partial value only; expand only along winning slices; no blanket architecture claim |
| **SURVIVE** | Best generative strictly beats classical by \(>\delta\) on primary \(U_{R★}\) | Generative value in R★ supported for that regime only; still not NanoScribe product authorization |
| **VOID** | Protocol/data/builder violation, leakage, or undecidable \(U\) | No scientific update; fix protocol; do not interpret as SURVIVE |

This table is precommitted so a future E4 cannot redefine success after seeing results.
It does **not** authorize Stage 4 execution.
