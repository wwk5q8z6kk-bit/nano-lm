# NanoScribe Architecture Gap Analysis

*Do not treat the fabric vertical slice as the cognitive architecture.*

## Program gate

Architecture expansion is explicitly STOP after E1 KILL (`papers/NANOSCRIBE_VNEXT.md`, `EMPIRICAL_FOUNDATION.md`, `MASTER_PLAN.md` override). vNext is an architecture sketch, not an active build queue.

## Element audit

| Element | Status | Required now? |
|---------|--------|---------------|
| Control kernel | conceptual / specified | No — unnecessary until product path survives E4 |
| Factorized state S=I×K×R×M×P×V×E | specified | No (useful later if product) |
| Intent management | conceptual | No |
| Task DAG | specified / Next | No |
| Capability routing | specified | No (E4 GRADED might need later) |
| Context compiler | specified | No |
| External / symbolic tools | conceptual | No |
| Permissions / write auth | specified (invariant 6) | No for science track |
| Memory classes | specified | No |
| Validated memory writes | specified | No |
| Relational / graph memory | conceptual | Speculative |
| Contradiction graph | local only (per-claim CONTRADICTED) | Partial already in fabric |
| Distributed workers | specified future | Speculative |
| Module registry | absent | Speculative |
| Recovery semantics | specified | Speculative |
| Human review UI | absent (stats only) | No for current science |
| Observability / dashboard | minimal JSONL+JSON | Partial / later |
| Storage (SQLite evidence graph) | Next only | Speculative |
| Release / deploy product | blocked by E1 | Not now |

## What exists vs what was sold

| Believed (from plans) | Actual |
|-----------------------|--------|
| Intent→Control→Generator→Ledger→Memory | Generator→Claims→Verifier→JSONL |
| Typed memory + authorization | No memory module |
| Calibrated risk controller | Fixed if/else |
| Append-only evidence DB | Truncating JSONL files |
| Scalable cognitive system | ~470-line regression harness + docs |

## Disposition

- Genuinely required now for research claims: none of the missing architecture — empirical track + fabric regression harness suffice.
- Useful later: only if E4 SURVIVE/GRADED licenses a product wedge.
- Speculative / remove from "exists" language: kernel, DAG, compiler, distributed workers, UI, graph memory.
