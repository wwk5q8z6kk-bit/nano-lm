# Corrections written into a private slot: results

Run 2026-11-22 under [PROTOCOL-MEMRULE-PRIVATEFIX.md](PROTOCOL-MEMRULE-PRIVATEFIX.md), committed in `9df7ddf` before any confirmatory run. Seeds 4101–4105, two separate value tables (shared typed slots and private random slots, read and summed), learning rate 1e-3, 200 steps per phase on new facts, then a correction-only pass. Last 10 of 40 phases. Raw files: `results/memory_rule_privatefix/`. [Observed]

## Means over five seeds

| Correction pass | Corrections after the pass | Later inference, new entities | Later inference, old entities | Acquisition | Retention | Collateral at writing (own, peers) |
|---|---|---|---|---|---|---|
| private table, 200 steps | 0.029 | 0.735 | 0.731 | 0.892 | 0.820 | 0.000, 0.000 |
| private table, 2,000 steps | 0.891 | 0.714 | 0.712 | 0.887 | 0.881 | 0.000, 0.000 |
| both tables, 2,000 steps | 0.998 | 0.659 | 0.655 | 0.872 | 0.874 | 0.007, 0.001 |

## Against the registered predictions

- P1 (private 2,000-step corrections at least 0.80): **held**, five of five (0.850–0.919).
- P2 (private 2,000-step later inference no more than 0.03 below the 200-step baseline): **held**, five of five (0.017–0.025 below).
- P3 (both-table 2,000-step later inference at least 0.05 below the baseline): **held**, five of five (0.069–0.085 below).
- P4 (private 2,000-step collateral at writing at most 0.01): **held**, five of five (zero on the corrected entities' other facts; at most 0.001 on peers).

## Reading

Writing the correction into a private slot keeps nine in ten corrections and avoids most of the later cost. Inference falls by 0.02 instead of 0.08 here, or 0.13 in the shared-only memory of the write-pass test. It also raises retention, from 0.82 to 0.88, because corrected facts now stay correct. A private correction still needs a larger budget than a normal write, because in the summed read it has to outweigh the shared path's accumulated value. With the normal budget it takes no more often than before (0.03).

Taken with the earlier tests, this is a consistent account of the typed fixed-address memory in this world, on five seeds each:
1. Shared slots carry generalisation, and resist individual corrections.
2. Corrections fail for lack of budget relative to the shared accumulation, not from conflict within a phase or from unreachable values.
3. Forcing them into shared slots works at the moment but costs later generalisation. Forcing them into private slots keeps both.

## What this does not show

- **The edit label is given.** The correction pass knows which facts are corrections and is told to write them privately with a larger budget. A learner that must detect corrections itself faces the routing problem of `INVESTIGATION.md`, Question 8, where a detector with no oracle recovered about a third of the oracle's inference gain.
- **The private table is storage.** It is per-fact slots, close to a hashed exact store, and doubles the value storage. This is an augmented-memory result in the sense of the architecture ledger's three scoreboards, not a gain in what a fixed-size learned model knows.
- Type information is given, not learned. Value tables, frozen addresses, a synthetic world. Not evidence about a language model.
- Thresholds were set after one development seed.
- All fifteen runs finished on the first attempt.
