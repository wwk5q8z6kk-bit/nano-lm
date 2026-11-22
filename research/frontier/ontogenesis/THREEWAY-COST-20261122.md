# The three-way router's stable-rule cost has two sources (2026-11-22)

`PROTOCOL-THREEWAY-COST.md`, predictions C1–C5 and scorer committed in `a50fc4e` before any confirmatory run; development exploration on seed 1 is recorded there. Seeds 7401–7405 (new). Raw files: `results/threeway_cost/`; scored output: `results/threeway_cost/verdict.json`. The masking diagnostic (private read removed for unseen keys, which no private write ever targeted) was proposed by the memory-line session. [Observed]

| Arm | Stable rule | Shared table only | Old entities revised | Exceptions kept |
|---|---|---|---|---|
| No routing | 81.7% | 82.2% | 18% | 37% |
| Oracle | 66.5% | 69.4% | 98% | 99.9% |
| Oracle, private table ×16 | 69.2% | 69.4% | 98% | 99.9% |
| Oracle, world without rule changes | 76.9% | 81.5% | — | 99.9% |
| Three-way | 71.8% | 74.3% | 71% | 89% |
| Three-way, private table ×16 | 73.6% | 73.8% | 73% | 87% |

## Against the registered predictions

- C1 (private leakage: in the oracle, shared-only minus full ≥ 0.02): **held**, 0.029, on 5 of 5 seeds.
- C2 (a ×16 private table removes it, gap ≤ 0.01): **held**, 0.002, on 5 of 5 seeds.
- C3 (the shared table carries a cost: oracle shared-only ≥ 0.08 below no routing): **held**, 0.128, on 5 of 5 seeds.
- C4 (that cost needs rule changes: without changes, shared-only within 0.02 of no routing): **held**, 0.007, on 5 of 5 seeds.
- C5 (a ×16 table lifts the router's stable rule by ≥ 0.02): **failed narrowly**, +0.019, on 3 of 5 seeds.

## What this shows

The stable-rule cost found in `THREEWAY-20261122.md` has two separable sources.
1. **Collision leakage in the private table, about 3 points.**
   - Spread private addresses put each fact on 8 of 16,384 slots, so strong private writes leak into other facts' reads.
   - A table 16 times larger removes it completely.
   - It is an engineering term, fixable by table size, at a memory cost.
2. **Collateral from writing rule changes into the shared table, about 13 points.**
   - Type-wide revisions spill into neighbouring rules that share typed slots.
   - Without rule changes it is absent. This is the same mechanism as the memory-line session's write-pass result (a strong shared write costs later inference, W4), at type scale.
   - This is the main cost and the open problem.

Even the oracle, with perfect labels, pays both. The cost is therefore a property of where and how strongly revisions are written in this memory, not of detector errors. That supports `nano-lm-ad`'s reading in the ledger.

## Also observed

On these seeds the router revised 71% of old entities, against 86% on the registered seeds 7301–7305 of the earlier study. That is a large difference between two five-seed sets drawn from the same world. Seed-to-seed variance in this router is larger than its five-seed means suggest; treat its revision figure as roughly 70–86%.

## Next

Reduce the shared-revision collateral without losing revision:
- a narrower write, revising only the slots unique to the changed rule;
- fewer steps with a larger learning rate;
- typed addresses with less overlap between rules.

The study's outcome is the trade-off curve between revision and stable-rule cost.

## Pooled over all ten seeds (7301–7305 and 7401–7405), at the ontogenesis session's request

Same code and settings in both sets. Each cell shows the mean, with the minimum and maximum over the 10 seeds in brackets.

| Arm | Old entities revised | New entities, 1–5 phases | Exceptions kept | Noise resisted | Stable rule |
|---|---|---|---|---|---|
| none | 17% [11%–27%] | 31% [19%–44%] | 35% [29%–42%] | 92% [90%–95%] | 82% [81%–85%] |
| oracle | 98% [91%–100%] | 96% [87%–99%] | 100% [100%–100%] | 89% [85%–92%] | 68% [64%–72%] |
| threeway | 78% [58%–91%] | 79% [65%–91%] | 89% [84%–92%] | 91% [88%–95%] | 72% [70%–75%] |

The router's revision of old entities ranges widely across seeds, so its headline figure should be read with that range, not as a single number.
