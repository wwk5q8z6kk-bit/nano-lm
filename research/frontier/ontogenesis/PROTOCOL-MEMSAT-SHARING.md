# Protocol: does address concentration explain the gap that remains without carried optimiser state?

Written 2026-11-21, before any confirmatory run. Follows [RESULTS-MEMSAT-FACTORIAL.md](RESULTS-MEMSAT-FACTORIAL.md) (`767857e`), which left this open: with Adam reset each phase the original addresses reach 0.55–0.69 late acquisition while hashed addresses reach 1.00.

## What I had seen when writing this

- All results of the factorial.
- Parity runs on the exploratory seed 3001 only, not results: `memory_saturation_sharing.py measure 3001` gave 4,619 slots addressed, 526.0 effective slots and mean top read weight 0.164 at M = 0, matching the figures recorded in `memory_saturation.py`; `train 0 3001` gave late acquisition 0.8609 and retention 0.8267, matching the recorded reset arm.
- The same measurement at M = 2, 4, 6, 8 on seed 3001: effective slots 901, 1,902, 5,267, 13,002; mean slots shared by two same-relation queries (as a fraction of 8) 0.044, 0.019, 0.005, 0.0006, against 0.076 at M = 0. No training run at M > 0 on any seed.
- A correction to how an earlier figure has been read. "Two queries with the same relation share 0.60 of their 8 slots" in `memory_saturation.py` is a count of 0.60 slots, that is 0.076 of the 8, not 60% of them. The uniform-hash figure beside it, 0.004, is also a count (64 / 16,384). Pairwise overlap is small; the concentration shows in the effective slot count.

## Design

Script `memory_saturation_sharing.py`. Same stream, model, step budget, seeds (3101–3105) and endpoint as the factorial. Optimiser: Adam at 1e-3, reset each phase, in every arm, so carried state is not in play.

Dose: for each query, M of its 8 slots, at random positions, are replaced with uniformly random slots. The query's read weights are kept. M ∈ {2, 4, 6, 8}. The replacement uses its own random stream, so the stream of facts is identical across doses for a seed.

- M = 0 is the original address function. Its five results are reused from the factorial (`original/adam_reset`); the parity run above shows the two scripts agree at M = 0.
- M = 8 keeps only the original read weights on fully random slots.
- The factorial's `hash/adam_reset` (uniform weights, random slots, 1.000 on every seed) is the reference ceiling.

Twenty new training runs. Each result file also records, without training, the slots addressed, effective slot count and mean same-relation and other-relation overlap for its seed and dose.

Primary endpoint: late acquisition, phases 46–60. Secondary: late retention.

## Predictions

"Holds" means on at least four of the five seeds.

- G1. Late acquisition does not fall as M rises: for each seed, each step up in M changes it by no less than −0.02.
- G2. M = 8 reaches at least 0.95. The non-uniform read weights are not the cause of the gap.
- G3. The response is graded, not a step: M = 4 is at least 0.05 above M = 0 and at least 0.05 below M = 8.
- G4, descriptive, no pass rule with five points: across seeds at M = 0, the ordering of effective slot count matches the ordering of late acquisition seen in the factorial (3101 lowest, 3105 highest).

## What each outcome would mean

- G1 and G2 hold: the remaining gap is a property of where the projection sends queries, not of the weights or the optimiser.
- G2 fails: read-weight non-uniformity matters; the "concentration" account is incomplete.
- G3 fails because M = 2 is already near the ceiling: the limit is a small set of heavily loaded slots, and spreading a quarter of each query's reads is enough. That would favour a cheap partial-hash remedy over replacing learned addresses.
- G3 fails because M = 4 is still near M = 0: acquisition needs nearly all reads decongested.
- G1 fails: something other than concentration is going on, and this account is wrong.

## Limits, stated in advance

- Replacing slots changes each query's address, so at M > 0 these are not "the original addresses with less sharing"; they are a mixture. The dose is a manipulation of concentration, not a clean isolation of it.
- Random replacement also destroys whatever similarity structure the projection addresses carry. This task has arbitrary facts and no use for that structure, so the test cannot say what a partial hash would cost a learner that generalises through its addresses.
- Same toy as before: a value table on arbitrary facts that an exact lookup table solves. Not evidence about a language model.
- The laptop is very heavily loaded (load average above 200 at the parity run). Timings are not measurements.
- Thresholds are mine and were set after the factorial. Effect sizes will be tabulated.
- Results are written with exclusive creation under `results/memory_saturation_sharing/`.
