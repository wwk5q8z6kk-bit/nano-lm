# Address concentration dose-response: results

Run 2026-11-21 under [PROTOCOL-MEMSAT-SHARING.md](PROTOCOL-MEMSAT-SHARING.md), committed in `c492b0e` before any confirmatory run. Seeds 3101–3105, Adam reset each phase in every arm. Raw files: `results/memory_saturation_sharing/`. Scored by `memory_saturation_sharing_score.py`. M = 0 is reused from the factorial, as registered. [Observed]

## Late acquisition by dose

M is the number of each query's 8 slots replaced with uniformly random slots; read weights kept.

| M | 3101 | 3102 | 3103 | 3104 | 3105 | Mean acquisition | Mean late retention | Effective slots, range over seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.547 | 0.630 | 0.640 | 0.665 | 0.694 | 0.635 | 0.600 | 212–313 |
| 2 | 0.842 | 0.913 | 0.859 | 0.886 | 0.924 | 0.885 | 0.846 | 369–549 |
| 4 | 0.971 | 0.995 | 0.969 | 0.990 | 0.995 | 0.984 | 0.955 | 799–1,166 |
| 6 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.987 | 2,599–3,659 |
| 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.996 | 12,977–13,045 |

## Against the registered predictions

- G1 (acquisition does not fall as M rises): **held**, five of five seeds.
- G2 (random slots with the original read weights reach at least 0.95): **held**, all 1.000. The non-uniform read weights are not the cause of the gap.
- G3 (graded response: M = 4 at least 0.05 from both ends): **failed**, none of five. M = 4 is 0.30–0.42 above M = 0 but within 0.04 of M = 8. The response saturates early.
- G4 (descriptive): at M = 0 the seed with the fewest effective slots (3101, 212) has the lowest acquisition and the seed with the most (3105, 313) the highest. The middle three are not in the same order (3103 has fewer effective slots than 3102 and slightly higher acquisition). Five points; no rule was set.

## Reading, by the outcomes the protocol named

G1 and G2 held, so the gap that remains without carried optimiser state is a property of where the projection sends queries. G3 failed in the direction the protocol called "M = 2 is already near the ceiling": spreading a quarter of each query's reads recovers 25 of the 36 missing points, and spreading half recovers nearly all of them. Retention moves with acquisition, from 0.60 to 0.85 to 0.96.

Together with the factorial, the acquisition collapse in this learner now has an account that holds on five seeds:

1. Queries concentrate on a few hundred effective slots out of 16,384.
2. Carried Adam state on those slots turns that congestion into near-total failure (0.02).
3. Removing the carried state recovers about two thirds (0.64). Decongesting half of each query's reads recovers the rest (0.98).

## Not predicted

- Effective slot count tracks late acquisition across seeds and doses on one scale. Seed 3001 at M = 0 has 526 effective slots and reached 0.861; the five confirmatory seeds at M = 2 have 369–549 and reached 0.842–0.924. So the one-seed figure of 86.1% in the original diagnostic was high because seed 3001 drew much less concentrated addresses (526 effective slots against 212–313), not because of noise in training. This was seen after the fact and is a hypothesis: effective slot count predicts acquisition regardless of how the concentration arises.
- `address_stats.jsonl` holds the statistics for every seed and dose.

## Deviations and limits

- No run was interrupted or restarted. All twenty finished on the first attempt.
- At M > 0 the addresses are a mixture of projection and random slots, not the original addresses with less sharing. The dose manipulates concentration; it does not isolate it.
- Random replacement destroys any similarity structure in the addresses. This task has arbitrary facts and cannot show what that would cost a learner that generalises through its addresses. That is the next thing to measure before recommending partial hashing anywhere else: `cab_s.py` has rules to generalise and is the place to test it.
- Same toy as before; an exact lookup table solves it. Not evidence about a language model.
- Thresholds were set after the factorial. G3's failure is a statement about the shape I guessed, not about the effect, which is large.
- The laptop was very heavily loaded; timings are not measurements.
- Pairwise overlap between same-relation queries is small at every dose (0.10–0.12 of 8 slots at M = 0 on these seeds). The earlier phrase "share 0.60 of their 8 slots" is a count of 0.6 slots, as the protocol records.
