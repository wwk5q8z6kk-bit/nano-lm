# RESULT — delimitation clause CONFIRMED on fresh instances

**Prereg:** `artifacts/PREREG-delimit-confirmatory.md`, committed at `1e064bd`. The same
commit contains the fresh data, and both precede any run.
**Cells:** P1 (length-matched placebo) and D (shortest-span clause). Each covers 36
instances × 16 slots = 576 slots. Greedy decoding, local MPS, $0.

## Validity gates, both passed

- **Determinism:** the i0–i11 slots in the new runs reproduce the earlier `cal-P1` and
  `cal-D` runs exactly: raw line and joint cell match, with **0 mismatches across 384
  slots**. The expansion changed nothing for the historical instances. The i0–i11 prompt
  hash was also identical before and after the change.
- **Credit guard:** both cells are eligible on all three axes. Format compliance is
  576/576.

## Primary endpoint — fresh instances i12–i35 only

D vs P1, paired per slot, exact two-sided McNemar:

| axis | slots | D-only | P1-only | p | verdict |
|---|---|---|---|---|---|
| **grounded** | 384 | **15** | **1** | **5.2 × 10⁻⁴** | ✓ primary met |
| selection (located) | 240 | 16 | 18 | 0.86 | no degradation |
| correct abstention | 384 | 0 | 3 | 0.25 | no *significant* degradation (see below) |
| unbound assertion | 384 | 14 | 110 | 1.2 × 10⁻¹⁹ | improved |

**CONFIRMED** by the pre-registered rule: grounded rises at p < 0.05, and neither
selection nor abstention degrades significantly.

This is the **first confirmed eligible improvement on the delimitation axis** in the
program. The test used data that did not exist when the hypothesis was first tested,
against a length-matched control, with the credit guard satisfied.

## Descriptive rates (fresh)

| | P1 placebo | D |
|---|---|---|
| grounded / gold-present | 3/240 = 1.2% | **17/240 = 7.1%** |
| delimitation given location | 1.6% | **8.9%** |
| selection | 80.4% | 79.6% |
| abstention on absent | 16.7% | 14.6% |
| unbound / asserted | 40.6% | **14.4%** |

The fresh estimate replicates the first run closely: delimitation given location was 7.7%
there and is 8.9% here.

## What the confirmation does not change

- **Abstention leans the wrong way.** The split is 0 to 3 against D. That is not
  significant, and the rule was written as "no significant degradation," so the verdict
  holds. The direction still goes on record, because it is the same axis where every
  added clause has pushed the model toward committing.
- **"No degradation" means not detected at this n, not shown to be absent.** At 0 vs 3 discordant slots the test only fails at around 6 or more in one direction. So the abstention clause passes because the signal is small, not because it was proven clean.
- **The strength of the claim is capped as pre-registered.** The clause says almost
  exactly what the metric scores. What is confirmed is that the model can delimit when
  asked, which makes over-extension a default behaviour. It is not confirmed that the
  clause teaches delimitation.
- **The absolute level is still low.** Delimitation given location is about 9%, up from
  about 2%. Most located evidence is still returned at the wrong extent. The axis moves,
  but it is far from solved.
- **Unbound assertions** fall from 40.6% to 14.4% again. That makes three independent
  measurements consistent with the hypothesis that ungrounded assertion partly reflects
  default verbosity.

## Also fixed in this round

The stem-level label-disjointness invariant had been documented and implemented
(`stems_overlap`) but was never called by any test. It let through three real
collisions in the new data. The data was corrected, and `ConceptLabelStemTest` now
enforces the invariant.

## Provenance

| cell | run |
|---|---|
| P1, 36 instances | `5242ec03-1025-4595-8092-2d174d9eadfb` |
| D, 36 instances | `1dd04c3a-f46c-4ffd-b991-b7cf0319c281` |

Per-slot exports are in `artifacts/slots/slots-conf-{P1,D}.json`, with SHA256SUMS
refreshed.
