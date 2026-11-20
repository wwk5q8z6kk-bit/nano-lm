# RESULT — fresh-only promotion: length artifact CONFIRMED; verbatim is MIGRATION, not a clean gain

**Prereg:** `artifacts/PREREG-fresh-promotion.md` (`52d8248`), committed before either run.
**Method:** fresh instances i12–i35 only, exact two-sided McNemar, with `conf-P1` as the
shared control.
**Gates passed:** determinism is 0/192 mismatches for both P0 and V against the earlier
cells; the credit guard is eligible on all three axes; format compliance is 576/576.

## Claim 1 — prompt length raises commission: CONFIRMED

| | commission on fresh absent slots |
|---|---|
| P0 (no clause, 39 words) | 99/144 = 68.8% |
| P1 (placebo clause, 56 words) | 120/144 = 83.3% |

P1-only 21, P0-only 0, **p = 9.5 × 10⁻⁷**. Adding one non-instructing sentence makes the
model commit on absent slots more often, and every discordant slot moves in that
direction. This was exploratory on i0–i11; it is now confirmed on fresh data. **Every
arm needs a length-matched control.**

## Claim 2 — the verbatim clause: both endpoints confirmed, migration check FAILED

| axis | V-only | P1-only | p | |
|---|---|---|---|---|
| unbound assertion | 11 | 100 | 4.1 × 10⁻¹⁹ | improved ✓ |
| selection (located) | 18 | 0 | 7.6 × 10⁻⁶ | improved ✓ |
| grounded | 2 | 0 | 0.5 | flat |
| **correct abstention** | **0** | **6** | **0.031** | **degraded ✗** |

Selection rose from 80.4% to 87.9%, and unbound assertion fell from 40.6% to 15.6%.
Both replicate the exploratory result.

**However, the pre-registered migration check fails.** On fresh data, V gives up 6
correct abstentions and wins none back, p = 0.031. On i0–i11 abstention was exactly
flat, at 16.7% versus 16.7%. That earlier null came from too few slots to detect this
effect.

**Verdict: MIGRATION.** V improves selection and faithfulness of quotation, and pays for
it partly on abstention. This is the pattern the research question asks about, so it
cannot be reported as a clean gain.

The same direction showed up in the D round: 0 vs 3 against D, not significant. Every
single clause tested on fresh data has cost abstention, and one of them significantly.
This is **inferred, not yet measured as a pooled effect**, but it points to a general
cost of clause-appending on this model's willingness to decline, separate from the
length effect.

## What changes

- `ANALYSIS-verbatim-three-axis.md` said verbatim left abstention "exactly unchanged"
  and called it the first affirmative answer to the research question. That is
  superseded. On confirmatory data V migrates.
- **D remains the only arm confirmed without significant migration.** It has the same
  caveat: not detected at this n.

## Provenance

| cell | run |
|---|---|
| P0, 36 instances | `ae13b37e-e381-4a0e-beca-c8daacb1b301` |
| V, 36 instances | `19232762-e0c4-41e5-9244-f4f366327ed4` |
