# PREREG — delimitation clause, confirmatory re-test on fresh instances

**Committed in the same commit as the 24 new instances, before any cell runs.**

## Why a new prereg

The first D-vs-P1 test (`RESULT-delimit-clause.md`) came out 5 to 0 in favour of D on
grounded, p = 0.0625. That misses the pre-registered 0.05 threshold. The result is
**NOT CONFIRMED** and stays that way.

Extending instances i0–i11 and re-testing the pool would be optional stopping, because
those 12 instances are already known to favour D by 5 to 0. **The confirmatory test uses
fresh instances only.**

## Fresh set

`FRESH_CONFIRMATORY_IDS` = **i12–i35** (24 instances, 16 slots each = 384 slots). It is
declared in `nanoscribe/campaign_instances.py`, so the list cannot drift away from this
document.

These values were authored after the first result had been seen, **without inspecting
which slots flipped**. Only aggregate counts had been printed. All invariants are
enforced by tests: exactly one occurrence per present value, zero occurrences per absent
value, no reserved values, and label disjointness at both substring and stem level. The
stem-level test is new in this commit; it was documented earlier but never actually
asserted.

The prompts for i0–i11 were hashed across all 16 flag combinations before and after the
expansion. The hashes are **identical** (`ab3d99fc…`), so the historical cells remain
comparable.

## Power, stated up front

The observed rate is about 5 discordant slots per 12 instances. At that rate, 24 fresh
instances give about **10 expected discordant slots**. An exact two-sided McNemar test
needs at least 6 discordant slots, all in one direction, to reach p < 0.05. With a split
near 9 to 1 it clears comfortably. With 12 fresh instances the expected count would have
been about 5, which is the same near-miss as before, so 24 were authored.

## Metric semantics (Amendment 1 rule)

`grounded` means a slot whose `joint_cell == "asserted_grounded"`: the model asserted,
its quote binds, and the span and state are exactly correct. For D this is the
end-to-end measure of delimitation success.

## Primary endpoint

**Grounded, D versus P1, exact two-sided McNemar, restricted to instance IDs i12–i35.**

- **CONFIRMED** if p < 0.05 and D wins more discordant slots, **and** selection (located)
  and correct abstention on the same fresh slots show no significant degradation
  (p ≥ 0.05, or a movement in D's favour).
- **MIGRATION** if grounded rises but selection or abstention falls significantly.
- **NOT CONFIRMED** otherwise, including a correct direction with p ≥ 0.05.

## Secondary (descriptive only, not confirmatory)

- Results pooled over all 36 instances.
- A determinism check: the i0–i11 slots in the new runs must reproduce the earlier
  `cal-P1` and `cal-D` runs slot for slot. Greedy decoding and identical prompts should
  make them byte-identical. A mismatch invalidates the run.

## Cells

The same flags as before, re-run on the 36-instance suite:
- P1: `PLACEBO_CLAUSES=1`, D/V/S off, leak channels off.
- D: `DELIMIT_CHECK=True`, `PLACEBO_CLAUSES=0`, V/S off, leak channels off.

Both flag sets are read back from git before launch.
