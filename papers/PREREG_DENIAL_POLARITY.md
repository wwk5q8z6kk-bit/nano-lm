# Preregistration — DP-1: denial-polarity correction on the state decision

**Frozen 2026-08-05, before any measurement on the calibration partition.**
Thresholds below are fixed now and will not move after results are seen.

## 1. Origin, and its status

`papers/FINDING_DENIAL_RECOGNITION.md` characterised — on **already-spent
development data**, exploratorily — that 177 of 413 `absent` fields were
labelled `supported`, 176 of them with the span exactly right, and that
`contract.py::_is_field_denial` matches 176 of those 177 while flipping 0 of
3,833 correct `supported`.

That analysis **cannot** support a gate. This preregistration exists to test the
same rule on data not yet spent for this purpose, with criteria fixed in
advance.

## 2. The rule under test

For a field the model proposes as `SUPPORTED` with evidence spans `S`:

```
if any(_is_field_denial(field, s.text) for s in S):
    state := ABSENT          # the span denies rather than asserts
```

Applied **only** to `SUPPORTED` proposals. No other state is rewritten, no span
is changed, no confidence is altered, and the model is not retrained. The rule
reuses `nano_ai/contract.py:93` verbatim — the same detector already used at
`contract.py:313` to validate ABSENT claims.

## 3. Partition and procedure

- **Partition:** calibration (`artifacts/nano_h5/data`, 800 examples / 4,000
  fields). Development is spent and will not be re-opened.
- **Checkpoint:** `seed-20260805/epoch-2` (H6's frozen selector choice).
- **Inference:** the loader path in
  `nano_ai/training/run_threshold_sweep.py::_load` — same loader, encoder, and
  shared inference authority the trainer uses.
- **Device:** CPU. **Cost:** $0.
- Rule applied post-hoc to stored proposals; nothing in the training or
  evaluation authority is modified.

## 4. Preregistered criteria

Let `absent_before` / `absent_after` be joint-exact counts on gold-`absent`
fields, and `supported_before` / `supported_after` on gold-`supported`.

| # | metric | requirement |
|---|---|---|
| **C1** | absent recovery | `absent_after ≥ absent_before + 0.60 × (mislabelled-as-supported)` |
| **C2** | supported regression | `supported_after ≥ supported_before` — **exactly zero loss** |
| **C3** | no other state degrades | `conflicting`, `uncertain`, `missing` joint counts unchanged |
| **C4** | rule specificity | of all fields the rule rewrites, ≥ 90% have gold state `absent` |
| **C5** | suite | `nano_ai/tests` and `fabric` green |

**ACCEPT** iff all five hold.

C2 is deliberately absolute. A rule that buys `absent` accuracy by damaging the
3,837-field majority class is not a fix, however good its headline looks — the
same asymmetry `PREREG_ABSTENTION_W1.md` established for correct abstentions.

C4 exists because C1 and C2 could both pass while the rule fires widely and
harmlessly on states we are not measuring; specificity says the rule does what
it claims rather than what merely scores.

C1 is set at 60% of the recoverable population rather than the 99.4% seen in
exploration, because that 99.4% is precisely the number most likely to be
optimistic on spent data. Clearing a materially lower bar on fresh data is the
honest test.

## 5. What acceptance would and would not license

**Would:** adopting the rule in the decision path under its own change record,
and treating denial polarity as a solved sub-problem of the state decision.

**Would not:** any claim that H6's gate is cleared. The exploratory estimate put
absent at 375/413 against a requirement of 383 — **still short by 8** — and this
preregistration measures a different partition, so it speaks to the mechanism,
not to H6's verdict, which stands as REJECTED.

**Would not:** any claim of domain generality. `_DENIAL_PATTERNS` are v0 and
tuned to synthetic clinic phrasing. Real documents deny in far more ways, and
C2's zero-loss result should not be expected to survive a corpus change.

## 6. Falsification

If C1 fails on calibration while it held on development, the exploratory result
was partition-specific — likely overfitting of the v0 patterns to development
phrasing. That outcome is recorded, the rule is rejected, and the diagnosis
moves to span semantics per `papers/ENHANCED_PLAN_20260805.md` §6.

If C2 fails, the rule is rejected outright regardless of C1.
