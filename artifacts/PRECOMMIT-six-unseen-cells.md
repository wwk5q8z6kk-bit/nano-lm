# Pre-commitment — predicted pattern for the six unseen cells

**Written 2026-08-25 while all six runs were `running` and none had been read.**
Two cells had been read at the time of writing (L000, L010); the other six had
not. Verified immediately before writing: `orx runs` showed L111, L011, L101,
L001, L110, L100 all `running`.

This exists so the interaction claim is a prediction rather than a description.
If it holds, it was predicted. If it breaks, that is the more interesting result
and this document is what makes it possible to say so honestly.

---

## What was already known

**Measured (real model, Qwen2.5-1.5B, 192 slots each):**

| cell | C1 | C2 | C3 | exact_span | state_ok | correct_abstain | unbound_assert |
|---|---|---|---|---|---|---|---|
| L000 | off | off | off | 16/192 | 66/192 | 32 | 39 |
| L010 | off | on | off | 18/192 | 66/192 | 32 | 39 |

Perfect-reader ceiling 120/192. Parrot floor in these two cells: 0/192.

**Parrot floor across all eight cells** (a fact about a copier, measured before
any model ran, reproduced digit-for-digit by two independent implementations):

| cell | C1 | C2 | C3 | parrot exact | parrot abstain |
|---|---|---|---|---|---|
| L111 | on | on | on | 109/192 | 0/72 |
| L101 | on | off | on | 109/192 | 0/72 |
| L110 | on | on | off | 109/192 | 0/72 |
| L100 | on | off | off | 109/192 | 0/72 |
| L011 | off | on | on | 108/192 | 36/72 |
| L001 | off | off | on | 108/192 | 36/72 |
| L010 | off | on | off | 0/192 | 72/72 |
| L000 | off | off | off | 0/192 | 72/72 |

The floor collapses **only** when C1 and C3 are both off. Either channel alone
suffices to feed a copier.

---

## The prediction

The substitutability the parrot exposes is a property of the *prompt*, not of
the parrot. A real model that can copy should exploit it the same way. So:

### P1 — super-additive interaction, not two additive main effects

Closing C1 alone or C3 alone should cost little; closing **both** should collapse
the score. Formally, for the C1×C3 sub-design at fixed C2, the interaction

```
I = S(C1on,C3on) − S(C1off,C3on) − S(C1on,C3off) + S(C1off,C3off)
```

is predicted **strongly positive** on `asserted_grounded`, estimated per instance
and averaged over the 12 paired estimates.

### P2 — a bimodal split on exact_span

- **Six leaky cells** (any of C1 or C3 on): `exact_span ≥ 80/192`.
- **Two closed cells** (L000, L010): the only cells below `40/192`. Already
  observed at 16 and 18.

The gap between the two groups should be large and clean, not graded.

### P3 — state_ok rises much less than exact_span

A leaked surface string tells the model *which characters to emit*, not whether
the patient asserted, denied or hedged. Both closed cells sit at `state_ok`
66/192 while `exact_span` is 16/192 — state is already well above span. So:

- `state_ok` in leaky cells: predicted **≤ 95/192**, i.e. a rise of under ~30
  points, materially smaller than the exact_span rise.
- If `state_ok` instead tracks `exact_span` up to near ceiling, P3 is refuted and
  the leak is doing more than surface-form supply.

### P4 — abstention DEGRADES under leakage

The parrot's `correct_abstain` goes 72/72 → 0/72 when C1 turns on, because the
answer template pushes toward STATED on slots whose value is absent. Predicted
for the real model:

- `correct_abstain` **falls** in the four C1-on cells relative to L000's 32.
- `unbound_assert` **rises** in those same cells relative to L000's 39.

This is the prediction I hold most strongly and consider most important: it means
the leak does not merely inflate a span score, it actively **degrades the safety
property** — the instrument was rewarding a model for asserting things that are
not in the source.

### P5 — C2 stays bounded everywhere

`quote_absent` was 2/192 in both cells read so far. Predicted `quote_absent ≤ 6`
in every remaining cell, so the C2 contrast stays below the 3-slot decision
threshold throughout and cannot produce a CONFIRMED verdict on its own.

---

## What would falsify this

- **P1 falsified** if `I ≈ 0` with the per-instance SEM excluding a
  super-additive effect — i.e. C1 and C3 turn out additive rather than
  substitutable for a real model, even though they are substitutes for a copier.
  That would be a genuine difference between copier and model and would mean the
  parrot floor over-predicts real leakage.
- **P2 falsified** by any leaky cell below 80, or any closed cell above 40.
- **P3 falsified** by `state_ok` tracking `exact_span` to near ceiling.
- **P4 falsified** if `correct_abstain` is flat or rises under C1.
- **P5 falsified** by `quote_absent > 6` anywhere.

**Most likely to break:** P2's 80/192 threshold. It is extrapolated from the
parrot's 109, and a real model is not a parrot — it may partly ignore the hint,
or hedge, or produce quotes that fail to bind for reasons a parrot never hits
(the closed cells already show 39 unbound assertions, which the parrot does not
produce there). A leaky cell landing between 40 and 80 would leave P2 in a
genuinely ambiguous band, so it is recorded as a threshold rather than a claim.
