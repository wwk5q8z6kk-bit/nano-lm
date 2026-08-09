# Result — model/rule disagreement is a usable escalation signal

**2026-08-06.** Subtask 15 / plan P3. EXPLORATORY: selects nothing, gates
nothing, trains nothing. Artifact `artifacts/nano_h6/analysis/disagreement.json`.

## The question

`RESULT_DP1_AND_THE_VOCABULARY_CEILING.md` showed `contract._is_field_denial` is
high-precision and low-recall: 4/4 correct firings on calibration, 0 of 3,833
correct `supported` flipped, but only ~3% of denial phrasings from two
independent MIT-licensed clinical lexicons recognised. That shape is wrong for a
decision rule. `PLAN_20260805_SURFACE_ROBUSTNESS.md` §3 proposed it might be
right for an **escalation signal** instead — routing on *disagreement* between
the model's state and the rule's verdict rather than on the rule's output.

That proposal had no evidence. This is the test, and it was specified to be able
to kill the idea: if disagreement does not predict error, the router dies here.

## Result — it replicates on both seeds

Per field: `rule_fires` = any span the model returned is a recognised denial;
`model_says` = the model proposed ABSENT; disagreement = the two differ.

| seed | agree n | agree error | disagree n | disagree error | lift |
|---|---|---|---|---|---|
| 20260805 | 4,784 | 18.7% | **216** | **94.0%** | **5.02×** |
| 20260806 | 4,726 | 22.8% | **274** | **98.9%** | **4.34×** |

Read as a router, with coverage reported beside accuracy per
`papers/SELECTIVE_VOCABULARY.md`:

| seed | fields flagged | flag precision | error recall |
|---|---|---|---|
| 20260805 | 216 / 5,000 = **4.3%** | 203/216 = **94.0%** | 203/1,099 = **18.5%** |
| 20260806 | 274 / 5,000 = **5.5%** | 271/274 = **98.9%** | 271/1,348 = **20.1%** |

**Escalating ~5% of fields captures ~19% of all errors, and roughly 19 in 20 of
those escalations are genuinely wrong.** For a system whose product claim is
trustworthiness, a 5%-coverage escalation that is ~95% precise is a good trade —
it is cheap to act on and it almost never wastes attention.

Both seeds clear `MIN_DISAGREEMENTS_FOR_CLAIM = 30` by an order of magnitude, so
unlike DP-1's C1 this ratio is not computed over a denominator the data drove to
nothing. That threshold was fixed in the probe before it was run.

## What is inside the flagged set

| seed | gold states among flagged fields |
|---|---|
| 20260805 | absent 199, conflicting 13, uncertain 4 |
| 20260806 | absent 197, conflicting 3, **supported 43**, uncertain 31 |

Mostly `absent`, as designed — the rule only knows denials. But the seed-06
column is the caution: 43 gold-`supported` and 31 gold-`uncertain` fields also
disagreed there and were almost all wrong. The signal is not purely an
absent-detector; what it really flags is *the model's state decision being
unreliable on this field*, which is more useful and less predictable.

## Honest limits

- **The rule's recall bounds the router's recall.** It cannot flag a denial it
  does not recognise, and it recognises ~3% of independent clinical phrasings.
  On real documents the ~19% error recall should be expected to fall sharply.
  The precision may survive; the coverage will not.
- **Two seeds, and they differ in composition** (seed 06 flags `supported` and
  `uncertain` fields that seed 05 does not). The aggregate replicates; the
  breakdown does not, consistent with the seed instability measured in
  `RESULT_SURFACE_HARNESS_RUN1.md`.
- **Synthetic clinic dialogue only.** No claim about real documents.
- **This is a measurement, not a shipped router.** Wiring it into the decision
  path needs its own preregistration, and that preregistration must denominate
  its gate in something neither the model nor the partition can drive to zero.

## Verdict

The router survives its falsification test. It is retained in the plan as P3,
now evidence-backed rather than assumed — and notably it is the *only* proposed
use of `_is_field_denial` that its 3%-recall/near-100%-precision shape actually
fits. Promoting the same rule into the decision path remains rejected.
