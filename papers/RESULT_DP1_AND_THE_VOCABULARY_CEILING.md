# Result — DP-1, and the vocabulary ceiling it exposed

**2026-08-05.** Reports `papers/PREREG_DENIAL_POLARITY.md` against its frozen
criteria, then the investigation the result forced.

**Verdict: DP-1 passes all five criteria, and the pass is vacuous.** The
follow-on finding matters far more than the rule did.

---

## 1. DP-1 against the frozen criteria

Checkpoint `seed-20260805/epoch-2`, calibration partition (800 examples / 4,000
fields), CPU, $0. Artifact: `artifacts/nano_h6/analysis/denial_probe_calibration.json`.

| # | criterion | required | measured | |
|---|---|---|---|---|
| C1 | absent recovery | ≥ 318.4 | **320** / 330 | pass |
| C2 | supported regression | ≥ 2944 | **2944** / 3070, 0 false flips | pass |
| C3 | other states unchanged | equal | conflicting 175, uncertain 173, missing 200 — all equal | pass |
| C4 | rule specificity | ≥ 90% | **100%** (4 of 4 firings gold-absent) | pass |
| C5 | suite | green | green | pass |

**ACCEPT** on the letter of the preregistration. Overall joint accuracy moved
3808 → 3812 of 4,000.

### Why the pass carries almost no information

`absent` was **already 316/330 = 95.8% correct before the rule ran.** Only four
fields on the whole partition were recoverable. The rule fired four times, was
right four times, and moved the needle by four.

C1's bar was `absent_before + 0.60 × recoverable`. With `recoverable = 4` the
bar was 318.4 — and had `recoverable` been 0, the bar would have collapsed to
`after ≥ before`, which a rule that does nothing satisfies.

**I preregistered a threshold denominated in a quantity that the data can drive
to zero.** That is precisely the degeneracy this project documented at
`fabric/slice.py:247` (`presented_err / max(1, presented)` — abstain on
everything, score 0.0%, pass the gate) and wrote a plan track to fix. I
reproduced it in my own gate, one day later, while holding the finding in hand.

The lesson generalizes past this instance and is the one worth keeping: *a
criterion must be denominated in a quantity neither the system nor the data
partition controls.* C1 should have been stated as an absolute floor on
`absent_after / absent_total` **plus** a minimum `recoverable` for the test to
count as informative at all — a data-sufficiency gate (`rules/math-toolkit.md`
§7). Without the second clause the first is unfalsifiable.

**No threshold was moved after seeing results.** DP-1 is recorded as passing,
and separately as uninformative. Both are true and the second is the useful one.

---

## 2. The contradiction that forced the investigation

The same checkpoint scores `absent` at **95.8% on calibration** and **48.2% on
development**. A 47.6-point spread on one model cannot be a property of the
model. It is a property of the partitions.

The development manifest states the cause in its own metadata:

```json
"isolation": {
  "denial_phrases_disjoint": true,
  "uncertainty_phrases_disjoint": true,
  "open_value_lexicons_disjoint": true,
  "answer_templates_disjoint": true,
  "question_templates_disjoint": true,
  "transcripts_disjoint": true,
  "worlds_disjoint": true
}
```

Development is not a random held-out split. It is a **deliberate lexical
generalization stress test**, and the H6 gate was measuring generalization all
along without anything in the plan naming it as such.

### Per-state, in-distribution vs held-out

`artifacts/nano_h6/analysis/generalization_gap.json`, one checkpoint, joint
(state **and** spans exact) on both sides:

| state | cue it depends on | in-distribution | held-out | gap |
|---|---|---|---|---|
| supported | open value | 2944/3070 = 95.9% | 3119/3837 = 81.3% | −14.6 |
| **absent** | **denial phrase** | **316/330 = 95.8%** | **199/413 = 48.2%** | **−47.6** |
| missing | *nothing to recognize* | 200/200 = 100% | 250/250 = 100% | **0.0** |
| uncertain | hedge phrase | 173/200 = 86.5% | 190/250 = 76.0% | −10.5 |
| conflicting | two-span structure | 175/200 = 87.5% | 143/250 = 57.2% | −30.3 |

The reproduction is exact: 199/413 matches H6's sealed `absence` gate figure,
and the 177 supported-mislabels match `FINDING_DENIAL_RECOGNITION.md`. The
pipeline is faithful.

**`missing` is the only state that does not degrade — and the only state whose
decision requires recognizing no surface form at all** (`∄ mention` → empty
evidence). Every state that must locate and read a span loses accuracy when the
vocabulary changes. That is a cleaner account than "accuracy tracks logical
complexity" (`ENHANCED_PLAN_20260805.md` §2): complexity and lexical dependence
were confounded, and this table separates them.

### The vocabulary is ten strings

| partition | allergy denials | medication denials |
|---|---|---|
| calibration (training distribution) | `I denied any allergy.` · `I deny allergies.` · `No allergies!` · `Not that I know of!` | `I denied taking medicine.` · `I deny taking medications.` · `No nothing yet!` · `No, nothing.` |
| development (held out) | `None whatsoever.` | `Nothing at all.` |

Four phrasings per field to learn from, one novel phrasing per field to be
tested on. The model's `absent` failure is: **shown two synonyms it had not
seen, it read them as assertions of a value.**

---

## 3. The rule is not the answer either

The obvious response — "use the deterministic detector, it recovered 176/177" —
required its own test, because `contract._DENIAL_PATTERNS` was hand-authored by
someone with access to the generator's vocabulary. Its coverage of the closed
pools may be enumeration rather than generalization.

**Provenance check:** `nano_ai/contract.py` was committed `a296f6f` on
2026-08-04 and **has never been modified since**. The patterns were not tuned
after H6's development results (produced 2026-08-05). So the comparison is not
retrofitted — but the author did have the generator's vocabulary in view when
writing them, so closed-pool coverage still proves little.

Three measurements settle it:

| test | denial phrasings | rule recognizes |
|---|---|---|
| both synthetic pools (all 743 gold denial spans) | 10 distinct | **743/743 = 100%** |
| realistic phrasings, hand-written, from neither pool | 24 | **0/24 = 0%** |
| **negspacy `en_clinical` triggers** (MIT, external) | 68 triggers exercised | **2/68 = 2.9%** |

The external inventory is `data/external/negspacy/en_clinical_termset.json`
(negspacy 1.1.0, MIT, upstream sha256 `1b3b8dc…`, vendored with provenance,
evaluation-only, never used for training). It was written years earlier by
people with no knowledge of this project. Only `no` and `denied` are recognized.

What it misses, verbatim:

```
MISS   'Denies medications.'          <- the most common phrasing in a clinical note
MISS   'Negative for allergies.'
MISS   'No known drug allergies.'     <- NKDA, the standard abbreviation's expansion
MISS   'Without medications.'
MISS   'No signs of allergies.'
MISS   'I have no allergies.'         <- differs from trained 'No allergies!' by a prefix
```

The last one is diagnostic of the mechanism: `_is_field_denial` uses
`pattern.fullmatch(text)`, so any denial embedded in a longer utterance fails.
The rule is a closed list matched end-to-end, not a polarity judgement.

---

## 4. What this means

**Neither component has learned polarity. Both have learned a list.**

- The model memorized 8 phrases and generalized to 2 unseen ones at 48%.
- The regex enumerates ~10 phrases and generalizes to independent clinical
  phrasing at 3%.

The consequence is uncomfortable and load-bearing: **the H-cycle has been
optimizing against a concept whose entire surface realization is ten strings.**
H1 through H6 measured, gated, and rejected architectures on a benchmark that
cannot distinguish learning the concept from learning the list. A model that
scored 100% on `absent` here would tell us nothing about whether it recognizes
`Denies medications.`

This does not invalidate the H-cycle's rejections — a rejection on a weak
benchmark is still a rejection, and the copying results stand. It invalidates
the *next* experiment, if that experiment is another architecture measured the
same way. Scaling the model, adding a state head, or adopting the rule into the
decision path would each be tuned against ten strings.

**The binding constraint is the evaluation vocabulary, not the model, and not
the architecture.** That is where the next work goes.

## 5. Honest limits of this result

- The 2.9% external-trigger figure uses utterances assembled by crossing
  external triggers with field nouns in fixed frames; some combinations are
  ungrammatical (`I absence of allergies.`), which is why the frame-independent
  trigger-level number is reported rather than the raw 14/2726 = 0.5%.
- The hand-written 24 phrasings are my judgement of what a patient would say,
  not a sampled corpus. They corroborate the external number; they are not
  independent of me.
- `conflicting`'s 30-point gap is **not** explained by lexical disjointness —
  its cue is structural. Its span accuracy (0.572) is a separate, genuine
  retrieval problem that none of this addresses.
- Everything here remains synthetic clinic dialogue. No claim is made about real
  documents.
