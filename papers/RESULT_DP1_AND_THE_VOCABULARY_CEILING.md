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
evidence). That single observation is what the table supports, and it is worth
having: it shows the degradation is tied to reading surface forms rather than to
the state machinery.

It does **not** support a general law. `conflicting` drops 30.3 points with no
disjoint phrase pool at all — its cue is structural — so the four non-zero gaps
do not rank by anything measured here. `ENHANCED_PLAN_20260805.md` §2 read this
ordering as "accuracy tracks logical complexity"; complexity and lexical
dependence are confounded in these numbers and neither reading is established.

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

## 4. The controlled experiment — and the claim it refuted

The three measurements above tempt an obvious conclusion: *neither component
learned polarity, both learned a list.* That conclusion rests on **two** held-out
strings for the model against **68** external triggers for the rule, and the
two behave very differently (allergy 33.5%, medication 62.0%). n=2 cannot carry
it. So I measured the model on the same independent vocabulary.

**Design.** Hold the development transcripts, worlds, fields, and gold structure
fixed. Substitute *only* the denial phrase, in both transcript and target. Any
difference is then attributable to the surface form alone.
`artifacts/nano_h6/analysis/lexical_substitution.json`, 413 gold-absent fields
per arm, same frozen `epoch-2` checkpoint.

| arm | denial phrasing | absent accuracy | rule recognises |
|---|---|---|---|
| TRAIN[2] | `I deny taking medications.` / `I deny allergies.` | **99.3%** | yes |
| TRAIN[3] | `I denied taking medicine.` / `I denied any allergy.` | 99.0% | yes |
| TRAIN[1] | `No nothing yet!` / `Not that I know of!` | 93.2% | yes |
| TRAIN[0] | `No, nothing.` / `No allergies!` | 92.2% | yes |
| EXTERNAL[6] | `I cannot take medications.` / `No signs of allergies.` | 93.2% | **no** |
| EXTERNAL[2] | `I never took medications.` / `I never had allergies.` | 70.0% | **no** |
| EXTERNAL[5] | `I didn't take medications.` / … | 69.2% | **no** |
| EXTERNAL[1] | `I'm not on any medications.` / `I'm not allergic.` | 67.5% | **no** |
| EXTERNAL[0] | `I don't take medications.` / `I don't have allergies.` | 67.1% | **no** |
| EXTERNAL[3] | `I have no medications.` / `I have no allergies.` | 64.4% | **no** |
| **DEV** | **`Nothing at all.` / `None whatsoever.`** | **48.2%** | yes |
| EXTERNAL[7] | `Absence of medications.` / `Absence of allergies.` | 42.1% | **no** |
| EXTERNAL[4] | `Negative for medications.` / `Negative for allergies.` | 28.1% | **no** |

Means: TRAIN **95.9%**, EXTERNAL **62.7%**, DEV 48.2%.

> **These are seed-20260805 only.** §5 replicates every arm on a second seed and
> finds the *arm-level* figures unstable (Kendall τ = 0.00 between seeds) while
> the aggregates hold. Read the per-arm ordering below as one draw, not as a
> property of the model; the two-seed numbers in §5 supersede it.

**Three things follow, and the first one refutes what I wrote above.**

**(a) The model did not merely memorise.** On eight phrasings it had never seen,
drawn from a lexicon authored years earlier for another purpose, it averages
62.7% — far above chance, and *higher than* its 48.2% on the two sealed
development strings. Partial polarity generalisation is real. The strong claim
is withdrawn.

**(b) On independent vocabulary the model beats the rule outright.** The regex
recognises **none** of the eight external phrasings; the model handles them at
62.7%. In-distribution the ordering reverses — rule 100%, model 95.9%.

> **The rule-versus-model comparison inverts under distribution shift, and every
> in-repo comparison to date measured only in-distribution.** E1's headline
> result — deterministic solver 0.999 against generative 0.925 — was a
> closed-vocabulary measurement. It does not license "prefer rules", and
> `ENHANCED_PLAN_20260805.md` §3, which proposed replacing composite model
> judgements with deterministic ones, rests on exactly that inference.

**(c) The two development strings are far harder than the training
distribution.** DEV scores 48.2% against a TRAIN mean of 95.9% — and this is the
comparison that replicates: on seed 20260806 it is 52.3% against 98.4%. H6's
absence gate required 383/413 = 92.7%; **both seeds clear that gate on
in-distribution phrasings** (seed 05: 93.2 / 99.0 / 99.3; seed 06: 99.8 / 99.8 /
99.3). The gate outcome turned on two specific strings.

*Not claimed:* that DEV is harder than the external phrasings. That was a
single-seed rank statement, and §5 shows the external ranking has no
seed-to-seed stability, so it does not survive. The stable reference is the
training distribution, not the external arms.

**This does not reverse H6's rejection.** The threshold was frozen in advance,
the measurement was correct, and the verdict stands. What changes is the
*diagnosis*: H6 was not rejected because a state-conditioned residual cannot
represent absence. It was rejected because absence was scored on two adversarial
surface forms, and no one — including me, until today — knew that was what the
gate measured.

**The binding constraint is the evaluation vocabulary.** Ten strings decide the
verdict, a 71-point accuracy swing separates the best phrasing from the worst,
and nothing in the H-cycle distinguished the concept from its surface forms.
Scaling the model, adding a state head, or promoting the rule into the decision
path would each be tuned against that. Fix the benchmark first.

## 5. The seed replication — the falsification partly succeeded

The §4 arms share one checkpoint and one seed, so the 71-point spread might be
variance rather than sensitivity. H6 trained a second seed. Running the identical
arms on `seed-20260806/epoch-2`:

| arm | seed 20260805 | seed 20260806 | \|Δ\| |
|---|---|---|---|
| TRAIN[0..3] | 92.2 / 93.2 / 99.3 / 99.0 | 99.8 / 94.9 / 99.8 / 99.3 | **mean 2.5** |
| DEV | 48.2 | 52.3 | 4.1 |
| EXTERNAL[0..7] | 67.1 / 67.5 / 70.0 / 64.4 / 28.1 / 69.2 / 93.2 / 42.1 | 99.0 / 38.0 / 98.3 / 96.6 / 69.5 / 98.8 / 47.0 / 54.7 | **mean 31.5** |

**Mean absolute seed-to-seed difference: 2.5 points in-distribution, 31.5 points
out-of-distribution — a 12.6× ratio.** Individual arms swing as much as 46.2
points. Seed 05's best external phrasing (`I cannot take medications.`, 93.2%) is
seed 06's second-worst (47.0%); seed 05's worst (`Negative for…`, 28.1%) is
mid-table on seed 06 (69.5%).

Rank correlation between the two seeds across the eight external arms:
**14 of 28 concordant pairs, Kendall τ = 0.00** — exactly chance.

**The falsification succeeded in part and made the finding worse.** Per-arm
out-of-distribution accuracy is *not* a stable property of the training recipe;
at n=1 seed it is close to unmeasurable, so the §4 arm-level spread must not be
read as a property of the model. What survives — and strengthens — is the
aggregate, which replicates on both seeds:

- in-distribution mean **95.9% / 98.4%**, out-of-distribution mean **62.7% /
  75.2%**: a ~23-point gap present in both seeds;
- the DEV arm is the low outlier under both seeds (48.2%, 52.3%);
- both seeds clear H6's absence gate (92.7%) on in-distribution phrasings.

So: two models differing only in seed agree on the phrasings they were trained
on and are **uncorrelated** on the phrasings they were not. Any surface
instrument must therefore aggregate over seeds *and* arms, and report no
arm-level number from a single seed.

*Sample caution: two seeds, eight arms, one scale, one synthetic corpus. τ over
8 items from 2 seeds is itself a noisy statistic. The claim is that per-arm OOD
accuracy is consistent with being seed-determined — not that τ is exactly 0 in
the population.*

## 6. Honest limits of this result
- **The eight external phrasings are patient-voice sentences I constructed around
  external triggers**, not sampled utterances. The triggers are independent; the
  framing is mine. `Negative for …` and `Absence of …` are clinician register in
  a patient's mouth, which plausibly explains their position at the bottom — a
  register effect, not only a polarity effect.
- **Substitution can perturb tokenisation.** The tokenizer was fit on the
  synthetic corpus, so external wordings may fragment differently. This is part
  of the phenomenon rather than a confound to remove, but it is not separated
  here.

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
