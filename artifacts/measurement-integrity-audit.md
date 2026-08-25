# Measurement-integrity audit — span-port leakage line

**Status:** written before any ablation cell has run. No cell has launched.
**Scope:** the `nanoscribe/` P1 span-port evaluator and the Stage P pointer-head
manipulation check. The Paper-α lineage runs through `trajectory/`, not
`nanoscribe/`, and is scoped out except where explicitly stated below.

This is a negative audit in two of its three parts. A negative audit with its
scope stated is worth more than no entry, because the next session otherwise
re-derives the same question from scratch and may reach a different answer.

---

## 1. Three corrections to earlier claims made in this line of work

### 1.1 The environment-variable "split" does not exist

**Claimed:** `NANOSCIBE_QWEN_WEIGHTS` (misspelled) is used in the runner while
`NANOSCRIBE_` (correct) is used in `fabric/schemas.py` and
`scripts/check_docs_integrity.py` — a live inconsistency.

**Correct:** there is no competing environment variable. The grep that produced
the claim matched the *filename* `papers/NANOSCRIBE_VNEXT.md`, referenced as a
path string in `scripts/lint_claim_auth.py:146` and in a comment in
`fabric/schemas.py:8`. `scripts/check_docs_integrity.py` contains no occurrence
of either spelling. Every reader and writer of the weights variable uses the
same misspelled name: `nanoscribe/qwen_inference.py`, `nanoscribe/run_eval.py`,
`nanoscribe/smoke_qwen_baseline.py`, `nanoscribe/test_adapt.py`.

**Consequence:** this is a latent hazard, not a live defect. A well-meaning
spell-fix in one file would stop the runner finding its weights, and it would
fail *open* — `Qwen25BaselineAdapter` falls back to fixture lines when the path
does not resolve, so the run would silently score the fixture and report
success. Pinned by `EnvVarSpellingTest` in
`nanoscribe/test_campaign_instances.py`, which asserts the set of files using
the name and that no file uses the corrected spelling. Behaviour deliberately
unchanged; renaming is a separate change that must update every site at once
and re-pin the test.

### 1.2 The `unbound_assertion` retroactive audit is negative

**Claimed:** every archived result reporting `correct_abstention` was measured
on the laundering metric and is suspect.

**Correct:** the blast radius is one file, and that file is unaffected.

`correct_abstention` was introduced in `be937c1` ("constrained evidence
transport and support evaluation"), which post-dates the frozen evidence
lineage. Counts of files containing the field:

| tree | files |
|---|---|
| `artifacts/` | 0 |
| `trajectory/` | 0 |
| `papers/` | 0 |
| `fabric/` | 0 |
| `scribe/` | 0 |
| `frontier/` | 2 |

Of the two in `frontier/`, one (`p1_qwen_baseline_smoke_v0.json`) is a manifest
that names `weights_env`, not the metric. The single real result,
`frontier/p1_qwen_baseline_smoke_v0_results.json`, reports
`correct_abstention: 0`. Laundering can only *inflate* that count — a
hallucinated assertion being miscounted as a correct abstention adds to it — so
a floor value of 0 cannot have been inflated.

**Consequence: no archived claim requires correction.** The fix is still
load-bearing for everything measured from here on, and the defect is real: a
model asserting a value that occurs nowhere in the source was credited as
`correct_abstention`, which inflates precisely the safety property this program
exists to measure. That belongs in Paper β as a named finding, not as a bugfix
line. But no prior number moves.

### 1.3 C1 and C2 did not arrive together

**Claimed:** `dc3b310` introduced both the prompt-hint channel and the parser
fallback channel.

**Correct:** a gold-value-in-prompt channel predates `dc3b310`. At `09745ec`
the prompt already interpolated `spec.raw_value` into the question and the
system prompt already used gold answers as its format examples
(`Example: STATED: "neck"` is the gold answer for `enc-1/atom-neck`). `dc3b310`
widened the prompt channel and added the parser fallback; it did not create the
leak.

**Consequence:** the taint window for prompt-side leakage opens earlier than
`dc3b310`. Any claim scored on the span metrics between `09745ec` and the
current tip is affected, not only those after `dc3b310`. This does not change
1.2's conclusion — those are different metrics — but it does mean the
"0% → ~83% coverage" claim in `dc3b310`'s own message is measuring a widening
of an existing channel rather than the introduction of one.

---

## 2. The parrot sweep is a methodological finding, not a fixed bug

The pre-registered REFUTED branch for H-leak was *"the contrast moves ≤1 slot
across all four cells."*

A prompt-surface parrot — an adversary that discards the transcript and answers
only from the instruction text — scores as follows on 12 instances / 192 slots,
against a perfect-reader ceiling of 120/192 exact:

| cell | exact | state_ok | abstain | unbound |
|---|---|---|---|---|
| L111 | 109/192 | 72/192 | 0/72 | 82 |
| L101 | 109/192 | 72/192 | 0/72 | 82 |
| L011 | 108/192 | 72/192 | 36/72 | 36 |
| L001 | 108/192 | 72/192 | 36/72 | 36 |
| L110 | 109/192 | 72/192 | 0/72 | 82 |
| L100 | 109/192 | 72/192 | 0/72 | 82 |
| L010 | 0/192 | 0/192 | 72/72 | 0 |
| L000 | 0/192 | 0/192 | 72/72 | 0 |

Two results follow, both settled before the model runs:

**C1 and C3 are substitutes for a copier.** Either channel alone yields
~109/192; both closed yields exactly 0. Single-channel ablation was never going
to be informative, because the surviving channel rescues the answer. This is the
retrospective justification for the whole redesign.

**C2 cannot move a copier at all.** L111 = L101 and L011 = L001 exactly, because
a parrot always emits a quote and the fallback fires only on quote-less output.
For any model that reliably quotes, the C2 contrast is empty by construction.

**Why this is a finding rather than a bug report.** The original four-cell
design had a channel open in all four cells. Under that design the parrot moves
≤1 slot — which is the REFUTED branch. A decision rule whose null branch is
indistinguishable from total confound would have fired REFUTED and read as
reassurance. This belongs in Paper α under measurement reliability, alongside
`unbound_assertion` in Paper β.

---

## 3. Stage P pointer-head manipulation check — re-audited, NOT the same defect

**Why audited:** the first manipulation check written for this line built its
answers from `spec.raw_value`, so it was cell-invariant by construction and
could not fail. If construction-from-gold were a pattern in this codebase, the
Stage P check would be equally non-binding — and the Stage P REFUTE (item-gap
25 pts, held-value top-1 21% vs 92% seen) rests on it.

**Verdict: the Stage P check is binding. It does not share the defect.**

Two independent grounds.

**Structural.** `manipulation_check` in `scribe/pointer/gate2.py:89` measures
`copy_share`, computed in `pointer_model2.py:36-52` entirely from the model's
own forward pass: `h = s.t.hidden(x)`, then `alpha, p_gen = s._copy(...)`, then
`copy_share = ((1 - p_gen) * pcopy_tgt) / P_tgt`. Gold enters only to select
*which positions* to measure at and which vocabulary entry's probability to
read. That is the correct use of gold — it defines where the measurement is
taken, not what it returns. The banned shape is the opposite: the measured
quantity itself is constructed from gold.

**Empirical, and decisive.** The check discriminated two real models with the
same code and the same n=118 held-value tokens:

| arm | M (copy-share) | p_gen | verdict |
|---|---|---|---|
| P1 (unsupervised head) | 0.181 | 0.825 | `VOID(unused)` |
| P2 (copy-supervised) | 0.969 | 0.087 | `EXERCISED` |

A check that actually failed once, on real data, and changed the conclusion, is
binding. The banned shape cannot produce two different verdicts at all.

**Residual limitation, stated honestly.** The check measures pathway
*engagement* under teacher forcing. In the P2 arm, M=0.97 under explicit
copy-supervision with gate-bias −2 is close to guaranteed by the training
objective, so `EXERCISED` there mainly confirms that training did what it was
told. That is adequate for its actual job — ruling out the P1 failure mode where
the pathway was never engaged and a null would have been meaningless — but it is
not independent evidence about generalisation, and should not be read as such.

**Stage M is not blocked by this audit.**

---

## 4. Program invariant, now mechanically enforced

> An adversarial baseline must consume **exactly** the channel under test and no
> other source. Its score must **vary** across the ablation cells. A baseline
> that scores identically everywhere is not measuring the channel — it is
> reporting its own construction.

Enforced by `nanoscribe/test_adversarial_baseline_invariant.py`:

1. the prompt-consuming parrot varies across the 8 cells;
2. it reaches exactly 0 when the channels close, so it does not merely wobble;
3. the gold-constructed shape is asserted to be cell-invariant, kept executable
   as the counter-example.

A codebase-wide scan for the banned shape found exactly one site — this line's
own retired echo adapter. It is gone; the manipulation check in
`test_leakage_ablation.py` now reads the prompt.

**Operational rule:** the parrot is the instrument's acceptance test, not a
diagnostic. Re-run it after every prompt-plumbing change. The instrument is
clean only when the parrot collapses to floor.
