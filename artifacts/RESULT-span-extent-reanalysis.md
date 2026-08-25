# RESULT — the span-port bottleneck is delimitation, not retrieval

**Date:** 2026-08-25
**Status:** re-analysis of landed data. **No new runs.** Every number below is
computed from the per-slot records of runs already in the store.
**Relates to:** `artifacts/RESULT-span-port-leakage-ablation.md` (`ddb5ce6`) —
this document **corrects one number and reinterprets the headline**, and
**confirms** that artifact's C1 verdict.

---

## 1. The claim

`RESULT-span-port-leakage-ablation.md` reports `asserted_grounded` 16/192
against a perfect-reader ceiling of 120 and reads it as *"the model reads, but
poorly"* — 13% of ceiling. That reading is wrong, and the error is not in the
number but in what the metric measures.

`asserted_grounded` and `exact_gold_span` are **exact-extent** predicates. They
score two very different failures identically at zero:

- the model quoted the wrong part of the transcript — it did not find the evidence;
- the model quoted a span that **contains** the gold span — it found the evidence
  and returned the wrong extent.

Separating them (`nanoscribe/analyze_span_extent.py`):

| L000, all channels closed (`ba18cf04`) | count | of ceiling |
|---|---|---|
| `grounded_exact` | 16 | 13.3% |
| **`located_over_extended`** (quote contains gold span) | **79** | |
| `located_under_extended` | 0 | |
| `not_located` | 21 | |
| `no_quote` | 4 | |
| **LOCATED (any extent)** | **95/120** | **79.2%** |

**79/104 = 76.0%** of the non-grounded gold-bearing slots contain the gold span
inside the model's quote. <run id="ba18cf04-3984-4322-95ec-c5d4b378fb9b" label="L000 grounded 16/192, LOCATED 95/120" />

## 2. Bounding LOCATED — the containment test alone proves nothing

Containment is trivially satisfiable by quoting the whole transcript, so a bare
"79.2%" is not reportable. Length statistics over the 95 over-extended slots
(unified-form L000, the strictest cell):

| | median | mean | max |
|---|---|---|---|
| gold span length (chars) | 8 | 8.8 | 24 |
| model quote length (chars) | 29 | 27.5 | 41 |
| `len(gold) / len(quote)` | **0.320** | | min 0.108 |
| quote / **enclosing-turn** length | **1.000** | | max 1.03 |
| quote / full-transcript length | 0.276 | | max 0.52 |

**The model returns exactly one conversational turn.** The median quote is
1.000× the length of the turn containing the gold span; only 1 of 95 quotes
exceeds its enclosing turn, and no quote reaches even 52% of the transcript.
The quote is ~3× the gold span, not 20×.

This is stable across all three cells analysed (over-extended quote/turn median
1.000 in every one; gold/quote median 0.30–0.35).

**The bounded claim, which is the only form this may be cited in:**

> With all three leak channels closed, the model selects the correct
> conversational turn for **79% of gold-bearing slots** (95/120) but delimits
> the gold span within it for only **13%** (16/120). Located quotes are
> turn-scale: median 29 characters against gold spans of median 8, and a median
> ratio of exactly 1.00 to the enclosing turn.

Never as a bare 79.2%. That is precisely how `dc3b310`'s 83% propagated.

## 3. The decisive test: LOCATED survives what exact-extent does not

Two runs carry the **identical condition label** `C1off_C2off_Qon_QSoff` and
differ only in question form:

| | question | `asserted_grounded` | **LOCATED** |
|---|---|---|---|
| `ba18cf04` @ `7447df5` | *"What is the place the patient says is hurting?"* | **16**/120 | **95**/120 (79.2%) |
| `e04b3016` @ `9a3ecd4` | *"What does the patient say about the place…?"* | **2**/120 | **97**/120 (80.8%) |

**Exact-extent moves 8× (16 → 2). LOCATED does not move at all (95 → 97).**
<run id="e04b3016-c00b-4c0d-b328-486c07e9177e" label="unified-form L000: grounded 2/192, LOCATED 97/120" />

The model's ability to find the right evidence is invariant to the question
form; only its answer *extent* moves, because *"what does the patient say
about X"* invites a clause and *"what is X"* invites a noun. The exact-extent
metric is substantially measuring **question phrasing**, not evidence-finding.

This is the strongest available evidence that the bottleneck is delimitation.

## 4. It is not merely a quoting habit — the semantics are right too

Of the 79 over-extended slots in `ba18cf04`, **50 also have
`assertion_state_correct`** (29 do not). So 66/120 slots — **55% of ceiling** —
locate the correct turn *and* assign the correct assertion state, and score
zero solely on extent.

## 5. Consequence for the program's central diagnosis

Four standing hypotheses for the surviving bottleneck — induction-circuit
capacity, the SFT objective, pretraining data diversity, morphology — are all
about **locating** the value. None predicts a model that reliably finds the
right turn and cannot delimit within it.

It also offers a unifying reason why Stage C (curriculum), Stage S (scale), and
Stage P/P2 (an explicit copy-dominant pointer head, 21% teacher-forced held-value
first-token top-1) all failed to move the gap: **each was aimed at retrieval.**
Feeding this into task 3 rather than asserting it here.

## 6. Corrections to `ddb5ce6`

### 6.1 The headline baseline is question-form dependent — establish canonical L000

`ddb5ce6` was committed 03:03:21. **Both** L000 and L100 were re-run at 03:10
under a "unified wh question form". The artifact therefore cites a run the peer
subsequently superseded, and the headline moves 16/192 → 2/192 under the
replacement form. **This needs a decision and a correction notice, not a quiet
edit.** Either form is defensible; what is not defensible is a cited baseline
whose value depends 8× on an undocumented instrument choice.

Recommendation: adopt the **unified** form as canonical (it is the peer's later
choice and it holds form fixed across cells), publish **2/192 exact-extent
alongside 97/120 LOCATED**, and retain 16/192 as the earlier-form reading.

### 6.2 The C1 contrast is NOT confounded — I was wrong, and I am recording it

I flagged the artifact's C1 contrast as cross-form confounded, the same defect
that voided C3. **That flag was wrong.** Verified:

```
git diff --stat 7447df5 a91ce6b   ->  nanoscribe/leakage.py | 2 +-   (only)
git diff --stat 9a3ecd4 9523bf4   ->  nanoscribe/leakage.py | 2 +-   (only)
```

`prompt.py` is **byte-identical** within each pair; the sole difference is the
C1 flag. The artifact's pair (`ba18cf04` × `5b1a17ae`, both @ 02:20) is
form-matched and clean. The cross-form pairing was **mine** — I had compared the
old L000 against the new L100. The hazard is real, but it was my error, not the
artifact's.

### 6.3 The C1 verdict survives exact McNemar — the CIs are the wrong shape, the conclusion is right

Per-slot outcomes are binary over a slot set identical in both cells under
greedy decoding, so the exact conditional-binomial null applies rather than a
paired *t* at df=11 over 12 instance means.

**Artifact's own pair** (`ba18cf04` × `5b1a17ae`), n=192 slots:

| outcome | ref | trt | b | c | p (exact) | p (DEFF 1.44) |
|---|---|---|---|---|---|---|
| `asserted_grounded` | 16 | 30 | 16 | 2 | 1.3e-03 | 6.4e-03 |
| `asserted_unbound` | 39 | 71 | 38 | 6 | 9.4e-07 | 6.0e-05 |
| `abstained_correct` | 32 | 19 | 1 | 14 | 9.8e-04 | 3.9e-03 |
| `assertion_state_correct` | 66 | 63 | 7 | 10 | 0.63 | 0.75 |

**Form-matched later pair** (`e04b3016` × `22244713`):

| outcome | ref | trt | b | c | p (exact) |
|---|---|---|---|---|---|
| `asserted_grounded` | 2 | 25 | **23** | **0** | 2.4e-07 |
| `asserted_unbound` | 43 | 86 | 55 | 12 | 1.0e-07 |
| `abstained_correct` | 24 | 2 | 0 | 22 | 4.8e-07 |
| LOCATED (any extent) | 97 | 108 | 18 | 7 | 4.3e-02 |

**On both pairs the co-movement rule fires and the verdict is
`UNRESOLVED — coverage shift`.** The artifact's reported deltas
(+1.17 / +2.67 / −1.08 per instance) reproduce its pair exactly. Its **verdict
stands**; its intervals should be restated as exact McNemar discordant counts,
which are both assumption-free and sharper — the later pair's grounded contrast
is **23 gains against 0 losses**, a cleanliness a *t*-interval cannot express.

Note LOCATED is the one outcome C1 barely moves (p=0.043, failing the clustering
sensitivity check at 0.077): C1 buys *assertions*, not *evidence-finding* — which
is the same story as §3 from the other direction.

## 7. Provenance

| artifact | contents |
|---|---|
| `artifacts/span_extent_L000.json` | per-slot categories, `ba18cf04` |
| `artifacts/span_extent_L000_unified.json` | per-slot categories, `e04b3016` |
| `artifacts/span_extent_L100.json` | per-slot categories, `22244713` |
| `artifacts/mcnemar_c1_artifact_pair.json` | McNemar, artifact's pair |
| `artifacts/mcnemar_c1_form_matched.json` | McNemar, later form-matched pair |
| `artifacts/mcnemar_question_form.json` | pure question-form contrast |

Code: `nanoscribe/analyze_span_extent.py`, `nanoscribe/mcnemar_c1.py`.
Gold spans reconstructed from each run's **own commit** via a detached worktree,
so no cross-revision gold leakage enters the join.
