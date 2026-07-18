# A held-out copying gap in small language models, and what it takes to measure it

*Working manuscript draft — Paper 1 (empirical + measurement). Claims are held to
what was measured; the causal question (scale vs. model family) is left open by
design. All numbers trace to archived per-run JSONs under `trajectory/` at git tag
`stage-t-v2-results`; instrument inputs are content-addressed in
`REPRODUCIBILITY.md`.*

## Abstract

We study a specific faithfulness failure in small language models finetuned to
convert short clinical dialogues into structured summaries: a **held-out copying
gap** — the model copies field values it saw during finetuning but errs on
held-out values under held-out phrasings, even though both are present verbatim in
the input. In our own from-scratch models (3.15M and 10M parameters) this gap is
large — **18.3±1.3 and 18.7±1.5 points** of recall on the same multi-instance
instrument used for the rest of the ladder (22–23 points on the single public
instance the anchors were first scored on; §6.1). Finetuning the Pythia open-weight
family (160M/410M/1B) on the identical task, the gap observed is **substantially
smaller** (single-digit points; 3.5±0.7 at 160M, 4.2±0.9 at 410M, and an interval of
[0,5] at 1B). We deliberately do **not** claim scale as the cause: the comparison also
changes pretraining corpus, tokenizer, architecture, and finetuning method. The
measurement itself is a second contribution. A pre-registered contamination check
caught that our initial single-instance evaluation was under-powered for the gap;
averaging over five larger evaluation instances resolved the middle rungs to
sub-point precision. At 1B, a determinism cross-check then revealed that
**training-run nondeterminism** — not evaluation noise — dominates the residual, so
we report the top rung as an interval rather than a point. Both measurement lessons
arose from empirical failures of the prior instrument, not from foresight.

## 1. Introduction

Production language systems for structured summarization (e.g. clinical scribing)
are judged not on average-case benchmark scores but on faithfulness under
distribution shift: whether the output contains only content grounded in the input,
including on inputs unlike the training data. We isolate one tractable component of
this — value copying — in a setting where faithfulness is measurable by
construction, and ask a single pre-registered question: **how does the copying gap
behave as model capability increases?**

The contribution is threefold and we keep the parts distinct:
1. **Empirical.** A large held-out copying gap in sub-10M models is much smaller in
   the tested Pythia models (§6.1).
2. **Measurement.** Two lessons about estimating such a gap reliably: single-instance
   evaluation was under-powered (§5.1), and training nondeterminism bounds the
   estimate at the top of the ladder (§5.2).
3. **Engineering (companion).** A verification architecture that routes model errors
   to human review at measured precision/review-load; reported separately, referenced
   in §7.

## 2. Task and benchmark

**Construction.** Synthetic clinic dialogues are rendered from a known fact tuple of
five fields (chief complaint, duration, severity, medication, allergy). Because the
generating tuple is known, faithfulness is exact — a model summary is scored field-
by-field against the tuple with no judge model. A field is a hit if it matches, an
*omission* if the model writes "none" for a present value, and a *hallucination*
otherwise.

**Contamination controls (two axes).** Evaluation dialogues use (i) held-out
*template families* — surface phrasings never seen in finetuning — and (ii) for half
the dialogues, held-out *slot values* (specific complaints/medications/allergies
excluded from all finetuning data). The **held-out copying gap** is
seen-value-recall minus held-out-value-recall, both measured under held-out
templates. It isolates value copying from phrasing familiarity.

**The benchmark is a generator.** The scientific object is the eval distribution
(a seeded generator), not any fixed set of dialogues. This matters after public
release: fresh instances can be drawn to defend against memorization of any specific
instance (§5.1), and the instance-difficulty distribution can be measured directly.

## 3. Prior stages (setup)

Two earlier results frame the question. **Stage C** tested whether the gap is a
training-curriculum artifact by adding an unmemorizable-value training slice; the
held-out gap was unmoved, refuting the curriculum explanation. **Stage S** scaled our
own stack from 3.15M to 10M parameters: the larger model passed the average-case
faithfulness gate (parse 100%, recall 88%, hallucination 7.5%) yet its held-out gap
was essentially unchanged (≈23 vs ≈22 points single-instance; **18.7±1.5 vs 18.3±1.3
on the multi-instance instrument**, §6.1 — the near-identity now measured with error
bars, not two single points). Stage S concluded that a 3.2× scale step does not
cross the failure — motivating a wider ladder.

## 4. Stage T design (pre-registered)

**Ladder.** Own models at 3.15M/10M (re-scored on the multi-instance instrument,
§6.1) plus the Pythia family at 160M/410M/1B, finetuned on the identical scribe
recipe with LoRA (r=16, α=32, LR 1e-4, 3 epochs), scored by the identical
faithfulness scorer.

**Why Pythia.** The question is scaling behavior, not model ranking. Pythia holds
architecture, tokenizer, pretraining corpus, and recipe fixed across sizes, so within
the family relatively few variables move with parameter count — the property a scaling
study needs. This choice does not isolate scale from the nano→Pythia *stack* change
(§7); it is the reference family for that axis, not a claim that scale is the cause.

**Pre-registration and falsifiers.** Bars, bands (PERSISTS/THRESHOLD/DIVERGENT),
seeds, and decision rules were fixed before measurement (`PREREG.md`, three
pre-measurement amendments). A base-model control (un-finetuned model must fail parse
< 50%) guards against a trivially-solvable task; it passed at every rung (base parse
0%). The measured code was adversarially reviewed before any run; three
measurement-invalidating defects were fixed pre-measurement.

## 5. The measurement instrument, and why it changed

The instrument evolved because the data demanded it. We present this as a result.

### 5.1 Single-instance evaluation was under-powered

The pre-registered contamination check (score the same model on the public instance
vs. a fresh instance; require per-metric agreement within 5 points) **flapped** at
the threshold across rungs (gap differences ≈5, 6, 4). Diagnosis: the gap rides on a
handful of hard held tokens, so a single 20-dialogue instance carries ±5–6 points of
sampling variance. Crucially, the *direction* of the discrepancy (the public instance
was **harder**, not easier) is the opposite of the contamination signature the check
was built to detect — ruling out memorization. The fix (T-v2) re-scores the same
frozen models on **five instances of 100 held dialogues each** and reports the gap as
a mean ± across-instance SD. This cut the SD to ~0.7–0.9 points at 160M/410M. We
applied the identical re-scoring to the own-stack anchors so the whole ladder shares
one instrument (`PREREG_anchors.md`, re-scoring only — frozen v0.1 checkpoints, their
native ChatML/greedy scorer; both anchors reproduced their canonical single-instance
readings byte-for-byte before the multi-instance pass). The anchors' single
public-instance gaps (22.4, 23.0) were themselves modestly hard draws — ~4 points
above their multi-instance means (18.3, 18.7) — the **same direction** (public
instance harder, not easier) found at the Pythia rungs, so the public instance is a
systematically hard draw across the entire ladder.

### 5.2 Training nondeterminism bounds the 1B estimate

T-v2 re-generates each frozen adapter by re-finetuning at the fixed seed; we verify
this reproduces the original model by re-scoring the byte-identical original instances
and checking the gaps match (a **determinism cross-check**). At 160M and 410M the
match was exact. At 1B it **failed**: on the byte-identical instance, one training run
gave a 5-point gap and another gave 0. With evaluation held constant, the difference
is training-run variance — fp16 with non-associative GPU reductions over ~1125 steps
of a 1B model that sits at the boundary of perfect held-out copying. The right
instrument for the 1B gap is therefore multiple training seeds, not multiple
evaluation instances; we report an interval.

## 6. Results

### 6.1 The gap across the ladder

All five rungs are scored on the **same multi-instance instrument** (five instances
of 100 held + 100 seen dialogues, v1 eval distribution, gap = mean ± across-instance
SD). The own-stack anchors' original single-instance readings are shown alongside for
provenance.

| Model | Params | Stack | Held-out gap (5×100 held) | Single-instance (inst0) | Basis |
|---|---|---|---|---|---|
| nano | 3.15M | own | **18.3 ± 1.3 pts** | 22.4 | determinism verified |
| scale | 10M | own | **18.7 ± 1.5 pts** | 23.0 | determinism verified |
| pythia-160m | 160M | Pythia | **3.5 ± 0.7 pts** | 7.0 | determinism verified |
| pythia-410m | 410M | Pythia | **4.2 ± 0.9 pts** | 8.0 | determinism verified |
| pythia-1b | 1B | Pythia | **[0, 5] pts** | 5.0 / 0.0 | training-run–bounded |

Model-side quality is high at every rung (own-stack anchors parse 95–100%, recall
80–91%, hallucination 7–12%; Pythia rungs parse 100%, recall 96–100%, hallucination
0.5–4%), on both the public and fresh instances. On this single consistent instrument
the large held-out copying gap **observed in our own-stack models (~18 points) was
substantially smaller in the tested Pythia models (3.5–4.2 points).** The reduction
(~14–15 points) is an order of magnitude larger than any noise source identified
(anchor SD ≈1.3–1.5, Pythia SD ≈0.7–0.9), so the direction is not in question. Note
that the single-instance column — the basis of the earlier headline — is a
uniformly hard draw (higher gap at every rung), which is precisely why mixing a
single-instance anchor with a multi-instance Pythia number would have overstated the
contrast; on the consistent instrument the gap is still large but the anchors read
~18, not ~22–23.

### 6.2 What the result does and does not establish

Within Pythia the gap is already small at 160M with no clean monotonic trend, so the
ladder captured a low-gap regime rather than a transition. Reconciling with Stage S:
the 3.2× step (3M→10M) did not move the gap (18.3±1.3 → 18.7±1.5 — a 0.4-point change
inside one SD, now measured rather than inferred from two single points); the 16×
step to 160M, **or** the stack change, coincides with a much smaller gap — these are
not separated here. The
formal band is not PERSISTS (the top-rung interval reaches 0, far from the ≥10 that
PERSISTS requires); it is consistent with a small residual or none.

## 7. Limitations

- **Scale vs. family confound (primary).** The nano→Pythia comparison changes
  parameter count *and* pretraining corpus, tokenizer, architecture, and finetuning
  method simultaneously. We therefore claim only that the gap is much smaller under
  the Pythia pipeline, not that scale per se removes it. Isolating scale needs a
  within-family run (below).
- **1B point estimate.** Bounded, not identified, by training nondeterminism (§5.2).
- **Single task and single scale family.** One synthetic clinical-summarization task;
  one scaling family. Generality across domains and families is untested here.
- **Anchor precision (resolved).** The 3M/10M anchors were re-scored on the same
  multi-instance instrument as the rest of the ladder (18.3±1.3, 18.7±1.5); the whole
  ladder now shares one instrument, and the anchor gaps carry across-instance SD like
  the Pythia rungs.

## 8. Future work (axes, not replacements)

The program extends as a matrix of one-variable-at-a-time questions, keeping Pythia as
the reference scaling family:

| Stage | Variable changed | Held fixed |
|---|---|---|
| T (done) | scale | Pythia family |
| F (family) | model family | ~similar size |
| O (objective) | SFT / DPO / RL / continual pretrain | model + size |
| R (retrieval) | RAG / pointer / copy head | model |
| M (mechanism) | — (ask *why*) | model + task |

**Stage F** is the direct test of §7's primary confound: hold capability ~fixed
(~160–500M) and vary the family (Pythia vs. SmolLM/OLMo/TinyLlama) to ask whether the
reduction tracks size or pipeline. **Frontier models** (GPT/Claude/Gemini) belong as
*external validation* — "does the phenomenon still appear in production-grade
systems?" — not as ladder rungs, since they change every variable at once. **Stage M**
(mechanism) is deliberately deferred until the behavioral phenomenon is fully frozen,
so it asks "why does this residual occur?" rather than carrying the burden of
justifying Stage T.

## 9. Conclusion

A severe held-out copying gap in sub-10M models was substantially smaller in the
tested Pythia models; the cause (scale vs. pipeline) is left open by design. The
measurement story is inseparable from the result: single-instance evaluation was
under-powered, and at the top of the ladder training nondeterminism — not evaluation
noise — sets the precision floor. Both were surfaced by pre-registered checks
(equivalence, determinism) that failed productively. The contribution is less any
single number than a research unit that repeatedly incorporated its own measurement
limitations into its conclusions rather than explaining them away.

---

*Status: draft. The anchors are now re-scored on the multi-instance instrument
(`results_anchors_v2_{nano,scale}.json`), so the whole ladder shares one instrument
and the public-instance-hard-draw holds at every rung (single-instance gap ≥
multi-instance mean at all five). Open manuscript decisions before submission —
precise abstract numbers pending a decision on multi-seed 1B; figures (gap-vs-params
with error bars/interval; instance-difficulty histogram showing the public instance
as a hard draw) to be produced from the archived JSONs; related-work section to be
written.*
