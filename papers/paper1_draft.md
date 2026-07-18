# Held-out value copying in small language models: a field-localized failure mode and the instrument to measure it

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
instance the anchors were first scored on; §6.1). The failure is not diffuse: it
localizes *entirely* to the three open-vocabulary fields (complaint, medication,
allergy) and is **exactly zero** in the two closed-value fields (duration, severity),
which act as a built-in control showing the effect is specifically held-out-*value*
copying in open slots, not a generic degradation under unfamiliar phrasing. Finetuning
the Pythia open-weight family (160M/410M/1B) on the identical task, the gap observed is
**substantially smaller** (single-digit points; 3.5±0.7 at 160M, 4.2±0.9 at 410M, and
an interval of [0,5] at 1B). We deliberately do **not** claim scale as the cause: the comparison also
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
including on inputs unlike the training data (Maynez et al., 2020; Ji et al., 2023). We isolate one tractable component of
this — value copying — in a setting where faithfulness is measurable by
construction, and ask a single pre-registered question: **how does the copying gap
behave as model capability increases?**

The contribution is threefold and we keep the parts distinct:
1. **Empirical.** A large held-out copying gap in sub-10M models is much smaller in
   the tested Pythia models, and localizes cleanly to the open-vocabulary fields (§6.1).
2. **Measurement.** Two lessons about estimating such a gap reliably: single-instance
   evaluation was under-powered (§5.1), and training nondeterminism bounds the
   estimate at the top of the ladder (§5.2).
3. **Engineering (companion).** A verification architecture that routes model errors
   to human review; its precision/review-load trade-off is measured in a companion
   write-up (not in this paper), referenced in §7.

## Related work

*Draft section — citations verified against primary sources (§ References); final
placement/numbering is a formatting pass. Each paragraph notes the relation to this
work.*

**Scaling laws and emergence.** Language-model loss follows smooth power laws in
parameters, data, and compute (Kaplan et al., 2020; Hoffmann et al., 2022), yet
specific *task* behaviors can change sharply with scale: Wei et al. (2022) catalogue
"emergent abilities" absent in small models and present in large ones. Schaeffer et
al. (2023) argue much apparent emergence is a *measurement artifact* of discontinuous
metrics (exact-match, multiple-choice) rather than a property of the model. Our study
runs in the opposite direction — a behavior (the held-out copying gap) that is *large*
in small models and *smaller* at larger scale — but Schaeffer et al.'s caution applies
directly, since our gap is an exact-match recall difference; we therefore report it
with across-instance error bars and, at 1B, separate metric behavior from training-run
variance (§5.2). Where the emergence literature asks "does scale switch an ability
*on*?", we ask "does scale switch a faithfulness failure *off*?", and find the honest
answer is confounded with the training stack (§7).

**The Pythia platform.** Biderman et al. (2023) release Pythia, 16 models spanning
70M–12B trained on identical data in identical order, expressly to make scale a
controlled variable. We use the 160M/410M/1B rungs as our scaling axis for exactly
this property — within Pythia, architecture, tokenizer, pretraining corpus, and data
order are fixed across sizes. The limitation we foreground (§7) is that our own-stack
anchors are *not* in that controlled family, so the nano→Pythia step moves the stack
as well as the scale — a confound Pythia's internal control cannot remove.

**Copying, pointer, and retrieval mechanisms.** The behavior we measure is value
*copying*: reproducing a field value present verbatim in the input. Explicit copy
pathways have a long lineage in sequence-to-sequence models — CopyNet (Gu et al.,
2016) and pointer-generator networks (See et al., 2017) add input-copying to
abstractive generation. In transformers, induction heads implement a pattern-completion
copy (given "…[A][B]…[A]", attend to the first A and predict B; Elhage et al., 2021;
Olsson et al., 2022), and *retrieval heads* — a sparse (<5%), apparently universal set
of attention heads — copy tokens from context into the output and mechanistically
explain long-context factuality (Wu et al., 2024). These are the natural mechanistic
hypotheses for our gap: a model that copies *seen* values but fails on *held-out*
values under held-out phrasing may have copy/retrieval circuitry that generalizes
poorly off-distribution. We deliberately defer this question (Stage M, §8) until the
behavioral phenomenon is deconfounded, so any circuit we implicate is attached to a
clean empirical contrast rather than a stack difference.

**Faithfulness and hallucination.** Faithfulness — output grounded only in the input —
is a central failure mode of abstractive summarization; Maynez et al. (2020)
distinguish intrinsic (misrepresenting source content) from extrinsic (unsupported)
hallucination, and Ji et al. (2023) survey the phenomenon across NLG. Most such work
relies on human judgment or model-based entailment scoring. Our task is constructed so
faithfulness is *exact*: because each dialogue is rendered from a known fact tuple, a
summary is scored field-by-field against ground truth with no judge model, and our
"hallucination" is precisely an extrinsic fabrication of a field value. This trades
ecological realism for a noise-free, reproducible faithfulness signal.

**Reproducibility and training nondeterminism.** Run-to-run variation from seeds and
low-level tooling can be large enough to reverse method rankings (Picard, 2021; Pham et
al., 2020), and even fixed-seed training is nondeterministic through non-associative
GPU reductions and library tooling (Zhuang et al., 2022); the standard prescription is
to report mean ± SD over multiple runs. Our 1B result is a concrete downstream
instance: two fixed-seed retrains of the same model produced gaps of 5 and 0 on the
byte-identical evaluation, so at that rung the *unit of uncertainty* shifts from the
evaluation instance to the training run, and we report an interval (§5.2). This
connects the reproducibility literature to a specific measured behavior (a faithfulness
gap) rather than to top-line accuracy.

**Evaluation reliability and contamination.** Public benchmarks risk contamination —
memorized test content inflating scores (Xu et al., 2024) — with held-out or rolling
test data the standard defense. Our benchmark is a *seeded generator* rather than a
fixed set, so fresh evaluation instances can be drawn at will, and our pre-registered
contamination check compares the public instance against fresh draws. We find the
public instance is *harder* (higher gap) than fresh draws at every rung — the opposite
of the contamination signature — which both rules out memorization and surfaces a
distinct hazard: a fixed public instance can be a systematically biased difficulty draw
even when content-addressed and reproducible.

## 2. Task and benchmark

*This section motivates the design; the precise protocol (generator, held-out split,
metric) is in Methods.*

**Faithfulness is exact by construction.** Synthetic clinic dialogues are rendered from
a known fact tuple of five fields (chief complaint, duration, severity, medication,
allergy), so each summary is scored field-by-field against ground truth with no judge
model — a noise-free faithfulness signal, traded against ecological realism.

**The held-out copying gap.** Evaluation dialogues are held-out along two axes — held-out
*template families* (all dialogues) and held-out *slot values* (half the dialogues) — and
the central object is the **held-out copying gap**: seen-value recall minus
held-out-value recall under held-out templates, which isolates value copying from
phrasing familiarity.

**The benchmark is a generator.** The scientific object is the eval *distribution* (a
seeded generator), not any fixed set of dialogues. Fresh instances can be drawn to
defend against memorization of any specific instance (§5.1), and the instance-difficulty
distribution can be measured directly — a property we exploit in both measurement lessons.

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

**Why Pythia.** The question is scaling behavior, not model ranking. Pythia (Biderman
et al., 2023) holds architecture, tokenizer, pretraining corpus, and data order fixed
across sizes, so within the family relatively few variables move with parameter count —
the property a scaling study needs. This choice does not isolate scale from the nano→Pythia *stack* change
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
of a 1B model that sits at the boundary of perfect held-out copying — the tooling-level
nondeterminism documented by Zhuang et al. (2022) (cf. Picard, 2021, on seed variance
exceeding effect sizes), here surfacing in a downstream faithfulness metric rather than
top-line accuracy. The right instrument for the 1B gap is therefore multiple training
seeds, not multiple evaluation instances; we report an interval. With only two retrains
we bound the *magnitude* of training-run variance at this rung (it is large — comparable
to the whole residual gap) but cannot characterize its distribution; the interval spans
the two observed outcomes rather than a modelled spread, and a fuller characterization
(how often a run tips to perfect held-out copying) would need more seeds.

## Methods (consolidated protocol)

*Authoritative, objective protocol; §2/§4/§5 give the motivation and the instrument's
evolution. Final section placement/de-duplication is a formatting pass (see
writing_audit.md). Every artifact below is content-addressed in `REPRODUCIBILITY.md`.*

**Datasets (a seeded generator).** Each example is a synthetic doctor–patient dialogue
rendered from a fact tuple of five fields — chief complaint (cc), duration (dur),
severity (sev), medication (med), allergy (alg) — paired with the target summary
`CC: … | DUR: … | SEV: … | MED: … | ALG: …`. Training values (seen): a fixed complaint
pool plus a compositional body-part×sensation pool (~190 complaints, so cc cannot be
solved as closed-set classification), 18 medications, 5 allergies, 3 severities.
Finetuning data: 12,000 examples, full supervised finetune, 3 epochs, LR 1e-4 (own
stack); the eval distribution is a separate v1 generator.

**Held-out protocol (two axes).** Evaluation dialogues are held-out along two
independent axes. (i) *Held template families*: doctor/patient surface phrasings never
seen in finetuning — used by **all** eval dialogues. (ii) *Held slot values*: six
specific values excluded from all finetuning data (complaints {toothache, neck pain,
heartburn}, medications {melatonin, throat lozenges}, allergy {sulfa drugs}) — used by
**half** the eval dialogues; the other half draw seen values. The **held-out copying
gap** = seen-value recall − held-out-value recall, both measured under held templates,
isolating value copying from phrasing familiarity.

**Model families and finetuning.** *Own stack*: from-scratch GPT (RoPE, GQA, SwiGLU,
RMSNorm pre-norm, tied embeddings, 4098-vocab BPE, 512 ctx) at 3.15M (d=192, L=6, H=6,
KV=2, ff=512) and 10M (d=320, L=8, H=8, KV=2, ff=864), pretrained on ~200M FineWeb
tokens (D≈20N; Hoffmann et al., 2022) and full-FT on the scribe task; the two anchor
checkpoints are the frozen v0.1 release assets (`scribe.pt`, `scale10m_scribe.pt`).
*Pythia*: EleutherAI/pythia-{160m,410m,1b}, adapted with LoRA (r=16, α=32, dropout 0,
targets {query_key_value, dense, dense_h_to_4h, dense_4h_to_h}), LR 1e-4, 3 epochs.

**Evaluation metric.** Greedy decoding; the output is parsed against the fixed field
template. A field is a *hit* (exact match), an *omission* ("none" for a present value),
or a *hallucination* (otherwise). Held/seen recalls are computed over the fields of
**parsed** dialogues only (identical convention across stacks). Two decode paths, each
matched to its stack's native format: own stack uses ChatML (`<|im_start|>…`) with a
token-by-token argmax loop stopping on `<|im_end|>`, max 64 new tokens; Pythia uses raw
text with Hugging Face greedy `generate`, EOS stop, max 64 — these are the checkpoints'
own inference formats, not a shared decoder.

**Multi-instance evaluation.** The powered instrument is K=5 fresh evaluation instances
(seeds 20260720–20260724), each 100 held + 100 seen dialogues from the v1 distribution.
A model's gap is the mean over the five instances; its uncertainty is the across-instance
SD. The single public instance (inst0, seed 7, 40 dialogues) and an auxiliary instance
(instT, seed 20260717) are retained for the cross-checks below.

**Determinism checks (why "re-score", not "re-run").** All measurements are re-scorings
of frozen models. Before the multi-instance pass, each model is re-scored on the
canonical single instance and required to reproduce its archived reference: own-stack
nano reproduced `gate_scribe_v2.log` byte-for-byte (held 68/95, seen 94/100, gap 22.4)
and scale reproduced Stage S exactly (parse 100%, recall 88%, gap 23.0) — the latter
even across a CUDA-T4→MPS device change. For Pythia, re-finetuning at the fixed seed
regenerates the frozen adapter (headless-T4 reproduced interactive-T4 byte-for-byte),
verified in-band by requiring the re-scored inst0/instT gaps to match the archived v1
JSONs. A check that fails (as at 1B) is treated as a finding, not smoothed over.

**Statistical reporting.** Powered rungs report gap mean ± across-instance SD (ddof=1);
each Pythia instance additionally carries a 95% bootstrap CI (10,000 resamples over
dialogues). The contamination check is direction-aware — it flags only if the public
instance gap is *below* the fresh-instance mean by more than 2 SD (memorization would
make the public instance easier); a higher public gap is a hard draw, not contamination.
The 1B rung is reported as an interval [0,5] from two fixed-seed training-run retrains,
not as a point.

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

Model-side quality is high at every rung (own-stack anchors parse 94–100%, recall
80–91%, hallucination 7–12%; Pythia rungs parse 100%, recall 96–100%, hallucination
0–4%), on both the public and fresh instances. On this single consistent instrument
the large held-out copying gap **observed in our own-stack models (~18 points) was
substantially smaller in the tested Pythia models (3.5–4.2 points).** The reduction
(~14–15 points) is an order of magnitude larger than any noise source identified
(anchor SD ≈1.3–1.5, Pythia SD ≈0.7–0.9), so the direction is not in question. Note
that the single-instance column — the basis of the earlier headline — is a
uniformly hard draw (higher gap at every rung), which is precisely why mixing a
single-instance anchor with a multi-instance Pythia number would have overstated the
contrast; on the consistent instrument the gap is still large but the anchors read
~18, not ~22–23.

**Where the gap lives (fieldwise).** Breaking the own-stack anchor gap down by field
(re-scored on the same five instances) localizes it sharply:

| Field | held-out values? | nano gap | scale gap |
|---|---|---|---|
| cc (chief complaint) | yes (3 held) | 45.3 ± 8.1 | 58.2 ± 3.5 |
| med (medication) | yes (2 held) | 23.8 ± 4.8 | 14.2 ± 2.3 |
| alg (allergy) | yes (1 held) | 22.5 ± 4.5 | 21.2 ± 4.5 |
| dur (duration) | no (numeric) | 0.0 | 0.0 |
| sev (severity) | no (closed 3-set) | 0.0 | 0.0 |

The entire gap sits in the three fields that *have* held-out vocabulary values (cc, med,
alg) and is **exactly zero** in the two whose values are all in-distribution (dur is
numeric; sev is the closed set {mild, moderate, severe}). The two zero-gap fields are an
internal control: under the *same* held-out templates, fields without held-out values
show no gap, so the effect is specific to copying novel lexical values rather than a
generic degradation on unfamiliar phrasing. It also explains the anchors' aggregate
near-identity (18.3 vs 18.7) masking field-level differences — nano and scale trade off
cc against med — and identifies the chief-complaint field, with its ~190 compositional
values, as the hardest copy. (Fieldwise breakdown for the Pythia rungs would need a
re-score with the adapters; a candidate appendix analysis.)

![Held-out copying gap vs scale on one consistent instrument. Own-stack anchors
(3.15M, 10M) sit at ~18 pts; the tested Pythia rungs (160M, 410M) at 3.5–4.2 pts;
1B is a training-run–bounded interval [0,5]. The shaded band marks the own→Pythia
stack change — the x-axis is not a pure scale axis (§7).](figures/fig1_gap_vs_scale.pdf)
*Figure 1. `papers/figures/fig1_gap_vs_scale.pdf` — generated by `papers/make_figures.py`
from the committed result JSONs.*

![Per-rung instance difficulty. The single public instance (inst0, ✗) exceeds the
five-instance mean ± SD at every rung; at 1B the two training runs (0 and 5) bracket
the interval. This is why a single-instance anchor read higher than the powered
mean.](figures/fig2_instance_difficulty.pdf)
*Figure 2. `papers/figures/fig2_instance_difficulty.pdf` — same generator.*

### 6.2 What the result does and does not establish

Within Pythia the gap is already small at 160M with no clean monotonic trend, so the
ladder captured a low-gap regime rather than a transition. Reconciling with Stage S:
the 3.2× step (3M→10M) did not move the gap (18.3±1.3 → 18.7±1.5 — a 0.4-point change
inside one SD, now measured rather than inferred from two single points); the 16×
step to 160M, **or** the stack change, coincides with a much smaller gap — these are
not separated here. The
formal band is not PERSISTS (the top-rung interval reaches 0, far from the ≥10 that
PERSISTS requires); it is consistent with a small residual or none.

### 6.3 Two distinct lessons

The paper carries two takeaways that are worth keeping separate, because they are
supported by different evidence and would survive independently.

**What we learned about language models.** The held-out copying gap is a *pronounced
small-model phenomenon*: ~18 points in from-scratch models at 3–10M, essentially flat
across that 3.2× scale step, and only single-digit points in the tested Pythia models
at 160M and above. The fieldwise breakdown makes the phenomenon precise rather than
diffuse: the gap lives *entirely* in the open-vocabulary fields and is exactly zero in
the closed-value fields, so what these small models fail at is specifically **copying a
held-out lexical value into an open slot**, not being unfaithful in general — and the
closed-value fields are a within-task control for that claim. What we do **not** get to
say is *why* it shrinks with the ladder — the nano→Pythia step changes scale together
with pretraining data, tokenizer, architecture, and finetuning method (§7), so the
honest claim is "the gap largely disappears by the Pythia pipeline," not "parameter
count removes it." The one clean causal statement is negative and comes from the
controlled within-stack step: at this recipe, going from 3M to 10M did not move the gap.

**What we learned about measurement.** Independently of the model result, estimating a
gap like this reliably required two corrections that the data forced on us. (i)
*Single-instance evaluation was underpowered, and worse, biased*: the public instance
is not merely noisy but *systematically hard* — its gap exceeds the multi-instance mean
at every rung — so a fixed public instance can skew an effect size even when it is
content-addressed and reproducible. (ii) *The precision floor moves with the regime*:
in the high-gap regime the dominant uncertainty is across evaluation instances, but by
1B the gap is small and the dominant uncertainty becomes the *training run* (two
fixed-seed retrains split 5/0), so the right replication unit changes from eval-instance
to training-seed. Both lessons are prescriptive for anyone measuring faithfulness gaps
in small models and hold regardless of how the scale-vs-stack question is eventually
resolved.

## 7. Limitations

- **Scale vs. stack confound (primary).** The nano→Pythia comparison changes parameter
  count *and* at least four other variables simultaneously, each a plausible alternative
  cause of the reduction: (i) **pretraining data quantity** (~200M own-stack tokens vs.
  Pythia's ~300B — a ~1500× difference), (ii) **tokenizer** (4098-vocab BPE vs. ~50k;
  larger vocabularies fragment field values like "ibuprofen" into fewer sub-tokens,
  which could itself change copy success), (iii) **architecture family**, and (iv)
  **finetuning method** (own-stack full fine-tune vs. Pythia LoRA r=16 — LoRA may
  preserve pretrained copying while suppressing the memorization pathway full FT
  exploits). We therefore claim only that the gap is much smaller under the Pythia
  *pipeline*, not that scale per se removes it. The pre-registered own-stack scale ladder
  (`PREREG_ownstack_160m.md`) is designed to separate scale from the stack bundle.
- **Transition point unobserved.** The gap drops somewhere between own-stack 10M (18.7)
  and Pythia-160M (3.5), but those endpoints are on different stacks; within Pythia the
  gap is already low at 160M with no monotonic trend, so "by Pythia scale" is safe while
  "as models scale up" is not yet supported.
- **1B point estimate.** Bounded, not identified, by training nondeterminism (§5.2).
- **Exact-match metric.** The gap is a difference of exact-match recalls; per Schaeffer
  et al. (2023) such metrics can exaggerate scale-linked transitions, which is part of
  why we report across-instance SD and, at 1B, a training-run interval.
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
justifying Stage T; the natural hypotheses are copy/retrieval circuitry — induction
heads (Olsson et al., 2022) and retrieval heads (Wu et al., 2024) — that generalize
poorly from seen to held-out values.

## 9. Conclusion

A severe held-out copying gap in sub-10M models — specifically a failure to copy
held-out lexical values into open slots, with the closed-value fields a zero-gap
internal control — was substantially smaller in the tested Pythia models; the cause
(scale vs. pipeline) is left open by design. The measurement story is inseparable from
the result: single-instance evaluation was under-powered, and at the top of the ladder
training nondeterminism — not evaluation noise — sets the precision floor. Both were surfaced by pre-registered checks
(equivalence, determinism) that failed productively. The contribution is less any
single number than a research unit that repeatedly incorporated its own measurement
limitations into its conclusions rather than explaining them away.

---

*Status: draft. The anchors are now re-scored on the multi-instance instrument
(`results_anchors_v2_{nano,scale}.json`), so the whole ladder shares one instrument
and the public-instance-hard-draw holds at every rung (single-instance gap ≥
multi-instance mean at all five). Figures produced (`papers/figures/`, regenerable via
`papers/make_figures.py`). Related-work drafted (§ Related work) with verified
citations. Running writing audit (unsupported claims, citations-needed, hypotheses,
reviewer limitations) in `papers/writing_audit.md`. Open decisions before submission —
precise abstract numbers pending a decision on multi-seed 1B; intro/abstract to be
tightened last (per plan); final section numbering/placement of Related work.*

## References

*Draft bibliography — titles/venues/years verified against primary sources; author
lists and exact arXiv IDs to be double-checked before submission (see writing_audit.md).*

- Biderman, S., Schoelkopf, H., Anthony, Q., et al. (2023). Pythia: A Suite for
  Analyzing Large Language Models Across Training and Scaling. *ICML 2023.* arXiv:2304.01373.
- Elhage, N., Nanda, N., Olsson, C., et al. (2021). A Mathematical Framework for
  Transformer Circuits. *Transformer Circuits Thread (Anthropic).*
- Gu, J., Lu, Z., Li, H., Li, V. O. K. (2016). Incorporating Copying Mechanism in
  Sequence-to-Sequence Learning. *ACL 2016.* arXiv:1603.06393.
- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large
  Language Models (Chinchilla). arXiv:2203.15556.
- Ji, Z., Lee, N., Frieske, R., et al. (2023). Survey of Hallucination in Natural
  Language Generation. *ACM Computing Surveys 55(12).* arXiv:2202.03629.
- Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). Scaling Laws for Neural
  Language Models. arXiv:2001.08361.
- Maynez, J., Narayan, S., Bohnet, B., McDonald, R. (2020). On Faithfulness and
  Factuality in Abstractive Summarization. *ACL 2020.* aclanthology 2020.acl-main.173.
- Olsson, C., Elhage, N., Nanda, N., et al. (2022). In-context Learning and Induction
  Heads. *Transformer Circuits Thread (Anthropic).* arXiv:2209.11895.
- Pham, H. V., Qian, S., Wang, J., et al. (2020). Problems and Opportunities in Training
  Deep Learning Software Systems: An Analysis of Variance. *ASE 2020.*
- Picard, D. (2021). torch.manual_seed(3407) is all you need: On the influence of random
  seeds in deep learning. arXiv:2109.08203.
- Schaeffer, R., Miranda, B., Koyejo, S. (2023). Are Emergent Abilities of Large Language
  Models a Mirage? *NeurIPS 2023.* arXiv:2304.15004.
- See, A., Liu, P. J., Manning, C. D. (2017). Get To The Point: Summarization with
  Pointer-Generator Networks. *ACL 2017.* arXiv:1704.04368.
- Wei, J., Tay, Y., Bommasani, R., et al. (2022). Emergent Abilities of Large Language
  Models. *TMLR.* arXiv:2206.07682.
- Wu, W., Wang, Y., Xiao, G., et al. (2024). Retrieval Head Mechanistically Explains
  Long-Context Factuality. arXiv:2404.15574.
- Xu, C., Guan, S., Greene, D., Kechadi, M.-T. (2024). Benchmark Data Contamination of
  Large Language Models: A Survey. arXiv:2406.04244.
- Zhuang, D., Zhang, X., Song, S. L., Hooker, S. (2022). Randomness in Neural Network
  Training: Characterizing the Impact of Tooling. *MLSys 2022.*
