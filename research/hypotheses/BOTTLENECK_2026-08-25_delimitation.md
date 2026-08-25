# The surviving bottleneck — five hypotheses, and the one experiment that separates them

**Date:** 2026-08-25
**Inputs:** `stage_m/PREREG_induction_curriculum.md`, `papers/RESEARCH_PROGRAM.md`,
`artifacts/RESULT-span-extent-reanalysis.md`, the landed span-port grid, and the
`trajectory/` JSONs behind Stages C / S / P2 / C-1b / C-3.

---

## 0. The correction that has to come first

The program treats "the gap" as one phenomenon. It is two, on two instruments,
and conflating them is why the fifth hypothesis below has never been on the list.

| | **scribe line** (Paper α) | **span-port line** (P1) |
|---|---|---|
| task | dialogue → `CC: … \| DUR: … \| MED: …` | per-atom: assertion label + evidence quote |
| model | nano 3.15M / scale 10M / Pythia / own-stack 160M | Qwen2.5-1.5B-Instruct |
| scored | exact field-value match | exact evidence span |
| **can an extent error even occur?** | **No** — the field template delimits the answer | **Yes** — the model chooses the quote's extent |

**In the scribe task a delimitation error is inexpressible.** The template
supplies the boundaries; the model emits a value into a slot. So the scribe
line's gap cannot be a delimitation failure, and the four standing hypotheses
are correctly scoped *to that line*.

The span-port line is where extent is a free parameter — and there, extent is
where essentially all the loss lives.

## 1. What each line's evidence actually shows

**Scribe line — retrieval, and the evidence is decisive.** Stage P2's readout is
**teacher-forced held-value *first-token* top-1: 21% held vs 92% seen**
<file path="stage_m/PREREG_induction_curriculum.md" lines="68-75" />. First-token
top-1 is a pure content-addressed *selection* measure — it is scored before any
extent decision exists. A model that could locate the value and merely
mis-delimit it would score high here. It scores 21%. **The scribe line's
bottleneck is retrieval.** That result stands untouched by anything below.

**Span-port line — delimitation.** On the canonical form, with all three leak
channels closed: the model selects the correct conversational turn for
**97/120** gold-bearing slots (81%) and delimits the gold span within it for
**2/120**. Located quotes are turn-scale — median 29 chars against gold median 8,
median quote/enclosing-turn ratio **1.000**, 1 of 95 exceeding its enclosing
turn. And the dissociation: a question-form edit moved exact-extent **8×**
(16 → 2) and moved LOCATED **not at all** (95 → 97)
<file path="artifacts/RESULT-span-extent-reanalysis.md" />.

## 2. The five hypotheses, their discriminating predictions, and what already tests each

| # | Hypothesis | **Discriminating prediction** (what only it predicts) | Existing artifact that partially tests it | Standing |
|---|---|---|---|---|
| H1 | **Induction-circuit capacity** — the trunk lacks a general content-addressed copy circuit | Held-value **first-token** top-1 stays at chance-ish *regardless* of output format, and an induced copy circuit lifts it | Stage P2: an explicit copy-dominant pointer head reached 21% vs 92%. Stage M is built to test the lift but has **no RESULT** | **Live on the scribe line.** Predicts nothing about extent — silent on the span-port finding |
| H2 | **SFT objective** — full-FT destroys or fails to build the copy pathway | The gap tracks *adaptation regime*, not capacity: LoRA ≫ full-FT at matched data | `C_ADAPT_DATA_CELLS`: at 159M own-stack, 16.9 (200M/full-FT) vs 7.1 (200M/LoRA) vs 4.2 (3.2B/LoRA). Strongly suggestive; E2 would identify it but is **GATED/STOP** | **Live on the scribe line.** Best-supported of the four |
| H3 | **Pretraining data diversity** — low diversity selects a positional shortcut over an induction head | OOD failures carry a **positional signature** — the model emits the value at a *fixed offset* (`(ℓ₁+ℓ₂)/2 + 2`, or leftmost) rather than the content-matched one, and the failure rate falls as context-length diversity rises | `C_DIVERSITY`: the D5→D20→D80 slot sweep, +66.7 pts. Consistent, but the sweep varied *slot-value* diversity, not context-length diversity, and **no error census was run for a positional signature** | **Live, and under-tested.** The specific signature has never been looked for |
| H4 | **Morphology** — re-inflection is the causal residual | Misses concentrate on morphologically variable values and vanish for invariant ones | C-3 error census: ~44% of core-cell misses. `C_MORPH_DESC` is **descriptive only**; `C_MORPH` is `SPECULATION` / wording `FORBIDDEN` | **Live but weakest.** No causal test exists |
| **H5** | **Delimitation / lexical-unit boundary** — the model addresses content at *unit* granularity (turn, clause, lexical item) and cannot resolve sub-unit boundaries | Retrieval and extent **dissociate**: manipulations that move extent leave locating untouched, and vice versa. Errors are *containment* errors, not *wrong-location* errors | **Already partially tested, and it passed:** `artifacts/span_extent_L000_unified.json` — LOCATED 97/120 invariant while exact-extent moved 8× under a phrasing edit; 79/104 non-grounded slots contain the gold span; 50 of 79 over-extended slots *also* have the correct assertion state | **New. Live on the span-port line only** |

## 3. Why H5 is not exotic — the mechanism is already described

The dual-route model of induction identifies **two** induction circuits: token-level
heads that copy verbatim token sequences, and concept-level heads that copy
*meaning*, whose attention consolidates on the **last token of a multi-token
lexical unit** — i.e. concepts are addressed as whole units, not as spans with
boundaries ([Feucht et al., *The Dual-Route Model of Induction*](https://www.alphaxiv.org/abs/2504.03022)).
Ablating the token-level route in that work leaves the concept route intact and
the model **paraphrases** rather than copying — right meaning, wrong surface
form, which the authors liken to deep dyslexia.

That is the mechanistic shape of what the span-port data shows: content resolved
correctly at unit granularity, surface extent wrong. A model relying on the
concept route to find the evidence, with no boundary-resolving mechanism to
delimit within the located unit, produces exactly a median quote/turn ratio of
1.000.

H3 also has a sharper form than the program has used. The phase-transition
analysis predicts the low-diversity failure is a **positional shortcut** with a
specific arithmetic error signature, and that biasing pretraining toward *longer*
contexts induces the head more cheaply than diversifying short ones
([*From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers*](https://www.alphaxiv.org/abs/2512.18634)).
Neither the signature nor the context-length lever has been tested here.

## 4. It also explains a standing puzzle — with a caveat

Stage C (curriculum), Stage S (scale) and Stage P/P2 (pointer head) all failed to
move the scribe gap. A unifying reading is that all three targeted retrieval.
**But that reading is only available for the span-port line**, and P2's
first-token evidence says retrieval genuinely *is* the scribe line's problem. So
the honest statement is narrower than "the program's diagnosis is inverted":

> The four standing hypotheses are correctly scoped to the scribe line, where
> P2's first-token result confirms a retrieval deficit. The span-port line has a
> different dominant bottleneck — delimitation — and the program has been
> carrying results between the two lines as though they measured one thing.

## 5. The single next experiment

**E-DELIMIT — forced-choice span selection.** Hold retrieval fixed; make
delimitation trivial; see whether the loss disappears.

Same 12 instances × 16 slots, same model, same greedy decoding, same leak
channels closed, canonical unified question form. One factor, three levels,
scored on the identical slot set so the contrast is paired:

| arm | output the model must produce |
|---|---|
| **A (control)** | current free-form quote |
| **B (menu)** | the located turn's candidate sub-spans enumerated; model picks an index |
| **C (offsets)** | character start/end offsets into the transcript |

Arm B is the discriminator. It removes *generation* of the boundary and leaves
only *selection* of it, while leaving retrieval exactly as hard.

**Pre-registered predictions.**
- **H5 true:** B converts most located slots into grounded ones — `asserted_grounded`
  in B ≥ **60% of LOCATED** (≥ 58/97), against 2/120 in A.
- **H1/retrieval true:** B changes almost nothing; the model cannot pick the right
  sub-span because it never had the value, so grounded stays < 25% of LOCATED.

**Pre-registered KILL condition for H5:**

> If arm B's `asserted_grounded` is **< 25% of LOCATED (< 24/97)**, H-delimit is
> **REFUTED for this model**. The boundary is then not merely unexpressed but
> absent from the representation, and the span-port line rejoins the retrieval
> hypotheses rather than standing apart from them.

**Between 25% and 60% → WEAKENED**; report both arms and do not round toward a
verdict.

**Guards, from `docs/RUNBOOK_contrast_hygiene.md`.** (R1) arms must differ only in
the output-format module — `git diff` must show nothing else, and in particular
the *question* is byte-identical across arms. (R2) all three arms must carry the
same prompt-template hash; if one arm is re-run, all are stale. (R4) LOCATED is
reported only with its length bound. (R5) the manipulation check is a
**menu-parrot** that picks index 0 always — it must score at chance, not at
ceiling, or arm B is measuring menu construction rather than the model.

**Power.** Paired binary over ≥97 informative slots, exact McNemar
(`nanoscribe/mcnemar_c1.py`); d ≥ 6 one-directional gives p < 0.05. The predicted
effect (2 → ~58) is an order of magnitude past that, so this is well-powered at
K=12 with no re-pilot needed — the one contrast in this program that is not
effect-size limited.

**Cost.** Three arms × 192 slots, local MPS, ~6.5 min/arm, \$0.

**What it does *not* settle.** Nothing about the scribe line. If H5 is confirmed,
the scribe line still needs Stage M for H1 and a re-scoped E2 for H2 — and the
program should stop transporting conclusions between the two instruments without
an argument.
