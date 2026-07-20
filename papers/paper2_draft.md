# Scale doesn't buy copying: a within-stack control isolates the training stack in a held-out value-copying failure

*Working draft — Paper 2 (causality). Companion and sequel to Paper 1 ("Held-out value
copying in small language models"). All numbers trace to immutable JSONs under
`trajectory/`; instrument identical to Paper 1 (5×(100 held + 100 seen), gap = seen −
held recall, mean ± across-instance SD). Status: full-FT, LoRA, Chinchilla, the 200M+LoRA seed duplicate, the diversity
sweep, and the 3.2B+LoRA factorial corner (both training seeds) have all landed —
Phase A is closed. The corner compounds the two escapes to ≈Pythia level
(4.2 ± 0.9 diluted), while a shared ~15–18-pt clean residual persists in both
stacks. Phase C (residual mechanism) is also closed: C-1b (lexical interference)
REFUTED; its follow-up C-3 (transition/boundary/length) REFUTED two of three
registered factors and left the third UNRESOLVED but noise-dominated — see §3.5.*

## Abstract (draft)

Paper 1 measured a severe held-out value-copying failure in small from-scratch language
models (~18 points diluted, ~80–87 points on the value-level metric at 3–10M) that is
far smaller under the Pythia pipeline (3.5–4.2 points at 160M–410M) — but the comparison
confounded parameter count with the entire training stack. Here we run the pre-registered
within-stack control: a **160M-parameter model in the same architecture family, tokenizer,
pretraining recipe (~200M FineWeb tokens), and finetuning method** as the small anchors,
measured on the identical instrument. The gap **does not close: 16.9 ± 1.7** diluted
(66.6 ± 5.0 value-level) — statistically indistinguishable from the 3.15M anchor. Across
**50× of within-stack scale** (3.15M → 10M → 160M) the copying gap is flat
(18.3 → 18.7 → 16.9), while Pythia at the *same parameter count* reads 3.5. The
pre-registered decision rule fires **stack-dominant**: parameter count alone does not
produce the collapse; properties of the training stack do. Two further arms then vary one factor each: swapping only the finetuning method
(full FT → LoRA r=16 on the same checkpoint) drops the gap to **7.1 ± 1.2** diluted
(29.6 ± 3.7 value-level); separately, scaling only the pretraining data to
Chinchilla-optimal (200M → 3.2B tokens, full FT retained) drops it to **7.0 ± 1.0**
(29.4 ± 4.0) — *indistinguishable at our current resolution (one training run per
cell), from two entirely different interventions*. Data quantity and adaptation method are therefore **substitutes, not
additive components**: the large gap is specifically the interaction of an
*under-trained base* with *aggressive full-parameter adaptation*, and fixing either
factor alone recovers the same ~10 points. Both finetuning methods drive training loss
to ≈0 — the difference is not whether the finetuning set is memorized but whether the
fit destroys the pretrained copy pathway (full FT on a weak base) or leaves it intact
(LoRA, or full FT on a robust base). A residual ~2× versus Pythia-160M (3.5 ± 0.7)
remains for the still-unseparated breadth/tokenizer bundle. Per-slot structure sharpens the
attribution: the residual failure concentrates in the lowest-diversity slot in every
own-stack configuration and at Pythia-410M (total failure on the single held allergy
type), though not universally — one Pythia-1B training draw largely solves it — a
pattern consistent with a per-slot competition between memorization and copying — a
**binding-and-coverage** account: reliable copying requires both sufficient slot
diversity and adequate token coverage, with the allergy slot the strongest instance of
the mechanism rather than its definition. A
pre-registered, type-controlled diversity sweep then tested this directly: raising one
slot's training diversity from 5 → 80 values, at fixed everything-else, lifts held-type
recall by **66.7 points** with categorical per-type flips (position controlled; the
fully-token-covered probe type flips earliest) — the slot-diversity hypothesis is
**supported**, with a residual token-coverage factor at the margin. A training-seed
duplicate bounds per-cell run variance at ±~1.3 pts, inside the pre-registered rule, so
the single-run cells stand.

## 1. The question Paper 1 left open

Paper 1's primary limitation was explicit: the nano→Pythia comparison changes parameter
count *and* pretraining data (~1500×), tokenizer (4098 vs ~50k vocab), architecture
family, and finetuning method (full FT vs LoRA) simultaneously. Its one controlled
within-stack step (3.15M → 10M, 3.2×) moved nothing. The open question: **is the collapse
a scale effect or a stack effect?**

## 2. Design (pre-registered)

`trajectory/PREREG_ownstack_160m.md`, fixed before any run, including the decision rule:
diluted gap ≥ 14 → STACK-dominant; ≤ 6 → scale-plausible within family; 6–14 → extend the
ladder (40M/80M).

- **Model:** own-stack GPT family (RoPE, GQA 4:1, SwiGLU, RMSNorm, tied embeddings,
  4098-vocab BPE, 512 ctx) at d=1024, L=14, H=16, KV=4, hd=64, ff=2752 → **159.3M**.
- **Held identical to the anchors:** pretraining recipe (~200M FineWeb tokens, D≈20N for
  the anchors; deliberately *not* rescaled — the "identical recipe" comparison), scribe
  finetune (v2 data, byte-identical generator, full FT, 3 epochs, LR 1e-4), scorer,
  eval instances (m0–m4 + inst0), both metrics (diluted + clean).
- **Deviation (pre-authorized):** effective batch 32 realized as micro-8 × accum-4
  (T4 memory); optimizer trajectory unchanged.
- **Method arm (2×2):** the same pretrained 160M checkpoint finetuned with LoRA r=16
  α=32 (98 wrapped modules, 4.03M trainables) instead of full FT — isolating the
  finetuning-method member of the stack bundle.

## 3. Results

### 3.1 The within-stack curve is flat across 50×

| model | params | diluted gap | clean (value-level) gap |
|---|---|---|---|
| nano | 3.15M | 18.3 ± 1.3 | 87.3 ± 2.7 |
| scale | 10M | 18.7 ± 1.5 | 79.5 ± 2.1 |
| **own-160M (full FT)** | **159.3M** | **16.9 ± 1.7** | **66.6 ± 5.0** |
| pythia-160m (reference) | 162M | 3.5 ± 0.7 | 14.7 ± 2.1 |

Model quality scales normally — pretrain val loss 2.86 (vs 3.28 at 10M, 3.96 at 3.15M),
scribe parse 100%, base control parse 0% — yet the copying gap barely moves. The
pre-registered rule fires **stack-dominant**. (The same N=1 caution as §3.2 applies to
this verdict itself: the 16.9 cell is a single training run — the pre-registered
duplicate-finetune study bounds this; the *direction* is safe at ~6 instance-SDs from
Pythia's 3.5, but the point value is provisional.) The clean metric declines modestly with
scale (87.3 → 79.5 → 66.6) — capacity helps at the margin — but at 160M own-stack the
pure held-out-value failure is still **4.5×** the Pythia-160M value; per-field, the
allergy slot remains at **100.0** (clean) at all three own-stack scales.

### 3.2 Factor isolation — data and method are substitutes

| own-stack 160M, diluted (clean) | 200M tokens | 3.2B tokens (Chinchilla) |
|---|---|---|
| **full FT** | 16.9 ± 1.7 (66.6 ± 5.0) | **7.0 ± 1.0 (29.4 ± 4.0)** |
| **LoRA r=16** | **7.1 ± 1.2 (29.6 ± 3.7)** | **4.2 ± 0.9 (17.7 ± 3.2)** |

Reference: pythia-160m (~300B tokens, LoRA) 3.5 ± 0.7 (14.7 ± 2.1).

Each single-factor intervention was run against the 200M+full-FT corner, and the result
is striking: **7.1 ± 1.2 vs 7.0 ± 1.0** — LoRA-on-a-weak-base and full-FT-on-a-strong-base
land on indistinguishable gaps — though they change *entirely different* variables.
(Our own §5.2-style lesson demanded a seed bound, and it has now been measured:
a pre-registered duplicate of the 200M+LoRA cell at a different training seed reads
5.76 ± 1.31 vs. the original 7.08 ± 1.22 — |Δ| = 1.32 pts, inside the pre-registered
≤2-pt rule, so **single-run cells stand** with a ±~1.3-pt training-seed band. The two
~7s are "equal" at that resolution; the 16.9 → ~7 separation is far outside it.)
A naïve additive reading of the method arm alone would have attributed ~73% of the
own→Pythia difference to the finetuning method; the Chinchilla control refutes
additivity — data quantity alone recovers the *same* ~10 points. The correct structure is
an **interaction**: the large gap lives specifically in the under-trained-base ×
full-parameter-adaptation cell, and either escape route out of that corner recovers most
of it. The factorial corner — both escapes at once — completes the grid: **3.2B + LoRA
reads 4.2 ± 0.9** (clean 17.7 ± 3.2), firing the pre-registered ≤4.5 rule: the two
escapes *compound to the Pythia level itself* (3.5 ± 0.7; clean 14.7 ± 2.1), so the
entire own-stack↔Pythia difference is attributable to **data quantity + adaptation
method, with tokenizer and architecture ~innocent**. The grid reads 16.9 (neither) →
7.0/7.1 (either) → 4.2 (both). Most strikingly, the corner's per-slot fingerprint is
*identical to Pythia's* — cc 0.0, med 0.0, alg 100.0: a fully "Pythia-like" own-stack
model, down to the residual's identity. The corner's seed duplicate returned
**behaviorally identical** results (|Δ| = 0.00; verified distinct finetune) — and its
per-instance vector is *identical to pythia-410m's*: three models, two stacks, different
seeds, metrically indistinguishable because they occupy the same categorical flip state
(cc ✓, med ✓, alg ✗) and aggregate metrics are composition arithmetic over that state.
Contrasted with |Δ| = 1.32 at the weak-base 200M+LoRA cell (boundary types present),
this confirms **variance is boundary-localized at both poles** — and that the flip
matrix, not the scalar gap, is the fundamental object. Two
sharpening observations:

1. **Both methods memorize.** Full FT and LoRA both reach ≈0 training loss and 100%
   parse — the difference is not *whether* the finetuning set is fit but *how*: the
   full-parameter fit overwrites the pretrained copy pathway; the 4M-parameter low-rank
   fit reaches the same training loss while leaving it intact — an implicit-regularization
   account of the failure, not a capacity account.
2. **The slot gradient survives the own-stack interventions — with one important
   exception elsewhere.** Under LoRA at 160M, clean per-field (means ± SD over the five
   instances): cc **0.0 ± 0.0** (solved — the complaint-copy pathway *exists* in the
   own-stack pretrained model; full FT on the weak base was destroying it: full FT reads
   cc 64.4 ± 7.1), med 47.1 ± 4.0, alg **100.0 ± 0.0**. Under Chinchilla pretraining
   with full FT: cc 9.2 ± 4.4, med 25.5 ± 6.1, alg **100.0 ± 0.0** again. The allergy
   slot is at total failure in **all five own-stack configurations** and at pythia-410m;
   pythia-160m reads 83.6 ± 4.6; but the pythia-1b *third training draw* reads
   **24.6 ± 10.3 — largely solved**, and the Pythia sequence (83.6 → 100.0 → 24.6) is
   non-monotonic — and a fourth 1B training draw (peer-run, comparability again
   failed-as-designed: another distinct draw inside [0,5]) reads alg clean **15.8**
   with every other field at 0.0. Across four 1B draws the allergy slot spans ~0–25
   while cc/med sit at 0.0 in all of them: **training-run variance concentrates in the
   hardest slot** — the "copying boundary" behind §5.2-style bistability is per-slot,
   and at 1B it is specifically the allergy slot. Two honesty notes follow. (i) Every alg number is a *type-level n=1*
   measurement — "sulfa drugs" is the only held allergy type, so the ±0.0 instance-SDs
   are repeated measures of one string, not precision about the slot. For med (two held
   types) this is now *proven*, not surmised: LoRA's and full FT's per-instance med gaps
   are **bit-identical on all five instances** — because both models copy melatonin 100%
   and throat-lozenges 0%, making each instance's med gap exactly
   100 × share(lozenges), a pure function of eval-instance composition (verified
   instance-by-instance to the last digit). Copy competence at this granularity is
   *categorical per held type*, and the ±4.0 "SD" is composition noise, not behavioral
   variance; the Chinchilla cell breaks the composition function (25.5 ± 6.1),
   i.e. data-scaling partially rescues lozenges. The same composition function holds
   *bit-identically* at the 10M anchor — so across the whole own stack the med story is
   a single **type flip**: melatonin fails at 3.15M (med = 100.0), becomes categorically
   solved by 10M, and stays solved under every own-stack configuration, while
   throat-lozenges never flips in-stack under any scale, method, or (fully) data budget.
   Copy competence appears to be acquired *per lexical type*, discretely. (ii) The
   diversity gradient (190/18/5) is three correlated points, confounded with held-type
   count, subword fragmentation (every held value contains ≥1 token never emitted in
   any training output), and field position (alg is template-final). The slot-diversity
   hypothesis therefore remains a *hypothesis*: the designed sweep must vary held-type
   identity and control position/tokenization before "diversity governs copy-vs-classify"
   can be asserted.

### 3.3 Familiar structure at the new rung

The public instance is again the hard draw (inst0 28.0 vs fresh 16.9; by the corner
run the streak reached 9/9 rungs) — and the type-composition lens *explains* it: inst0's
held dialogues over-sample exactly the never-flipping types (held-allergy 40% vs the
fresh instances' 21%, held-cc 85% vs 68%, lozenges-heavy med), so its gap is
mechanically elevated at every rung. The "systematically hard draw" of Paper 1 §5.1 is
composition, not chance —
and the per-slot pattern replicates (clean means: cc 64.4 ± 7.1, med 47.1 ± 4.0, alg
100.0 ± 0.0) — the slot gradient shapes the failure at 159M as at 3M.

### 3.4 The slot-diversity sweep — diversity causally induces copying

The pre-registered type-controlled sweep (`PREREG_slot_diversity.md`; scale-10M frozen
base, full FT per arm, 6 fixed held types, ~83 items/type) delivered its verdict:
**H-slot SUPPORTED** — diversity effect D80−D5 = **66.7 pts** (rule required ≥30),
monotonic (0 → 24.5 → 66.7), position innocent (|D20pos−D20| = 3.1 ≤ 5).

| held type | D5 | D20 | D80 |
|---|---|---|---|
| bee stings | 0 | 15 | **100** |
| ibuprofen (token-probe) | 0 | **69** | **100** |
| wool | 0 | 0 | **100** |
| strawberries | 0 | 62 | **100** |
| sulfa drugs | 0 | 0 | **0** |
| ragweed pollen | 0 | 0 | **0** |

Three readings. (i) Raising ONE slot's training diversity, at fixed scale, data size,
method, and eval, causally induces held-out copying on that slot — and the flips are
**categorical per type**, as the type-flip account predicts. (ii) D5's zero across all
six types kills the "sulfa-drugs-specific string" alternative for the baseline regime.
(iii) Two types never flip even at 80 training values, while the ibuprofen probe (whose
tokens are fully train-output-covered) flips earliest — consistent with a *second*,
token-coverage factor at the margin (hypothesis; analyzable from this data without new
runs). The position control also localizes: mean-level position effects are excluded,
though individual type flips vary across arms/draws (type-level draw sensitivity, noted).

### 3.5 Phase C — residual mechanism: lexical interference, then transition/boundary/length

Two pre-registered follow-ups asked *why* a held value, once diversity makes copying
possible, still sometimes fails to complete. Both are mechanically closed.

**C-1b (lexical interference, `PREREG_token_coverage.md`):** REFUTED. flip(I-iso) −
flip(I-contain) = 71% − 75% = **−4 pts** (rule ≤15). Containment substitution signature
0/77. Failure-mode census (402 misses) found the dominant pattern was **tail truncation
at word boundaries** (38%) — and specifically that the *novel* half of a compound value
survives while the *trained* half is dropped (e.g. ragweed pollen → "ragweed"), the
opposite of what the interference account predicted. This motivated C-3.

**C-3 (transition/boundary/length, `PREREG_C3_binding_probe.md`):** a T×B×L factorial
(8 cells × ≥5 types, 3 FT seeds, orthogonality hard-gated) tested three named mechanistic
accounts against C-1b's truncation pattern. **None reached SUPPORTED.** H-transition
(does junction-bigram availability predict completion) REFUTED, +1.7pts. H-boundary
(does a subword vs. whitespace junction predict completion) REFUTED, −8.3pts. H-length
(short vs. long) UNRESOLVED at +25pts, but Wilson intervals on the underlying cell rates
(n=2–6/cell) overlap for every matched short/long pair — the point estimate is
noise-dominated at this sample size, not a suppressed trend. A dedicated truncation-locus
check (does a B-space miss actually truncate at the whitespace junction, the mechanism
C-1b's descriptive pattern implied) confirmed only 12.4% of cases (rule ≥60%) — **C-1b's
truncation pattern did not survive controlled factorial manipulation** at this
resolution. Because H-length is UNRESOLVED rather than REFUTED, the pre-registered
H-stochastic gate (all three REFUTED + ≥20% seed-unstable ⇒ promote a representation-level
probe) does not fire, despite 24.7% seed instability being independently consistent with
it — stated as a frozen-rule outcome, not a judgment call.

The mechanistic result instead points somewhere the design wasn't built to test: across
strata, the dominant miss is not truncation but **morphological re-inflection** — chiefly
the model emitting a differently-inflected form of the correct value (daisy seed →
"daisy seeds", rose hips → "rose hip"; ~44% of core-cell misses, ~54% restricted to the
truncation-locus population specifically) rather than truncating, substituting, or
omitting. Full verdict table, uncertainty bands, and error census in
`PREREG_C3_binding_probe.md` RESULT (independently triple-cross-checked: the kernel, a
from-scratch recompute, and a second independently-authored harness, `recompute_c3.py`,
all agree exactly on every frozen number after a bug in the latter — an unstable-type
exclusion that was computed but never applied — was found and fixed).

## 4. What this establishes, and what it doesn't

**Established:** within this stack and recipe, parameter count from 3.15M to 159M does
not produce the low-gap regime that the Pythia pipeline exhibits at the same size. The
collapse Paper 1 measured is a **stack effect**, not a scale effect, under the tested
conditions. Combined with the slot structure, the natural reading is that copying is
induced by *pressure* (slot diversity) and *pretraining breadth*, not by capacity alone.

**Established by the factor isolations:** the large gap is an **interaction** — it
requires *both* an under-trained base *and* full-parameter adaptation; removing either
(LoRA, or Chinchilla-scaled data) recovers the same ~10 points to ≈7. Additive
attributions ("X% method, Y% data") are refuted by the substitutability. A ~2× residual
versus Pythia persists under every single-factor fix.

**Not established:** the missing factorial corner (3.2B + LoRA — do the two escapes
compound toward Pythia's 3.5, or floor at ~7? base checkpoint preserved, ~30-min run);
the method effect at the *small* anchors (does LoRA rescue 3–10M models, or does escape
require 160M-scale capacity?); which member of the remaining breadth/tokenizer bundle
carries the final 2×; training-run variance (single run per cell; the Chinchilla cell
also changes venue — H100 vs T4).

**Established by Phase C (§3.5):** none of the three named lexical/positional mechanisms
(junction-transition availability, boundary type, value length) explain residual
completion failures at the ≥40pt threshold this program requires before naming a cause.
C-1b's own motivating descriptive pattern (word-boundary truncation) does not survive
controlled factorial manipulation. H-length remains formally UNRESOLVED (not REFUTED) but
the Wilson-band evidence argues the design was underpowered there, not that a real length
effect is being masked.

A post-hoc (not pre-registered) corpus check on the dominant unpredicted failure mode —
morphological re-inflection, chiefly pluralization — found a specific, concrete candidate
mechanism: the held phrases themselves never appear in training in either grammatical
form (properly held out), but their bare stems do, asymmetrically, via *unrelated*
genuinely-trained values sharing the same stem ("sting" 0× / "stings" 64× via the trained
value "wasp stings"; "seed" 0× / "seeds" 145× via "poppy seeds"/"sunflower seeds"; "hip"
304× / "hips" 0× via incidental complaint-text mentions). In 3 of 4 testable stems, the
model's emitted (wrong) inflection is exactly the corpus-majority bare-word form,
regardless of the held value's true number — consistent with a word-level (not
phrase-level) inflectional prior from incidental stem exposure overriding faithful
reproduction. Descriptive and hypothesis-generating (n=4 stems), not a confirmed causal
claim requiring its own decision rule. **Not established by Phase C:** whether this
generalizes beyond the 4 stems checked, or whether it is itself a cause or a downstream
symptom of the same representational bottleneck the three refuted factors were trying to
locate.

## 5. Next decompositions (designed, not run)

1. **The missing factorial corner: 3.2B + LoRA** — the preserved Chinchilla base makes
   this a ~30-minute finetune. Compounding (→ ~3.5–5) would reduce the entire own↔Pythia
   difference to data×method with the tokenizer innocent; flooring (≈7) would implicate
   the breadth/tokenizer bundle as a hard residual.
2. **LoRA at the anchors** — LoRA-finetune the frozen 3.15M/10M bases (base-matched:
   nano from dpo.pt, scale from scale10m_pretrain.pt): does escape require capacity?
3. **Tokenizer swap** — own-stack with a ~50k vocab: is the residual value fragmentation?
4. **Slot-diversity intervention** — vary the allergy slot's training diversity at fixed
   scale: the direct test of the hypothesis (type-controlled: multiple held types per condition, position varied).
5. **Duplicate finetunes** — training-run variance per cell (single run each; Chinchilla
   cell additionally changes venue, H100 vs T4).
6. **Morphological re-inflection follow-up (Phase C successor, not yet pre-registered)** —
   C-3's dominant unpredicted failure mode. The cheap no-run corpus-statistics check is
   done (§3.5/§4): 3 of 4 testable stems show the model defaulting to the corpus-majority
   bare inflected form via incidental exposure through unrelated trained values, not a
   candidate-vocabulary artifact. Next: a registered probe testing whether this
   generalizes beyond the 4 stems checked, and whether it is upstream cause or downstream
   symptom of the same bottleneck the three refuted T/B/L factors were targeting. C-3's
   own decision rule does not mandate this — H-stochastic did not fire — so pursuing it
   is an owner-level choice, not a promoted next stage.

---
*All artifacts: `results_ownstack_v2_160m_fullft.json` (immutable), kernels
`kaggle_ownstack_160m{,_lora}.py`, decision rule in `PREREG_ownstack_160m.md`.*
