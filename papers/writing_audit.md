# Paper 1 — writing audit (running list)

Maintained during drafting per the write-first plan. Four lists: unsupported claims,
citations-needed, hypotheses (must be labeled, not concluded), reviewer-facing
limitations. This feeds the Limitations section and guards against overstatement.
Update as the draft evolves. Legend: ☐ open · ☑ handled · ⚠ needs owner decision.

## 1. Unsupported / soft claims (need evidence or hedging)

- ☑ "order of magnitude larger than any noise source" (§6.1) — quantified: reduction
  ~14–15 pt vs anchor SD ~1.3–1.5 and Pythia SD ~0.7–0.9 (≈10×). Keep as the ratio.
- ☑ Model-side quality ranges (§6.1) — verified against per-rung JSONs (inst0+fresh);
  corrected own-stack parse 95→**94**–100% and Pythia hallucination 0.5→**0**–4% (1B is
  0.0%). Recall 80–91% (own) and 96–100% (Pythia), hallucination 7–12% (own) confirmed.
- ☐ Companion verification architecture "at measured precision/review-load" (§1, §7) —
  those metrics live in the companion write-up, not this paper; ensure the sentence
  doesn't imply they are measured *here*. Either cite the companion or soften.
- ☐ "first model to pass the pre-registered faithfulness bars" (Stage S, §3) — true
  within this project's own bars; make sure it doesn't read as a field-wide claim.
- ☐ Abstract point numbers (18.3±1.3 etc.) are final for the anchors/160M/410M but the
  1B representation (interval vs point) is still an open decision (⚠ below).

## 2. Citations needed (spot → reference)

- ☑ Pythia as controlled scaling platform (§4) → Biderman et al., 2023.
- ☑ Faithfulness / intrinsic-vs-extrinsic hallucination (§1) → Maynez et al., 2020; Ji
  et al., 2023.
- ☑ 1B training nondeterminism / tooling-level (§5.2) → Zhuang et al., 2022; Picard, 2021.
- ☑ Exact-match metric can exaggerate scale transitions (§7) → Schaeffer et al., 2023.
- ☑ Copy/retrieval mechanism hypotheses (§8) → Olsson et al., 2022; Wu et al., 2024;
  (lineage: Gu et al., 2016; See et al., 2017 — in Related work).
- ☑ Contamination framing / generator benchmark (§5.1, Related work) → Xu et al., 2024.
- ☐ Chinchilla D≈20N pretraining budget (Related work "pretraining data quantity",
  PREREG) → Hoffmann et al., 2022 — add in-text where token/param ratios are discussed.
- ☐ Verify author lists + exact arXiv IDs for all References entries (esp. Pham et al.
  2020 ASE, Zhuang et al. 2022 MLSys, Wu et al. 2024, Xu et al. 2024) before submission.

## 3. Hypotheses — must be labeled as hypotheses, NOT conclusions

- ☐ "the gap rides on a handful of hard held tokens" (§5.1) — diagnostic hypothesis for
  the single-instance variance; supported by the SD drop but not directly isolated.
  Phrase as "we attribute this to…", not "because".
- ☐ 1B bifurcation "sits at the boundary of perfect held-out copying" (§5.2) — a
  mechanistic conjecture for why two retrains split 5/0; label as conjecture.
- ☐ Retrieval/induction heads as the mechanism (§8) — explicitly a hypothesis for Stage
  M, not a result of this paper. Already framed as "natural hypotheses" — keep.
- ☐ Alternative causes of the reduction (pretraining-data quantity, tokenizer
  granularity, LoRA-vs-full-FT) (§7) — these are *candidate* explanations the current
  data cannot rank; present as an enumerated hypothesis set, not ranked causes.
- ☑ Scale-vs-stack: correctly left OPEN, not concluded (abstract, §6.2, §7, §9).

## 4. Reviewer-facing limitations (feeds §7; consolidated from draft + consult panel)

Ranked by expected reviewer severity (from the 10-model consult, 2026-07-18):
1. ☑ **Scale-vs-stack confound (primary).** Enumerated sub-confounds now in §7:
   pretraining-data quantity (~200M vs ~300B, ~1500×), tokenizer (4098 vs ~50k vocab →
   value fragmentation), architecture, and finetuning method (full-FT vs LoRA r=16).
2. ☑ **Transition point unobserved** — endpoints on different stacks; "by Pythia scale"
   safe, "as models scale" not (§7).
3. ☑ **LoRA-vs-full-FT** — surfaced as a first-class confound in §7 (panel flagged it as
   the most under-weighted). The pre-reg confound experiment should control it (2×2).
4. ☑ **1B interval interpretation** — training-run-bounded [0,5], not a point (§5.2, §7).
5. ☑ **Exact-match metric artifact caution** (§7; Schaeffer et al., 2023).
6. ☐ **Undertraining risk** — relevant to the *planned* own-stack 160M rung: 200M tokens
   is far below Chinchilla for 160M; a high-gap result there must report train/val
   curves + token/param ratios and be framed as "within this recipe", not "architecture
   bad". (Lives in PREREG_ownstack_160m.md; note here so the paper's future-work text
   pre-empts it.)
7. ☐ **Single task / single generator / single scale family** — generality untested (§7).
8. ☑ **Field heterogeneity** — RESOLVED for the anchors (`fieldwise_anchors.py`,
   `results_fieldwise_anchors.json`, added to §6.1). The gap is entirely in the three
   open-vocabulary fields (cc/med/alg) and **exactly 0** in the two closed-value fields
   (dur/sev) — a built-in control confirming the effect is held-out-*value* copying, not
   generic template degradation. Strengthens, not weakens, the phenomenon. Pythia
   fieldwise still open (needs adapters) — appendix candidate.

## Drafting progress (write-first plan)

- ☑ Related work (verified refs, relevance notes) + References — committed b5ff27f.
- ☑ Methods (consolidated protocol: datasets, held-out, model families, metric,
  multi-instance, determinism, statistics) — the objective section.
- ☑ Results numbers verified against JSONs; §6.1 quality ranges corrected.
- ☑ Discussion §6.3 "Two distinct lessons" — the models-vs-measurement two-contributions
  framing the owner asked for, developed as independently-surviving takeaways.
- ☐ Introduction + Abstract — tighten LAST (per plan), once sections settle.
- ☐ De-duplicate §2/§4/§5 narrative against the consolidated Methods (formatting pass).
- ☑ Fieldwise breakdown (item 8) — done for anchors, folded into §6.1.
- ☑ Open/closed control ELEVATED (owner-recommended reframing) — the phenomenon is now
  framed as "held-out-value copying into open slots, with closed-value fields as a
  built-in control" consistently across abstract, §6.1, §6.3, and §9. ☑ Title set (owner
  chose A, tighter form): "Held-out value copying in small language models: a
  field-localized failure mode and the instrument to measure it".
- ☐ 1B multi-seed decision (⚠); Pythia fieldwise (appendix); §2/§4/§5-vs-Methods de-dup.

## Open owner decisions (⚠)

- ⚠ **1B representation**: interval [0,5] vs a point from added seeds. Affects the abstract.
  Current plan: report interval, seeds are a precision follow-up (recorded in FINDINGS.md).
- ⚠ **Related-work placement**: keep as post-intro unnumbered section vs promote to §2
  (renumbering cascade) — a formatting pass, deferred.
- ⚠ **Confound experiment before submission?** Owner chose freeze-and-write; pre-reg is
  drafted and ready (`PREREG_ownstack_160m.md`) as the highest-priority follow-up.
