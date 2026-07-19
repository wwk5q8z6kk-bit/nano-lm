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
- ☑ Companion verification architecture (§1) — softened: now reads "its precision/
  review-load trade-off is measured in a companion write-up (not in this paper)".
- ☑ "first model to pass..." — N/A: that phrasing is in scale/AUDIT.md, NOT in the
  manuscript; the paper says only "passed the average-case faithfulness gate" (§3),
  which is scoped to the project's own bars. No field-wide claim to fix.
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
- ☑ Author lists + arXiv IDs verified against primary sources (2026-07-18). All four
  uncertain entries confirmed CORRECT as written: Pham/Qian/Wang… (ASE'20), Zhuang/Zhang/
  Song/Hooker (MLSys'22, added arXiv:2106.11872), Wu/Wang/Xiao/Peng/Fu (2404.15574),
  Xu/Guan/Greene/Kechadi (2406.04244). No citation errors found.

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

## Council decision — submission readiness (DWA, 2026-07-18)

5-member Debate-Weighted-Aggregation council (methodologist, skeptical reviewer,
statistician, PI/strategist, interp specialist). DWA scores: **SUBMIT_AS_IS 1.278** >
PYTHIA_FIELDWISE 1.041 > OWNSTACK_160M 0.580 > (1B_MULTISEED, STAGE_M_PILOT = 0).
Aggregate confidence 44% (low → venue-conditional, not incoherent); HHI 0.363.

**Decision: submit as-is now to a workshop / *ACL-Findings / short-paper venue.**
Riders:
- Main-conference target → run OWNSTACK_160M (pre-registered deconfounder) first; it is
  the natural Paper 2.
- Cheapest freeze-and-write-compatible strengthener = PYTHIA_FIELDWISE (re-score frozen
  adapters; tests the titular "field-localized" claim on the Pythia side; Kaggle-gated).
  ☑ KERNEL PREPARED: trajectory/kaggle_pythia_fieldwise.py (parses; v2-prefix exec
  verified; same deterministic finetune as arm1_v2, per-field scorer over m0-m4). Run
  on Kaggle T4 per its header cell — no local creds, so owner launches. Emits
  results_fieldwise_pythia_{160m,410m,1b}.json. PREREQUISITE (owner): this session's
  commits are LOCAL only (origin/master still at e8c6f05); the kernel checks out `master`
  (NOT the frozen stage-t-v2 tag, which lacks the script), so `git push origin master`
  is required before the run — and it publishes the draft to the public repo, an owner
  call. Finetune inputs are byte-identical to stage-t-v2 so the adapter regenerates.
- Rejected: 1B_MULTISEED (misrepresents the bistable [0,5], weakens the nondeterminism
  contribution — the 1B gap is 0-or-5, so the interval is the correct representation) and
  STAGE_M_PILOT (premature under the confound).
- Free writing follow-through (optional): one-sentence hedge that the n=2 1B interval
  lower-bounds training-variance magnitude but cannot characterize its distribution.

## AAEA code-correctness audit (2026-07-18)

Two parallel correctness reviewers on the paper-critical scripts (931 LOC). **No
arithmetic bug found** — numbers reproduce; anchor↔Pythia gap conventions consistent;
rescore reproduces gate_scribe_v2.log byte-exact; the two Pythia scripts are
finetune-equivalent (byte-diff of training constants + stage-t-v2↔master inputs = 0).
Findings were validity/methodology, now addressed:

- ☑ **A1 (HIGH, validity):** dur/sev "exactly 0" is partly structural (no held-out
  dur/sev values exist). Fixed §6.1 to frame them as a **template-vs-value control**
  (they undergo the held-*template* shift but show no gap → gap driven by held *values*,
  not phrasing), explicitly NOT a value-robustness claim.
- ☑ **A2 (MEDIUM, validity):** dialogue-level seen/held split dilutes the held bucket —
  VERIFIED directly: only **68% cc / 30% med / 21% alg** of held-bucket items carry a
  real held-out value. So per-field gaps are conservative, not cross-field comparable;
  aggregate ~18 is a lower bound. Added to §6.1 + a §7 limitations bullet. (Caught &
  corrected a wrong "≈50%" figure I'd first written for med/alg.)
- ☑ **A3 (MEDIUM):** parsed-only + nano's differential held/seen parse rate → conservative
  bound. Folded into the same §7 bullet.
- ☑ **B2 (MEDIUM, code):** hardened kaggle_pythia_fieldwise.py with a comparability
  self-check (emits `agg_from_fieldwise` per instance + asserts == published fresh_gaps;
  input fingerprints) so the pending run self-validates instead of silently drifting.
  B1/B3/B4 (RNG order, model.train, max(1,·)) verified benign; added model.train() for
  hygiene.

**RESOLVED (plan T1-A / O1, 2026-07-18):** the CLEAN per-field held-out-value gap was
computed (`undilute_anchors.py` → `results_undilute_anchors.json`): nano **87.3±2.7**,
scale **79.5±2.1** aggregate; med/alg **≈0% recall on real held-out values** (100-pt gap).
Folded into §6.1 (distinctly labeled "value-level/clean" vs the "dialogue-level/diluted"
ladder spine) and §6.3, per the approved plan option A — diluted ladder kept as the spine,
clean reported anchor-only, clean *ladder* deferred to O3 (kaggle_pythia_fieldwise.py now
emits both metrics). No clean-anchor-vs-diluted-Pythia comparison (S2 discipline).

## T1.5 — internal Reviewer #2 pass (2026-07-18)

Fresh-eyes full-draft review (subagent, no author-blindness) + spot-check vs JSONs.
**Verdict: submission-worthy AS-IS for workshop / *ACL-Findings / short-paper** after
text-only fixes — the contribution stands without OWNSTACK_160M (legs (a) clean anchor
phenomenon and (c) measurement lessons are unconfounded; (b) is labeled confounded).
All headline numbers traced to JSONs and confirmed. Fixes applied:

- ☑ **MUST-FIX:** §6.1/§6.3 said "≈0% on held-out medication AND allergy" — contradicted
  by the clean table (scale med = 47.1). Corrected: ≈0% on held-out allergy (both) and
  held-out med (nano); scale recovers ~half its held-out meds. Also dropped the
  apples-to-oranges "~4–5×" ratio (3-field clean vs 5-field diluted) for "substantially
  understates" + both numbers.
- ☑ §1 broken "referenced in §7" companion cross-ref → "(companion paper, in preparation)".
- ☑ Hypothesis labeling: §5.1 "Diagnosis:" → "We attribute this to"; §5.2 "sits at the
  boundary" → "we conjecture, sits near".
- ☑ Stale References caveat ("to be double-checked") → "verified (2026-07-18)".

**Deferred to submission/camera-ready time (SHOULD/NICE, not blocking):** strip draft
scaffolding (italic meta-notes, "Status: draft" block); de-dup §2/§4/§5 vs Methods;
final section renumbering; footnote the ~68/30/21% denominator; add Hoffmann cite at the
§7 token-ratio line. These are LaTeX-template-time tasks; the working draft keeps them.

## Post-review addition (2026-07-18, after the T1.5 pass)

- ☑ **Failure-mode characterization folded into §6.1 + §6.3** (owner-authorized resume of
  the offered option). Source: `results_failure_mode_anchors.json`; numbers verified
  internally consistent with the clean metric (scale-med 71/151 misses ↔ clean gap 47.1).
  Claim strength: behavioral characterization only, hedged ("consistent with"), circuit
  account explicitly deferred to §8/Stage M. NOTE: this text post-dates the Reviewer #2
  pass — give it one extra read at submission time (numbers: alg 100% omission both;
  cc@10M 86% substitution 258/300; cc@3M 98% other; med 66% omit @3M / 100% other @10M;
  pooled sub rates nano 1% vs scale 54%).

## Open owner decisions (⚠)

- ⚠ **1B representation**: interval [0,5] vs a point from added seeds. Affects the abstract.
  Current plan: report interval, seeds are a precision follow-up (recorded in FINDINGS.md).
- ⚠ **Related-work placement**: keep as post-intro unnumbered section vs promote to §2
  (renumbering cascade) — a formatting pass, deferred.
- ⚠ **Confound experiment before submission?** Owner chose freeze-and-write; pre-reg is
  drafted and ready (`PREREG_ownstack_160m.md`) as the highest-priority follow-up.

## Pythia fieldwise landed (2026-07-19, Kaggle runs complete)

- ☑ Comparability self-check EXACT on both rungs (max_abs_diff 0.0) — regenerated
  adapters reproduce the published fresh_gaps; numbers fully ladder-comparable.
- ☑ HEADLINE ENRICHMENT folded into md+tex (abstract, §6.1 incl. extended clean-ladder
  table, §6.2, §6.3): the entire Pythia residual is the ALLERGY slot (diluted alg
  17.6/21.2, all other fields ≈0; clean alg 83.6→100.0 at 160m→410m) while cc/med are
  SOLVED (clean 0.0). Clean ladder: 87.3/79.5 → 14.7±2.1/17.7±3.2.
- ☑ Slot-diversity hypothesis added, labeled as hypothesis (~190/18/5 training values;
  copy-vs-classify settled per-slot by diversity; predicts alg failure persists at
  larger scale on this recipe). Falsifiable via varying slot diversity at fixed scale.
- NOTE: these additions post-date the T1.5 review — flag for one extra read.
- ⏳ 1b fieldwise kernel running (nano-lm-fieldwise-1b-run).

## 1B fieldwise landed (2026-07-19)

- ☑ Comparability check FAILED AS DESIGNED (max diff 2.2): regenerated 1B adapter is a
  different training draw (aggregates 0.6–2.2 vs published all-zeros; inside [0,5]).
  Folded into §6.1 (md+tex) as a THIRD TRAINING DRAW, explicitly not placed in the
  ladder — the §5.2 lesson operating live. Pattern robust: clean cc/med 0.0, residual
  entirely alg (24.6±10.3). Single-slot localization is training-draw-stable.
- All three fieldwise runs complete; the Kaggle arc for Paper 1 is done.

## P2 deconfounder landed (2026-07-19) — STACK-dominant

- ☑ OWNSTACK_160M full-FT ran (7.0h T4; probe caught OOM first, micro8×accum4 fix):
  diluted 16.9±1.7, clean 66.6±5.0, alg clean still 100. Decision rule ≥14 →
  STACK-dominant. Own-stack flat across 50× scale (18.3→18.7→16.9) vs Pythia-160M 3.5.
- ☑ P1 §7 post-freeze addendum added (md+tex, one paragraph, clearly marked): the
  pre-registered control has run once and fires stack-dominant; full report = Paper 2.
- ☑ RESEARCH_PROGRAM: H0b SUPPORTED (single run), H0a supported within-stack.
- Next: LoRA arm (2×2, needs peft-wrap validation), then Paper 2 drafting. inst0
  hard-draw now 6/6 rungs (28.0 > 16.9).
