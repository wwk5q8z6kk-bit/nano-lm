# Paper α submission packet — one-click readiness

**Status:** Everything below is prepared; the only missing input is the owner
saying "submit." Council decision of 2026-07-18: SUBMIT_AS_IS to a workshop /
*ACL-Findings / short-paper venue (score 1.278, top of three options).

## Step 1 — arXiv (no deadline; do first)

- **Files:** `papers/latex/paper1.tex` + `refs.bib` (source upload) or
  `papers/latex/paper1.pdf` (365 KB, built 2026-07-31, same mtime as source —
  current). Source upload preferred; arXiv compiles TeX.
- **Title:** Held-out value copying in small language models: a
  field-localized failure mode and the instrument to measure it
- **Author:** Hassan El Jesr (summer.say3y@icloud.com)
- **Categories:** `cs.CL` (primary), `cs.LG` (cross-list)
- **License:** recommend CC BY 4.0 (owner may prefer arXiv non-exclusive;
  choose at upload).
- **Metadata abstract** (the in-paper abstract is 472 words — over the arXiv
  field limit; use this condensed version, full abstract stays in the PDF):

> We identify a held-out value copying gap in small language models finetuned
> to convert clinical dialogues into structured summaries: models copy field
> values seen in finetuning but fail on held-out values present verbatim in
> the input. In from-scratch 3.15M/10M models the gap is 18.3±1.3 and
> 18.7±1.5 recall points (value-level: ~80–87), localizing entirely to
> open-vocabulary fields and exactly zero in closed-value fields — a built-in
> control. Pythia 160M–1B shows much smaller, still field-localized gaps. We
> deliberately do not claim scale as the cause: a within-stack ~160M control
> keeps the gap at 16.9±1.7, while LoRA or Chinchilla-scale data each reduce
> it to ~7 and together to 4.2±0.9 — an adaptation×data interaction. A
> pre-registered slot-diversity sweep lifts held-type recall +66.7 points at
> fixed scale. On a pre-registered utility, a non-generative baseline exceeds
> the best generative reference (0.999 vs 0.925); we report this honestly and
> do not advocate a generative substrate. Measurement reliability is a second
> contribution: single-instance evaluation was under-powered, and at 1B
> training-run nondeterminism dominates the residual.

## Step 2 — venue (after arXiv ID exists)

Per the council: workshop / Findings / short-paper class. The one dated match
found 2026-08-05 is off-domain (NLLP 2026, legal NLP, deadline 2026-08-11 —
skip). Scan sources for the right home (interpretability/memorization/eval
workshops around EMNLP 2026): aideadlines.org and the EMNLP 2026 workshops
page. Decision rule: first venue whose scope names memorization, evaluation
methodology, or trustworthy/faithful generation, with a deadline ≥7 days out.

## Pre-flight checklist (all verified 2026-08-05)

- [x] Author + email in `paper1.tex` (no placeholders remain)
- [x] PDF current with source (identical mtimes)
- [x] Corrections synced (32.8M token budget; schedule-aware scale language;
      E1/E3 scoping — see `EMPIRICAL_FOUNDATION.md`)
- [x] Post-freeze addendum (stack-dominant) marked in §7
- [x] Claims match `EVIDENCE_MANIFEST.json` v2 (`paper_alpha_spine`:
      SUPPORTED_WITH_RECORDED_LIMITATIONS)
- [ ] Owner: arXiv account login + license choice + click submit
- [ ] Owner: venue pick from the scan (or delegate the scan result)

## What submission does NOT require

No new experiments, no repo push, no code release decision (the paper stands
alone; repo-public is a separate owner decision), no Paper 2 resolution.
