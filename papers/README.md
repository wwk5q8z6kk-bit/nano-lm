# Paper α — Held-out value copying in small language models

**Title:** Held-out value copying in small language models: a field-localized failure mode and the instrument to measure it

**Author:** Hassan El Jesr (<summer.say3y@icloud.com>)

**Status:** Camera-ready draft — public GitHub release; arXiv pending (endorsement / category). Intended as a short empirical / negative-result paper (workshop or Findings-style venue).

**License (this manuscript):** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code and other repository assets follow the repository license.

## Abstract

We study a specific faithfulness failure in small language models finetuned to convert short clinical dialogues into structured summaries: a **held-out copying gap** — the model copies field values it saw during finetuning but errs on held-out values under held-out phrasings, even though both are present verbatim in the input. In our own from-scratch models (3.15M and 10M parameters) this gap is large — **18.3±1.3 and 18.7±1.5 points** of recall on the same multi-instance instrument. The failure localizes entirely to open-vocabulary fields and is exactly zero in closed-value fields. On the Pythia ladder the observed gap is substantially smaller, with a field-localized residual on the lowest-diversity open slot; a within-stack control and a pre-registered diversity sweep separate stack/adaptation/data effects from a pure scale story. On a pre-registered utility, non-generative baselines dominate official generative LM references (**KILL**); this paper reports that result honestly and does not advocate a generative substrate. Primary metrics are exact string match; human-accepted equivalence is an explicit unvalidated limitation.

## Scope (locked)

This is **measurement only** — an empirical / negative-result account of when small LMs fail held-out emission, plus instrument lessons (under-powered single-instance eval; training nondeterminism at 1B).

- **§0 kill-gate** is part of the paper (non-generative baselines dominate utility).
- **Exact-match construct limitation** is stated in §0 and Limitations.
- **Not** a systems, architecture, product, or “generative substrate wins” paper.
- E2 / fabric / residual continua remain **gated** and are not part of this release.

See `EMPIRICAL_FOUNDATION.md` for the lockfile.

## Artifacts

| Path | Role |
|------|------|
| [`latex/paper1.pdf`](latex/paper1.pdf) | Camera-ready PDF (14 pages) |
| [`latex/paper1.tex`](latex/paper1.tex) | LaTeX source |
| [`latex/refs.bib`](latex/refs.bib) | Bibliography |
| [`figures/`](figures/) | Figures used by the PDF |
| [`paper1_draft.md`](paper1_draft.md) | Locked markdown source of truth |
| [`EMPIRICAL_FOUNDATION.md`](EMPIRICAL_FOUNDATION.md) | Scope / evidence lockfile |

Build (from `papers/latex/`):

```bash
pdflatex paper1.tex && bibtex paper1 && pdflatex paper1.tex && pdflatex paper1.tex
```

## Citation

```text
Hassan El Jesr. Held-out value copying in small language models: a field-localized
failure mode and the instrument to measure it. 2026.
GitHub: https://github.com/wwk5q8z6kk-bit/nano-lm
(tag: paper-alpha-v1).
```

When an arXiv identifier exists, replace the GitHub line with the arXiv URL in this README and in citations.
