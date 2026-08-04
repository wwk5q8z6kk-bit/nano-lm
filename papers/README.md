# Paper α — held-out value copying in small language models

**Title:** *Held-out value copying in small language models: a field-localized
failure mode and the instrument to measure it*
**Author:** Hassan El Jesr
**Status:** Camera-ready draft; public repository release; arXiv pending
**Manuscript license:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Scope

Paper α is an empirical negative-result paper about held-out exact value copying
in small language models trained to convert short synthetic clinical dialogues
into structured summaries.

On the multi-instance instrument, the measured held-out versus seen exact-copy
gaps were 18.3 ± 1.3 points for the 3.15M model and
18.7 ± 1.5 for the 10M model. The measured gap localized to
open-vocabulary fields; closed-value fields were approximately zero. The
repository also records within-stack, adaptation, data-diversity,
interference, binding, and pointer-head tests.

The paper includes the E1 result: under frozen utility, the classical M1
baseline beat the official generative M0, so the generative-substrate
hypothesis received a scoped **KILL** for that closed task.

This is not a clinical-validity, deployment-readiness, universal scaling, or
general anti-generation claim. Exact match is the primary construct.
Independent human or clinician validation and inter-rater agreement have not
been completed; E3 was a bounded agent-applied rubric audit only.

## Manuscript artifacts

| Path | Role |
|---|---|
| [`latex/paper1.pdf`](latex/paper1.pdf) | Camera-ready PDF |
| [`latex/paper1.tex`](latex/paper1.tex) | LaTeX source |
| [`latex/refs.bib`](latex/refs.bib) | Bibliography |
| [`figures/`](figures/) | Manuscript figures |
| [`paper1_draft.md`](paper1_draft.md) | Markdown manuscript source |
| [`EMPIRICAL_FOUNDATION.md`](EMPIRICAL_FOUNDATION.md) | Scope and evidence lock |
| [`../trajectory/REPRODUCIBILITY.md`](../trajectory/REPRODUCIBILITY.md) | Artifact and environment reproduction notes |

Build from `papers/latex/`:

```bash
pdflatex paper1.tex
bibtex paper1
pdflatex paper1.tex
pdflatex paper1.tex
```

## Project documents

| Document | Purpose |
|---|---|
| [`STRATEGIC_RESET.md`](STRATEGIC_RESET.md) | Nano AI capability contract and current strategy |
| [`WEDGE_V1.md`](WEDGE_V1.md) | Supporting document-evidence component contract |
| [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md) | Canonical supported claims and limitations |
| [`EXECUTION_QUEUE.md`](EXECUTION_QUEUE.md) | Current scribe-first work |
| [`DECISION_GATES.md`](DECISION_GATES.md) | Experiment and promotion criteria |
| [`CLAIM_GLOSSARY.md`](CLAIM_GLOSSARY.md) | Required claim scoping |

AI engineering and Paper α share evidence discipline, but they are not the same
claim surface. Nano is the small local scribe AI; Wedge is supporting validation
infrastructure. Wedge evaluations remain component-development results unless
they receive a separate traceable scientific promotion.

The research record exists to guide Nano's successive improvement. Each result
should inform a bounded change to its data, training, architecture, inference,
grounding, verification, or abstention, followed by a held-out comparison and
an integrate-or-reject decision.

## Citation

```text
Hassan El Jesr. Held-out value copying in small language models: a
field-localized failure mode and the instrument to measure it. 2026.
https://github.com/wwk5q8z6kk-bit/nano-lm (tag: paper-alpha-v1).
```
