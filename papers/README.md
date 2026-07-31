# Paper α — Held-out value copying in small language models

**Title:** Held-out value copying in small language models: a field-localized failure mode and the instrument to measure it

**Author:** Hassan El Jesr (<summer.say3y@icloud.com>)

**License (this manuscript):** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code and other repository assets follow the repository license.

## Status

Paper α is a camera-ready empirical draft released from the public
`paper-alpha-v1` tag. An arXiv submission is pending.

```text
PAPER_ALPHA_MANUSCRIPT = CAMERA_READY
PAPER_ALPHA_TAG = PUBLIC          # paper-alpha-v1
ARXIV = PENDING
POST_ALPHA_REMEDIATION = INCOMPLETE_OR_PENDING_COMMIT
FINAL_PUBLIC_EVIDENCE_FREEZE = NOT_YET_ESTABLISHED
```

The paper reports a completed measurement study of held-out value copying
in small language models. Its conclusions are scoped to the evaluated
datasets, model configurations, adaptation recipes, and frozen metrics.

Subsequent repository programs—including Nano Runtime, Fabric extensions,
R★/E4, and broader product or architecture work—are not part of this paper.

Post-publication archival remediation is tracked separately and does not
authorize new experiments or alter the historical `paper-alpha-v1` tag.

## Abstract

We study a specific faithfulness failure in small language models finetuned to convert short clinical dialogues into structured summaries: a **held-out copying gap** — the model copies field values it saw during finetuning but errs on held-out values under held-out phrasings, even though both are present verbatim in the input. In our own from-scratch models (3.15M and 10M parameters) this gap is large — **18.3±1.3 and 18.7±1.5 points** of recall on the same multi-instance instrument. The failure localizes entirely to open-vocabulary fields and is exactly zero in closed-value fields. On the Pythia ladder the observed gap is substantially smaller, with a field-localized residual on the lowest-diversity open slot; a within-stack control and a pre-registered diversity sweep separate stack/adaptation/data effects from a pure scale story. Across evaluated own-stack configurations the gap did not collapse monotonically with parameter count; pretraining exposure was not fully isolated (3.15M used 32.8M tokens vs ~200M / 3.2B later). On a pre-registered utility, **M1** exceeds the best evaluated generative reference under frozen \(U\) (**KILL**); this paper reports that result honestly and does not advocate a generative substrate. Primary metrics are exact string match; dual-clinician equivalence is unvalidated (agent-rubric audit reported).

## Scope

This paper is an empirical and negative-result account of:

- held-out value-copying failure under held-out phrasings;
- localization of that failure across open- and closed-value fields;
- the limitations of under-powered single-instance evaluation;
- training nondeterminism observed in larger reference runs;
- and the E1 utility result showing that a classical method outperformed
  the evaluated generative reference on the old closed task.

It is not:

- a product paper;
- a clinical-validation paper;
- a systems or architecture paper;
- evidence that parameter count alone determines the failure;
- evidence for a LoRA mechanism;
- or an argument that a generative substrate should be deployed.

## Principal findings

- Own-stack 3.15M / 10M models show large held-out copying gaps on the measured task.
- The gap localizes to open-vocabulary fields; closed-value fields show zero measured gap.
- Scale was not cleanly isolated (unequal pretraining token budgets).
- E1 killed the generative reference for the old closed task under frozen utility.
- Exact string match remains the primary metric.
- The agent-rubric exercise is not clinician or independent-human validation.
- The paper does **not** advocate a generative product substrate.

## Artifacts and reproduction

| Path | Role |
|------|------|
| [`latex/paper1.pdf`](latex/paper1.pdf) | Camera-ready PDF (14 pages) |
| [`latex/paper1.tex`](latex/paper1.tex) | LaTeX source |
| [`latex/refs.bib`](latex/refs.bib) | Bibliography |
| [`figures/`](figures/) | Figures used by the PDF |
| [`paper1_draft.md`](paper1_draft.md) | Locked markdown source of truth |
| [`EMPIRICAL_FOUNDATION.md`](EMPIRICAL_FOUNDATION.md) | Scope / evidence lockfile |
| [`PAPER_ALPHA_CORRECTION_NOTE.md`](PAPER_ALPHA_CORRECTION_NOTE.md) | Post-freeze factual sync (token budget, ρ, E3 arm) |

Build (from `papers/latex/`):

```bash
pdflatex paper1.tex && bibtex paper1 && pdflatex paper1.tex && pdflatex paper1.tex
```

**Selected evidence pointers (Paper α spine):**  
E1 KILL — `../trajectory/results_e1_utility.json` · diversity +66.7 — `../trajectory/results_sweep_10m.json` · E3 construct — `../trajectory/results_e3_normalize_construct.json`

## Citation

```text
Hassan El Jesr. Held-out value copying in small language models: a field-localized
failure mode and the instrument to measure it. 2026.
GitHub: https://github.com/wwk5q8z6kk-bit/nano-lm
(tag: paper-alpha-v1).
```

When an arXiv identifier exists, replace the GitHub line with the arXiv URL in this README and in citations.

## Limitations

Exact-match construct limitation; unequal pretraining budgets across scale cells; E3 Stage-1 is agent-applied rubric, not dual-clinician IAA; no clinical deployment claim; no LoRA-mechanism identification (E2 has no RESULT).

## Related repository work

Paper α is a frozen publication boundary. Later research, archival
remediation, and product exploration are maintained separately in the
repository and must not be interpreted as claims of this paper.

In particular:

- E2 has no RESULT artifact and remains gated.
- R★/E4 is **outside Paper α**. A post-α utility artifact exists at
  `../trajectory/results_e4_utility.json` (regime-scoped; commit history
  records Gate 4 **KILL** on tested R★ v1). That artifact is **not** a
  Paper α claim, does not license generative product resurrection, and
  further E4 execute remains blocked without new owner authorization.
  (An earlier remediation note that said “protocol only / no RESULT” is
  stale relative to the tracked artifact.)
- Fabric refers only to a separately scoped verification prototype.
- Nano Runtime and Wedge v1 are subsequent product-engineering work.
- Active development authority: [`../frontier/DEVELOPMENT_PLAN.md`](../frontier/DEVELOPMENT_PLAN.md)
  and [`PROGRAM_AUTHORITY.md`](PROGRAM_AUTHORITY.md).

Repository archive entry (not this paper): [`../README.md`](../README.md).
