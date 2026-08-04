# Empirical foundation

This document is the compact scientific record behind nano-lm. It separates
measured results from interpretation and from the active AI direction. Exact
claim wording lives in `papers/EVIDENCE_LEDGER.md`; file-level provenance lives in
`papers/EVIDENCE_MANIFEST.json`.

## Scope

The original instrument is a synthetic structured-summarization task with exact
field-value scoring. It is useful for studying held-out copying, field binding,
adaptation, and abstention under controlled conditions. It is not a clinical
validation set, a general document-intelligence benchmark, or evidence of
deployment readiness.

Paper α is frozen at tag `paper-alpha-v1` (commit
`0e01d73205e9c35ea32925fd4d6c7e5fceb61137`). Post-α corrections and primary E1/E3
evidence are anchored at `post-alpha-reconciled-evidence-freeze-2026-07-31`
(commit `67bf87b1f968a38e68c0225b2b556f7bba5ea1cc`). E4 has its own verdict tag,
`verdict/E4-kill@6af178d`.

## Paper α results

### Held-out copying gap

Small LM pipelines exhibited a held-out-versus-seen exact-copy gap on the
synthetic instrument. Fieldwise scoring localized the gap mainly to the
open-vocabulary fields (`cc`, `med`, `alg`); closed-value controls (`dur`, `sev`)
were approximately gap-free under the recorded protocol.

This supports a behavioral observation on this distribution. It does not prove a
general transformer limitation or identify a neural mechanism.

### Slot diversity

In the preregistered 10M allergy-slot sweep, moving from D5 to D80 increased mean
held-type recall by 66.7 percentage points, with a position control. The result is
limited to one task family, slot, seed structure, and the recorded diversity
intervention. It is consistent with diversity affecting copying behavior; it does
not establish a circuit-level explanation.

### Scale and adaptation

The measured diluted gaps across tested own-stack configurations were:

| Configuration | Diluted gap |
|---|---:|
| 3.15M parameters / 32.8M tokens | 18.3 ± 1.3 |
| 10M parameters / about 200M tokens | 18.7 ± 1.5 |
| 159M parameters / about 200M tokens / full fine-tune | 16.9 ± 1.7 |

These runs do not isolate parameter count because token budgets and training
schedules differ. “Scale removes the failure” is therefore unsupported.

At 159M parameters, the observed cells were approximately 16.9
(200M-token/full-FT), 7.1 (200M-token/LoRA), 7.0 (3.2B-token/full-FT), and 4.2
(3.2B-token/LoRA). The pattern is consistent with a weak-base × full-FT
interaction, but whether pretraining and LoRA are mechanistic substitutes remains
unresolved. E2 produced no valid result.

### Residual and architecture probes

- The preregistered isolated-versus-contained lexical-interference contrast did
  not reach its support threshold.
- The C3 transition-availability and boundary-type effects did not reach their
  preregistered 40-point thresholds; the length factor remained underpowered.
- Morphological re-inflection was the largest post-hoc descriptive C3 error class
  (about 44% of core-cell misses). This is descriptive, not a causal mechanism.
- Pointer P1 was void because its manipulation check failed. P2 did not close the
  OOD gap for the tested implementation. This says nothing universal about pointer
  architectures.

## Post-α decision experiments

### E1: old-task substrate test

E1 compared a generative reference with classical methods under the frozen
utility

\[
U = P - 0.5M - 0.3\rho - 0.02L - 0.05C,
\]

where \(\rho\) is review load, not hallucination rate. Reported utility was about
0.999 for M1, 0.925 for the official M0 reference, and 0.886 for M2. The M1–M0
margin was about +0.074, above the frozen 0.05 decision margin, and the
one-at-a-time sensitivity grid did not flip the decision.

Verdict: **KILL** for the claim that a generative LM is the necessary or preferred
substrate for this closed synthetic task under the frozen utility. M1 is a
generator-aligned hand-template extractor; M2 is the train-dictionary/span
baseline. Runtime and compute reconstruction is partial. The result is scoped to
this task and utility.

Primary files: `trajectory/PREREG_E1_nonlm_baseline.md`,
`trajectory/results_e1_utility.json`, and `artifacts/e1/MANIFEST.json`.

### E3: exact-match construct audit

Frozen normalize-then-match rescued 0 of 486 M0 exact failures. A bounded
single-agent rubric pass assigned `faithful` to 0 of 100 sampled exact errors, so
the recorded decision was `EXACT_SURVIVES`.

This was an agent-applied rubric audit, not independent human or clinician
evaluation. Inter-rater agreement and broader semantic-equivalence validity remain
unmeasured.

Primary files: `trajectory/PREREG_E3_faithfulness_construct.md`,
`trajectory/results_e3_normalize_construct.json`,
`trajectory/results_e3_human.json`, and `artifacts/e3/MANIFEST.json`.

### E4: R★ v1 substrate test

On the locked R★ v1 regime, best classical utility was about 0.638 (C-M2) and the
generative-plus-verification reference was about −1.623. The sensitivity analysis
did not flip the decision.

Verdict: **KILL** for the generative substrate on this tested regime. The verdict
does not establish that generation can never add value; it means this implementation
did not justify it under the frozen comparison.

Primary files: `trajectory/PREREG_E4_Rstar_killgate.md`,
`trajectory/e4/recipe_freeze.json`, and `trajectory/results_e4_utility.json`.

## Fabric result

On the closed synthetic `inst0` fixture, the rules-strong v2
propose→verify→abstain slice reduced presented error to zero. That result is scoped
to the recorded verifier relation and distribution. Fabric is a regression harness
for typed claims, verification, and abstention; it is not evidence of open-world
soundness or a complete Nano architecture.

## What the evidence changes

The synthetic five-field task remains a research instrument, while scribe
intelligence remains Nano's build target. The research exists to guide Nano's
successive improvements: localize a failure, choose a bounded change to data,
training, architecture, inference, grounding, or verification, evaluate it, and
carry the result into the next step. The instrument does not establish clinical
validity or open-world generalization.

Classical methods are mandatory baselines for extraction-like work. They set a
floor, expose residuals, and may contribute verification scaffolds; they are not
a replacement destination for Nano's trained intelligence. A learned or
generative change earns integration only when it improves matched held-out
utility after grounding, abstention, error rate, coverage, latency, compute,
privacy, and maintenance are counted.

The active AI question is therefore:

> Can a small, local, verification-gated scribe intelligence turn a supplied
> conversation into an accurate, evidence-bound structured representation while
> abstaining on unsupported fields?

Nano is the AI being trained and improved to answer that question. Paper α,
Wedge, Fabric, and the solver comparisons are feedback instruments for that
work. Wedge mechanisms may be reused when Nano acceptance tests justify them;
Wedge is not a replacement AI direction.
