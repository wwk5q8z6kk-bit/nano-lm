# Wedge v1 — document-evidence validation runtime

**Status:** Supporting component engineering
**Implementation:** [`wedge_v1/`](../wedge_v1/README.md)
**Strategy:** [`STRATEGIC_RESET.md`](STRATEGIC_RESET.md)

## Relationship to Nano

**Nano is the small local scribe AI itself.** Wedge is a document-domain test bed
for evidence binding, retrieval, contradiction handling, abstention, private
review, and re-verifiable local state. Its mechanisms may support Nano when an
AI-core acceptance test proves that they fit; its research-document workflow is
not Nano and does not replace Nano's identity.

Wedge research is useful to this project only when it reveals a mechanism,
failure, measurement, or verification method that can improve Nano's next
training or engineering step. Wedge is an instrument in Nano's development
loop, not an independent destination.

Wedge's runnable commands exercise supporting component infrastructure. They do
not run Nano's scribe intelligence or define its inference surface.

## Component validation contract

A researcher points the Wedge runtime at a local folder of technical or
biomedical documents, asks for facts or comparisons, and receives typed claims
with source evidence. The runtime surfaces contradictions and abstains when it
cannot support an answer.

This is a local, verification-gated validation environment—not Nano's scribe
intelligence, a general chatbot, an autonomous agent, or a clinical system.

## Version-one scope

- **Corpus:** 10–50 Markdown, text, or PDF documents; target extracted text at
  or below 5 MB.
- **Default execution:** offline and classical-first.
- **Operations:** metadata extraction, exact find, evidence-backed question
  answering, multi-document comparison, contradiction detection, review, and
  saved questions.
- **Outputs:** typed claims with document identity and source spans; explicit
  `SUPPORTED`, `DISPUTED`, `ABSTAIN`, `MISSING`, or equivalent
  machine-readable states.
- **Document scope:** query and report operations accept repeatable exact
  corpus-relative document IDs. An empty or partly unknown explicit scope
  abstains with zero claims; it never broadens to the full corpus.
- **Persistence:** saved answers are bound to task, query, exact scope,
  selected-source digest, solver fingerprint, and result digest. A cache hit
  requires a successful live audit; stale or failed refreshes fail closed.
- **Validation studies:** representative evaluation requires an explicit local
  directory, 10–50 readable documents, and 10–20 genuine unique tasks whose
  exact scopes cover at least 10 documents; comparisons require at least two
  documents and the pack includes at least one recall task. That recall task
  counts once in the pack and must record one `REFRESHED` solver execution
  followed immediately by a zero-solver `CACHE_HIT`.
  Corpus, task-pack, solver, review instrument, PDF extractor when used, and the
  exact card, saved-recall, and private raw-result artifacts are frozen before
  review. Verification rederives every task once with the deterministic solver
  over its exact live scope and with persistence disabled, then independently
  re-audits the frozen result and reproduces its reviewer card. The aggregate
  binds the exact review artifact used to produce it.
- **Public data:** synthetic or clearly redistributable only. Private corpus
  contents, genuine task packs, and detailed review outputs stay outside Git.

## Task surface

The original 40-task pack fixes the intended capability mix:

| IDs | Family | Examples | Default solver |
|---|---|---|---|
| T01–T08 | Metadata and structure | title, authors, date, identifiers, headings | metadata and structural parsers |
| T09–T18 | Span-grounded extraction | numbers, units, entities, key-value lines, citations | regex, lexicons, schema parsers |
| T19–T26 | Retrieval and evidence-bound QA | locate passages, mention checks, corpus unions | exact search, inverted index, BM25 |
| T27–T34 | Comparison and contradiction | schema fill, numeric conflicts, aliases, disputed merge | extraction plus symbolic verification |
| T35–T40 | Stress and optional flexibility | paraphrastic retrieval, implied change, OCR, tables, coreference, extractive explanation | normalization and small deterministic plugins first |

The stress tasks may reveal a need for a model; they do not reserve work for one.
Free generation without evidence binding is outside v1.

## Runtime pipeline

```text
documents
  -> normalize and preserve offsets
  -> index and classify the requested operation
  -> run the cheapest capable solver
  -> create minimal typed claims
  -> bind each factual claim to evidence
  -> verify support and inspect conflicts
  -> present, dispute, review, or abstain
  -> save only re-verifiable local state
```

Required invariants:

1. Every presentable factual claim has exactly one matching claim identity in
   the claim/evidence envelope and at least one valid source evidence atom.
2. Evidence is selected before any optional paraphrase; citations are not added
   after free generation.
3. Conflicting values remain visible and are never silently merged.
4. Retrieval miss and evidence absence are distinct failures.
5. Unsupported or unauditable output becomes abstention or review, not a
   plausible sentence or unbound retrieval excerpt.
6. An explicit document scope never falls back to unselected documents.
7. A saved answer is reused only after live audit against its task, scope,
   selected sources, solver, and result fingerprint; otherwise it refreshes
   once or abstains.
8. Verified reports preserve the same auditable envelope as the underlying
   answer.
9. A representative study never falls back to public examples, accepts empty or
   unreadable inputs, writes global owner/CoE state, or exports anything beyond
   content-free aggregate decisions. Study verification may rederive a result
   only through the exact scoped deterministic solver with persistence disabled;
   this is separate from the recorded two-pass recall transition.

## Utility and solver admission

The wedge uses the component-development score
$U = Q - 0.5E - 0.3R - 0.02L - 0.05C$.

| Symbol | Meaning |
|---|---|
| $Q$ | precision of presented claims |
| $E$ | miss or wrong rate on fields that should emit |
| $R$ | review fraction |
| $L$ | p50 latency in seconds |
| $C$ | relative compute versus the classical baseline |

The current wedge score is a draft component metric, not a scientific ledger
claim. Freeze the corpus, labels, thresholds, and cost normalization before
using it for a consequential comparison.

A new solver remains only when it improves matched utility over the strongest
cheaper solver and does not weaken evidence coverage or liability controls. For
an LM, the default admission margin is $\Delta U > 0.05$. A model is never
admitted merely because a task contains natural language.

## Recorded component evaluations

| Evaluation | Recorded result | Boundary |
|---|---|---|
| Synthetic classical baseline | 50/50 checks; $Q=1.0$, $E=0$, $R\approx0.196$, draft $U\approx0.891$ | Eight-document synthetic mini-corpus; not an Evidence Ledger result |
| E-class probes | T35, T36, and T39 passed using query expansion, symbolic comparison, and lightweight coreference; no LM invoked | Aggregate draft $U\approx0.870$, below the recorded classical baseline; one of 50 checks failed |
| Noisy diagnostic | Raw synthetic-noise $U\approx0.458$; normalization recovered to $U\approx0.859$; no LM invoked | Diagnostic track, not the clean primary comparison or proof about real OCR |
| Repository dogfood | 8/8 checked questions behaved as expected | Tests the repository’s own papers, not representative user usefulness |
| Agent-applied owner-corpus component check | Five current cards: three `CONTRADICTION_HANDLED`, one `USEFUL`, and one `CORRECT_ABSTENTION` | Owner-delegated agent labels; not independent human or clinician validation, and too small to establish representative usefulness |
| 2026-08-02 scoped pilot | In a frozen 10-task repository-corpus run, semantic candidate preflight moved D01, D03, and D05 from `OVER_ABSTENTION` to `USEFUL`; reviewed over-abstention fell from 6 to 3 while all 4 correct abstentions remained | Agent-applied labels with no manual baseline; not representative/private-corpus, human/clinician, time-saved, scientific, or solver-superiority evidence |
| Exact document scope | Scoped operations and provenance use only selected document IDs; invalid scopes abstain without corpus fallback | Engineering behavior, not evidence of task usefulness |
| Verified saved recall | Task/scope/source/solver/result fingerprints plus live audit distinguish `REFRESHED`, `CACHE_HIT`, and safe refresh failure | Engineering behavior; it does not yet show that recall saves meaningful user time |
| Representative-use instrument | Readability and scope coverage, exact input/result/review identity, isolated execution, typed timed review, deterministic exact-scope rederivation, independent live audit, drift detection, and content-free summaries are regression-tested | The scoped pilot does not replace this track; no representative owner-reviewed study with a manual comparator has been completed yet |

These results are a library of mechanisms and diagnostic evidence that may
inform Nano. They do not independently justify more Wedge work, establish Nano
AI capability, open-world generalization, clinical validity, open-world
factuality, or the superiority of this runtime on unseen real corpora.

## Acceptance criteria

A Wedge component increment is ready when:

- all presented factual claims carry valid source evidence;
- planted unsupported and conflicting cases abstain or surface disagreement;
- existing Wedge and Fabric regressions remain green;
- latency, review burden, coverage, and failures are reported separately;
- private input is not written into tracked artifacts;
- study summaries omit queries, document IDs, evidence spans, paths, and
  correction text;
- the change addresses a measured Nano-relevant mechanism failure and defines
  an explicit transfer test against Nano's acceptance contract.

## Recorded learning state

The fail-closed claim boundary, exact document scoping, and freshness-gated
saved recall now exist. A five-card agent-applied check closed the first small
component-review loop without a negative label, but it is not representative
user evidence or independent human or clinician validation.

The study lifecycle now distinguishes a strict representative-use study from an
identity-bound `AGENT_APPLIED_SCOPED_PILOT`. The narrower class keeps exact
scope, audit, isolation, timing, provenance, recall, and privacy requirements,
but cannot claim representative readiness, manual comparison, or time saved.

The frozen 2026-08-02 scoped pilot contained 10 owner-confirmed repository
questions over 10 documents. The pre-fix review recorded six
`OVER_ABSTENTION` and four `CORRECT_ABSTENTION` outcomes. A bounded semantic
candidate preflight then recovered D01, D03, and D05 without weakening claim
audit, exact scope, or recall checks; the post-fix review recorded three
`USEFUL`, three `OVER_ABSTENTION`, and four `CORRECT_ABSTENTION`. The remaining
D02/D08 lexical-coverage/low-margin retrieval misses and D06 atom-coverage plus
paragraph-span/semantic-binding mismatch are still amenable to cheaper
deterministic work, so an LM is not earned. A proposed remedy remains a
hypothesis until reviewed; term occurrence alone cannot replace conjunction and
semantic-relation verification.

No standalone Wedge milestone is active. New Wedge work must begin with a
measured Nano failure, a mechanism-transfer hypothesis, a bounded test, and a
Nano acceptance gate. A result enters Nano only when it passes that gate and is
integrated into the AI's common inference surface. The private-corpus study and
remaining Wedge residuals stay parked until such a Nano-linked question exists.

Current work is tracked in [`EXECUTION_QUEUE.md`](EXECUTION_QUEUE.md).
