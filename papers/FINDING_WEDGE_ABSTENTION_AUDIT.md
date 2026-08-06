# Finding — the wedge over-abstention diagnosis was correct, stale, and overstated

**2026-08-06.** Independent audit of hurdle H-6. Read-only; measured by re-running
the failing cards, not inferred.

## Three corrections to claims this repo has been repeating

**1. "3/10 useful on real documents" — the documents were ours.**
Source located: `wedge_v1/.studies/p5-repo-papers-20260802-r4/summary.json`,
`by_label: {CORRECT_ABSTENTION: 4, OVER_ABSTENTION: 3, USEFUL: 3}` over 10 tasks
**on this project's own repo papers**, not third-party documents. Every paper
citing "3/10 on real documents" — including `FORWARD_PLAN_20260806.md` H-6 —
overstates the external validity. The dogfood corpus is in-house prose.

**2. The `_relevant_claim` diagnosis is stale.** It was exactly accurate before
commit `fe6f95b3` (2026-08-05). A prior session already diagnosed it,
preregistered `PREREG_ABSTENTION_W1/W2.md`, and shipped a two-scope fix
(`runtime.py:187-235`) accepted under W-ABSTAIN-2. That fix recovered **1 of 3**
failures. The `len(tokens)==2` branch (`runtime.py:220-222`) still uses the
original unforgiving rule and is a recorded open gap.

**3. `_relevant_claim` is not the sole cause.** Measured by re-running the three
OVER_ABSTENTION cards live:

| card | attribution |
|---|---|
| D08 | **already fixed** by the shipped patch — now SUPPORTED |
| D02 | 100% attributable to `_relevant_claim` (35/35 candidates rejected) — but root cause is a query/corpus spelling mismatch (`M1_template` vs `M1 template`), not the span-conjunction design |
| D06 | `_relevant_claim` **passes** a candidate; a *downstream* COE evidence-binding invariant (`EVIDENCE_CREATED_WITH_CLAIM`) is what rejects it |

So "wedge over-abstains because of `_relevant_claim`" was one-third true, and the
remaining third is a spelling mismatch rather than a design flaw.

## The gate is load-bearing, which the previous framing missed

A live ablation (`_relevant_claim` forced to `True`) recovers all three failures
**and creates false positives on 2 of 4 correct-abstention guard cards.** The
gate is not merely overcautious — removing it trades three recovered answers for
two ungrounded assertions. For a product whose claim is *never assert what you
cannot ground*, that trade is bad. Any fix must be measured on both sets.

## Two new defects found

- **`wedge_v1/classical/solvers.py::keyword_paragraph` can emit an 11,106-character
  claim value**, which trivially contains any two query tokens and so passes
  `_relevant_claim` vacuously. An unbounded value defeats a relevance test the
  same way an empty denominator defeats a ratio.
- **`wedge_v1/run_classical_baseline.py:122`** computes
  `R = abstain_count / max(1, len(claims))` where `claims` is a battery the
  script itself constructs — the `fabric/slice.py:247` shape again, **fifth
  instance this cycle**. Worse, that script never calls `runtime.ask()`, so it is
  structurally blind to the failure mode it appears to measure. Currently
  self-labelled `DRAFT_NOT_SCORING_FROZEN` and read by no CI gate, so it gates
  nothing today — but it would mislead anyone who promoted it.

## Next measurement (smallest, no rewrite)

Extend the existing `wedge_v1/eval/margin_sweep.py` pattern into a sibling
`relevance_ablation.py` sweeping `_relevant_claim` variants — current /
full-ablation / two-scope-extended-to-2-tokens / bounded `keyword_paragraph`
value — against the same frozen r4 cards, scored on **both** the recovered and
the held (correct-abstention) sets. Reuses existing discipline; adds no corpus.
