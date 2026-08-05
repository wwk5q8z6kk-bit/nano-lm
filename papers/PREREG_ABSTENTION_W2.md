# Preregistration — W-ABSTAIN-2: two-scope relevance, scope-matched denominator

**Frozen 2026-08-05, before measuring.** Supersedes the *decision rule* of
`PREREG_ABSTENTION_W1.md`; that document's REJECTED result is retained in full
and is not overwritten.

## 1. Why a second preregistration

W-ABSTAIN-1 tested a change that alters behaviour **only for queries with ≥3
content tokens** (§3 of that document says so explicitly), but scored it
against all three recorded OVER_ABSTENTION cards — two of which are 2-token
queries the change cannot affect by construction, and one of which
(`E1 KILL M1_template`) is unanswerable as written, since `M1_template` occurs
**0** times in the corpus (the text reads `M1 template`).

Result: 1/3 recovered against a required 2/3 → **REJECTED**, correctly, under
the rule as written. The threshold was not moved.

The defect is demonstrable **without reference to the outcome** — token counts
are a property of the queries, fixed before any measurement — which is the only
thing that licenses a corrected instrument rather than a rationalisation.

## 2. Scope of the change under test (unchanged from W1 §3)

Two-scope relevance in `wedge_v1/runtime.py::_relevant_claim`:

- **document scope** — every query token must appear somewhere in the source
  document (preserves the anti-false-positive property the filter exists for);
- **span scope** — a majority (`max(2, ceil(n/2))`) must appear in the claim
  value + evidence (keeps the answer anchored in what is shown).

Relevant iff both hold. **Queries with fewer than 3 content tokens are
untouched** — this is the scope boundary the corrected denominator respects.

## 3. Preregistered decision rule

Measured on the frozen r4 pack (`task_pack_digest 939e7ad8…`, `corpus_digest
f768fbac…`).

**Primary metric — in-scope recovery.** Of the recorded OVER_ABSTENTION cards
with **≥3 content tokens**, how many now return a span-supported claim.
In-scope population, fixed here before measurement: exactly one card,
`Nano Runtime smallest sufficient solver` (5 tokens).

| metric | requirement |
|---|---|
| **in-scope recovery** | **1 of 1** |
| **held** — recorded CORRECT_ABSTENTION still ABSTAIN | **4 of 4, no exceptions** |
| **regressions** — recorded USEFUL still answer | **3 of 3** |
| **out-of-scope unchanged** — the two 2-token cards still ABSTAIN | **2 of 2** |
| **suite** — `wedge_v1` tests | **no new failures (≥355 passing)** |

**ACCEPT** iff all five hold. **REJECT** otherwise. A single lost
CORRECT_ABSTENTION rejects the change outright even at full recovery —
trading trustworthiness for coverage is the one trade this project does not
make.

The fourth row is deliberate: if a change scoped to ≥3-token queries alters a
2-token result, the scope claim is false and the change must be re-examined
regardless of its recovery number.

## 4. Honest limits of this instrument

- **n is tiny.** One in-scope card. This is a smoke test that the mechanism
  works on the one case that motivated it, not evidence of general improvement.
  No coverage or AURC claim may be made from it.
- **Labels are agent-applied** (`review_evidence_kind: AGENT_APPLIED_RUBRIC`,
  `representative_ready: false`). The r4 study says so itself.
- **The 2-token gap remains open.** `E4 KILL 0.638` has its terms present
  verbatim and still abstains; the conjunctive rule at n=2 requires both, and
  `KILL` sits in a different paragraph from `0.638`. That is a separate
  question, recorded here and deliberately not folded in.
- Accepting this does **not** license a claim that over-abstention is fixed.
  The product-level evidence remains 3/10 useful on the only real corpus.

## 5. Next question if accepted

Whether the 2-token conjunction should also become two-scope. That needs its
own preregistration and, ideally, a corpus with more than ten tasks — the
open-licensed dogfood corpus in `PRODUCT_THESIS.md` §5b.

---

## 6. RESULT — ACCEPTED (2026-08-05)

Measured against the criteria frozen in §3, on the frozen r4 pack:

| metric | requirement | observed | |
|---|---|---|---|
| in-scope recovery | 1 of 1 | **1 of 1** | pass |
| held (CORRECT_ABSTENTION) | 4 of 4 | **4 of 4** | pass |
| regressions (USEFUL) | 3 of 3 | **3 of 3** | pass |
| out-of-scope unchanged | 2 of 2 | **2 of 2** | pass |
| `wedge_v1` suite | ≥355 passing | **355 passed** | pass |

**ACCEPTED.** `Nano Runtime smallest sufficient solver` now returns SUPPORTED
with span-bound claims; it previously abstained while retrieval had already
promoted the answering paragraph at margin 3.728. No correct abstention was
lost, no useful answer regressed, and the two out-of-scope 2-token cards are
bit-for-bit unchanged — confirming the scope claim in §2 rather than assuming it.

### What this does and does not establish

**Establishes:** the conjunctive span filter was a real over-abstention
mechanism for multi-token queries, and two-scope relevance fixes it without
weakening the anti-false-positive property the filter exists for.

**Does not establish:** that over-abstention is fixed. n=1 in scope, labels are
agent-applied, and the product-level evidence remains **3/10 useful** on the
only real corpus ever tested. The 2-token conjunction (`E4 KILL 0.638`, terms
present verbatim, still abstaining) is untouched and remains open per §5.

The honest summary: one mechanism identified, fixed, and pinned — out of a
failure class that is not yet closed.
