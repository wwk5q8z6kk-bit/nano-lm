# Preregistration — W-ABSTAIN-1: span-conjunctive relevance is the over-abstention cause

**Frozen 2026-08-05, before implementing or measuring any fix.** Thresholds
below are fixed now and are not to be moved after seeing results.

## 1. Diagnosis (established, not hypothesised)

`wedge_v1/runtime.py::_relevant_claim` requires, for any query with ≥3 content
tokens, that **every** token appear inside the claim's own value + evidence
blob (evidence text is truncated to 240 chars in `_bm25_claims`):

```python
if len(tokens) >= 3:
    need = len(tokens)          # conjunctive AND over a ~240-char window
```

Measured on the frozen r4 corpus (`p5_repo_papers_20260802`), the three
recorded OVER_ABSTENTION cards all die here, *after* retrieval has already
promoted the correct paragraph:

| query | top promoted paragraph | hits/need | missing |
|---|---|---|---|
| `Nano Runtime smallest sufficient solver` | contains "Use the smallest sufficient solver…" | 3/5 | `Nano`, `Runtime` |
| `E4 KILL 0.638` | contains "best classical utility was about 0.638" | 1/2 | `KILL` |
| `official M0 utility 0.925` *(a USEFUL card)* | contains the 0.925 table row | 2/3 | `utility` |

The third row is the load-bearing observation: **even the query that succeeds
has its BM25 claims dropped by this filter**, and survives only via the
numeric-literal `FIND` path. Queries lacking an exact numeric token have no
fallback. This is why retrieval-margin sweeps recover nothing —
`wedge_v1/eval/margin_sweep.py` at τ=0.0 recovers 0/3 with a passing control.

**Mechanism:** natural queries put the *subject* in a document heading or
earlier section and the *predicate* in the answering sentence. Demanding both
inside one 240-char span is a conjunction the corpus structurally cannot
satisfy.

## 2. Why the filter exists (do not break this)

Its docstring is explicit: *"so governance prose mentioning NanoScribe cannot
'answer' clinical-accuracy questions."* It prevents a document that merely
shares vocabulary from answering an unrelated question. Any fix must preserve
that property — this project's differentiator is that it does not assert what
it cannot ground.

## 3. The change under test

**Two-scope relevance.** Evaluate token coverage at two scopes instead of one:

- **document scope** — all query tokens must be present somewhere in the
  source document (preserves the anti-false-positive property: a governance
  document still cannot answer a clinical query);
- **span scope** — a majority (`ceil(n/2)`, minimum 2 for n≥3) must be present
  in the claim value + evidence (keeps the answer content anchored in the span
  actually being presented).

A claim is relevant iff **both** hold. Single- and two-token queries keep their
current thresholds unchanged.

## 4. Preregistered decision rule

Measured on the frozen r4 task pack (`task_pack_digest 939e7ad8…`,
`corpus_digest f768fbac…`) via `wedge_v1/eval/margin_sweep.py`-style scoring:

| metric | definition | requirement |
|---|---|---|
| **recovered** | of the 3 recorded OVER_ABSTENTION queries, how many now return a span-supported claim | **≥ 2 of 3** |
| **held** | of the 4 recorded CORRECT_ABSTENTION queries, how many still ABSTAIN | **4 of 4 — no exceptions** |
| **regressions** | of the 3 USEFUL queries, how many still answer | **3 of 3** |
| **adversarial** | `wedge_v1/eval/adversarial.py` suite | **6/6 unchanged** |
| **suite** | `wedge_v1` test suite | **no new failures** |

**ACCEPT** iff all five hold. **REJECT** otherwise — in particular, any single
lost CORRECT_ABSTENTION rejects the change outright, even if all three
OVER_ABSTENTIONs are recovered. Trading trustworthiness for coverage is the
one trade this project does not make.

If REJECTED, the next candidate is span-window widening (evidence text beyond
240 chars) rather than threshold relaxation — recorded here so the fallback is
also preregistered.

## 5. Scope

Product layer only (`wedge_v1`). Does **not** touch the model, the H-cycle
gates, `nano_ai/`, or `fabric/`. Related but distinct: H6's rejection on
absence/conflict/uncertainty is the same disease in the *model*
(`papers/H6_TERMINAL_DECISION.json`, `papers/THE_REFRAME_20260805.md`); this
preregistration does not address that and must not be cited as evidence about it.
