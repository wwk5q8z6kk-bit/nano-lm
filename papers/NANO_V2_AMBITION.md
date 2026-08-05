# Nano v2 ambition — the full contract (owner-set, 2026-08-05)

**Owner directive:** the next-phase target is *more than* a general structured
scribe. This document records the maximal ambition and the measured ladder
that reaches for it. Ambition lives here; evidence lives in preregistrations.
Nothing in this file authorizes compute.

## The vision (ceiling, not next step)

Nano v2 is a **general text-intelligence scribe**: given any transcript,
document, or stream of human working conversation, it can —

- **understand and organize**: extract who/what/when/decisions/actions/values
  across any domain (clinical, corporate, technical, personal);
- **reason before writing**: internal deliberation (think-arm) that plans the
  output before emitting it;
- **produce structured artifacts**: hierarchical notes, markdown tables,
  SOAP-style charts, agile backlogs, timelines, and **compile-valid Mermaid**
  (flowchart/sequence/gantt) — machine-checkable structure, never broken syntax;
- **stay evidence-bound**: every asserted value grounded in a source span,
  explicit missing/conflicting/uncertain states, abstention when unsupported —
  the verification-first identity that already distinguishes Nano from generic
  LLMs (fabric v2: 0.0% presented error);
- **run local-first** in a compact envelope; audio deferred (cascade later
  with an off-the-shelf ASR front-end; from-scratch ASR only if a measured
  failure of that cascade ever demands it).

## The ladder (each rung preregistered, measured, kill-gated)

| Rung | Capability added | Instruments | Status |
|---|---|---|---|
| L0 | 5-field clinical scribe, evidence-bound (today's Nano) | frozen H-cycle suite, fabric | **exists**; H6 verdict pending (Kaggle run live) |
| L1 | Real language capability at 160M (licensed pretrain) + generalized field extraction beyond the clinic-synthetic world | held-out-value instruments carried forward; ACI-BENCH-class external check | = SCALE_PROGRAM_PREREG rung 1 (blocked on license + budget approvals) |
| L2 | Structured outputs: tables, SOAP/backlog charting, Mermaid — every generated artifact machine-validated (mermaid-cli compile gate; table parser gate) at data-generation AND evaluation time | synthetic structured-pair suites w/ compile-validation; structure-exactness metrics | design in rung-1 finetune; full at rung 2 (1B) |
| L3 | Reasoning/think-arm: deliberation traces before output; register `<think>`/`</think>` in the tokenizer from day one so the option is never architecturally foreclosed | ablation arm: with/without think-traces, same budget — measured, not assumed | small preregistered arm at rung 2 |
| L4 | Breadth and robustness: messy multi-speaker transcripts, interruptions, noise injection, cross-domain transfer | noisy-instrument variants (the W5 ingest-SLA discipline, applied to transcripts) | rung 2+ |
| L5 | Beyond: multi-document working memory, incremental/streaming scribing, self-verification loops | to be preregistered when L2–L4 evidence exists | vision |

## Standing rules carried into v2 (unchanged)

Dataset gate (no training without a complete, verified, licensed path through
`nano_ai/pretraining/`); smallest sufficient solver; every corpus pinned +
hashed + contamination-digested; sealed evaluation partitions; kill gates
named before spend; negative results retained; runtime pinned to
provider-controllable identities only; process frozen at current size.

## Immediate consequences

1. Rung-1 finetune target broadens from the 5-field clinical contract to
   **general structured contract v0**: field extraction + hierarchical notes +
   markdown tables, with clinical as one profile. Mermaid enters at L2.
2. Tokenizer decision for rung 1 must reserve special tokens
   (`<think>`, `</think>`, structural symbols) NOW — costless, prevents a
   future retrain.
3. Data additions to work through `sources.py` gates: Stack-v2
   markdown/Mermaid slices; MeetingBank/QMSum (license review); synthetic
   structured pairs with compile-validation; frontier-API dialogue synthesis
   (spend approval). MIMIC remains prohibited without compliance review.
4. Audio: deferred entirely; revisit as a cascade integration after L2.
