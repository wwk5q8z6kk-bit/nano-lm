# Decision — span port takes route (b): generate then relocate

**2026-08-06.** Resolves hurdle H-2 in `FORWARD_PLAN_20260806.md`, which was
named as the objection most likely to invalidate the pretrained direction.
Independent read-only design review; every claim below cites code that was read.

## Decision

**Route (b): the model generates the span as text; the span is relocated in the
document by unique string match.** Route (a) — porting the pointer head onto a
pretrained encoder — is rejected.

## Why (b), on evidence

**The verification and abstention layer is already route-agnostic.**
`state_span.py::_verify_proposal` / `_rejected_field` and `pointer_span.py` both
consume the same `StateSpanProposal` and are checked against `_extract_fields`
(`deterministic_v0.py:299-301`), a pure regex binder with **zero model
dependency**. Both routes inherit the same safety net; neither gets an advantage
there.

**Route (b)'s core already exists and is tokenizer-free.**
`_locate_unique_patient_span` / `parse_state_span_summary`
(`state_span.py:76-172`) have no tokenizer or model coupling at all —
`StateSpanSolver.predict` is a bare `Callable[[str], object]`. Swapping the base
model does not touch it.

**Route (a) is pinned to Nano's own trunk.**
`NanoEvidenceQueryPointerModel.__init__` (`evidence_query_model.py:73-119`) hard-
asserts Nano's exact 192-wide / 6-layer geometry, and `load_frozen_tokenizer` /
`_require_exact_tokenizer` hash-pin Nano's tokenizer
(`anchor_checkpoint.py:26`). Porting needs a new encoder-attachment class, a new
tokenizer-identity scheme, a new supervision-data builder, and a training regime
with no precedent in this repo.

**The evaluation metric is unchanged either way** — `_proposal_exact` /
`_span_key` (`evaluate_pointer.py:915-930`) compare a set of
`(start, end, text, speaker)` keys, which route (b) still produces after
relocation.

**Production practice converged here already.** Anthropic's Citations API (2025)
and the documented generate-then-locate pattern both refuse to trust a
generator's self-reported offsets and verify against the source, returning null
rather than guessing — the same principle Nano's contract already encodes.

## A defect this review found in my own code

`build_lora_control_data.py:80-99` parses gold spans via
`parse_state_span_summary` — so `gold.spans` is in hand — and then writes only
`_EXPECTED[gold.state]`, a bare label, as the training target. **The spans were
available and discarded.**

That is the verified, specific reason the tuned model in `RESULT_LORA_CONTROL.md`
§3b "produces no span at all." It is not a research gap. It is a ~20-line change
to an existing script, and it means route (b)'s data work is essentially done.

**Not fixed yet, deliberately.** The P0 transfer curve is running against this
exact dataset. Changing the training target mid-run would break the
constant-recipe guarantee that makes the curve's points comparable. The
span-bearing dataset is a **v2 build** produced after the curve completes.

## Falsifier

LoRA a candidate base on the corrected span-bearing target, run generations
through the unmodified parser, and report **two separate rates**:

- `no_match_rate` — generated span text absent from the document (paraphrase
  failure). Route (a) is structurally immune to this; route (b) is not. **This
  is the number that decides the route.**
- `ambiguous_rate` — text present more than once. Both routes hit this, at
  different layers.

Low `no_match_rate` → route (b) is cheap and sufficient. High → the
recommendation flips to route (a) or grammar-constrained decoding.

Both failures degrade to **abstention**, not to a wrong answer, because
`_locate_unique_patient_span` refuses a non-unique or absent match. That is the
correct failure mode for a system whose claim is *never assert what you cannot
ground* — and it is the property that made route (b) preferable even before cost
was considered.
