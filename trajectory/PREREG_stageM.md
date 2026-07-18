# PREREG — Stage M (Paper 3): mechanism of held-out-value copying failure

*Drafted 2026-07-18. Methodology pre-registration. NOT executed. Scoped to the
**non-gated** part of the mechanism question; the gated part is deferred to post-P2.*

## Two mechanism questions — one is gated, one is not

The program (RESEARCH_PROGRAM.md, S4) gates Stage M on P2 deconfounding. That gate applies
to the **scale-collapse** question — *why does the gap collapse nano→Pythia?* — because
a circuit found on the confounded nano→Pythia contrast is entangled with the stack change
(a smallness / tokenizer / full-FT artifact could masquerade as the cause). That stays
deferred.

But there is a second, **non-confounded, within-stack** question the fieldwise/clean
result opened and P1 does not answer:

> **Q(M): In a single frozen own-stack scribe model, what mechanism copies a *seen*
> value correctly yet fails near-totally on a *held-out* value under the same template?**

This needs no cross-stack comparison — it is one model, held vs. seen *values*, template
held fixed. It is the ~0%-recall-on-held-out-med/allergy finding (§6.1 clean metric) asked
at the circuit level. This pre-reg covers Q(M); the scale-collapse contrast is a slot
filled after P2 (below).

## Targets (frozen, already in hand)

- **Local, inference-only:** the frozen own-stack anchors `scribe.pt` (3.15M) and
  `scale10m_scribe.pt` (10M) — runnable on MPS with the validated `rescore_anchors.py`
  model + generate path. No training, no Kaggle.
- Pythia rungs: deferred (need the adapters; and cross-stack mechanism is the gated part).

## Method (activation patching / head ablation, pre-specified)

Following the induction-head (Olsson et al., 2022) and retrieval-head (Wu et al., 2024)
methodology, on a matched pair of prompts that differ *only* in the field value
(seen-value vs held-out-value, same template, same field):

1. **Localize the copy.** For each attention head, **ablate** (zero / mean-patch) its
   output and measure the drop in *seen-value* recall. Heads whose ablation collapses
   seen-value copying are the candidate copy/retrieval heads (expect a sparse set,
   per Wu et al.: <5%).
2. **Test the failure.** On held-out-value prompts, measure whether those same heads
   **fire and attend to the correct input span** (the held-out value token) but the
   value fails to surface, vs. **fail to attend** at all. Patch the head's attention
   pattern / value-vector from the seen-value run into the held-out run and measure
   **gap recovery** (how much of the held-out failure the patch repairs).
3. **Distinguish two failure modes (pre-registered hypotheses):**
   - **H-M1 (routing failure):** the copy heads attend to the right span but the held-out
     token's representation is not routed to the output — patching the *attention pattern*
     recovers little, patching the *value/OV pathway* recovers the gap.
   - **H-M2 (representation failure):** the held-out token's residual representation is
     impoverished (rare/fragmented subwords) — patching the *residual at the value
     position* recovers the gap, attention patterns are already correct.
   - **H-M3 (distributional override):** the output distribution is dominated by a
     memorized seen-value prior — logit-lens on the copy heads shows the correct held-out
     token present in the residual but out-competed at the unembedding.

## Metric

Per head / per intervention: **gap recovery** = (patched held-out recall − unpatched
held-out recall) / (seen recall − unpatched held-out recall), in [0,1]. A single
intervention with recovery ≥ 0.5 on ≥ 2 held fields = a located mechanism. Report per-head
recovery, the minimal head set, and which of H-M1/H-M2/H-M3 the pattern supports.

## Decision rule (pre-specified)

- If a sparse copy-head set is located and one intervention recovers the gap → mechanism
  identified; name it (routing vs representation vs override) per the H-M pattern.
- If no sparse set / distributed effect → report distributed-mechanism, which itself
  constrains the capacity-threshold hypothesis (abstraction is not one head).
- Null: if seen-value copying has no localizable head set at 3–10M, that is evidence the
  small models copy via a non-head (e.g. MLP-memorization) route — directly relevant to
  the memorization-vs-abstraction mission.

## P2-conditional slot (the gated scale-collapse contrast — fill AFTER P2)

Once P2 (OWNSTACK_160M) resolves scale-vs-stack, add the cross-model contrast:
- **If own-stack-160M stays high-gap:** compare the *same* mechanism between own-stack-10M
  (high) and own-stack-160M (high) — why scale within the stack did *not* install the fix.
- **If own-stack-160M collapses:** compare own-stack-10M (high) vs own-stack-160M (low) —
  what circuit appears at the transition. This is the clean, deconfounded mechanism
  contrast the gate was protecting.

## Feasibility / cost

Q(M) on the frozen anchors is **inference-only, local (MPS)** — a hook-based patching
harness over the `rescore_anchors.py` model, no Kaggle, no training. A pilot is the cheap
first step whenever P3 is greenlit; the P2-conditional contrast waits on P2.

## Status

Drafted, not executed. Q(M) is runnable now (local); the scale-collapse contrast is gated
on P2. This closes the P3 design gap in the program roadmap without pre-committing the
gated comparison.
