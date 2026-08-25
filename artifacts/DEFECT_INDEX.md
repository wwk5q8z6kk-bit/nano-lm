# Defect index — measurement and training integrity

**Purpose.** Single indexed record of every known defect that inflated a result
in this program. Before this file existed the set was scattered across commit
messages, one JSON finding, a document that had been written but never landed in
the working tree, and two session transcripts. It is the evidence base for Paper
α's measurement-reliability contribution.

**Status:** current as of `frontier/accelerated-research-campaign-v2`.

**Two framing facts that are the actual contribution:**

1. **Every defect below biased a result in the favourable direction.** Not one
   produced a number that looked wrong. That is not coincidence — a defect that
   depresses a metric gets investigated the same day, so the surviving
   population of undetected defects is precisely the flattering ones. Any
   program without adversarial checks is therefore accumulating a biased
   residual, silently.
2. **No primary metric caught any of them.** Each was found by an adversarial
   check, a differential test, or by reading code while fixing something else.
   Four of the five were found while fixing a different defect.

**Counting note.** The program refers to "five defects." That is the count of
*fix threads*, not of distinct failure sites. Three of the five threads bundle
multiple independent defects that were found together and fixed in one commit,
so the true site count is **twelve**. They are enumerated as `N.x` below, because
a bundled sub-defect is exactly the kind that gets lost when someone later reads
only the headline. Two of the twelve (D1.2 and D2.3) are independently capable of
voiding a whole wave's conclusions and neither appears in any headline summary.

---

## Index

| id | site | mechanism | bias | fixed in | taints |
|---|---|---|---|---|---|
| **D1.1** | `Block.forward` | `nn.MultiheadAttention(h,h,h)` with no `attn_mask`/`is_causal` in a next-token decoder — every position attends to its own label | **Inflates.** Objective solvable by copying the future; loss → 0.002 | `c98e4ad` | every native30/native100 training loss and any learned-capability claim derived from it |
| **D1.2** | `losses.py` arm objectives | `span_port`/`evidence_align`/`assertion_state` were `lm*0.5`, `lm*0.25`, `lm*0.1` — scalar multiples of one number, so each arm's total was an affine function of `lm` | **Voids comparison.** All three arms shared an identical gradient direction and differed only in effective LR (1.0× / 1.5× / 1.15×) | `c98e4ad` | the entire native30 three-arm comparison — it measured learning rate, not objective |
| **D2.1** | `losses.py` budgeting | `(prompt_ids + target_ids)[:max_seq]` truncates right; prompts are 519–642 chars vs `max_seq=512`, so the target was discarded | **Inflates.** `final_loss` 0.017–0.084 was next-char prediction on templated prompt text | `35ad570` | 19,194/19,194 examples in the 2026-08-24 native30 wave |
| **D2.2** | `losses.py` labels | No prompt masking — `labels = seq[1:]` supervised every prompt position | **Inflates.** `final_loss` could not measure target prediction even when the target survived | `35ad570` | same wave; compounds D2.1 |
| **D2.3** | `analyze_revalidation.py` | Binary verdict fallthrough: total output collapse became `NOT_SEPARATED` — an inferential claim — when coverage was 0/150 and three metrics had `eligible=0` | **Fabricates a null.** Six false `NOT_SEPARATED` nulls reported as findings | `35ad570` | six cells of the 2026-08-24 wave, now correctly `INVALID_NO_SIGNAL` |
| **D2.4** | `eval_one_run` | Discarded the eval subprocess returncode and stderr | **Hides failure.** An eval crash was indistinguishable from an eval scoring zero | `35ad570` | any wave cell whose eval crashed — unknown which, by construction |
| **D3.1** | `hash_tokens: text[:64]` | Hard-truncated every input to 64 chars; prompts average 530 | **Inflates.** Model saw 12.1% of each prompt; gold value usually invisible → task impossible by construction, loss fit 96 memorised fragments | `native_tokenizer_defect_v1.json` | native30 8-way round-1 tournament; native100 round-2 promotion ranking; native100 extended runs; all `exact_gold_span`/support metrics from constrained eval |
| **D3.2** | `_char_for_token: range(32,127)` | `\n`, `\t`, `\r` had no preimage and decoded to `?` | **Corrupts input.** Turn separators destroyed in 224/224 corpus sources | same | same as D3.1 |
| **D3.3** | `hash_tokens` vs `max_seq=512` | Character-level tokenization against a 512 context: **82.7%** of `p1_screening_eval_v1` prompts (median 530 tok, max 576) and **100%** of the training corpus (median 615, max 694) exceed the window. D3.1 removed the 64-char cap but never addressed the underlying mismatch — **D3 is only partly closed** | **Confounds.** Median 4.4% / worst 11.1% of eval content dropped; scale and truncation move together, so any parameter result is measured through a truncating tokenizer | **OPEN** — fix is an existing asset, `sft/tokenizer.json` (own-stack BPE, same vocab 4098): measured 2.60x compression, median 204 tok, **0/150 over context**, embeddings unchanged | the 2026-08-25 causalfix capability-floor conclusion — it stands as written because it names the tokenizer, but "30M is below the floor" is an impermissible restatement |
| **D4** | `nanoscribe/evaluate.py` | `unbound_assertion` laundered into `correct_abstention` — a hallucinated assertion scored as a *correct abstention* | **Inflates safety.** Model-level commission credited to the binder's save | `be937c1` lineage | **No new taint.** Frozen Paper-α lineage: **0** — recounted across all 11 frozen trees (`c1b`, `c3_primary`, `c3_replication`, `e1`, `e3`, `pointer_p1/p2`, `slot_diversity`, `durable_raw`, `stage_t_v2`, `ownstack_corner`) plus `trajectory/`, `fabric/`, `scribe/`. 38 files in `artifacts/campaign/` do carry the field, but those are native-line results already void from D1–D3. `papers/`: 1, and it *describes* the defect rather than reporting a result |
| **D5.1** | `prompt.py` answer template | `reply STATED: "{raw_value}"` hands over the exact string; system-prompt examples are themselves gold answers | **Inflates.** A model copying its instructions scores as one that read the transcript | orx leakage thread | `dc3b310`'s "campaign_v1 coverage 0% → ~83%" |
| **D5.2** | `adapt.candidate_from_span_port_line` | Substitutes gold `raw_value` as the model's quote when the model emitted a label with no quote | **Inflates.** A bare `STATED` resolves to the exact gold span | orx leakage thread | same — but measured **inert**: `quote_absent` 2/192, Δ`asserted_grounded` 0.000 |
| **D5.3** | `prompt.py topic_for_spec` | The **question** interpolated `spec.raw_value!r` — 14/16 slots — and was classified as task specification, so it was pinned ON in all four original ablation cells | **Inflates, and voids the ablation.** The cell named "leakage-free" still leaked; a question-only parrot scored 9/16 against a 10/16 ceiling, identical in all four cells | orx leakage thread | the original 2×2 design itself; forced the 2×2×2 redesign |

---

## What the index is for

**D1.2 and D2.3 are the entries to re-read.** Neither appears in any headline
summary of "the five defects," and each independently voids a wave's
conclusions — D1.2 because the arm contrast measured learning rate rather than
objective, D2.3 because it manufactured six nulls that were reported as
findings. A summary that lists five defects and omits these is how they get lost
a second time.

**D4 is the counter-example and is why the "taints" column exists.** The
retroactive audit for it came back *negative* for the frozen lineage: the metric
post-dates it, and a recount confirms 0 occurrences across all eleven frozen
trees plus `trajectory/`, `fabric/` and `scribe/`. Without a recorded scope the
next session re-derives the question and may answer it differently — and the
safe-looking answer ("assume everything is tainted") is itself a claim that
needs evidence.

**One correction made while building this index.** The recovered audit reports
`artifacts/` as 0 files for D4. That is stale: 38 files under
`artifacts/campaign/` carry `correct_abstention`. It changes no conclusion —
those are native-line results already void from D1–D3, so D4 adds no *new*
taint — but the flat "0" would have been read as "nothing anywhere," and the
audit was written before those results landed. Recorded here rather than
silently corrected, because the original number is still sitting in the audit
document and someone will read it again. **Recheck:**
`for d in artifacts/c1b artifacts/c3_primary artifacts/c3_replication artifacts/e1 artifacts/e3 artifacts/pointer_p1 artifacts/pointer_p2 artifacts/slot_diversity artifacts/durable_raw artifacts/stage_t_v2 artifacts/ownstack_corner trajectory fabric scribe; do grep -rl correct_abstention "$d" 2>/dev/null | wc -l; done`
— every line must read 0.

**D5.2 is the counter-example in the other direction.** The channel was real and
open, but measured **inert** on the live instrument: the parser fallback only
fires on quote-less output, and the model quoted in 190/192 slots. A channel
being open is not evidence it was exploited. That distinction is the difference
between the parrot floor (an upper bound on what leakage *can* buy) and an
effect size (what it *did* buy).

## Standing rule this history produces

> A check that has passed once is not protection. All of D1.1, D2.1 and D3.1
> shipped as passing code. Convert every defect into a runtime assertion in the
> entrypoint, and prefer a differential test against a known-correct reference
> over a derived metric — a derived metric can itself be wrong, and this program
> has already shipped one non-binding check.

## Sources consolidated here

- `c98e4ad`, `35ad570`, `3991096` commit messages
- `artifacts/campaign/native_tokenizer_defect_v1.json`
- `artifacts/measurement-integrity-audit.md` — **was written but never landed in
  the working tree**; recovered from WIP commit `1b241d6` while building this
  index. It was one of the three cited sources and had been silently lost.
- the orx span-port leakage ablation thread (8 cells, `artifacts/RESULT-span-port-leakage-ablation.md`)

---

## Result status carried by this index (added 2026-08-25)

**native30 causalfix wave — below capability floor, tokenizer suspected.**
Prereg + RESULT: `trajectory/PREREG_causalfix_wave_arm_split.md`.

- Coverage 6/150 (**4.0%**) constrained in all nine runs; **0/150** unconstrained
  in all nine. Models abstain on 144 of 150 atoms.
- The `DENIED`-vs-`ASSERTED` arm split was **seed noise** — it came entirely from
  `evidence_bottleneck_s0`; seeds 1 and 2 read 0/6 like every other arm. Wilson
  intervals overlap and direction held in 1/3 seeds, both pre-registered as
  disqualifying.
- Capability-floor clause fired, **with the D3.3 confound attached**.

**Do not re-run architecture arms from here.** The arms did not separate, and the
input does not fit the context (D3.3). The next variable is the tokenizer, not
scale or architecture: scaling through a truncating tokenizer confounds scale
with truncation. Ceiling calibration — Qwen2.5-1.5B on the related span-port task
selects the correct turn for ~4 of 5 gold slots and still rarely delimits the
exact span; a 30M char-level model at 4% coverage is a different regime, not a
few increments below.

## Pre-commitment as a detection mechanism (added 2026-08-25)

Two independent live uses this week, each converting a plausible false positive
into a null:

1. **Span-port co-movement rule** — caught one on its first live use, on a
   contrast that looked like a clean doubling.
2. **Causalfix arm-split prereg** — registered while only `s0` of two arms was
   observed; the s0-only view was 0/6 vs 6/6 with a mechanistic story attached.
   Seeds 1-2 refuted it.

This is a *pattern*, not two anecdotes, and it is the counterpart to the
survivorship argument at the top of this file. Survivorship explains why the
undetected defect population is systematically flattering; pre-commitment is the
mechanism that removes the flattering reading before the data is seen. Cost in
both cases: minutes. Note the asymmetry — a guard written *after* seeing `s0`
could have been fitted to it.
