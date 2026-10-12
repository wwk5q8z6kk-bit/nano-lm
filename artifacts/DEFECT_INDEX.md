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
so the true site count is **sixteen**. Figures in this table are re-measured in
this repository wherever possible; where a widely-quoted number could not be
reproduced it is marked as such rather than repeated. They are enumerated as `N.x` below,
because a bundled sub-defect is exactly the kind that gets lost when someone
later reads only the headline. Two of them (D1.2 and D2.3) are independently
capable of voiding a whole wave's conclusions and neither appears in any
headline summary.

**D3.3 is now ASSERTED AT RUNTIME even though its corpus-level fix is still
open.** `integrity.assert_context_budget` runs on every training run: it raises
if the target alone cannot fit or if any gold span is lost to prompt
left-truncation, and it reports the over-budget fraction into every run's
integrity block. `corpus/validate.py::sequence_budget` had computed this since
`3991096` but only at corpus *build* time, which is exactly how D3.3 outlived
D3.1 — the 64-char cap was removed while the underlying mismatch went unmeasured
in every run report. Measured on the shipped corpus: **96/96 rows over budget,
median 551 tokens against `max_seq` 512, `relies_on_left_truncation: true`, and
`spans_visible: true`** — the last being why the corpus is learnable at all.

**Two entries are OPEN**, both deliberately: D3.3 (below) and D8 (init logit
scale). Neither is fixed in place, because both change the measurement
instrument and this program's rule is that an instrument change gets its own
preregistered wave rather than a silent patch.

**One of them:** D3.3 — character-level tokenization against a
512-token context, with 82.7% of eval prompts and 100% of the training corpus
over the window. D3.1 removed the 64-char cap but never addressed the underlying
mismatch, so **D3 is only partly closed**. It is preregistered as a separate
wave in `artifacts/PREREG-D33-context-tokenizer-wave.md` and deliberately *not*
folded into the completed revalidation. Everything else here is fixed and
asserted at runtime.

---

## Index

| id | site | mechanism | bias | fixed in | taints |
|---|---|---|---|---|---|
| **D1.1** | `Block.forward` | `nn.MultiheadAttention(h,h,h)` with no `attn_mask`/`is_causal` in a next-token decoder — every position attends to its own label | **Inflates.** Objective solvable by copying the future. The commonly-quoted `loss → 0.002` is **not reproducible** from this description (see `bpb_floor_derivation_v1.md`); isolated, D1.1 gives a 15% loss improvement (3.989 vs 4.718), which is real but not a collapse | `c98e4ad` | every native30/native100 training loss and any learned-capability claim derived from it |
| **D1.2** | `losses.py` arm objectives | `span_port`/`evidence_align`/`assertion_state` were `lm*0.5`, `lm*0.25`, `lm*0.1` — scalar multiples of one number, so each arm's total was an affine function of `lm` | **Voids comparison.** All three arms shared an identical gradient direction and differed only in effective LR (1.0× / 1.5× / 1.15×) | `c98e4ad` | the entire native30 three-arm comparison — it measured learning rate, not objective |
| **D2.1** | `losses.py` budgeting | `(prompt_ids + target_ids)[:max_seq]` truncates right; prompts are 519–642 chars vs `max_seq=512`, so the target was discarded | **Inflates.** `final_loss` 0.017–0.084 was next-char prediction on templated prompt text | `35ad570` | 19,194/19,194 examples in the 2026-08-24 native30 wave |
| **D2.2** | `losses.py` labels | No prompt masking — `labels = seq[1:]` supervised every prompt position | **Inflates.** `final_loss` could not measure target prediction even when the target survived | `35ad570` | same wave; compounds D2.1 |
| **D2.3** | `analyze_revalidation.py` | Binary verdict fallthrough: total output collapse became `NOT_SEPARATED` — an inferential claim — when coverage was 0/150 and three metrics had `eligible=0` | **Fabricates a null.** Six false `NOT_SEPARATED` nulls reported as findings | `35ad570` | six cells of the 2026-08-24 wave, now correctly `INVALID_NO_SIGNAL` |
| **D2.4** | `eval_one_run` | Discarded the eval subprocess returncode and stderr | **Hides failure.** An eval crash was indistinguishable from an eval scoring zero | `35ad570` | any wave cell whose eval crashed — unknown which, by construction |
| **D3.1** | `hash_tokens: text[:64]` | Hard-truncated every input to 64 chars; prompts average 530 | **Inflates.** Model saw 12.1% of each prompt; gold value usually invisible → task impossible by construction, loss fit 96 memorised fragments | `native_tokenizer_defect_v1.json` | native30 8-way round-1 tournament; native100 round-2 promotion ranking; native100 extended runs; all `exact_gold_span`/support metrics from constrained eval |
| **D3.2** | `_char_for_token: range(32,127)` | `\n`, `\t`, `\r` had no preimage and decoded to `?` | **Corrupts input.** Turn separators destroyed in 224/224 corpus sources | same | same as D3.1 |
| **D3.3** | `hash_tokens` vs `max_seq=512` | Character-level tokenization against a 512 context: **82.7%** of `p1_screening_eval_v1` prompts (median 530 tok, max 576) and **100%** of the training corpus (median 615, max 694) exceed the window. D3.1 removed the 64-char cap but never addressed the underlying mismatch — **D3 is only partly closed** | **Confounds.** Median 4.4% / worst 11.1% of eval content dropped; scale and truncation move together, so any parameter result is measured through a truncating tokenizer | **OPEN at the corpus level; ASSERTED at runtime** by `assert_context_budget`, and the BPE path is now implemented behind `NativeTrainConfig.tokenizer` (`b78a9e5`, default `hash` so frozen classes stay reproducible). Asset re-measured in this repo, superseding the figures first recorded here: vocab **4096** (not 4098; still fits the 4098-row embedding), compression **2.537×**, median **216 tok**, **0/96 over context** on the training corpus vs char-level median 551 / 96-96 over | the 2026-08-25 causalfix capability-floor conclusion — it stands as written because it names the tokenizer, but "30M is below the floor" is an impermissible restatement |
| **D4** | `nanoscribe/evaluate.py` | `unbound_assertion` laundered into `correct_abstention` — a hallucinated assertion scored as a *correct abstention* | **Inflates safety.** Model-level commission credited to the binder's save | `be937c1` lineage | **No new taint.** Frozen Paper-α lineage: **0** — recounted across all 11 frozen trees (`c1b`, `c3_primary`, `c3_replication`, `e1`, `e3`, `pointer_p1/p2`, `slot_diversity`, `durable_raw`, `stage_t_v2`, `ownstack_corner`) plus `trajectory/`, `fabric/`, `scribe/`. 38 files in `artifacts/campaign/` do carry the field, but those are native-line results already void from D1–D3. `papers/`: 1, and it *describes* the defect rather than reporting a result |
| **D5.1** | `prompt.py` answer template | `reply STATED: "{raw_value}"` hands over the exact string; system-prompt examples are themselves gold answers | **Inflates.** A model copying its instructions scores as one that read the transcript | orx leakage thread | `dc3b310`'s "campaign_v1 coverage 0% → ~83%" |
| **D5.2** | `adapt.candidate_from_span_port_line` | Substitutes gold `raw_value` as the model's quote when the model emitted a label with no quote | **Inflates.** A bare `STATED` resolves to the exact gold span | orx leakage thread | same — but measured **inert**: `quote_absent` 2/192, Δ`asserted_grounded` 0.000 |
| **D5.3** | `prompt.py topic_for_spec` | The **question** interpolated `spec.raw_value!r` — 14/16 slots — and was classified as task specification, so it was pinned ON in all four original ablation cells | **Inflates, and voids the ablation.** The cell named "leakage-free" still leaked; a question-only parrot scored 9/16 against a 10/16 ceiling, identical in all four cells | orx leakage thread | the original 2×2 design itself; forced the 2×2×2 redesign |
| **D6** | `integrity.py` BPB floor | A flat `BPB_HARD_FLOOR = 0.01` was shipped claiming it "cannot false-positive and still fires on the defect". The known-leaking archived `reval30_decoder_control_s0` finished at loss 0.00823 → **0.0119 bpb** and would have passed | **Would have cleared a leaking run.** True at 40 steps, false at 1800 — the quoted "loss ≈ 0.002" signature is a *40-step* number retold as if it described converged training | `11c907c` | nothing archived — caught in the same session it was introduced, before the gated wave ran |
| **D7** | `train.py` BPB numerator | Total nats computed as `lm * target_chars`. `lm` is a mean over supervised **token** positions, so the char count is only correct for a character-level tokenizer | **Weakens the gate under a tokenizer change.** Under the 2.60x BPE D3.3 proposes this overestimates total nats ~2.6x and inflates BPB by the same factor, moving the measurement *away* from the floor — it would have fired exactly when the instrument changed. Also wrong by 6.3% in the char case (9.4848 -> 10.0840), since `n_supervised` includes the boundary position | this session | nothing archived — the only BPB figures recorded so far are from char-level runs, where the error is 6.3% and no verdict rests on it |
| **D8** | `model.py` tied output projection | `logits = h @ token_emb.weight.T` with **no 1/sqrt(d) scaling**, against `nn.Embedding`'s default N(0,1) init. Measured at d_model=256: logit std **16.2 = sqrt(256)**, absmax 201. Init CE is **39.7 nats/token against ln(vocab)=8.3** — the model starts ~4.8x worse than uniform | **Confounds scale comparisons.** Init loss grows as sqrt(d_model), so a wider model starts worse for reasons unrelated to capability. Not a flattering defect — it depresses early loss quality — but it is scale-coupled, and this program makes cross-scale claims | **OPEN, deliberately unfixed** — fixing it changes the instrument, so it needs its own prereg like D3.3 | **not** the native30 wave (all nine arms are 30M, so the offset is common to every cell and cancels in the contrasts). It **would** confound any native30-vs-native100 loss/BPB comparison |

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

## D6 in full, and the measurement it produced

D6 is the only defect here found in *this program's own instrumentation* rather
than in the thing being instrumented, and it is the one worth reading twice: the
check was written specifically to catch D1.1, was tested, passed its tests, and
would still have cleared the very run it was designed to catch.

The fix was to scope the claim rather than move the constant. `floor_for_step()`
is 0.5 at ≤100 steps — where the model has seen each example at most once, so a
near-zero loss has no innocent explanation — and the 0.01 impossibility bound
after. Nudging 0.01 upward until it separated the two archived points would have
produced a number that fit the data and meant nothing.

**The gated wave then confirmed the scoping empirically at n=27.** Final
training loss across the three provenance classes overlaps heavily, and
`decoder_control_s1` in the GATED class — attention leakage proven exactly 0.0,
matching the causal reference to 3.58e-07 — finished at **0.00962, lower than
eight of the nine known-leaking archived runs**. Medians: archived 0.028,
causalfix 0.086, gated 0.061.

> **Converged training loss is not a leak detector.** Memorisation and label
> leakage are indistinguishable in it. The detector is the startup assertion,
> which is exact rather than statistical. Full table:
> `artifacts/RESULT-native30-gated-revalidation.md`.

**Strengthened further by a fourth population found while auditing disk.**
`artifacts/campaign/kaggle_native30_download/` turned out not to be a copy of the
archived wave but a **second independent execution of it** — same run_ids,
byte-identical configs, same 1800 steps, same date, different hardware, and
materially different results. Converged loss for the same config differs between
the two executions by up to **27.1×** (Kaggle median 0.063, range 0.006–0.753;
local median 0.028, range 0.008–0.536).

That is run-to-run variance *within the defective population alone*, and it
settles D6 beyond the n=27 argument: no converged-loss threshold can separate
leaking from clean runs when a single leaking config reproduces across a 27×
spread. Metadata preserved independently of the weights in
`artifacts/campaign/native30_kaggle_replicate_metadata.json`.

## Standing rule this history produces

> A check that has passed once is not protection. All of D1.1, D2.1 and D3.1
> shipped as passing code. Convert every defect into a runtime assertion in the
> entrypoint, and prefer a differential test against a known-correct reference
> over a derived metric — a derived metric can itself be wrong, and this program
> has already shipped one non-binding check.

And the rule D6 adds, which is sharper:

> **Calibrate a threshold against the archived failure it claims to catch, not
> against the anecdote about that failure.** D6 was calibrated on a remembered
> "loss ≈ 0.002" and would have cleared the real run, whose converged loss was
> 4× higher. The archived checkpoints carried the true number the whole time.

> **A threshold's anchor must be reproducible, or the threshold is folklore.**
> D6 said calibrate against the archived failure rather than the anecdote about
> it. §7.3 then found the archived failure itself could not be reproduced from
> the recorded descriptions — seven reconstructions landed 3.99–401 bpb against
> an anchor of 0.002. The gate was kept and its *claim* narrowed, which is the
> move: a backstop that admits what it cannot detect is worth more than a
> detector that is trusted for something it never did.

> **A build-time gate is not a runtime gate.** D3.3's invariants were computed
> and correct from `3991096` onward — and nothing ran them per-run, so the
> corpus could drift out from under them and no run report would show it. If a
> check matters, it belongs in the entrypoint, not only in the builder.

> **A gate is only calibrated for the instrument it was measured on.** D7 was
> correct for the character tokenizer it was written against and silently wrong
> for the one the program is about to adopt. When the instrument changes,
> re-derive the thresholds rather than carrying them across — the D3.3 prereg
> makes that a blocking pre-run item rather than a hope.

> **Never merge "the gate passed" with "the result replicated."** The first is
> measurement validity, the second is an empirical claim; each can hold while
> the other fails. The gated wave established both, separately, and the write-up
> keeps them in separate sections for that reason.

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
