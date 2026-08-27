# PREREG — tokenizer context-fit on the native30 instrument

**Registered 2026-08-26, before any arm ran.**
**Standard:** question-before-architecture-v1
**Rule source:** `research/decision_records/2026-08-26-question-before-architecture.md`;
candidate 1 in `docs/RESEARCH_STATUS.md` § NEXT CANDIDATE EXPERIMENTS
**Base commit:** **TBD — this branch cannot run this experiment.** See field 15.

> **Status: NOT AUTHORIZED.** This document supplies the preregistration that
> `NANO_VNEXT_MASTER_SPEC.md` §25 requires *before* a candidate may be proposed
> for launch. It is not a request to run and confers no authorization.
>
> This is also the **first document written to the eighteen-field standard**, and
> was written partly to test whether the standard is fillable. Field 8 is not
> fully fillable from existing evidence; that finding is recorded there rather
> than papered over.

---

## 1. Product question

Medical scribing (§2) requires reading a clinical encounter transcript and
transporting evidence out of it. Every downstream P1 capability is bounded by
whether the model can see the whole encounter at once. If the input does not fit
the context, no scribing capability measured on that instrument is a measurement
of the model.

## 2. Scientific question

Is the native30 capability floor a property of the **model** (30M parameters at
1800 steps is too small), or an artifact of the **instrument** (a character-level
tokenizer that cannot fit the prompts into a 512-token context)?

This is a **foundational instrument question, not an architecture rung** — per
the *Question before architecture* record, it is resolved separately and
precedes architectural claims at this scale. It tests no mechanism.

## 3. Instrument

`p1_screening_eval_v1` — 150 prompts, the frozen screening eval already used by
the native30 wave. Training corpus `native_corpus_screen_v1` (n = 19,194
prompt+target pairs). Harness: `nanoscribe/native/p1_eval.py::campaign_cases`,
tokenizers `nanoscribe/native/tokenize.py::hash_tokens` and `sft/tokenizer.json`
via `tokenizers.Tokenizer`, prompts built by
`nanoscribe/prompt.py::build_span_port_prompt`.

Verdicts are read from the same `NOT_SEPARATED` / `INVALID_NO_SIGNAL` machinery
and the same integrity block (attention leakage measured against the
`scaled_dot_product_attention(is_causal=True)` reference) as the gated wave.

## 4. Measured bottleneck

**Truncation, measured — not inferred.** Under char-level `hash_tokens` (one
token per character) against `max_seq=512`:

| set | median | max | exceeds 512 |
|---|---|---|---|
| `p1_screening_eval_v1` (150 prompts) | 530 | 576 | **124/150 = 82.7%** |
| `native_corpus_screen_v1` train, prompt+target (n=19,194) | 615 | 694 | **19,194 = 100.0%** |

Content dropped on the eval set: median **4.4%**, worst **11.1%**.

**What is measured is the truncation, and only that.** That truncation *causes*
the capability-floor result is the hypothesis (field 5), not an established
fact. The source artifact is explicit: *"Whether 30M with a fitting tokenizer is
below the floor is untested."* The two must not blur — this is precisely the
distinction the standard's field 4 exists to enforce.

## 5. Hypothesis

The capability-floor verdict for 30M at 1800 steps is **confounded with residual
truncation on 82.7% of eval atoms**. Causal prediction: swapping to a tokenizer
that fits the prompts will move measured capability on `p1_screening_eval_v1`,
because the model will for the first time receive the whole prompt.

The competing hypothesis is that 30M at 1800 steps is genuinely below the floor
and the fit makes no difference.

## 6. Baseline / control

**Baseline:** the existing char-level arms at the same parameter count, steps,
corpus and eval — the gated wave under `reval_results_fixed/`, whose integrity
block is already verified (attention leakage exactly 0.0).

**Adversarial baselines on the same instrument, required by §22:** a
majority-class and a constant-output baseline scored through the identical
analyzer. The native30 line has produced `NOT_SEPARATED` at effect +0.0000
before; without a floor baseline a null is unreadable.

**Held fixed:** parameter count, architecture, step count, seeds, corpus
content, eval set, analyzer, decision machinery.

## 7. Manipulation

Exactly one thing changes: the tokenizer.
`nanoscribe/native/tokenize.py::hash_tokens` → `sft/tokenizer.json` (own-stack
BPE, Paper-α ladder), **same vocabulary size 4098**.

Measured consequence of the swap, already established at
`de188a2:artifacts/campaign/TOKENIZER_CONTEXT_CONFOUND.md`:

| set | median | max | exceeds 512 |
|---|---|---|---|
| eval | **204** | 226 | **0/150** |
| train | **232** | 256 | **0/19,194** |

Compression **2.60×** (eval), **2.65×** (train) — measured, not the ~4× rule of
thumb. Median prompt then occupies 39.8% of context, ~60% headroom.

## 8. Invariance requirements

**Fillable, and checked before the primary endpoint is read:**

- **Embedding parameter count identical.** Assert *equality*, not approximate
  equality: vocab is 4098 in both arms, so the embedding matrix shape must be
  unchanged. Any inequality ⇒ VOID, because the contrast would then vary model
  size as well as tokenizer.
- **Total trainable parameter count identical.** Same assertion, same
  consequence.
- **Eval set membership identical** — the same 150 prompts, same atom specs,
  compared by hash.

**Not fully fillable from existing evidence — recorded, not papered over.**

The R8-shaped requirement here is that the swap changes *how the prompt is
encoded* without changing *what the eval is asking*. Length and overflow are
measured; **semantic preservation is not**, and I cannot state a defensible
numeric bound for it from the current record. A round-trip decode check
(`decode(encode(p)) == p` on all 150 prompts) is the obvious candidate and is
cheap, but it bounds lossless-ness of the tokenizer, not equivalence of the
task — the same gap that let arm B's format-feasibility gate pass at ~96% while
the task collapsed.

**Consequence under the readiness gate:** gate question 4 is answered only
partially. This experiment is **not ready to launch** until the round-trip
criterion is either adopted with a stated bound and a rationale for why it is
sufficient here, or replaced with a better one. That is a blocking finding, not
a caveat.

## 9. Confound analysis

| Competing explanation | How the design distinguishes it |
|---|---|
| 30M is genuinely below the floor | The swap holds parameters/steps/corpus/eval fixed; if the floor is real, the fitting tokenizer will not move it |
| The BPE tokenizer is simply a *better* tokenizer independent of context fit | Partially distinguished only. Compression and overflow are separable in principle but move together here — a real limitation, stated in field 18 |
| Seed noise | Multi-seed with the pre-registered `seed_spread` comparison; effect below spread ⇒ NULL, not support |
| Instrument produces nothing | Coverage is read before accuracy; zero coverage ⇒ `INVALID_NO_SIGNAL`, not a null |

The experiment separates *capability floor* from *context-fit confound* — the
single largest ambiguity in the current record, which is why it ranks first.

## 10. Outcome measures

Capability: coverage first, then the `p1_screening_eval_v1` primary metric and
per-arm verdicts. Quality: proportion of prompts fitting context (expected
150/150 vs 26/150). Compute, latency, memory: wall-clock and peak memory per
arm, plus tokens/step — the shorter sequences change step cost, so cost per
step is *not* comparable across arms and total cost is reported instead. Cost:
dollars per arm.

## 11. Decision rule

Fixed here, before any run.

- **SUPPORTED** — capability improves on the fitting tokenizer by more than the
  measured `seed_spread`, with coverage > 0 in both arms.
- **REFUTED** — the fitting tokenizer does not move capability beyond
  `seed_spread` while coverage is non-zero and the invariance checks passed. The
  floor is then not a context artifact.
- **NULL / INCONCLUSIVE** — effect present but at or below `seed_spread`, or
  arms return `NOT_SEPARATED` with adequate coverage. Underpowered, not a null.
- **Methodological failure (VOID)** — see field 12.

Post-hoc movement of these bars is bar-chasing and is forbidden.

## 12. Kill condition

Distinct from the falsifier: these say *the instrument never tested the
hypothesis*, so the endpoint is not read and the result is VOID.

- Either invariance assertion in field 8 fails (parameter counts differ).
- Coverage is zero in either arm — `INVALID_NO_SIGNAL`, which this line has
  produced before (unconstrained cells, pooled coverage 0, malformed 150/150).
- The integrity block does not show attention leakage 0.0 against the
  `is_causal=True` reference.
- Both arms return `NOT_SEPARATED` with effect below `seed_spread` — the
  native30 line has produced this twice, and it leaves "no effect" and
  "underpowered" indistinguishable.

## 13. Falsifier

The fitting tokenizer is applied, coverage is non-zero, invariance holds, and
measured capability does not move beyond seed spread. That is genuine evidence
**against** the context-fit account and for a real capability floor at 30M.

## 14. Authorization

**NOT AUTHORIZED. Exploratory vs confirmatory undeclared — pending owner.**

That classification is an owner gate (`docs/ACTIVE_NOW.md`), and it determines
which rule applies: a confirmatory evidential run needs prereg **plus**
experiment-scoped authorization; a materially costly run needs experiment-scoped
authorization. **Cheapness does not decide this** — a \$0 local run can still be
confirmatory.

## 15. Provenance

- Measurement source, **pinned by SHA**:
  `de188a2:artifacts/campaign/TOKENIZER_CONTEXT_CONFOUND.md`. Read at
  `de188a2bd5951d93c5a6839b21dcaa87001b7f3e`. That branch
  (`frontier/accelerated-research-campaign-v2`) is live and moving — it was
  `09621fe` when the spec's §0 audit table was written.
- Prior conclusion this qualifies:
  `de188a2:trajectory/PREREG_causalfix_wave_arm_split.md` (CONFOUND NOTICE).
- **This prereg is authored on a branch that cannot run it.**
  `nanoscribe/native/p1_eval.py` and `nanoscribe/native/tokenize.py` do not exist
  on `work/question-before-architecture`; they resolve only on
  `frontier/accelerated-research-campaign-v2`. `sft/tokenizer.json` and
  `nanoscribe/prompt.py` are present here. **Base commit is therefore TBD**
  pending a port or a decision to run from the frontier branch. Recorded as a
  provenance fact, per §0's rule that a document must not claim paths exist on a
  branch where they do not.
- To reproduce the length measurement, the recheck block in the source artifact
  runs against the frontier branch's tree.

## 16. Resource accounting

One asset swap plus a re-run of an existing wave at unchanged parameter count and
step count. Sequences shorten ~2.6×, so per-step cost falls; total cost is
expected at or below the gated wave, which is already characterised.

**Proportionality:** this resolves *the single largest ambiguity in the record*
for the cost of a re-run using an asset already in the repository. No new
tokenizer is written or vendored. Against candidate 2, which is **not ready**
under the readiness gate at all, this is the proportionate next spend — but the
field-8 gap means it is not launch-ready either.

## 17. Reproducibility

Multi-seed, matching the existing wave's seed set, with `seed_spread` computed
the same way — an effect smaller than seed spread is not a tokenizer effect.
Both arms scored by a **single analyzer over both payloads**; the E-DELIMIT
97-vs-98 discrepancy was analyzer-vs-analyzer, not run-vs-run, and cost a
session to settle. Integrity block required per arm.

## 18. Interpretation boundary

**Would establish:** whether the measured capability floor for 30M at 1800 steps
survives when the prompts fit the context.

**Would not establish:** that 30M is or is not "enough" in general — only at this
step count, corpus and eval. It would not separate *context fit* from *tokenizer
quality*, which move together in this design (field 9); a positive result licenses
"a fitting tokenizer moves the floor", not "context length was the mechanism".
It says nothing about the span-port line, whose questions are independent, and
nothing about any `NEURAL_CANDIDATES` mechanism.

**Remains untested regardless:** every architectural hypothesis. This is an
instrument-repair experiment. Its value is that architectural claims at this
scale are uninterpretable until it lands.

---

## Status

**Registered, not resolved. NOT AUTHORIZED, and NOT READY** — field 8's
invariance requirement is incompletely specified and the readiness gate treats an
unanswered question as a stop. Append a RESULT section below without editing
anything above it. The result closes in exactly one verdict: SUPPORTED /
REFUTED / NULL-INCONCLUSIVE / VOID / PENDING / NOT-AUTHORIZED.
