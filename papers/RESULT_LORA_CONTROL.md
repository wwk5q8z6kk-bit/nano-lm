# Result — the LoRA control: pretraining, not finetune vocabulary, carries transfer

**2026-08-06.** The control specified in `RESULT_CROSSMODEL_CONTROL.md` §4.
EXPLORATORY: selects nothing, gates nothing, produces no Nano checkpoint.
Local MLX, $0. Authority record: `papers/DATASET_AUTHORITY_LORA_CONTROL.md`.

## 1. The design

Hold task training constant, vary only the starting point.

- **Nano**: 13M parameters, trained from scratch on `artifacts/nano_h5/data/fit.jsonl`.
- **Tuned 3B**: `Llama-3.2-3B-Instruct-4bit` + LoRA (8 layers, 3.47M trainable,
  0.108%) on the **same** `fit.jsonl`, same 3-way task, same output format.
  400 iters, val loss 3.674 → 0.205.
- **Base 3B**: the same model with no adaptation, zero-shot.

All three are measured on the **identical denial arms over the identical sealed
development documents**. A build-time leak guard verified no training transcript
contains any development or held-out arm phrasing.

## 2. The result

| | base 3B | **tuned 3B** | Nano 13M |
|---|---|---|---|
| in-distribution arms (mean) | 53.3% | 89.2% | **97.2%** |
| **held-out arms (mean)** | 12.2% | **90.3%** | 60.0% |
| held-out sensitivity | 90.0% | **63.3%** | 59.8% |
| DEV arm | 0.0% | 46.7% | 48.2% |
| balanced control | passes | **passes** (worst class recall 0.40) | — |

The tuned model answers **10 of 12 external phrasings at 93–100%** — wordings
absent from its finetune data. Given the *same* eight training phrasings, Nano
reaches 60.0% on those arms and the tuned 3B reaches 90.3%.

**And the two models trade places.** Nano is better in-distribution (97.2% vs
89.2%) and much worse out of it. That is the signature of memorising the
training phrasings versus generalising from prior lexical knowledge.

## 3. What this changes

`DECISION_MEMO_20260806.md` proposed **H7-V — widen Nano's finetune vocabulary**
— reasoning that Nano's failures track lexical unfamiliarity. The reasoning was
right; the remedy was aimed at the wrong stage.

**The transfer comes from pretraining, not from finetune breadth.** The tuned 3B
saw exactly the same eight denial phrasings Nano did and still generalised to
twelve unseen ones, because its base already knew what denial looks like in
English. Widening Nano's finetune set would add phrasings to memorise; it would
not supply the prior that makes unseen phrasings work.

This **partially resurrects the pretraining track retired in
`RESULT_PER_STATE_DIAGNOSIS.md` §5.** That retirement said "a bigger model
trained on 8 denial phrasings learns 8 denial phrasings." The control refutes it
directly: a *pretrained* model trained on those same 8 generalises to 12 unseen.
The retirement was correct about *scale alone* and wrong about *pretraining*,
and those were conflated.

**Revised recommendation.** The next Nano experiment should initialise from a
small **pretrained** language model rather than from scratch — the smallest base
that carries English lexical priors — and finetune on the existing task data.
That is a different and cheaper hypothesis than H7-V, and it is the one the
evidence supports. H7-V becomes secondary: worth doing, but as vocabulary
insurance rather than as the primary fix.

## 3b. Correction — the comparison is state-only, and Nano's number is joint

**The two figures do not measure the same thing.** Nano's 60.0% is *joint* —
state **and** spans exact, via `_proposal_exact`. The tuned 3B emits a bare
label and produces **no span at all**, so 90.3% is state-only. A joint metric is
strictly harder.

How much does this matter? For gold-`absent` specifically, H6's diagnostics show
that of 177 fields Nano mislabelled, **176 (99.4%) had the span exactly right**
— Nano's span is almost always correct whether or not its state is. So for this
state joint ≈ state, and the comparison is *approximately* fair. That is an
inference from the error structure, **not a measurement**, and it holds only for
`absent`; it should not be assumed for `conflicting`, whose span accuracy is
0.572.

**The deeper consequence is architectural, not numerical.** Nano's product value
is evidence-bound extraction: a state *plus* the span that grounds it. The tuned
3B did not do that job. It did the easier half. Nothing here shows a pretrained
base can produce grounded spans under Nano's contract — see §4.

## 4. What is not established

- **The tuned 3B is not a Nano candidate.** It is 230× the parameter budget and
  violates the local-first thesis. It is a control that isolates a cause, not a
  proposed product.
- **Sensitivity is reduced, not eliminated.** 63.3 points of spread remain, and
  the DEV arm still sits at 46.7%. Pretraining moves the mean a long way and
  leaves real brittleness.
- **Task training cost something.** Control accuracy fell 73.3% → 66.7% and
  `missing` recall 0.47 → 0.40. The denial gain is not free, and a longer or
  better-balanced run might or might not recover it.
- **One base model, one adapter configuration, one seed, 15 documents per arm.**
  No arm-level ordering is claimed. The comparison that carries weight is the
  aggregate held-out mean, which moved 12.2 → 90.3 — far outside anything the
  measured noise in this project could produce.
- **The intermediate point is untested.** Nothing here shows what a *small*
  pretrained base (100M–500M) achieves. That is exactly the experiment the
  revised recommendation calls for, and its result is not predictable from a 3B.

## 5. Method note worth keeping

The balanced control block ran first and passed on both models (worst class
recall 0.40 > 0.20 floor), so the arm figures are interpretable. On the base
model it also disproved the obvious objection — that model never emitted
`DENIED` at all, so its low scores were not a one-sided-scoring artifact. The
tuned model emits all three labels. Without that block, "held-out 90.3%" could
have been a model that learned to say `DENIED` and nothing else.
