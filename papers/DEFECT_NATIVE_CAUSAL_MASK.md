# DEFECT — non-causal attention in the native track

**Defect ID:** `D_NATIVE_CAUSAL_MASK`
**Found:** 2026-08-25
**Severity:** invalidating for every result produced by `nanoscribe/native/`
**Fixed:** `c98e4ad` (`fix(native): restore causal masking; make the three arm objectives distinct`)

This is a **named defect with retroactive scope**, not a bugfix line. Any result
derived from a model built by `nanoscribe/native/model.py` before `c98e4ad` is
void, regardless of how it was reported.

## The defect

`nanoscribe/native/model.py`, `Block.forward`:

```python
attn_out, _ = self.attn(h, h, h, need_weights=False)
```

`nn.MultiheadAttention` invoked with **no `attn_mask` and no `is_causal`** — full
bidirectional self-attention inside a decoder trained with next-token
prediction. Every position could attend to its own label, so the training
objective was solvable by copying the future.

## Measurement

On the shipped 30M config, before the fix:

| probe | before | after |
|---|---|---|
| change tokens at positions 6-7, measure max abs logit delta at positions 0-5 | **20.15** | **0.0** |
| append 3 tokens, measure max abs logit delta at positions 0-7 | **24.99** | 1.2e-4 (float32 kernel numerics) |
| append *different* content, same lengths, delta at positions 0-7 | — | **0.0** |

The third probe discriminates numerics from information flow: identical
lengths with different content produce exactly zero change, so the residual
1.2e-4 in probe 2 is shape-dependent float32 kernel selection, not leakage.

## Why it inflates, and in which direction

Leakage makes the training objective trivially solvable, so **training loss is
biased toward zero** while genuine capability is unmeasured. Free-running
generation — where no future exists to copy — collapses. This is an
optimism-inflating defect: it makes the model look better on the reported
number and worse in reality.

Observed signature in the 2026-08-24 wave: `final_loss` 0.017-0.084 with
100% degenerate generation.

## Affected artifacts (retroactive scope — enumerated)

Everything trained through `nanoscribe/native/` before `c98e4ad`:

**native30 revalidation wave 1** (2026-08-23/24, MPS, 9 runs, ~5 h):
`reval30_decoder_control_{s0,s1,s2}`, `reval30_evidence_bottleneck_{s0,s1,s2}`,
`reval30_span_port_{s0,s1,s2}` — checkpoints under
`artifacts/native_checkpoints/`, results under
`artifacts/campaign/reval_results/`, summary
`native30_revalidation_summary_v1.json`. **All six arm×mode verdicts void.**

**native100** (2026-08-23, 4 run dirs):
`native100_evidence_bottleneck_{s0,s1}`, `native100_span_port_{s0,s1}`.
Note `native100_evidence_bottleneck_s0` is an **incomplete run** — it wrote
step 6 at 10:51:49Z and never advanced (its sibling s1 wrote step 6 five
seconds later and reached step 200). Its only surviving trace is a
git-tracked `step_000006.json`; `artifacts/native_checkpoints/` is gitignored
(`.gitignore:62`) and three such metadata files were force-added, which is why
they reappear on branch checkout. No recoverable artifact is missing.

These runs are additionally affected by two further defects fixed in the same
sequence — see `artifacts/campaign/reval_results/FALSE_NULL_DIAGNOSIS.md`:
target truncated out of the loss for 100% of rows (`35ad570`), and the three
arm objectives being scalar multiples of one another (`c98e4ad`).

## NOT affected — boundary verified, not assumed

The Paper α ladder does **not** share this code path. Verified by direct
inspection:

| file | line | call |
|---|---|---|
| `sft/model_nano.py` | 29 | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` |
| `pretrain/train.py` | 38 | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` |
| `scale/kaggle_scale_test.py` | — | causal |

Therefore the following remain untouched by this defect: `C_GAP_EXISTS`,
`C_DIVERSITY`, `C_OWNSTACK_200M_FULLFT_GATE`, `C_ADAPT_DATA_CELLS`,
`C_INTERFERENCE`, `C_C3_TB`, `C_C3_L`, `C_POINTER_P1`, `C_POINTER_P2`,
`C_E1_MEASUREMENT`, `C_E1_GATE`, and the 3.15M/10M/159M/Pythia anchors.

**The blast radius is the native track only.** Do not read this defect as
retracting Paper α.

## Recheck

```bash
.venv/bin/python -m pytest nanoscribe/test_native_loss_target_budget.py -q
```

Pins: `test_future_tokens_cannot_change_earlier_logits`,
`test_appended_content_cannot_change_earlier_logits`. Both verified to FAIL
against the pre-fix model and pass after.

## Methods note — how it was caught

Not by any primary metric. `final_loss` reported 0.017 and the analyzer
reported six clean `NOT_SEPARATED` nulls; both were consistent with a healthy
experiment. The defect surfaced from an **adversarial self-test**: the
requirement that each objective respond only to its own region. Editing the
span moved the label loss, which is impossible under causal attention, and the
contradiction forced a direct causality probe.

Indexed as **D1.1** in `artifacts/DEFECT_INDEX.md`, the canonical record — 5 fix
threads across **12 distinct failure sites**, every one biased favourably, none
caught by a primary metric. The companion sub-defect found in the same commit,
**D1.2** (the three arm objectives being scalar multiples of one another), is
flagged there as independently wave-voiding and absent from every headline
summary. Methods argument: `papers/METHODS_ADVERSARIAL_INSTRUMENTATION.md`.
