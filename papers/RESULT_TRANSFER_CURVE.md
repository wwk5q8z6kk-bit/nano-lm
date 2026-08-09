# Result — the transfer curve is confounded by my own class imbalance

**2026-08-06.** Reports `papers/PREREG_TRANSFER_CURVE.md` against its frozen
bar. SCREENING. Artifact: `artifacts/nano_h6/transfer_curve/curve.json`.

**Verdict: the experiment does not answer its question.** Four of five bases are
uninterpretable, the cause is a defect in the training data I built, and the
"smallest sufficient base" remains unmeasured.

## 1. Against the frozen bar

| base | held-out | in-dist | DEV | control | worst-class recall | verdict |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 15.3% | 22.5% | 13.3% | 43.3% | **0.00** | uninterpretable |
| SmolLM2-360M | 43.6% | 41.7% | 36.7% | 46.7% | **0.00** | uninterpretable |
| Qwen2.5-0.5B | 8.3% | 50.0% | 0.0% | 50.0% | **0.00** | uninterpretable |
| **Qwen2.5-1.5B** | **100.0%** | 100.0% | 100.0% | 70.0% | 0.40 | **CLEARS 75.0%** |
| SmolLM2-1.7B | 99.7% | 97.5% | 100.0% | 50.0% | **0.00** | uninterpretable |

**One interpretable point out of five.** A single point is not a curve, and the
saturation threshold the experiment existed to find is not measured.

## 2. The guard earned its place, twice

**SmolLM2-1.7B scored 99.7% held-out and is excluded.** It never emits
`NOT_MENTIONED`. Without the balanced control block that point would have been
plotted at 99.7%, and together with 1.5B's 100% it would have drawn a clean
saturation curve with a sharp threshold near 1B.

That reading would have been wrong twice over: **SmolLM2-1.7B is larger than
Qwen2.5-1.5B and fails where the smaller model passes**, so control failure is
not monotone in size, and the "capability threshold" story dies on its own data.

## 3. The actual cause — mine

Every failure is the **same class**:

| base | `supported` recall | `missing` recall | emits `NOT_MENTIONED`? |
|---|---|---|---|
| SmolLM2-135M | 0.87 | **0.00** | never |
| SmolLM2-360M | 0.93 | **0.00** | never |
| Qwen2.5-0.5B | 1.00 | **0.00** | never |
| Qwen2.5-1.5B | 1.00 | **0.40** | 6 of 30 |
| SmolLM2-1.7B | 1.00 | **0.00** | never |

My LoRA training set is `supported` 14,420 / `absent` 4,620 / `missing`
**1,120 — 5.6%**. Four of five models collapsed on exactly the minority class.

**This is not a property of the bases. It is a property of the dataset I built**
(`build_lora_control_data.py`), which took the natural label distribution without
asking whether it could support a three-way decision. Sixth measurement defect
this cycle, and again mine.

## 4. What survives

- **Qwen2.5-1.5B (Apache-2.0) clears the preregistered bar** at 100% held-out
  with the control passing. That is one real, interpretable data point: a ~1.5B
  pretrained base, LoRA'd on Nano's own fit partition, recognises all twelve
  external denial phrasings. It does **not** establish that 1.5B is the *smallest*
  such base.
- **The bar was not moved.** 75.0% was frozen before any base was trained and is
  reported as frozen. The uninterpretable points are excluded, not re-scored.
- **The LoRA control's headline is untouched** — that comparison used
  Llama-3.2-3B, whose control block passed (worst-class recall 0.40).

## 5. The rerun that would answer the question

1. **Rebalance the training set** — cap the majority class or upsample `missing`
   so no label is under ~20%. One flag in `build_lora_control_data.py`.
2. **Rerun all five bases** on the balanced set. Same arms, same bar, same guard.
3. **If control still fails at small sizes on balanced data**, the conclusion
   flips to a genuine capability threshold — and that would then be a real
   finding rather than an artifact.

Until then the honest statement is: **the smallest sufficient base is unknown,
and one base ≥1.5B is sufficient.**

## 6. Method note

The control block has now caught four distinct failures in one cycle: an
always-`DENIED` scorer (probe v1), a guard that failed a *correct* model (v2),
a collapsed base at 135M, and a 99.7%-scoring base at 1.7B that would otherwise
have anchored a false threshold claim. Every one would have produced a
publishable-looking number. That is the argument for running the control
*first* and gating interpretation on it, rather than reporting it alongside.
