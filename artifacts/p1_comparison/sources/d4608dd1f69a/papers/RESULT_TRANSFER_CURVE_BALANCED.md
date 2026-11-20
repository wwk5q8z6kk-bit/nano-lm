# Result — the transfer curve, after repairing my class imbalance

**2026-08-06.** Rerun of `PREREG_TRANSFER_CURVE.md` on a class-balanced training
set, against the **same frozen 75.0% bar** and the **same guard**. SCREENING.
Artifact: `artifacts/nano_h6/transfer_curve_balanced/curve.json`.

**The repair worked. Interpretable points went from 1 of 5 to 4 of 5, and a real
curve appeared where the confounded run showed noise.**

## 1. The curve

| base | held-out | control | worst-class recall | interpretable | clears 75.0% |
|---|---|---|---|---|---|
| SmolLM2-135M | 78.9% | 26.7% | 0.07 | **no** | — |
| SmolLM2-360M | **7.2%** | 63.3% | 0.53 | yes | no |
| Qwen2.5-0.5B | **40.8%** | 56.7% | 0.27 | yes | no |
| **Qwen2.5-1.5B** | **99.7%** | 80.0% | 0.60 | yes | **YES** |
| **SmolLM2-1.7B** | **91.1%** | 60.0% | 0.20 | yes | **YES** |

Among interpretable points the ordering is monotone in size:
**7.2% → 40.8% → 91.1–99.7%**, with the transition between **0.5B and 1.5B**.

## 2. Answer to the preregistered question

> *What is the smallest pretrained base that recovers most of the transfer?*

**Between 0.5B and 1.5B.** Qwen2.5-0.5B reaches 40.8% — clearly insufficient
against a 75.0% bar. Qwen2.5-1.5B reaches 99.7% and SmolLM2-1.7B 91.1%; **both
clear**, from two different model families, which is the first evidence here that
the result is not one family's quirk.

Nano-13M-from-scratch scores 60.0% on these arms. **A 1.5B pretrained base,
LoRA'd on Nano's own fit partition, reaches 99.7%.** The gap is pretraining, and
it is now bracketed rather than asserted.

## 3. What the imbalance was hiding

The confounded run had SmolLM2-1.7B at 99.7% held-out with worst-class recall
**0.00** — excluded. Balanced, it posts 91.1% with recall **0.20** and is
admitted. The earlier exclusion was an artifact of my 5.6% `missing` class, not a
property of the model.

**135M swapped which class it fails.** Imbalanced: `missing` 0.00, `supported`
0.87. Balanced: `missing` 0.47, `supported` **0.07**. It cannot hold all three
labels at once under either mix, which is the honest signal that 135M is below
the task's capability floor — a conclusion the first run could not support
because every base failed the same way for the same avoidable reason.

## 4. What this still does not establish

- **One seed per base.** SCREENING, as preregistered. The 31.5-point per-arm seed
  instability measured earlier means adjacent points are not separable — 99.7%
  vs 91.1% is **not** evidence Qwen2.5-1.5B beats SmolLM2-1.7B. Both clear; that
  is all.
- **The balanced set is also smaller** — 3,360 rows against 20,160. Balance and
  size co-vary, so a shift cannot be attributed to balance alone. The recipe
  draws 1,600 samples (400 iters × batch 4), so neither set is exhausted, but the
  confound is real and unmeasured.
- **State-only, still.** Every point scores a bare label. Nothing here shows any
  base can carry Nano's span contract — that is route (b), and the span-bearing
  v2 dataset now exists to test it.
- **Nothing below 360M is interpretable**, so the true floor could sit anywhere
  between 135M and 360M and is not measured.

## 5. Consequence

`FORWARD_PLAN_20260806.md`'s recommendation — initialise from the smallest
pretrained base carrying English lexical priors — now has a bracketed target:
**≥1.5B, with 0.5B insufficient.** At 4-bit that is roughly 1 GB, which runs
locally on a laptop. The local-first thesis survives; "small" now means ~1.5B
rather than 13M, and that is a genuine change to what Nano is.

The next experiment is therefore **not** another size sweep. It is the span port
(route (b)) on Qwen2.5-1.5B with the span-bearing target, measuring
`no_match_rate` against `ambiguous_rate` — because a base that recognises
denials but cannot ground them is not Nano.

## 6. Method note

The bar was frozen before any base was trained and never moved. Both runs are
reported. The first is retained as a confounded result rather than deleted,
because its failure mode — a metric that looked like a clean threshold and was
an artifact of the reporter's own dataset — is the more instructive half.
