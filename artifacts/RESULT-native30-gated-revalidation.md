# RESULT — Native30 gated revalidation wave

**Status:** complete. This is the **artifact of record** for the native30 30M screen.
**Branch:** `frontier/accelerated-research-campaign-v2`
**Wave:** PID 70762, MPS, 9 arms × 1800 steps.

Two claims are made below and they are **deliberately not merged**. Passing the
integrity gate establishes *measurement validity* — that the instrument was
sound while it produced these numbers. Agreement with the prior causalfix wave
is a *separate empirical result* about reproducibility. Neither implies the
other, and a reader who collapses them gets a false sense of confirmation.

---

## Artifact locations

| class | results | checkpoints | summary |
|---|---|---|---|
| **GATED (record)** | `artifacts/campaign/reval_results_fixed/` | `artifacts/native_checkpoints_fixed/` (18G) | `native30_revalidation_summary_fixed.json` |
| CAUSALFIX (pre-gate) | `artifacts/campaign/reval_results_causalfix/` | `artifacts/native_checkpoints_causalfix/` (18G) | `native30_revalidation_summary_causalfix.json` |
| ARCHIVED (leaking) | `artifacts/campaign/reval_results/` | `artifacts/native_checkpoints/reval30_*` (18G) | — |

All nine arms landed in every class. Archived checkpoint mtimes verified
unchanged after the gated wave (08-23/08-24; `decoder_control_s0` 08-25 01:53).

---

## Claim 1 — MEASUREMENT VALIDITY: the gate passed on all nine runs

Every training run carries an `integrity` block. No run was permitted to start
without one, and none was bypassed or relaxed.

| run | steps | attn leakage | ref max abs diff | supervised | prompt cap |
|---|---|---|---|---|---|
| `decoder_control_s0` | 1800 | **0.0** | 3.58e-07 | 305 | 512 = max_seq |
| `decoder_control_s1` | 1800 | **0.0** | 3.58e-07 | 305 | 512 |
| `decoder_control_s2` | 1800 | **0.0** | 3.58e-07 | 305 | 512 |
| `evidence_bottleneck_s0` | 1800 | **0.0** | 5.07e-07 | 305 | 512 |
| `evidence_bottleneck_s1` | 1800 | **0.0** | 4.17e-07 | 305 | 512 |
| `evidence_bottleneck_s2` | 1800 | **0.0** | 3.87e-07 | 305 | 512 |
| `span_port_s0` | 1800 | **0.0** | 2.98e-07 | 305 | 512 |
| `span_port_s1` | 1800 | **0.0** | 2.98e-07 | 305 | 512 |
| `span_port_s2` | 1800 | **0.0** | 3.58e-07 | 305 | 512 |

Attention leakage is **exactly 0.0** everywhere, and every block matched the
`F.scaled_dot_product_attention(is_causal=True)` reference to float32 precision.
D1.1, D2.1/D2.2 and D3.1 were each falsified at startup on all nine runs.

**Scope of this claim.** It says the instrument was sound. It says nothing about
whether the models learned anything.

---

## Claim 2 — EMPIRICAL: the scientific result replicated exactly

Compared against the causalfix wave, verdict by verdict:

| arm / mode | gated | causalfix | match |
|---|---|---|---|
| `evidence_bottleneck` / constrained | NOT_SEPARATED, eff +0.0133, cov 18 | NOT_SEPARATED, eff +0.0133, cov 18 | ✓ |
| `evidence_bottleneck` / unconstrained | INVALID_NO_SIGNAL, cov 0 | INVALID_NO_SIGNAL, cov 0 | ✓ |
| `span_port` / constrained | NOT_SEPARATED, eff +0.0000, cov 18 | NOT_SEPARATED, eff +0.0000, cov 18 | ✓ |
| `span_port` / unconstrained | INVALID_NO_SIGNAL, cov 0 | INVALID_NO_SIGNAL, cov 0 | ✓ |

**Verdicts, effect sizes and pooled coverage are identical.** Majority-class
baseline 0.7984; no arm approaches it. Three cells are `INVALID_NO_SIGNAL` at
zero coverage — D2.3's fix behaving correctly, refusing to emit the false
`NOT_SEPARATED` null the old fallthrough produced.

**What this does and does not license.** It shows the gate did not perturb the
measurement, and that the finding is stable across two independent runs of the
fixed code. It does **not** promote `NOT_SEPARATED` to a positive claim: with
pooled coverage 18/450 on the constrained arms and 0/450 unconstrained, this
wave is not powered to separate these arms, and "no separation detected" here is
much closer to "no signal" than to "no effect."

---

## Finding — training loss does NOT separate leaking runs from clean ones

Final training loss, all 27 runs across the three provenance classes:

| run | ARCHIVED (leaking) | CAUSALFIX (pre-gate) | GATED (record) |
|---|---|---|---|
| `decoder_control_s0` | 0.00823 | 0.05927 | 0.05379 |
| `decoder_control_s1` | 0.01713 | 0.01132 | **0.00962** |
| `decoder_control_s2` | 0.03947 | 0.04489 | 0.01924 |
| `evidence_bottleneck_s0` | 0.02227 | 0.17763 | 0.22079 |
| `evidence_bottleneck_s1` | 0.08401 | 0.08573 | 0.10522 |
| `evidence_bottleneck_s2` | 0.03691 | 0.13855 | 0.10317 |
| `span_port_s0` | 0.01061 | 0.09124 | 0.12524 |
| `span_port_s1` | 0.53607 | 0.10592 | 0.03844 |
| `span_port_s2` | 0.02776 | 0.04977 | 0.06062 |
| **median** | **0.02776** | **0.08573** | **0.06062** |

**`decoder_control_s1` in the GATED class finished at 0.00962 — lower than eight
of the nine known-leaking archived runs.** Its attention leakage is proven
exactly 0.0 and it matched the causal reference to 3.58e-07, so it is certainly
clean. A converged-loss threshold that flagged it would be wrong; one that
cleared it would also clear most of the leaking wave.

The distributions overlap heavily. The leaking class is *lower on average*
(median 0.028 vs 0.061) but its range spans 0.008–0.536 and contains only two
runs near the famous "loss ≈ 0.002" signature — which is a **40-step** number
that was retold as if it characterised converged training.

**Consequence.** Converged training loss, and therefore converged BPB, is not a
leak detector. The detector is the startup assertion, which is exact rather than
statistical. This confirms at n=27 what the two-point calibration in `11c907c`
inferred, and it is the empirical justification for the step-aware floor.

---

## Owed / caveats

- **Disk is at 99% (16–17 GiB free).** Three 18G checkpoint sets now coexist by
  design. The cheapest reclaim remains `artifacts/campaign/kaggle_native30_download/`
  (19G, re-downloadable, contains a nested `.git`); do not delete any of the
  three checkpoint classes, which are the provenance record.
- `NOT_SEPARATED` here is under-powered, not a null. The pooled coverage is
  18/450 constrained and 0/450 unconstrained.
- The unconstrained arms produce no scoreable output at all. That is upstream of
  any arm comparison and is the next thing to explain.

### Candidate explanation for the zero coverage — **inference, not measurement**

`DEFECT_INDEX.md` D3.3 (still **OPEN**) records that character-level
tokenization against a 512-token context leaves **82.7% of `p1_screening_eval_v1`
prompts over the window** (median 530 tokens, max 576) and **100% of the training
corpus** (median 615, max 694). D3.1 removed the 64-char cap but never closed
that mismatch.

If most eval prompts are still being truncated, a model that never sees the tail
of its prompt — where the question and answer instructions sit — would be
expected to emit nothing scoreable in free generation, which is exactly the
observed `coverage 0/450` on every unconstrained arm.

**This is a hypothesis, not a result.** It has not been tested here, and it is
consistent with but not established by this wave. The falsifier is cheap and
already available: D3.3 names `sft/tokenizer.json` (own-stack BPE, same vocab
4098, measured 2.60× compression, median 204 tokens, **0/150 over context**), so
re-running one unconstrained arm under that tokenizer would settle it. That
swap changes the instrument and therefore breaks comparability with all three
provenance classes above — it belongs in its own preregistered change, not as a
patch to this wave.
