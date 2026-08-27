# PREREG — D3.3 context/tokenizer wave

**Status:** written before any run. **Nothing has launched.** Requires owner
authorization before execution.
**Branch:** to be cut from `frontier/accelerated-research-campaign-v2`.
**Namespace:** `reval30_*_bpe_*` / `artifacts/native_checkpoints_bpe/` /
`artifacts/campaign/reval_results_bpe/`.

---

## 1. Round question

Does the 0/450 unconstrained coverage in the gated revalidation wave come from
**context truncation**, or from the models genuinely having nothing to emit?

## 2. Why this is a separate wave and not a patch

The proposed change swaps the tokenizer. That changes the measurement
instrument, not a hyperparameter. Every existing result was produced through a
character-level tokenizer against a 512-token context; a BPE tokenizer changes
what the model can see, how many tokens a prompt costs, and therefore what any
number means.

**Explicit comparison boundary.** Results from this wave are **not comparable**
to any of the four existing provenance classes:

| class | instrument | comparable to this wave? |
|---|---|---|
| ARCHIVED (leaking, local) | char-level, 512 | **no** |
| KAGGLE replicate (leaking) | char-level, 512 | **no** |
| CAUSALFIX (pre-gate) | char-level, 512 | **no** |
| GATED (record) | char-level, 512 | **no** |

Those four stay frozen. This wave does **not** retroactively reinterpret them,
and the completed revalidation must not be described as having tested this
hypothesis — it did not. The only claim this wave can license is about the
instrument it uses.

## 3. The change, and only it

Swap `hash_tokens` for the existing own-stack BPE asset `sft/tokenizer.json`
(vocab 4098 — unchanged, so embedding shapes are untouched).

Recorded properties of that asset, from D3.3:

- measured compression **2.60×**
- median prompt **204 tokens** (vs 530 char-level)
- **0/150** eval prompts over the 512 context (vs 82.7% over)

Nothing else changes: same corpus, same nine run_ids, same 1800 steps, same
arms, same seeds, same integrity gate.

## 4. Hypothesis and decision rule (pre-committed)

**H-context.** The unconstrained zero coverage is caused by prompt truncation.

Primary endpoint: **pooled unconstrained coverage**, out of 450.

- **CONFIRMED** if pooled unconstrained coverage rises to **≥ 45/450 (10%)**.
- **REFUTED** if it stays **≤ 4/450**, i.e. indistinguishable from the current 0.
- **UNRESOLVED** in between — report as such and do not round toward a verdict.

Secondary, reported but **not** verdict-bearing: constrained coverage (currently
18/450), the arm contrasts, and final BPB.

**Predicted direction, stated up front:** CONFIRMED. If 82.7% of prompts lose
their tail — where the question and answer instructions sit — a model with
nothing to condition on is expected to emit nothing scoreable. Recording the
prediction so a null is informative rather than deniable.

## 5. What a CONFIRMED result would and would not establish

It would establish that the **prior waves' unconstrained arms were
uninterpretable** — measuring truncation, not capability. It would **not**
retro-fit a capability claim onto any archived class, and it would **not** turn
the constrained `NOT_SEPARATED` cells into a positive finding; those remain
under-powered at 18/450 regardless of tokenizer.

## 6. Power, stated honestly

At n=450 pooled, the 10% CONFIRMED threshold is comfortably detectable; the
REFUTED band at ≤4/450 is likewise cleanly separated from it. The **arm
contrasts are not powered** at this n and no arm-level verdict may be drawn from
this wave, whatever the coverage does. If coverage rises, the correct follow-up
is a properly powered arm comparison **under the new instrument**, as its own
round.

## 7. Blocking pre-run checks

All must pass before launch; failure voids the wave rather than being worked
around.

1. **Context-fit assertion (the D3.3 gate).** A new startup assertion:
   post-tokenization prompt length ≤ `max_seq` for **every** row, not merely for
   a sample. This is the assertion whose absence let D3.3 persist after D3.1
   removed the 64-char cap.
2. **The existing gate stays on, unmodified** — attention leakage exactly 0.0,
   causal-reference equivalence, target present in the loss.
3. **BPB numerator must use the token count.** D7 corrected `lm * target_chars`
   to `lm * n_supervised`; under a 2.60× tokenizer the old form inflates BPB by
   ~2.6× and moves the measurement away from the floor. Pinned by
   `test_bpb_uses_the_token_count_not_the_char_count`. **The BPB floor must be
   re-derived for this instrument before it can gate anything** — the current
   values were calibrated on char-level runs.
4. **Round-trip fidelity.** The BPE tokenizer must decode losslessly over
   multi-line corpus sources, pinned as D3.2's fix is. A tokenizer that mangles
   turn separators reintroduces D3.2 under a new name.

## 8. Voiding conditions

- Any startup assertion or the BPB gate is bypassed, relaxed, or edited to fit a
  result.
- Results are written into any existing namespace, or any of the four frozen
  classes is modified.
- The BPB floor is carried over from the char-level calibration without
  re-derivation (item 7.3).
- Fewer than all nine arms complete.

## 9. Cost and authorization

Local MPS, ~6h by analogy with the gated wave, zero marginal cost — inside
`local_zero_cost_exploratory_training = ALLOWED`. **Disk is the binding
constraint, not compute:** ~17 GiB free against ~18 GiB of new checkpoints.
Space must be reclaimed before launch. No PHI; corpus unchanged and synthetic.

## 10. Recheck

```
python3 -m pytest nanoscribe/test_native_integrity.py -q
python3 -c "import json;d=json.load(open('artifacts/campaign/reval_results_bpe/reval30_decoder_control_s0_train.json'));print(d['integrity'],d['bits_per_byte'])"
```
