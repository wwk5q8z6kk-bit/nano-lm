# native30 revalidation wave 1 — FALSE NULL, root cause found

**Diagnosed 2026-08-25.** Wave completed 2026-08-24 ~02:08 (PID 26035, MPS, 9/9 imported).
Code read via `git show frontier/accelerated-research-campaign-v2:<path>` (no branch
switch). Model probes run from an extracted copy in a scratchpad against the real
checkpoints. Nothing in the shared working tree was modified.

## Verdict

`native30_revalidation_summary_v1.json` reports all six arm×mode cells `NOT_SEPARATED`,
effect `+0.0000`, pooled `0/450`. **This is not a null result. The models were trained on
an objective that excluded the labels entirely.** Do not bank these verdicts, and do not
retire span_port or evidence_bottleneck on them.

## Root cause 0 — the decoder was not causal (found 2026-08-25, deepest cause)

`nanoscribe/native/model.py` `Block.forward` called

```python
attn_out, _ = self.attn(h, h, h, need_weights=False)
```

`nn.MultiheadAttention` with **no `attn_mask` and no `is_causal`** — full
bidirectional self-attention in a decoder trained on next-token prediction.
Every position could attend to its own label, so the training objective was
solvable by copying the future.

Measured on the shipped 30M config before the fix: changing tokens at positions
6-7 moved logits at positions 0-5 by up to **20.1**, and appending three tokens
moved positions 0-7 by **25.0**. After the fix both are exactly **0.0** (a
residual 1.2e-4 on the length probe is float32 kernel numerics, not information
flow — appending *different* content gives delta 0.0).

This is why loss collapsed toward zero regardless of the truncation bug below,
and why free-running generation — where no future exists to copy — emitted
degenerate output. **The truncation fix alone would not have fixed training:**
with leakage intact the model would simply have read the restored targets
instead of learning them.

After both fixes, a 40-step CPU run on the real corpus descends normally
(lm 40.6 -> 22.4) instead of collapsing to ~0.002.

## Root cause 1 — the target was truncated out of the loss

`nanoscribe/native/losses.py:52-57`

```python
prompt_ids = hash_tokens(prompt, vocab)
target_ids = hash_tokens(target, vocab)
seq = (prompt_ids + target_ids)[: cfg.max_seq]   # cfg.max_seq = 512
input_ids.append(seq[:-1]); labels.append(seq[1:])
```

`hash_tokens` is character-level (one token per character). Measured over the actual
training corpus `artifacts/campaign/native_corpus_screen_v1_train.json` (n=19,194):

| quantity | value |
|---|---|
| prompt chars | min **519**, median 581, mean 583, max 642 |
| target chars | min 13, median 35, max 54 |
| `max_seq` | **512** |
| **targets fully truncated** | **19,194 / 19,194 = 100.0%** |
| targets partially truncated | 0 |
| targets intact | 0 |

Every prompt already exceeds `max_seq` before the target is appended. The slice therefore
keeps only the first 512 characters of the prompt and **discards the target completely for
100% of examples**. The model never saw a single target token.

There is also **no prompt masking**: `labels = seq[1:]` with `ignore_index=-100` applied
only to padding. So the training objective was plain next-character LM over truncated,
highly templated transcript text.

**This is what `final_loss` measured.** The reported 0.0171 / 0.0395 / 0.0840 / 0.0369
(span_port s1: 0.5361) are near-zero because continuing a template is trivial. They are
not evidence of task learning, and they must not be cited as convergence.

## Why the outputs look the way they do

Probed `reval30_span_port_s1` on the real eval prompt (`p1_screening_eval_v1`, case 0,
gold `raw_value='neck'`), CPU, loaded from `latest.pt`:

- first-step argmax token id = **11**; `_char_for_token(11)` = `chr(10)` = **newline**
  (logit 55.6, clear top; id 0 ranks 482nd)
- `_autoregressive_line` appends it, `"\n" in decoded` fires, loop breaks, and the final
  `.strip()` erases it → **`''`**. This is the exact unconstrained output in all 6 runs.
- generating *past* the newline stop yields
  `'\nQus mus courcous sorcordsorce swordswords "\n- s e s sourdsource sords'`
  — prompt-vocabulary babble, i.e. the model continuing prompt text, exactly as the
  truncated objective would produce. **It is not a good answer being truncated.**
- constrained mode scores a candidate set with a model that has no notion of the target
  format, and selects `NOT_MENTIONED` every time.

## Second defect — the three arms are not distinct objectives

`nanoscribe/native/losses.py:70-73`

```python
span_port       = lm * 0.5
evidence_align  = lm * 0.25 if cfg.evidence_aware else 0.0
assertion_state = lm * 0.1  if cfg.evidence_aware else 0.0
```

These are **scalar multiples of the same `lm` value**, not independent objectives. The
arms differ only by `cfg.evidence_aware` toggling two rescalings of one identical number,
so the total loss is an affine function of `lm` in every arm. Even with the truncation
fixed, this comparison cannot isolate an architecture effect.

## Third defect — the analyzer launders it into a clean null

`scripts/analyze_revalidation.py:141`

```python
"verdict": ("PROMOTION_CANDIDATE" if beats_control and beats_major else "NOT_SEPARATED")
```

A binary with no validity precondition. Under total output collapse both flags are False,
so every arm falls through to `NOT_SEPARATED` — an inferential claim about architecture —
when the truth is that nothing scoreable was produced. In the result JSONs
`exact_gold_span_eligible`, `assertion_state_correct_eligible` and
`support_direct_exact_eligible` are all **0**, so their `rate = 0.0` is 0/0 rendered as
zero, and `coverage_rate` is **0/150**. None of that reaches the printed table.

## Fourth defect — the eval failure was unobservable

`nanoscribe/campaign/native30_revalidation.py:153`

```python
subprocess.run(cmd, cwd=root, check=False, capture_output=True, text=True)
```

`check=False` with captured output that is then discarded. Any traceback from
`evaluate_native_nano.py` is silently swallowed. (Eval also forces `--cpu` while training
ran on MPS.)

## Fixes, in dependency order

1. **Truncate the prompt, never the target.** Reserve `len(target_ids)` (plus any
   separator) before slicing, or left-truncate the prompt. Add a corpus gate asserting
   `max(len(prompt)) + max(len(target)) <= max_seq`, which would have caught this at
   build time — the corpus already carries a `gates` block.
2. **Mask prompt positions** out of the loss so `final_loss` measures target prediction,
   not template continuation. Until then no loss number from this harness is meaningful.
3. **Make the arm objectives actually distinct**, or stop describing them as separate
   architecture arms.
4. **Add an `INVALID_NO_SIGNAL` verdict** gated on `coverage_count == 0` or `eligible == 0`,
   before separation verdicts compute; surface `coverage_rate` in the printed table.
5. **Propagate eval subprocess failures** (`check=True`, or log stderr).
6. Re-run the wave only after 1, 2 and 4. ~5 h of MPS compute (9 runs × ~35 min) produced
   no usable measurement.

## Precedent

Same family as two bugs already documented in `nanoscribe/native/tokenize.py`'s own
docstrings — a historical hard `text[:64]` truncation that "silently discarded ~88% of
every span-port prompt", and `\n`/`\t`/`\r` decoding to `?` and "corrupting the turn
separators in 100% of corpus sources" — and as the C-3 `recompute_c3.py` defect in
`CLAUDE-PROGRESS.md`, where a filter was computed but never applied. Silent truncation and
unguarded analysis code are this harness's recurring failure mode.

## Reproduce

```bash
# truncation census (repo venv, any branch)
.venv/bin/python -c "
import json; d=json.load(open('artifacts/campaign/native_corpus_screen_v1_train.json'))
E=d['entries']; print(sum(1 for e in E if len(e['prompt'])>=512), '/', len(E))"

# code (not present on frontier/p1-qwen-prompt-v0)
git show frontier/accelerated-research-campaign-v2:nanoscribe/native/losses.py
git show frontier/accelerated-research-campaign-v2:nanoscribe/native/inference.py
```
