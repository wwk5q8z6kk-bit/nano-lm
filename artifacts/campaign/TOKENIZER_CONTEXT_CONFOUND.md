# D3.3 — char-level tokenization does not fit this task in a 512 context

**Measured 2026-08-25.** Follow-up to D3.1 (`hash_tokens: text[:64]`). That entry
is recorded as fixed. **It is only partly closed**: the 64-char cap is gone, but
the underlying mismatch — a character-level tokenizer against `max_seq=512` on
prompts that are longer than 512 characters — was never addressed.

## Measurement

`hash_tokens` emits one token per character. Prompt lengths against `max_seq=512`:

| set | tokenizer | median | max | exceeds 512 |
|---|---|---|---|---|
| `p1_screening_eval_v1` (150 prompts) | char-level | 530 | 576 | **124/150 = 82.7%** |
| `native_corpus_screen_v1` train, prompt+target (n=19,194) | char-level | 615 | 694 | **19,194/19,194 = 100.0%** |

Content dropped on the eval set: median **4.4%**, worst **11.1%**.

## Consequence for the capability-floor result

`trajectory/PREREG_causalfix_wave_arm_split.md` concluded that 30M at 1800 steps
on a character-level hash tokenizer is below the capability floor for
`p1_screening_eval_v1`. That conclusion stands as written — it names the
tokenizer — but it is **confounded with residual truncation on 82.7% of eval
atoms**. It must not be restated as "30M is below the floor", because parameter
count is not the only thing varying. The honest form is:

> 30M at 1800 steps **with a character-level tokenizer that cannot fit 83% of
> eval prompts in its context** is below the floor. Whether 30M with a fitting
> tokenizer is below the floor is **untested**.

## The fix already exists in this repository

No new tokenizer needs to be written or vendored. `sft/tokenizer.json` is the
own-stack BPE used by the Paper-α ladder, at **the same vocabulary size (4098)**.
Re-measured on the identical prompts:

| set | tokenizer | median | max | exceeds 512 |
|---|---|---|---|---|
| eval | own-stack BPE (4098) | **204** | 226 | **0/150** |
| train (prompt+target) | own-stack BPE (4098) | **232** | 256 | **0/19,194** |

- **Compression: 2.60× (eval), 2.65× (train)** — measured, not the ~4× rule of thumb.
- **Overflow eliminated entirely** on both sets.
- Median prompt then occupies **39.8%** of the context, leaving ~60% headroom.

Vocabulary is unchanged at 4098, so embedding parameter count is unchanged and
the swap is not confounded with model size.

## Why this is the next variable, not scale

The context does not currently fit the input. Any parameter scaling is measured
through a truncating tokenizer, so scale and truncation move together. Fixing the
tokenizer is a larger effective change than any affordable parameter increase,
costs one asset swap, and removes a confound rather than adding one.

## Ceiling calibration

The span-port thread measured Qwen2.5-1.5B on the related task: it selects the
correct conversational turn for roughly four gold slots in five, and still
delimits the exact span only rarely. A 30M char-level model at 4% coverage is not
a few increments below that — it is a different regime. Size the next design
against that gap, not against the previous wave.

## Recheck

```bash
.venv/bin/python -c "
import statistics as st
from tokenizers import Tokenizer
from nanoscribe.native.tokenize import hash_tokens
from nanoscribe.native.p1_eval import campaign_cases
from nanoscribe.prompt import build_span_port_prompt
tok=Tokenizer.from_file('sft/tokenizer.json')
ch=[];bp=[]
for c in campaign_cases('p1_screening_eval_v1'):
    for s in c.atom_specs:
        p=build_span_port_prompt(c.model_input.source,s)
        ch.append(len(hash_tokens(p,4098))); bp.append(len(tok.encode(p).ids))
print('char over512:', sum(1 for x in ch if x>512), '/', len(ch))
print('bpe  over512:', sum(1 for x in bp if x>512), '/', len(bp))"
```

## Status

**Open.** The tokenizer swap is not implemented. This document records the
measurement and the confound only.
