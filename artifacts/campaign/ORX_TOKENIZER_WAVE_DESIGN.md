# Next wave design — fit the input in the context (orx)

**Drafted 2026-08-25.** Successor to the causalfix wave
(`trajectory/PREREG_causalfix_wave_arm_split.md`), which returned *below
capability floor* confounded with residual truncation (D3.3).

## Why not nine parallel nodes

The obvious shape — nine arms as nine sibling nodes — is the **flat fan** that
`orx skill experiment-tree` names as the primary failure mode: every result is
measured against the start, so wins never accumulate. Two orx cardinal rules also
bind:

- **Rule 2/3:** the run command and environment are a fixed contract; only
  committed code may differ between nodes. So seeds cannot be swept via env vars
  or command edits.
- **Rule 4:** fan *within a round* (the options of one decision), then descend
  onto that round's winner.

**Seeds therefore live inside the node, not across nodes.** Each node runs its 3
seeds internally and reports pooled coverage plus per-seed spread. That keeps the
round-1 fan at two, and it matches the causalfix prereg, where seed spread was
part of a single result rather than a separate hypothesis — the discipline that
caught the `s0`-only artifact.

## Round 1 — one decision, two options

**Decision: how do we get the input to fit in the context?** Currently 82.7% of
eval prompts and 100% of training rows exceed `max_seq=512` under character-level
`hash_tokens`.

| node | change (committed code only) | rationale |
|---|---|---|
| **A — own-stack BPE** | swap `hash_tokens` for `sft/tokenizer.json` (existing asset, **same vocab 4098**) | measured 2.60× compression: median 204 tok, max 226, **0/150 over context**, ~60% headroom, embeddings unchanged |
| **B — wider context, same tokenizer** | keep char-level, raise `max_seq` 512 → 704 | 704 > 694 (worst train row), so nothing truncates |

**B is not filler — it is the disambiguator.** Swapping the tokenizer alone
changes two things at once: the tokenization scheme *and* how much content fits.
B holds the tokenizer fixed and buys only the fit. If A ≈ B, the win was fitting
the input and the tokenizer is incidental. If A > B, compression carries
independent value beyond fit. Without B this wave would repeat the
two-things-at-once error that voided the peer's C3 arm.

Attention cost for B is ~(704/512)² ≈ 1.9× — affordable at 30M.

Everything else is pinned to the causalfix wave: 30M params, 1800 steps, vocab
4098, same corpus, same eval suite, same 3 seeds, MPS or a single GPU type for
the whole round (**never split a round across devices** — device would be
confounded with arm).

## Round 2 — descend on the winner

Only after round 1 resolves. Candidates, in the order the current evidence
supports: **steps** (1800 is short for an honest objective now that leakage is
gone), then **scale**. Scale stays last: scaling through a truncating tokenizer
confounds scale with truncation, which is exactly what round 1 removes.

## Pre-registered decision rule (fix before any run lands)

Primary metric is **constrained coverage** — the causalfix headline, `6/150 =
4.0%` pooled. It is the metric the floor result rests on and the one with room
to move.

- **A or B wins** if its pooled coverage Wilson-95 lower bound exceeds the
  char-level/512 baseline's upper bound (baseline pooled 54/1350 → Wilson upper
  ≈ 0.052), and the direction holds in 3/3 seeds.
- **No fit effect** if neither clears it. Then the capability floor stands
  *without* the D3.3 confound, and it becomes a statement about 30M rather than
  about the tokenizer — a strictly stronger result than we have today.
- `exact_gold_span` remains **banned** as a decision input under constrained
  mode (now suppressed to `None` at source).

Both outcomes are publishable. The failure mode this rule prevents is reading a
coverage bump as a capability gain when it is a fit gain, or vice versa.

## Ceiling calibration

Qwen2.5-1.5B on the related span-port task selects the correct conversational
turn for roughly four gold slots in five and still rarely delimits the exact
span. A 30M char-level model at 4% coverage is a **different regime**, not a few
increments below. Size expectations against that gap: round 1 is about removing a
confound, not about approaching that ceiling.

## Status and the one thing that blocks launch

**Design only. Nothing created, nothing launched.**

The existing orx project (`595cff63-…`, nano-lm) holds the **span-port/Qwen**
tree on `frontier/p1-qwen-prompt-v0`, whose nodes run
`NANOSCIBE_QWEN_WEIGHTS=… python3 nanoscribe/run_eval.py --suite campaign_v2`
(the missing `R` is consistent with `qwen_inference.py:31` — cosmetic, not a
bug). The native30 code lives on `frontier/accelerated-research-campaign-v2`,
and the native track has **never been run on orx compute**. Its run command and
environment there are unestablished.

`orx skill experiment-tree` is explicit: *do not reverse-engineer or guess the
setup from the repository — ask.* So the gating item is the run command and
environment for the native track on OpenResearch compute, plus whether this
subtree belongs in the existing project or a new one, given it is a different
track from the span-port work already there.

Locally the equivalent invocation is:

```bash
NANO_NATIVE_CHECKPOINT_DIR=<fresh tree> .venv/bin/python \
  scripts/run_native30_revalidation.py --results-dir <fresh> --out <summary> --analyze
```

~35 min/run on MPS, 9 runs — but that is the *local* recipe, not a verified orx
compute recipe.
