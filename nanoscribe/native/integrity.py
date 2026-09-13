"""Runtime integrity assertions for native training.

Every check here corresponds to a defect in `artifacts/DEFECT_INDEX.md` that
shipped as *passing code* and inflated a result in the favourable direction.
The standing rule from that index:

> A check that has passed once is not protection.

So these run at startup, on every run, and raise rather than warn. A run that
cannot prove its own instrument is sound does not get to produce a number.

Design notes, because two of these are subtler than they look:

`assert_causal_attention_matches_reference` is a **differential** test against
`F.scaled_dot_product_attention(is_causal=True)` — the shape nanoGPT's
`CausalSelfAttention` uses. The perturbation probe below it
(`measure_attention_leakage`) is a *derived metric*, and a derived metric can
itself be wrong; this program has already shipped one non-binding check. An
equivalence test against a known-correct reference cannot silently pass a
bidirectional block, because a bidirectional block computes a different
function, not a differently-measured one. Keep both: the reference test proves
the block is causal, the probe proves the wiring that reaches it is too.

`assert_bits_per_byte_plausible` gates on BPB rather than on raw loss. "Loss
descends 40.6 -> 22.4" is two magic numbers from a single run and transfers
across neither configs nor tokenizers nor arms. BPB has an interpretable scale
that does transfer: a uniform byte model sits at 8 bits/byte, and it is
per-BYTE, so it survives a tokenizer swap (measured: honest final BPB 6.61 on
the character tokenizer vs 5.01 on BPE, because BPE's 0.494 tokens/byte cancels
against its higher nats/token).

**Its claim has since been narrowed by measurement.** It was anchored to a
historical `loss ~= 0.002`, and
`artifacts/campaign/bpb_floor_derivation_v1.md` could not reproduce that from
the recorded defect descriptions: seven reconstructions of D1.1 / D2.2 / D3.1
land between 3.99 and 401 bpb, none within three orders of magnitude of it.
D1.1 in isolation does move loss the expected way (3.989 vs 4.718 — a 15% gain
from reading the future) but that is not a collapse. So BPB is a
**catastrophic-collapse backstop, not a leak detector**; the exact assertions
in this module are what actually catch the defects. Keeping the floor is still
right — it costs nothing and covers the failure no exact check anticipates —
but it must not be cited as evidence a run is clean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# BPB thresholds
#
# Equation:        bpb = total_nats / (total_bytes * ln 2)
# Purpose:         detect label leakage via an information-theoretic floor
# Output range:    [0, inf); a uniform byte model sits at 8.0
# Why this scale:  transfers across configs/tokenizers/arms, unlike raw loss
# Failure mode it
#   catches:       any objective solvable by reading the answer
# Calibration:     the observed causal-mask leak sat at ~0.003 bpb; the best
#                  published general text compressors reach ~0.9 bpb; a small
#                  model on highly templated text might legitimately reach
#                  ~0.1-0.2 bpb once it has memorised the template.
# Threshold:       HARD_FLOOR is set an order of magnitude below the most
#                  optimistic legitimate value and ~3x above the observed leak,
#                  so it cannot false-positive on a good model and still fires
#                  on the defect. SUSPICIOUS is advisory only.
# Recheck:         python3 -m pytest nanoscribe/test_native_integrity.py
# ---------------------------------------------------------------------------
BPB_HARD_FLOOR = 0.01
BPB_SUSPICIOUS_BELOW = 0.15
BPB_UNIFORM_BYTE_MODEL = 8.0

# Step-aware floor. Archived data points from this repository:
#
#   leaking  @1800 steps   loss 0.00823  -> 0.0119 bpb   (archived reval30 s0)
#   fixed    @1800 steps   loss 0.0593   -> 0.0856 bpb   (causalfix s0)
#   fixed    @  40 steps   loss 12.90    -> 9.4848 bpb   (a smoke run)
#   leaking  @  40 steps   loss 0.002    -> 0.0029 bpb   ** NOT REPRODUCIBLE **
#
# The flat 0.01 floor does NOT separate the first two: the leaking run at 1800
# steps sits at 0.0119, only 19% above it. On TRAINING data a converged model
# that has memorised its corpus and one reading its own labels both produce a
# near-zero loss, and BPB cannot tell them apart. Hence the step split: early,
# before memorisation is possible, a low reading has no innocent explanation.
#
# The fourth row is the one to be careful about. It is the number this floor was
# originally anchored to, and bpb_floor_derivation_v1.md could not reproduce it
# from the recorded defect descriptions — seven reconstructions of D1.1/D2.2/
# D3.1 all land 3.99-401 bpb. It is retained here as provenance, NOT as
# calibration. Do not re-derive thresholds from it without reproducing it first.
#
# The detectors are the exact assertions below (attention leakage, causal
# reference equivalence, target-in-loss, context budget). This floor is the net
# underneath them.
BPB_EARLY_STEP_LIMIT = 100
BPB_EARLY_FLOOR = 0.5


class IntegrityError(AssertionError):
    """A training run's instrument failed to prove itself sound."""


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Recorded into every run's output so the gate is auditable after the fact."""

    attention_leakage: float
    attention_matches_reference: bool
    attention_reference_max_abs_diff: float
    supervised_target_tokens: int
    sampled_examples: int
    prompt_tokens_at_cap: int
    max_seq: int
    context_budget: dict[str, Any] | None = None
    tokenizer: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention_leakage": self.attention_leakage,
            "attention_matches_reference": self.attention_matches_reference,
            "attention_reference_max_abs_diff": self.attention_reference_max_abs_diff,
            "supervised_target_tokens": self.supervised_target_tokens,
            "sampled_examples": self.sampled_examples,
            "prompt_tokens_at_cap": self.prompt_tokens_at_cap,
            "max_seq": self.max_seq,
            "context_budget": self.context_budget,
            "tokenizer": self.tokenizer,
        }


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch required for integrity checks") from exc
    return torch, F


# ---------------------------------------------------------------------------
# Defect D1.1 — missing causal mask
# ---------------------------------------------------------------------------


def reference_causal_attention(block: Any, x: Any) -> Any:
    """Recompute `block.attn` with a known-correct causal implementation.

    Mirrors `nn.MultiheadAttention`'s packed-projection math exactly, then
    delegates the attention itself to `F.scaled_dot_product_attention` with
    `is_causal=True`. Any block that is not causal computes a different
    function and will diverge here.
    """
    torch, F = _torch()
    mha = block.attn
    d = mha.embed_dim
    heads = mha.num_heads
    head_dim = d // heads
    bsz, seqlen, _ = x.shape

    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    # (B, S, D) -> (B, H, S, hd)
    q = q.view(bsz, seqlen, heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seqlen, heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seqlen, heads, head_dim).transpose(1, 2)

    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, d)
    return F.linear(attn, mha.out_proj.weight, mha.out_proj.bias)


def assert_causal_attention_matches_reference(
    model: Any, *, seqlen: int = 16, tol: float = 1e-4
) -> float:
    """Differential test of every block against the reference implementation.

    Returns the max absolute difference. Raises `IntegrityError` above `tol`.
    """
    torch, F = _torch()
    blocks = list(model.blocks)
    if not blocks:
        raise IntegrityError("model has no blocks to verify")

    d = blocks[0].attn.embed_dim
    generator = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(2, seqlen, d, generator=generator)

    worst = 0.0
    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for i, block in enumerate(blocks):
                h = block.ln1(x)
                causal = torch.triu(
                    torch.ones(seqlen, seqlen, dtype=torch.bool), diagonal=1
                )
                got, _ = block.attn(
                    h, h, h, need_weights=False, attn_mask=causal, is_causal=True
                )
                want = reference_causal_attention(block, h)
                diff = (got - want).abs().max().item()
                worst = max(worst, diff)
                if diff > tol:
                    raise IntegrityError(
                        f"block {i} attention diverges from the causal reference "
                        f"implementation by {diff:.3e} (tol {tol:.1e}). The block is "
                        f"not computing causal self-attention. See DEFECT_INDEX D1.1."
                    )
    finally:
        model.train(model_was_training)
    return worst


def measure_attention_leakage(model: Any, *, seqlen: int = 8) -> float:
    """Perturbation probe: do later positions move earlier-position logits?

    Complements the reference test. That one proves the block is causal; this
    proves the wiring around it is too — positional embeddings, residual paths
    and any evidence gate are all in scope here and none of them are in scope
    for a block-level equivalence test.
    """
    torch, _ = _torch()
    vocab = model.cfg.vocab_size
    generator = torch.Generator(device="cpu").manual_seed(0)
    base = torch.randint(1, vocab, (1, seqlen), generator=generator)

    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            ref = model(base)
            split = seqlen // 2
            perturbed = base.clone()
            # Change the second half to different tokens.
            perturbed[:, split:] = (perturbed[:, split:] + 1) % (vocab - 1) + 1
            got = model(perturbed)
            # Positions before `split` must be untouched by the change.
            delta = (ref[:, :split, :] - got[:, :split, :]).abs().max().item()
    finally:
        model.train(model_was_training)
    return float(delta)


def assert_no_attention_leakage(model: Any, *, tol: float = 1e-3) -> float:
    """Defect D1.1 gate. Leakage must be 0.0 up to float kernel numerics.

    `tol` is not slack for a partially-causal model — a bidirectional block
    moves earlier logits by O(10), four orders of magnitude above this. It
    absorbs float32 kernel non-determinism only; c98e4ad measured a 1.2e-4
    residual from exactly that.
    """
    leakage = measure_attention_leakage(model)
    if leakage > tol:
        raise IntegrityError(
            f"attention leakage {leakage:.4f} exceeds {tol:.1e}: future tokens are "
            f"moving past-position logits, so the next-token objective is solvable "
            f"by copying the answer. See DEFECT_INDEX D1.1 / c98e4ad."
        )
    return leakage


# ---------------------------------------------------------------------------
# Defect D2.1 / D2.2 — target truncated out of, or unsupervised in, the loss
# ---------------------------------------------------------------------------


def assert_target_present_in_loss(rows: Any, cfg: Any, *, sample: int = 16) -> tuple[int, int]:
    """Verify supervised target tokens actually reach the loss.

    D2.1 discarded the target for 19,194/19,194 examples by budgeting it last;
    D2.2 then supervised every prompt position so the loss could not measure
    target prediction even when the target survived. Both presented as a
    near-zero `final_loss` that looked like success.

    Returns (supervised_target_tokens, examples_checked).
    """
    from nanoscribe.native.tokenize import resolve_tokenizer

    hash_tokens, _ = resolve_tokenizer(getattr(cfg, "tokenizer", "hash"))
    sampled = list(rows)[:sample]
    if not sampled:
        raise IntegrityError("no rows sampled; cannot verify the loss contains a target")

    vocab = cfg.vocab_size
    total_supervised = 0
    for row in sampled:
        prompt = row["prompt"] if isinstance(row, dict) else row.prompt
        target = row["target"] if isinstance(row, dict) else row.target
        target_ids = hash_tokens(target, vocab)[: max(1, cfg.max_seq - 1)]
        prompt_budget = cfg.max_seq - len(target_ids)
        prompt_ids = hash_tokens(prompt, vocab)[-prompt_budget:] if prompt_budget > 0 else []
        seq = prompt_ids + target_ids
        n_prompt = len(prompt_ids)
        labels = [(-100 if i + 1 < n_prompt else tok) for i, tok in enumerate(seq[1:])]
        supervised = sum(1 for lab in labels if lab != -100)
        if supervised == 0:
            raise IntegrityError(
                "an example contributes zero supervised target tokens to the loss: "
                "the objective contains no label. See DEFECT_INDEX D2.1/D2.2."
            )
        total_supervised += supervised

    return total_supervised, len(sampled)


# ---------------------------------------------------------------------------
# Defect D3.1 — tokenizer silently capping the prompt
# ---------------------------------------------------------------------------


def assert_prompt_not_silently_capped(cfg: Any, *, probe_chars: int = 600) -> int:
    """Verify post-tokenization length honours the caller's max_seq.

    D3.1 hard-truncated to 64 characters inside `hash_tokens`, defeating every
    call site's own `[: cfg.max_seq]` cap. The check is deliberately phrased as
    "the tokenizer returns what the caller asked for" rather than "the constant
    64 is gone", so it also catches a *different* constant appearing later.
    """
    from nanoscribe.native.tokenize import resolve_tokenizer

    hash_tokens, _ = resolve_tokenizer(getattr(cfg, "tokenizer", "hash"))
    probe = "a" * probe_chars
    uncapped = hash_tokens(probe, cfg.vocab_size)
    if len(uncapped) != probe_chars:
        raise IntegrityError(
            f"tokenizer returned {len(uncapped)} tokens for a {probe_chars}-char input "
            f"with no cap requested: it is silently truncating. "
            f"See DEFECT_INDEX D3.1."
        )

    capped = hash_tokens(probe, cfg.vocab_size, max_len=cfg.max_seq)
    expected = min(probe_chars, cfg.max_seq)
    if len(capped) != expected:
        raise IntegrityError(
            f"tokenizer returned {len(capped)} tokens for max_len={cfg.max_seq}; "
            f"expected {expected}. See DEFECT_INDEX D3.1."
        )
    return len(capped)


# ---------------------------------------------------------------------------
# BPB smoke gate
# ---------------------------------------------------------------------------


def bits_per_byte(total_nats: float, total_bytes: int) -> float:
    """Convert summed cross-entropy in nats to bits per byte."""
    if total_bytes <= 0:
        raise IntegrityError("cannot compute bits-per-byte over zero bytes")
    return total_nats / (total_bytes * math.log(2))


def floor_for_step(step: int) -> float:
    """The BPB floor that is actually defensible at this point in training.

    Early (`step <= BPB_EARLY_STEP_LIMIT`) the model cannot have memorised its
    corpus, so a near-zero loss has no innocent explanation and a high floor is
    justified. Later, memorisation and label leakage are indistinguishable on
    training loss, so only the absolute impossibility bound applies and the
    real detector is `assert_no_attention_leakage`.
    """
    return BPB_EARLY_FLOOR if step <= BPB_EARLY_STEP_LIMIT else BPB_HARD_FLOOR


def assert_bits_per_byte_plausible(
    bpb: float, *, step: int, floor: float | None = None
) -> None:
    """Gate on an information-theoretic floor rather than a remembered number.

    Scope, stated rather than implied: this is a **catastrophic-collapse
    backstop, not a leak detector**. Measured in
    `artifacts/campaign/bpb_floor_derivation_v1.md`: seven reconstructions of
    D1.1 / D2.2 / D3.1 all land between 3.99 and 401 bpb — none within three
    orders of magnitude of the 0.002 signature this floor was anchored to. D1.1
    in isolation does move loss the expected way (3.989 vs 4.718, a 15% gain
    from reading the future) but that is not a collapse.

    So this fires only when an objective becomes trivially solvable. The actual
    detectors are exact and separate: assert_no_attention_leakage plus the
    causal-reference differential test (D1.1), assert_target_present_in_loss
    (D2.1/D2.2), assert_prompt_not_silently_capped and assert_context_budget
    (D3.1/D3.3). BPB is the net underneath those, not the thing catching them.

    Also a *smoke-time* measure.
    See the calibration block above — at 1800 steps the known-leaking run sits
    at 0.0119 bpb against a fixed run's 0.0856, which this metric cannot
    reliably separate on training data. Do not read a late-training pass here
    as evidence of a clean run; that evidence comes from the startup gate.
    """
    if not math.isfinite(bpb):
        raise IntegrityError(f"bits-per-byte is not finite ({bpb}) at step {step}")
    effective = floor_for_step(step) if floor is None else floor
    if bpb < effective:
        ratio = BPB_UNIFORM_BYTE_MODEL / max(bpb, 1e-12)
        early = step <= BPB_EARLY_STEP_LIMIT
        why = (
            f"the model has seen each example at most once by step {step}, so "
            f"memorisation cannot explain this"
            if early
            else "this is below the absolute impossibility bound"
        )
        raise IntegrityError(
            f"bits-per-byte {bpb:.5f} at step {step} is below the floor {effective}: "
            f"that implies {ratio:,.0f}:1 compression on text, and {why}. The "
            f"objective is almost certainly readable from its own input. "
            f"See DEFECT_INDEX D1.1."
        )


# ---------------------------------------------------------------------------
# Defect D3.3 — context budget (OPEN at the corpus level; asserted here)
# ---------------------------------------------------------------------------

_QUOTED_TARGET = None


def assert_context_budget(rows: Any, cfg: Any) -> dict[str, Any]:
    """Runtime context-budget gate — measured in TOKENS, not characters.

    `corpus/validate.py::sequence_budget` already computes this at corpus BUILD
    time, but nothing checked it per-run, which is how D3.3 survived D3.1: the
    64-char cap was removed while the underlying prompt-vs-context mismatch went
    on unmeasured in every run report.

    Two differences from the build-time gate, both deliberate:

    1. **Tokens, not characters.** `sequence_budget` uses `len(prompt)`, which
       equals the token count only for a character-level tokenizer — the same
       instrument coupling as D7. Tokenizing here keeps the check correct under
       the BPE swap D3.3 proposes.
    2. **Fail on the fatal invariant, report the confound.** A lost gold span
       makes exact emission impossible by construction, so that raises. Being
       merely over budget is survivable — the trainer budgets the target first
       and truncates the prompt from the left — so it is *reported into every
       run* rather than blocking. Reporting is the point: D3.3 was invisible
       precisely because no run surfaced it.
    """
    import re

    from nanoscribe.native.tokenize import resolve_tokenizer

    hash_tokens, detokenize = resolve_tokenizer(getattr(cfg, "tokenizer", "hash"))

    global _QUOTED_TARGET
    if _QUOTED_TARGET is None:
        _QUOTED_TARGET = re.compile(r'"([^"]+)"')

    vocab, max_seq = cfg.vocab_size, cfg.max_seq
    over_budget = span_lost = 0
    lengths: list[int] = []
    lost_examples: list[str] = []

    sampled = list(rows)
    for row in sampled:
        prompt = row["prompt"] if isinstance(row, dict) else row.prompt
        target = row["target"] if isinstance(row, dict) else row.target
        atom_id = (
            row.get("atom_id", "?") if isinstance(row, dict) else getattr(row, "atom_id", "?")
        )

        p_tok = hash_tokens(prompt, vocab)
        t_tok = hash_tokens(target, vocab)
        lengths.append(len(p_tok) + len(t_tok))

        if len(t_tok) >= max_seq:
            raise IntegrityError(
                f"target alone is {len(t_tok)} tokens against max_seq={max_seq}: "
                f"it cannot fit and will be truncated out of the loss. "
                f"See DEFECT_INDEX D2.1/D3.3."
            )
        if len(p_tok) + len(t_tok) > max_seq:
            over_budget += 1

        # Does the gold span survive the trainer's left-truncation? Mirror the
        # trainer's budgeting exactly, then decode the visible window back to
        # text and look for the span.
        m = _QUOTED_TARGET.search(target)
        if m:
            budget = max_seq - min(len(t_tok), max_seq - 1)
            visible = detokenize(p_tok[-budget:], vocab) if budget > 0 else ""
            if m.group(1) not in visible:
                span_lost += 1
                if len(lost_examples) < 10:
                    lost_examples.append(atom_id)

    if span_lost:
        raise IntegrityError(
            f"{span_lost}/{len(sampled)} rows lose their gold span to prompt "
            f"left-truncation (e.g. {lost_examples[:5]}): exact emission is "
            f"impossible by construction for those rows. See DEFECT_INDEX D3.1/D3.3."
        )

    lengths.sort()
    n = len(lengths)
    return {
        "rows_checked": n,
        "n_over_budget": over_budget,
        "frac_over_budget": round(over_budget / n, 4) if n else 0.0,
        "median_prompt_plus_target_tokens": lengths[n // 2] if n else 0,
        "max_prompt_plus_target_tokens": lengths[-1] if n else 0,
        "max_seq": max_seq,
        "spans_visible": True,
        # When true the corpus is learnable ONLY because the trainer budgets the
        # target first and truncates the prompt from the left. That coupling is
        # an unstated assumption unless it is surfaced here.
        "relies_on_left_truncation": over_budget > 0,
    }


def tokenizer_fingerprint(cfg: Any) -> dict[str, Any]:
    """Identify the tokenizer that ACTUALLY ran, not the one the name claims.

    `config_for_run` previously never set `tokenizer`, so a wave launched into
    the BPE namespace would have run character-level while reporting itself as
    BPE in its own provenance. Plumbing the flag fixes the cause; this makes the
    symptom detectable after the fact, which matters because the four frozen
    provenance classes are distinguished by exactly this property.

    The probe deliberately contains `\n` and `\t`: D3.2 was a whitespace
    round-trip defect, so whitespace belongs in any tokenizer identity check.
    """
    import hashlib

    from nanoscribe.native.tokenize import resolve_tokenizer

    name = getattr(cfg, "tokenizer", "hash")
    encode, decode = resolve_tokenizer(name)
    probe = 'clinician: any pain?\n\tpatient: my throat is sore.'
    ids = encode(probe, cfg.vocab_size)
    return {
        "name": name,
        "probe_token_count": len(ids),
        "probe_id_sha256_12": hashlib.sha256(
            ",".join(map(str, ids)).encode()
        ).hexdigest()[:12],
        "roundtrip_lossless": decode(ids, cfg.vocab_size) == probe,
        "max_id": max(ids) if ids else 0,
        "vocab_size": cfg.vocab_size,
    }


def assert_requested_tokenizer_honoured(cfg: Any) -> str | None:
    """The tokenizer the launcher ASKED for must equal the one in the config.

    This is the guard the fingerprint alone cannot provide. `tokenizer_fingerprint`
    derives everything from `cfg.tokenizer`, so it faithfully reports whatever the
    config says — including when the config is wrong. An independent second source
    is required, and the launch request is it.

    The defect this exists for was real and live: `config_for_run` had three
    `NativeTrainConfig(...)` return sites and none set `tokenizer`, so a wave
    launched with `NANO_NATIVE_TOKENIZER=bpe` into `reval_results_bpe/` would have
    run character-level and recorded `tokenizer: "hash"` in its own provenance.
    Every downstream reader would have seen a BPE-namespaced, char-level result.

    Returns the requested name, or None when nothing was requested (the default
    path, where `cfg.tokenizer` is authoritative and there is nothing to compare).
    """
    import os

    requested = os.environ.get("NANO_NATIVE_TOKENIZER")
    if not requested:
        return None
    resolved = getattr(cfg, "tokenizer", "hash")
    if requested != resolved:
        raise IntegrityError(
            f"tokenizer request was NOT honoured: launcher asked for "
            f"{requested!r} but the config resolved to {resolved!r}. A run would "
            f"execute {resolved!r} while its namespace and provenance claim "
            f"{requested!r}. Check every NativeTrainConfig return site in "
            f"config_for_run. See DEFECT_INDEX D3.3 / PREREG section 8."
        )
    return requested


def assert_tokenizer_identity(cfg: Any) -> dict[str, Any]:
    """Gate: the tokenizer must round-trip and stay inside the embedding."""
    requested = assert_requested_tokenizer_honoured(cfg)
    fp = tokenizer_fingerprint(cfg)
    fp["requested"] = requested
    fp["request_honoured"] = requested is None or requested == fp["name"]
    if not fp["roundtrip_lossless"]:
        raise IntegrityError(
            f"tokenizer {fp['name']!r} does not round-trip a probe containing "
            f"newline and tab: this reintroduces D3.2 under a new name."
        )
    if fp["max_id"] >= cfg.vocab_size:
        raise IntegrityError(
            f"tokenizer {fp['name']!r} produced id {fp['max_id']} >= vocab_size "
            f"{cfg.vocab_size}: the embedding cannot represent it."
        )
    return fp


def run_startup_gate(model: Any, rows: Any, cfg: Any) -> IntegrityReport:
    """Run every defect gate. Raises `IntegrityError` on the first failure.

    Call this from the training entrypoint before the first optimizer step.
    """
    ref_diff = assert_causal_attention_matches_reference(model)
    leakage = assert_no_attention_leakage(model)
    supervised, sampled = assert_target_present_in_loss(rows, cfg)
    capped = assert_prompt_not_silently_capped(cfg)
    budget = assert_context_budget(rows, cfg)
    tokenizer = assert_tokenizer_identity(cfg)
    return IntegrityReport(
        attention_leakage=leakage,
        attention_matches_reference=True,
        attention_reference_max_abs_diff=ref_diff,
        supervised_target_tokens=supervised,
        sampled_examples=sampled,
        prompt_tokens_at_cap=capped,
        max_seq=cfg.max_seq,
        context_budget=budget,
        tokenizer=tokenizer,
    )
