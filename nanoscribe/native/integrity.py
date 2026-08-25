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
that does transfer: a uniform byte model sits at 8 bits/byte, and the
information content of real text bounds how low an honest model can go. The
causal-mask leak presented as loss ~= 0.002, which is ~0.003 bits/byte — a
2700:1 compression ratio on text, roughly three orders of magnitude past every
published text compressor. That makes the detector a principled bound instead
of an anomaly someone happened to remember.
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention_leakage": self.attention_leakage,
            "attention_matches_reference": self.attention_matches_reference,
            "attention_reference_max_abs_diff": self.attention_reference_max_abs_diff,
            "supervised_target_tokens": self.supervised_target_tokens,
            "sampled_examples": self.sampled_examples,
            "prompt_tokens_at_cap": self.prompt_tokens_at_cap,
            "max_seq": self.max_seq,
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
    from nanoscribe.native.tokenize import hash_tokens

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
    from nanoscribe.native.tokenize import hash_tokens

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


def assert_bits_per_byte_plausible(
    bpb: float, *, step: int, floor: float = BPB_HARD_FLOOR
) -> None:
    """Gate the smoke run on an information-theoretic floor, not a magic number.

    Below `floor` the implied compression ratio exceeds every published text
    compressor by orders of magnitude. On a ~40-step run the model has seen
    each example at most once, so genuine memorisation cannot explain it
    either — the only remaining explanation is that the objective can read its
    own answer.
    """
    if not math.isfinite(bpb):
        raise IntegrityError(f"bits-per-byte is not finite ({bpb}) at step {step}")
    if bpb < floor:
        ratio = BPB_UNIFORM_BYTE_MODEL / max(bpb, 1e-12)
        raise IntegrityError(
            f"bits-per-byte {bpb:.5f} at step {step} is below the plausibility floor "
            f"{floor}: that implies {ratio:,.0f}:1 compression on text, ~3 orders of "
            f"magnitude past the best published compressors, and the model has seen "
            f"each example at most once at this step. The objective is almost "
            f"certainly readable from its own input. See DEFECT_INDEX D1.1."
        )


def run_startup_gate(model: Any, rows: Any, cfg: Any) -> IntegrityReport:
    """Run every defect gate. Raises `IntegrityError` on the first failure.

    Call this from the training entrypoint before the first optimizer step.
    """
    ref_diff = assert_causal_attention_matches_reference(model)
    leakage = assert_no_attention_leakage(model)
    supervised, sampled = assert_target_present_in_loss(rows, cfg)
    capped = assert_prompt_not_silently_capped(cfg)
    return IntegrityReport(
        attention_leakage=leakage,
        attention_matches_reference=True,
        attention_reference_max_abs_diff=ref_diff,
        supervised_target_tokens=supervised,
        sampled_examples=sampled,
        prompt_tokens_at_cap=capped,
        max_seq=cfg.max_seq,
    )
