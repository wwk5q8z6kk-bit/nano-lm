"""Regression pins for the native30 target-truncation defect (2026-08-24 wave).

The wave trained nine 30M models whose loss never saw a single label: every
prompt in native_corpus_screen_v1 is 519-642 characters against max_seq=512, and
`(prompt_ids + target_ids)[: cfg.max_seq]` truncates from the right, so the
target was discarded for 100.0% of 19,194 examples. The near-zero `final_loss`
that resulted was next-character prediction on templated prompt text.

Full diagnosis: artifacts/campaign/reval_results/FALSE_NULL_DIAGNOSIS.md
"""

from __future__ import annotations

import pytest

from nanoscribe.native.tokenize import hash_tokens

VOCAB = 4098


def _encode(prompt: str, target: str, max_seq: int) -> tuple[list[int], list[int], int]:
    """Mirror of the budgeting in losses.compute_batch_loss (torch-free)."""
    target_ids = hash_tokens(target, VOCAB)[: max(1, max_seq - 1)]
    prompt_budget = max_seq - len(target_ids)
    prompt_ids = hash_tokens(prompt, VOCAB)[-prompt_budget:] if prompt_budget > 0 else []
    seq = prompt_ids + target_ids
    if len(seq) < 2:
        seq = seq + [1]
    n_prompt = len(prompt_ids)
    labels = [(-100 if i + 1 < n_prompt else tok) for i, tok in enumerate(seq[1:])]
    return seq, labels, n_prompt


def test_target_survives_an_over_length_prompt() -> None:
    """The exact shape that broke the wave: prompt alone exceeds max_seq."""
    max_seq = 512
    prompt = "P" * 581  # median prompt length in native_corpus_screen_v1
    target = "ASSERTED: sore throat"
    seq, _labels, n_prompt = _encode(prompt, target, max_seq)

    target_ids = hash_tokens(target, VOCAB)
    assert len(seq) <= max_seq
    # The whole target is present, at the tail, uncorrupted.
    assert seq[n_prompt:] == target_ids
    assert len(seq[n_prompt:]) == len(target_ids)


@pytest.mark.parametrize("prompt_len", [519, 581, 642])
def test_no_corpus_length_loses_its_target(prompt_len: int) -> None:
    """Sweep the observed corpus range (min/median/max prompt chars)."""
    target = "ASSERTED: sore throat"
    seq, _labels, n_prompt = _encode("P" * prompt_len, target, 512)
    assert seq[n_prompt:] == hash_tokens(target, VOCAB)


def test_prompt_positions_are_masked_out_of_the_loss() -> None:
    """final_loss must measure target prediction, not template continuation."""
    max_seq = 512
    target = "ASSERTED: sore throat"
    seq, labels, n_prompt = _encode("P" * 581, target, max_seq)

    supervised = [i for i, lab in enumerate(labels) if lab != -100]
    assert supervised, "at least one position must be supervised"
    # Every supervised label predicts a target token...
    for i in supervised:
        assert i + 1 >= n_prompt
        assert labels[i] == seq[i + 1]
    # ...and the boundary (last prompt token -> first target token) is supervised,
    # so the model learns where the target begins.
    assert labels[n_prompt - 1] == seq[n_prompt]
    assert len(supervised) == len(hash_tokens(target, VOCAB))


def test_short_prompt_is_untouched() -> None:
    """Budgeting must not disturb inputs that already fit."""
    prompt, target = "Transcript:\npatient: hi", "NOT_MENTIONED"
    seq, _labels, n_prompt = _encode(prompt, target, 512)
    assert seq == hash_tokens(prompt, VOCAB) + hash_tokens(target, VOCAB)
    assert n_prompt == len(hash_tokens(prompt, VOCAB))


def test_prompt_is_truncated_from_the_left() -> None:
    """The question and answer instructions live at the prompt tail; keep them."""
    prompt = "DISCARD" * 100 + "KEEPTHISTAIL"
    seq, _labels, n_prompt = _encode(prompt, "NOT_MENTIONED", 512)
    kept = seq[:n_prompt]
    assert kept == hash_tokens(prompt, VOCAB)[-n_prompt:]
    assert kept[-len(hash_tokens("KEEPTHISTAIL", VOCAB)):] == hash_tokens("KEEPTHISTAIL", VOCAB)


def test_degenerate_target_longer_than_max_seq() -> None:
    """A pathological target must not produce an empty or oversized sequence."""
    seq, labels, _n_prompt = _encode("prompt", "T" * 5000, 512)
    assert 2 <= len(seq) <= 512
    assert any(lab != -100 for lab in labels)


def test_real_compute_batch_loss_actually_depends_on_the_target() -> None:
    """Behavioural pin on the shipped function, not a mirror of it.

    Under the pre-fix `(prompt_ids + target_ids)[: max_seq]`, an over-length
    prompt truncated the target away entirely, so the loss was *mathematically
    independent* of the target: swapping the label changed nothing. This test
    fails on the old code and passes on the new.
    """
    torch = pytest.importorskip("torch")
    from nanoscribe.native.config import smoke_config
    from nanoscribe.native.losses import compute_batch_loss
    from nanoscribe.native.model import build_native_model

    cfg = smoke_config()
    build = build_native_model(cfg)
    build.model.eval()

    prompt = "P" * 581  # exceeds cfg.max_seq on its own, like the real corpus
    with torch.no_grad():
        a = compute_batch_loss(build.model, [prompt], ["ASSERTED: sore throat"], cfg)
        b = compute_batch_loss(build.model, [prompt], ["NOT_MENTIONED"], cfg)

    assert a.lm.item() != pytest.approx(b.lm.item(), abs=1e-9), (
        "loss is independent of the target — the label is being truncated away"
    )


# ---------------------------------------------------------------------------
# Objective-distinctness pins.
#
# span_port/evidence_align/assertion_state were `lm * 0.5`, `lm * 0.25`,
# `lm * 0.1` — scalar multiples of one number. Every arm's total was therefore
# an affine function of `lm`, so the three revalidation arms shared one gradient
# direction and differed only in effective learning rate. The objective factor
# was unmeasurable by construction.
# ---------------------------------------------------------------------------

_PROMPT = (
    "Transcript:\nclinician: What brings you in?\n"
    "patient: I have sore throat.\n\nQuestion: about it\n"
    "- ASSERTED: quote\n- NOT_MENTIONED\nAnswer."
)


def _loss(cfg, model, prompts, targets):
    from nanoscribe.native.losses import compute_batch_loss

    return compute_batch_loss(model, prompts, targets, cfg)


def test_components_are_not_scalar_multiples_of_lm() -> None:
    torch = pytest.importorskip("torch")
    from nanoscribe.native.config import config_for_run
    from nanoscribe.native.model import build_native_model

    cfg = config_for_run("reval30_evidence_bottleneck_s0")
    build = build_native_model(cfg)
    build.model.eval()
    with torch.no_grad():
        r = _loss(cfg, build.model, [_PROMPT] * 3,
                  ["ASSERTED: sore throat", "NOT_MENTIONED", "DENIED: fever"])
    lm = r.lm.item()
    assert r.span_port.item() != pytest.approx(lm * 0.5, abs=1e-9)
    assert r.evidence_align.item() != pytest.approx(lm * 0.25, abs=1e-9)
    assert r.assertion_state.item() != pytest.approx(lm * 0.1, abs=1e-9)


def test_span_edit_moves_span_port_but_not_assertion_state() -> None:
    """Each objective must respond to its own region."""
    torch = pytest.importorskip("torch")
    from nanoscribe.native.config import config_for_run
    from nanoscribe.native.model import build_native_model

    cfg = config_for_run("reval30_evidence_bottleneck_s0")
    build = build_native_model(cfg)
    build.model.eval()
    with torch.no_grad():
        a = _loss(cfg, build.model, [_PROMPT], ["ASSERTED: sore throat"])
        b = _loss(cfg, build.model, [_PROMPT], ["ASSERTED: fever"])
    # same label region, different span region
    assert a.assertion_state.item() == pytest.approx(b.assertion_state.item(), abs=1e-6)
    assert a.span_port.item() != pytest.approx(b.span_port.item(), abs=1e-6)


def test_evidence_align_ignores_ungrounded_spans() -> None:
    """A span absent from the visible source contributes no evidence signal."""
    torch = pytest.importorskip("torch")
    from nanoscribe.native.config import config_for_run
    from nanoscribe.native.model import build_native_model

    cfg = config_for_run("reval30_evidence_bottleneck_s0")
    build = build_native_model(cfg)
    build.model.eval()
    with torch.no_grad():
        ungrounded = _loss(cfg, build.model, [_PROMPT], ["ASSERTED: zzzznotinsource"])
        grounded = _loss(cfg, build.model, [_PROMPT], ["ASSERTED: sore throat"])
    assert ungrounded.evidence_align.item() == 0.0
    assert grounded.evidence_align.item() > 0.0


def test_control_and_span_port_arms_have_different_gradient_directions() -> None:
    """The pin that matters: under the old loss these were exactly parallel."""
    torch = pytest.importorskip("torch")
    from nanoscribe.native.config import config_for_run
    from nanoscribe.native.model import build_native_model

    cfg_ctrl = config_for_run("reval30_decoder_control_s0")
    cfg_span = config_for_run("reval30_span_port_s0")
    build = build_native_model(cfg_ctrl)
    build.model.eval()
    prompts = [_PROMPT] * 2
    targets = ["ASSERTED: sore throat", "NOT_MENTIONED"]

    def grad_for(cfg):
        build.model.zero_grad(set_to_none=True)
        _loss(cfg, build.model, prompts, targets).total.backward()
        return torch.cat([
            p.grad.reshape(-1) for p in build.model.parameters() if p.grad is not None
        ]).clone()

    g_ctrl, g_span = grad_for(cfg_ctrl), grad_for(cfg_span)
    cos = torch.nn.functional.cosine_similarity(g_ctrl, g_span, dim=0).item()
    assert cos < 0.9999, (
        f"control and span_port arms share a gradient direction (cos={cos}) — "
        "the objective factor is not being varied"
    )


# ---------------------------------------------------------------------------
# Causality pins.
#
# Block.forward called nn.MultiheadAttention with no attn_mask, i.e. full
# bidirectional attention in a decoder trained on next-token prediction. Every
# position could attend to its own label, so the objective was solvable by
# copying the future: training loss collapsed toward 0 while free-running
# generation (no future available) emitted degenerate output. Measured on the
# shipped 30M config before the fix: changing tokens at positions 6-7 moved
# logits at positions 0-5 by up to 20.1.
# ---------------------------------------------------------------------------


def _tiny_model():
    pytest.importorskip("torch")
    from nanoscribe.native.config import config_for_run
    from nanoscribe.native.model import build_native_model

    build = build_native_model(config_for_run("reval30_evidence_bottleneck_s0"))
    build.model.eval()
    return build.model


def test_future_tokens_cannot_change_earlier_logits() -> None:
    torch = pytest.importorskip("torch")
    model = _tiny_model()
    base = [5, 9, 12, 40, 77, 23, 61, 8]
    alt = base[:6] + [99, 100]  # differs only at positions 6 and 7
    with torch.no_grad():
        a = model(torch.tensor([base]))[0, :6]
        b = model(torch.tensor([alt]))[0, :6]
    assert (a - b).abs().max().item() == 0.0, "decoder is attending to the future"


def test_appended_content_cannot_change_earlier_logits() -> None:
    """Length may perturb float32 kernel numerics; content must not leak at all."""
    torch = pytest.importorskip("torch")
    model = _tiny_model()
    base = [5, 9, 12, 40, 77, 23, 61, 8]
    with torch.no_grad():
        a = model(torch.tensor([base + [7, 7, 7]]))[0, :8]
        b = model(torch.tensor([base + [200, 300, 400]]))[0, :8]
    assert (a - b).abs().max().item() == 0.0, "future content leaks backward"
