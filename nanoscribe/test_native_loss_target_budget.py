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
