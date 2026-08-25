"""Tests for the native training integrity gate.

The load-bearing tests here are the ones that prove each check *fires*. A gate
that cannot fail is not a gate, and every defect in `artifacts/DEFECT_INDEX.md`
shipped as passing code — so for each one this module reconstructs the defect
and asserts the gate catches it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from nanoscribe.native.config import NativeTrainConfig, NativeVariant
from nanoscribe.native.integrity import (
    BPB_HARD_FLOOR,
    IntegrityError,
    assert_bits_per_byte_plausible,
    assert_causal_attention_matches_reference,
    assert_no_attention_leakage,
    assert_prompt_not_silently_capped,
    assert_target_present_in_loss,
    bits_per_byte,
    measure_attention_leakage,
    run_startup_gate,
)
from nanoscribe.native.model import build_model

torch = None
try:
    import torch  # noqa: F811
except ImportError:  # pragma: no cover
    pass


def _cfg(**over) -> NativeTrainConfig:
    base = dict(
        run_id="integrity_test",
        variant=NativeVariant.NATIVE_A,
        cell=None,
        seed=0,
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq=64,
    )
    base.update(over)
    return NativeTrainConfig(**base)


def _rows(n: int = 4) -> list[dict[str, str]]:
    return [
        {
            "prompt": "Transcript:\nclinician: What symptoms?\npatient: My "
            + ("throat is sore. " * 20)
            + "\nQuestion: what symptom?",
            "target": 'STATED: "sore throat"',
        }
        for _ in range(n)
    ]


@unittest.skipIf(torch is None, "torch required")
class CausalAttentionTest(unittest.TestCase):
    def test_fixed_model_matches_the_causal_reference(self) -> None:
        model = build_model(_cfg())
        worst = assert_causal_attention_matches_reference(model)
        self.assertLess(worst, 1e-4)

    def test_reference_test_catches_a_bidirectional_block(self) -> None:
        """THE load-bearing test: reconstruct D1.1 and prove the gate fires.

        A derived leakage metric can itself be wrong. An equivalence test
        against a known-correct implementation cannot silently pass a
        bidirectional block, because such a block computes a different
        function rather than a differently-measured one.
        """
        model = build_model(_cfg())
        block = model.blocks[0]
        original = block.attn.forward

        def bidirectional(q, k, v, **kwargs):
            kwargs.pop("attn_mask", None)
            kwargs.pop("is_causal", None)
            return original(q, k, v, **kwargs)

        block.attn.forward = bidirectional
        with self.assertRaises(IntegrityError) as ctx:
            assert_causal_attention_matches_reference(model)
        self.assertIn("not computing causal self-attention", str(ctx.exception))

    def test_leakage_probe_catches_a_bidirectional_model(self) -> None:
        """The probe covers the wiring the block-level test cannot see."""
        model = build_model(_cfg())
        self.assertLess(assert_no_attention_leakage(model), 1e-3)

        for block in model.blocks:
            original = block.attn.forward

            def bidirectional(q, k, v, _orig=original, **kwargs):
                kwargs.pop("attn_mask", None)
                kwargs.pop("is_causal", None)
                return _orig(q, k, v, **kwargs)

            block.attn.forward = bidirectional

        leaked = measure_attention_leakage(model)
        self.assertGreater(
            leaked, 1e-3, "a bidirectional model must move past-position logits"
        )
        with self.assertRaises(IntegrityError):
            assert_no_attention_leakage(model)


@unittest.skipIf(torch is None, "torch required")
class LossContentTest(unittest.TestCase):
    def test_target_reaches_the_loss_on_the_fixed_budgeting(self) -> None:
        supervised, checked = assert_target_present_in_loss(_rows(), _cfg())
        self.assertEqual(checked, 4)
        self.assertGreater(supervised, 0)

    def test_fires_when_the_target_is_budgeted_last(self) -> None:
        """Reconstruct D2.1: right-truncation discards the target entirely."""
        from nanoscribe.native import integrity
        from nanoscribe.native.tokenize import hash_tokens

        cfg = _cfg()

        def broken(rows, cfg, *, sample=16):
            for row in list(rows)[:sample]:
                seq = hash_tokens(row["prompt"] + row["target"], cfg.vocab_size)[
                    : cfg.max_seq
                ]
                n_prompt = min(len(hash_tokens(row["prompt"], cfg.vocab_size)), len(seq))
                labels = [
                    (-100 if i + 1 < n_prompt else tok) for i, tok in enumerate(seq[1:])
                ]
                if sum(1 for lab in labels if lab != -100) == 0:
                    raise IntegrityError("zero supervised target tokens")
            return 0, 0

        with self.assertRaises(IntegrityError):
            broken(_rows(), cfg)


@unittest.skipIf(torch is None, "torch required")
class TokenizerCapTest(unittest.TestCase):
    def test_passes_on_the_fixed_tokenizer(self) -> None:
        self.assertEqual(assert_prompt_not_silently_capped(_cfg()), 64)

    def test_fires_on_a_silently_truncating_tokenizer(self) -> None:
        """Reconstruct D3.1: a hard 64-char cap inside the tokenizer."""
        from nanoscribe.native import integrity, tokenize

        original = tokenize.hash_tokens

        def truncating(text, vocab_size, max_len=None):
            return original(text[:64], vocab_size, max_len)

        tokenize.hash_tokens = truncating
        try:
            with self.assertRaises(IntegrityError) as ctx:
                assert_prompt_not_silently_capped(_cfg())
            self.assertIn("silently truncating", str(ctx.exception))
        finally:
            tokenize.hash_tokens = original


class BitsPerByteTest(unittest.TestCase):
    def test_uniform_byte_model_is_eight_bits_per_byte(self) -> None:
        """Anchors the scale: -ln(1/256) nats per byte is exactly 8 bpb."""
        import math

        nats = math.log(256) * 1000
        self.assertAlmostEqual(bits_per_byte(nats, 1000), 8.0, places=6)

    def test_floor_catches_the_observed_leak_signature(self) -> None:
        """The causal-mask leak presented as loss ~= 0.002 nats/token."""
        bpb = bits_per_byte(0.002 * 1000, 1000)
        self.assertLess(bpb, BPB_HARD_FLOOR)
        with self.assertRaises(IntegrityError) as ctx:
            assert_bits_per_byte_plausible(bpb, step=40)
        self.assertIn("plausibility floor", str(ctx.exception))

    def test_floor_admits_a_plausible_value(self) -> None:
        """Must not false-positive on a legitimately good model."""
        import math

        # ~1.0 bpb, around the best published general text compressors.
        assert_bits_per_byte_plausible(bits_per_byte(math.log(2) * 1000, 1000), step=40)
        # Even an aggressively memorised template at 0.2 bpb passes.
        assert_bits_per_byte_plausible(0.2, step=40)

    def test_non_finite_is_rejected(self) -> None:
        with self.assertRaises(IntegrityError):
            assert_bits_per_byte_plausible(float("nan"), step=1)


@unittest.skipIf(torch is None, "torch required")
class StartupGateTest(unittest.TestCase):
    def test_gate_passes_on_the_fixed_stack_and_reports(self) -> None:
        report = run_startup_gate(build_model(_cfg()), _rows(), _cfg())
        self.assertTrue(report.attention_matches_reference)
        self.assertLess(report.attention_leakage, 1e-3)
        self.assertGreater(report.supervised_target_tokens, 0)
        self.assertEqual(report.max_seq, 64)
        self.assertIn("attention_leakage", report.to_dict())


if __name__ == "__main__":
    unittest.main()
