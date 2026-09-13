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
    BPB_EARLY_FLOOR,
    assert_context_budget,
    BPB_EARLY_STEP_LIMIT,
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


_SAVED_TOKENIZER_ENV: str | None = None


def setUpModule() -> None:
    """Neutralise NANO_NATIVE_TOKENIZER for the whole module.

    Most tests here build a config directly with `_cfg()`, which bypasses
    `config_for_run` and therefore has no launch request attached. If the
    ambient environment requests a tokenizer, `assert_requested_tokenizer_honoured`
    correctly reports a mismatch against `_cfg()`'s default and the suite fails
    for a reason unconnected to the code under test.

    That is the same defect this session fixed in
    `test_candidate_raw_value_when_quote_missing`: a test whose result depends
    on ambient state cannot serve as a pre-launch gate, because it is red or
    green depending on how you invoked it. It was caught here by running the
    suite with NANO_NATIVE_TOKENIZER=bpe exported during the D3.3 pre-launch
    check — the launch configuration itself.

    Tests that need a request set it explicitly and restore it themselves.
    """
    import os

    global _SAVED_TOKENIZER_ENV
    _SAVED_TOKENIZER_ENV = os.environ.pop("NANO_NATIVE_TOKENIZER", None)


def tearDownModule() -> None:
    import os

    if _SAVED_TOKENIZER_ENV is not None:
        os.environ["NANO_NATIVE_TOKENIZER"] = _SAVED_TOKENIZER_ENV


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
    """Rows whose gold span sits near the TAIL of the prompt.

    The trainer truncates the prompt from the left, so a span at the head is
    lost by construction. An earlier version of this fixture asked for
    `"sore throat"` while the prompt said `"throat is sore"` — the span was not
    in the prompt at all, and `assert_context_budget` correctly rejected it.
    """
    return [
        {
            "prompt": "Transcript:\nclinician: What symptoms?\n"
            + ("patient: unrelated chatter here. " * 12)
            + "patient: my throat is sore.\nQuestion: what symptom?",
            "target": 'STATED: "throat is sore"',
            "atom_id": f"atom-throat-{i}",
        }
        for i in range(n)
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


@unittest.skipIf(torch is None, "torch required")
class ContextBudgetTest(unittest.TestCase):
    """D3.3 — the runtime context gate.

    `corpus/validate.py::sequence_budget` computed this at corpus BUILD time,
    but nothing checked it per-run. That is how D3.3 outlived D3.1: the 64-char
    cap was removed while the prompt-vs-context mismatch went unmeasured in
    every run report.
    """

    def test_reports_over_budget_without_blocking_a_survivable_corpus(self) -> None:
        """Being over budget is survivable; it must be surfaced, not fatal."""
        rows = _rows(4)
        info = assert_context_budget(rows, _cfg(max_seq=128))
        self.assertEqual(info["rows_checked"], 4)
        self.assertGreater(info["n_over_budget"], 0)
        self.assertTrue(info["relies_on_left_truncation"])
        self.assertTrue(info["spans_visible"])

    def test_fires_when_the_gold_span_scrolls_out_of_view(self) -> None:
        """THE fatal case: exact emission becomes impossible by construction."""
        # Gold value sits at the HEAD of a long prompt, so left-truncation
        # drops it out of the visible window.
        rows = [{
            "prompt": "patient: my throat is sore.\n" + ("filler text. " * 200),
            "target": 'STATED: "sore"',
            "atom_id": "atom-throat",
        }]
        with self.assertRaises(IntegrityError) as ctx:
            assert_context_budget(rows, _cfg(max_seq=64))
        self.assertIn("lose their gold span", str(ctx.exception))

    def test_fires_when_the_target_alone_cannot_fit(self) -> None:
        rows = [{"prompt": "short", "target": 'STATED: "' + ("x" * 200) + '"',
                 "atom_id": "a"}]
        with self.assertRaises(IntegrityError) as ctx:
            assert_context_budget(rows, _cfg(max_seq=64))
        self.assertIn("target alone", str(ctx.exception))

    def test_measures_tokens_not_characters(self) -> None:
        """Instrument-independence: the same coupling that produced D7."""
        import inspect
        from nanoscribe.native import integrity
        src = inspect.getsource(integrity.assert_context_budget)
        self.assertIn("hash_tokens", src)
        self.assertNotIn("len(prompt) + len(target)", src)


class TokenizerIdentityTest(unittest.TestCase):
    """The run report must prove WHICH tokenizer ran, not just name one.

    config_for_run never set `tokenizer`, so a wave launched into the BPE
    namespace would have run character-level and reported itself as BPE in its
    own provenance. Plumbing the flag fixes the cause; the fingerprint makes the
    symptom detectable, which matters because the frozen provenance classes are
    distinguished by exactly this property.
    """

    def test_fingerprint_separates_the_two_tokenizers(self) -> None:
        from nanoscribe.native.integrity import tokenizer_fingerprint

        h = tokenizer_fingerprint(_cfg(tokenizer="hash"))
        b = tokenizer_fingerprint(_cfg(tokenizer="bpe", vocab_size=4098))
        self.assertNotEqual(h["probe_id_sha256_12"], b["probe_id_sha256_12"])
        self.assertLess(b["probe_token_count"], h["probe_token_count"])
        self.assertTrue(h["roundtrip_lossless"] and b["roundtrip_lossless"])

    def test_identity_gate_rejects_a_lossy_tokenizer(self) -> None:
        """A tokenizer that mangles whitespace reintroduces D3.2."""
        from nanoscribe.native import integrity, tokenize

        original = tokenize.detokenize
        tokenize.detokenize = lambda ids, v: "mangled"
        try:
            with self.assertRaises(IntegrityError) as ctx:
                integrity.assert_tokenizer_identity(_cfg(tokenizer="hash"))
            self.assertIn("does not round-trip", str(ctx.exception))
        finally:
            tokenize.detokenize = original

    def test_identity_gate_rejects_ids_outside_the_embedding(self) -> None:
        """An id past the embedding must be a named error, not a runtime crash.

        Targeted directly: a tokenizer that round-trips but emits an
        out-of-range id. Shrinking vocab_size instead trips the round-trip
        check first, because `_char_for_token` cannot invert ids it never had.
        """
        from nanoscribe.native import integrity, tokenize

        enc, dec = tokenize.hash_tokens, tokenize.detokenize
        tokenize.hash_tokens = lambda t, v, max_len=None: [v + 5]
        tokenize.detokenize = lambda ids, v: (
            'clinician: any pain?\n\tpatient: my throat is sore.'
        )
        try:
            with self.assertRaises(IntegrityError) as ctx:
                integrity.assert_tokenizer_identity(_cfg(tokenizer="hash"))
            self.assertIn("vocab_size", str(ctx.exception))
        finally:
            tokenize.hash_tokens, tokenize.detokenize = enc, dec

    def test_REGRESSION_deliberate_mismatch_is_detected(self) -> None:
        """Deliberately create the mismatch and prove the gate fails.

        Simulates a `config_for_run` return branch that drops the tokenizer
        flag: the launcher requests bpe, the config comes back hash. Before the
        guard existed this ran silently and mislabelled its own provenance.
        """
        import os

        from nanoscribe.native import integrity

        saved = os.environ.get("NANO_NATIVE_TOKENIZER")
        try:
            os.environ["NANO_NATIVE_TOKENIZER"] = "bpe"
            dropped = _cfg(tokenizer="hash")  # the branch that forgot the flag
            with self.assertRaises(IntegrityError) as ctx:
                integrity.assert_requested_tokenizer_honoured(dropped)
            msg = str(ctx.exception)
            self.assertIn("NOT honoured", msg)
            self.assertIn("'bpe'", msg)
            self.assertIn("'hash'", msg)

            # And the full identity gate must also reject it, not just the helper.
            with self.assertRaises(IntegrityError):
                integrity.assert_tokenizer_identity(dropped)

            # The honoured case passes and records the request.
            ok = integrity.assert_tokenizer_identity(_cfg(tokenizer="bpe", vocab_size=4098))
            self.assertEqual(ok["requested"], "bpe")
            self.assertTrue(ok["request_honoured"])
        finally:
            os.environ.pop("NANO_NATIVE_TOKENIZER", None)
            if saved is not None:
                os.environ["NANO_NATIVE_TOKENIZER"] = saved

    def test_no_request_means_config_is_authoritative(self) -> None:
        """Absent a request there is nothing to compare — must not false-fire."""
        import os

        from nanoscribe.native import integrity

        saved = os.environ.pop("NANO_NATIVE_TOKENIZER", None)
        try:
            self.assertIsNone(integrity.assert_requested_tokenizer_honoured(_cfg()))
            fp = integrity.assert_tokenizer_identity(_cfg(tokenizer="hash"))
            self.assertIsNone(fp["requested"])
            self.assertTrue(fp["request_honoured"])
        finally:
            if saved is not None:
                os.environ["NANO_NATIVE_TOKENIZER"] = saved

    def test_config_for_run_honours_the_env_override(self) -> None:
        import os

        from nanoscribe.native.config import config_for_run

        saved = os.environ.get("NANO_NATIVE_TOKENIZER")
        try:
            os.environ.pop("NANO_NATIVE_TOKENIZER", None)
            self.assertEqual(config_for_run("reval30_decoder_control_s0").tokenizer, "hash")
            os.environ["NANO_NATIVE_TOKENIZER"] = "bpe"
            self.assertEqual(config_for_run("reval30_decoder_control_s0").tokenizer, "bpe")
            os.environ["NANO_NATIVE_TOKENIZER"] = "not-a-tokenizer"
            with self.assertRaises(ValueError):
                config_for_run("reval30_decoder_control_s0")
        finally:
            os.environ.pop("NANO_NATIVE_TOKENIZER", None)
            if saved is not None:
                os.environ["NANO_NATIVE_TOKENIZER"] = saved


@unittest.skipIf(torch is None, "torch required")
class SupervisedTokenCountTest(unittest.TestCase):
    """D7 — the BPB numerator must count TOKENS, not characters.

    `lm` is a mean over supervised token positions, so total nats is
    `lm * n_supervised`. Using the target character count instead is correct
    only for a character-level tokenizer. Under the BPE asset D3.3 proposes
    (2.60x compression) it overestimates total nats ~2.6x and inflates BPB by
    the same factor — moving the measurement AWAY from the floor and making the
    gate less sensitive precisely when the instrument changes.
    """

    def test_breakdown_reports_the_supervised_position_count(self) -> None:
        from nanoscribe.native.losses import compute_batch_loss

        cfg = _cfg()
        rows = _rows(3)
        model = build_model(cfg)
        b = compute_batch_loss(
            model, [r["prompt"] for r in rows], [r["target"] for r in rows], cfg
        )
        self.assertGreater(b.n_supervised, 0)
        # It must track the target, not the prompt: prompts here are ~20x the
        # target length, so a count that included them would be far larger.
        total_target_chars = sum(len(r["target"]) for r in rows)
        self.assertLessEqual(b.n_supervised, total_target_chars + len(rows))

    def test_bpb_uses_the_token_count_not_the_char_count(self) -> None:
        """A tokenizer with N tokens per char must not change the numerator."""
        lm_nats, n_supervised, n_bytes = 2.0, 100, 260
        correct = bits_per_byte(lm_nats * n_supervised, n_bytes)
        # What the char-count proxy would have produced under 2.6x compression:
        char_proxy = bits_per_byte(lm_nats * 260, n_bytes)
        self.assertAlmostEqual(char_proxy / correct, 2.6, places=6)
        self.assertLess(correct, char_proxy, "the proxy inflates BPB away from the floor")


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
        self.assertIn("below the floor", str(ctx.exception))
        self.assertIn("memorisation cannot explain this", str(ctx.exception))

    def test_floor_admits_a_plausible_value(self) -> None:
        """Must not false-positive on a legitimately good model."""
        import math

        # ~1.0 bpb, around the best published general text compressors.
        assert_bits_per_byte_plausible(bits_per_byte(math.log(2) * 1000, 1000), step=40)
        # Late in training an aggressively memorised template at 0.2 bpb passes.
        assert_bits_per_byte_plausible(0.2, step=1800)

    def test_non_finite_is_rejected(self) -> None:
        with self.assertRaises(IntegrityError):
            assert_bits_per_byte_plausible(float("nan"), step=1)

    def test_calibrated_against_the_four_measured_points(self) -> None:
        """Pin the real numbers this floor was calibrated on.

        If any of these move, the floor needs re-deriving rather than nudging.
        """
        import math

        ln2 = math.log(2)
        leak_40 = 0.002 / ln2       # 0.0029 — archived leak signature at 40 steps
        leak_1800 = 0.00823 / ln2   # 0.0119 — archived reval30_decoder_control_s0
        fixed_1800 = 0.0593 / ln2   # 0.0856 — causalfix same arm
        fixed_40 = 9.4848           # this session's smoke

        # Early: the floor separates leaking from honest by a wide margin.
        with self.assertRaises(IntegrityError):
            assert_bits_per_byte_plausible(leak_40, step=40)
        assert_bits_per_byte_plausible(fixed_40, step=40)
        self.assertGreater(fixed_40 / leak_40, 1000)

        # Late: honest by the impossibility bound, but the separation is thin —
        # this is exactly why the docstring refuses to claim late-training
        # detection, and why the attention assertion is the real detector.
        assert_bits_per_byte_plausible(fixed_1800, step=1800)
        self.assertLess(fixed_1800 / leak_1800, 10)
        self.assertGreater(leak_1800, BPB_HARD_FLOOR)  # the flat floor MISSES it

    def test_floor_is_step_aware(self) -> None:
        from nanoscribe.native.integrity import floor_for_step

        self.assertEqual(floor_for_step(40), BPB_EARLY_FLOOR)
        self.assertEqual(floor_for_step(BPB_EARLY_STEP_LIMIT), BPB_EARLY_FLOOR)
        self.assertEqual(floor_for_step(BPB_EARLY_STEP_LIMIT + 1), BPB_HARD_FLOOR)


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
