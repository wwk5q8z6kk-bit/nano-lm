"""Synthetic, standard-library-only information-transport contract checks.

NO Nano LM weights, datasets, training, evaluation or licensed external code.
This is NOT an implementation or performance test of LoopRes, ALF, ResFormer,
or the user's Nano LM. It checks indexing, causality, document resets, and the
finite-pass approximation gap for a deliberately simple scalar toy operator.

Usage: python transport_contracts.py
"""
from __future__ import annotations

import math
import unittest


def loop_sources(loop: int, block: int, blocks: int, window: int) -> list[tuple[int, int]]:
    """Previously completed loop blocks plus earlier current-loop blocks.

    (-1, -1) is an always-available prelude. The real paper also uses a
    within-block partial sum; this minimal index model does not represent it.
    """
    if not (loop >= 0 and blocks >= 1 and 0 <= block < blocks and window >= 1):
        raise ValueError("invalid loop/block/window")
    sources = [(-1, -1)]
    for old in range(max(0, loop - window), loop):
        sources.extend((old, b) for b in range(blocks))
    sources.extend((loop, b) for b in range(block))
    return sources


def weighted_sources(values: list[float], identities: list[float],
                     content_query: float = 0.0, identity_query: float = 0.0) -> list[float]:
    """Scalar structural/content-key softmax fixture, not a learned router."""
    if len(values) != len(identities) or not values:
        raise ValueError("source identity mismatch")
    logits = [content_query * v + identity_query * k
              for v, k in zip(values, identities)]
    pivot = max(logits)
    weights = [math.exp(x - pivot) for x in logits]
    return [w / sum(weights) for w in weights]


def value_mix(first: float, current: float, local_coefficient: float) -> float:
    """Numerical value mix only. Coefficients are NOT information fractions."""
    return local_coefficient * current + (1 - local_coefficient) * first


def previous_depth_sources(depth: int, layers: int, window: int) -> range:
    """Previous-position depths at/above the receiving depth, locally bounded."""
    if not (0 <= depth < layers and window >= 1):
        raise ValueError("invalid depth/window")
    return range(depth, min(layers, depth + window))


def toy_token(token: float, previous: tuple[float, ...] | None,
              layers: int, window: int) -> tuple[float, ...]:
    """A linear toy with current-layer dependence + preceding-token feedback.

    Not ALF's learned low-rank source maps, source attention, input injection or
    transformer attention. Its only purpose is to make recurrence observable.
    """
    current = 0.4 * token
    states: list[float] = []
    for depth in range(layers):
        indices = previous_depth_sources(depth, layers, window)
        feedback = (sum(previous[i] for i in indices) / len(indices)) if previous is not None else 0.0
        current = 0.55 * current + 0.25 * token + 0.18 * feedback
        states.append(current)
    return tuple(states)


def sequential(tokens: list[float], doc_starts: set[int],
               layers: int = 4, window: int = 2) -> list[tuple[float, ...]]:
    """Exact sequential evaluation of the toy recurrence; caches one token."""
    result: list[tuple[float, ...]] = []
    previous = None
    for t, token in enumerate(tokens):
        if t in doc_starts:
            previous = None
        current = toy_token(token, previous, layers, window)
        result.append(current)
        previous = current
    return result


def jacobi(tokens: list[float], doc_starts: set[int], passes: int,
           layers: int = 4, window: int = 2) -> list[tuple[float, ...]]:
    """Toy parallel Jacobi passes: pass 1 has no feedback; next passes use the
    preceding token from the *previous pass*, not from the current pass.
    This is an abstract operator, not ALF's trained model or schedule.
    """
    if passes < 1:
        raise ValueError("passes must be >= 1")
    previous_pass = [toy_token(x, None, layers, window) for x in tokens]
    for _ in range(1, passes):
        old = previous_pass
        previous_pass = [toy_token(x, None if t in doc_starts or t == 0 else old[t - 1],
                                   layers, window)
                         for t, x in enumerate(tokens)]
    return previous_pass


def max_error(x: list[tuple[float, ...]], y: list[tuple[float, ...]]) -> float:
    if len(x) != len(y):
        raise ValueError("different sequence lengths")
    return max((abs(a - b) for row_x, row_y in zip(x, y)
                for a, b in zip(row_x, row_y)), default=0.0)


class TransportContracts(unittest.TestCase):
    def test_loop_source_window_and_no_future(self):
        for window in (1, 2, 3):
            for loop in range(9):
                for block in range(4):
                    sources = loop_sources(loop, block, 4, window)
                    self.assertEqual(sources[0], (-1, -1))
                    self.assertLessEqual(len(sources), 1 + window * 4 + block)
                    self.assertTrue(all(t < loop or (t == loop and b < block)
                                        for t, b in sources if t >= 0))
                    self.assertTrue(all(t >= loop - window for t, _ in sources if t >= 0))
                    self.assertEqual(len(sources), len(set(sources)))

    def test_loop_structural_identity_vs_content_tie(self):
        same_contents = [1.0, 1.0]
        uniform = weighted_sources(same_contents, [0.0, 1.0], content_query=0.8)
        keyed = weighted_sources(same_contents, [0.0, 1.0], content_query=0.8,
                                 identity_query=1.0)
        self.assertAlmostEqual(uniform[0], uniform[1])
        self.assertNotAlmostEqual(keyed[0], keyed[1])
        self.assertAlmostEqual(sum(keyed), 1.0)

    def test_value_coefficient_not_information_attribution(self):
        self.assertEqual(value_mix(5.0, 9.0, 0.0), 5.0)
        self.assertEqual(value_mix(5.0, 9.0, 1.0), 9.0)
        self.assertAlmostEqual(value_mix(5.0, 9.0, 0.09), 5.36)
        # The scalar coefficient alone cannot identify vector norm contribution.
        self.assertAlmostEqual(value_mix(100.0, 0.0, 0.09), 91.0)
        self.assertAlmostEqual(value_mix(0.0, 100.0, 0.09), 9.0)

    def test_feedback_depth_bounds(self):
        for layers in (2, 4, 8):
            for depth in range(layers):
                src = list(previous_depth_sources(depth, layers, 3))
                self.assertLessEqual(len(src), 3)
                self.assertTrue(all(depth <= s < layers for s in src))
                self.assertEqual(len(src), len(set(src)))

    def test_sequential_prefix_is_future_invariant(self):
        a = [1.0, 2.0, -1.0, 4.0]
        b = a[:3] + [1000.0]
        self.assertEqual(sequential(a, {0})[:3], sequential(b, {0})[:3])
        self.assertEqual(jacobi(a, {0}, 2)[:3], jacobi(b, {0}, 2)[:3])

    def test_document_reset_blocks_cross_document_leak(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [100.0, -200.0, 3.0, 4.0, 5.0]
        self.assertEqual(sequential(a, {0, 2})[2:], sequential(b, {0, 2})[2:])
        self.assertEqual(jacobi(a, {0, 2}, 5)[2:], jacobi(b, {0, 2}, 5)[2:])
        self.assertGreater(max_error(sequential(a, {0})[2:],
                                     sequential(b, {0})[2:]), 0.01)

    def test_jacobi_exact_prefix_and_eventual_equivalence(self):
        tokens = [1.0, -0.7, 0.4, 2.0, -1.0, 0.5]
        exact = sequential(tokens, {0})
        for passes in range(1, len(tokens) + 1):
            approx = jacobi(tokens, {0}, passes)
            self.assertLess(max_error(approx[:passes], exact[:passes]), 1e-12)
        self.assertLess(max_error(jacobi(tokens, {0}, len(tokens)), exact), 1e-12)
        self.assertGreater(max_error(jacobi(tokens, {0}, 2), exact), 1e-5)

    def test_invalid_args_rejected(self):
        for invalid in [(-1, 0, 2, 1), (1, 2, 2, 1), (1, 0, 2, 0)]:
            with self.assertRaises(ValueError):
                loop_sources(*invalid)
        with self.assertRaises(ValueError):
            previous_depth_sources(4, 4, 1)
        with self.assertRaises(ValueError):
            jacobi([1.0], {0}, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
