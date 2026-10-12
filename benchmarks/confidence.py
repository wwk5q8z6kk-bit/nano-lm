"""Binomial confidence bounds for rate-based gates.

Why this exists
---------------
`papers/SWE_CAPABILITY_EVALUATION_PROPOSAL.md` §5.1 requires that any gate on a
rate be stated as a **confidence bound, not an observed count**: "destructive-action
violations exceed zero" is unfalsifiable at small n, because zero observed
violations in a dozen runs is consistent with a high underlying rate.
`benchmarks/gold_metrics.py::aggregate_rate` deliberately returns ``(successes, n)``
rather than a float so that both terms reach a bound. Until this module existed,
that requirement was named in three places and implemented in none.

Choice of estimator
-------------------
Wilson score interval, over the alternatives:

- **Naive** ``k/n``: no uncertainty at all. Reports 0.0 for 0/12, which is the exact
  misreading the bound exists to prevent.
- **Wald** ``p̂ ± z·sqrt(p̂(1-p̂)/n)``: degenerate at p̂ ∈ {0, 1} — the interval
  collapses to a point, which is worst precisely where safety gates live (zero
  observed violations). Also known to undercover badly at small n.
- **Clopper–Pearson**: exact and conservative, but needs an incomplete beta
  function; wider intervals mean more runs to clear the same gate, and the extra
  conservatism is not earned here.
- **Wilson**: closed form, well-behaved at the boundaries, good small-n coverage.
  Chosen as the smallest mathematically honest tool that does the job.

No gate is promoted by this module. It provides the bound; thresholds and their
justification belong to whoever proposes a gate, under §5.3's promotion criteria.
"""

from __future__ import annotations

import math

#: Two-sided 95% normal quantile. Callers wanting a one-sided 95% bound should
#: pass z=1.645 explicitly rather than relying on this default.
Z_95 = 1.96


def _validate(successes: int, n: int) -> None:
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if successes < 0:
        raise ValueError(f"successes must be >= 0, got {successes}")
    if successes > n:
        raise ValueError(f"successes ({successes}) cannot exceed n ({n})")


def _wilson(successes: int, n: int, z: float, *, upper: bool) -> float | None:
    """Wilson score bound.

    Equation:
        p̂      = k / n
        centre = p̂ + z²/(2n)
        margin = z · sqrt( (p̂(1-p̂) + z²/(4n)) / n )
        bound  = (centre ± margin) / (1 + z²/n)

    Output range: [0, 1]. Returns None when n = 0 — a bound over no observations
    is undefined, not zero and not one.
    """
    _validate(successes, n)
    if n == 0:
        return None
    p_hat = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = p_hat + z2 / (2 * n)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4 * n)) / n)
    value = (centre + margin if upper else centre - margin) / denominator
    return min(1.0, max(0.0, value))


def wilson_upper(successes: int, n: int, z: float = Z_95) -> float | None:
    """Upper confidence bound on the underlying rate.

    Use when deciding whether a *failure* rate is reliably low enough — safety
    violations, regression escapes, scope breaches. The observed count is not the
    claim; this is.

    Interpretation: ``wilson_upper(0, 12) ≈ 0.24``. Twelve clean runs are
    consistent with a true violation rate of nearly one in four. Driving the bound
    below a strict ε is what forces an adequate sample size, which is the entire
    point of stating the gate this way.
    """
    return _wilson(successes, n, z, upper=True)


def wilson_lower(successes: int, n: int, z: float = Z_95) -> float | None:
    """Lower confidence bound on the underlying rate.

    Use when deciding whether a *success* rate is reliably high enough — the
    mirror case, e.g. a verification-attempted floor.
    """
    return _wilson(successes, n, z, upper=False)


def failure_gate_passes(
    successes: int, n: int, *, epsilon: float, n_min: int = 0, z: float = Z_95
) -> bool | None:
    """Does the observed failure count clear a rate gate at ε?

    Equation: ``wilson_upper(k, n, z) ≤ ε`` and ``n ≥ n_min``.
    Output: bool, or None when n = 0 or n < n_min — an unmet data-sufficiency
    requirement means the gate is **undecided**, which is a different answer from
    "failed" and must not be silently read as "passed".

    ``n_min`` is separate from ε because the two encode different requirements:
    ε is how safe is safe enough, n_min is how much evidence is enough before the
    question is even asked (`rules/math-toolkit.md` §7, data_sufficiency).
    """
    _validate(successes, n)
    if n == 0 or n < n_min:
        return None
    bound = wilson_upper(successes, n, z)
    assert bound is not None  # n > 0 checked above
    return bound <= epsilon


def runs_needed_for_zero_observed(epsilon: float, z: float = Z_95) -> int:
    """Smallest n such that ``wilson_upper(0, n) ≤ ε``.

    Answers the planning question a gate implies: how many clean runs are needed
    before "no violations observed" supports a claim at ε? For ε = 0.01 this is in
    the hundreds — a fact worth knowing before proposing the gate rather than
    after collecting twelve runs.

    Closed form for k = 0: ``wilson_upper(0, n) = z²/(n + z²)``, so
    ``n = ceil(z²(1 - ε)/ε)``. Verified against `wilson_upper` in the tests rather
    than trusted as algebra.
    """
    if not 0 < epsilon < 1:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    return math.ceil(z * z * (1.0 - epsilon) / epsilon)
