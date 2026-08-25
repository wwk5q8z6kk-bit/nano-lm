"""Replicate-count derivation for the C1xC2 span-port leakage 2x2.

Metric / Formula:  exact-McNemar power for a paired binary contrast, inverted to
                   solve for the number of eval instances K.
Purpose:           choose K BEFORE any cell runs, so the 2x2 is not run at n=1.
Inputs:            pi (per-slot discordance rate), slots_per_instance,
                   icc (within-encounter correlation), slots_per_encounter.
Output range:      K in {1, 2, ...}, or None when the contrast is unpowerable.
Interpretation:    K instances give >= `power` chance of a significant contrast.
Why this formula:  decoding is greedy and all four cells score the SAME slots, so
                   the design is fully within-item. The interaction contrast
                   c_i = (Y^11 - Y^01) - (Y^10 - Y^00) is a paired difference-of-
                   differences; under the mechanism (C1-on saturates both C2
                   cells) it collapses to a McNemar discordant-pair indicator.
                   At these sample sizes the normal approximation is not honest:
                   conditional on d discordant slots the exact null is
                   Binomial(d, 1/2), so a one-directional d needs d >= 6 for
                   p < 0.05 two-sided (d = 5 gives 0.0625).
Why not simpler:   a two-proportion z-test would treat the four cells as
                   independent samples. They are not — they are the same slots
                   rescored, and ignoring the pairing inflates the required N
                   several-fold while testing a hypothesis we are not asking.
Why not complex:   a GLMM with random encounter effects would model the
                   clustering properly, but with ~10 informative slots per
                   instance there is nowhere near enough data to fit one; the
                   Kish design effect is the honest small-sample substitute.
False positives:   treating a d >= 6 that arose from mixed-direction discordance
                   as evidence — `min_discordant_one_directional` is the floor
                   only when all discordant pairs point the same way.
False negatives:   pi over-estimated => K too small => a real effect missed.
                   This is why pi is piloted, not assumed.
Calibration:       pi measured on throwaway encounters disjoint from every
                   measurement instance (`pilot_quote_absent.py`).
Threshold:         alpha = 0.05 two-sided, power = 0.80.
Recheck command:   python3 -m pytest nanoscribe/test_power.py -q
When to retire:    if decoding ever becomes stochastic, the pairing breaks and
                   this module no longer describes the design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ALPHA = 0.05
POWER = 0.80


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))


def min_discordant_one_directional(alpha: float = ALPHA) -> int:
    """Smallest one-directional discordant count d with exact p < alpha.

    Exact McNemar: conditional on d discordant pairs, the null distribution of
    the count going one way is Binomial(d, 1/2). All d one way gives a
    two-sided p of 2 * (1/2)^d.
    """
    d = 1
    while 2 * (0.5**d) >= alpha:
        d += 1
        if d > 64:  # unreachable for any sane alpha; guards an infinite loop
            raise ValueError(f"no d satisfies alpha={alpha}")
    return d


def prob_at_least_d(n_slots: int, pi: float, d_min: int) -> float:
    """P(Binomial(n_slots, pi) >= d_min) — the power of the exact test."""
    if pi <= 0.0:
        return 0.0
    if d_min > n_slots:
        return 0.0
    return 1.0 - sum(_binom_pmf(k, n_slots, pi) for k in range(d_min))


def design_effect(icc: float, slots_per_encounter: float) -> float:
    """Kish design effect: DEFF = 1 + (m - 1) * ICC.

    Slots inside one encounter share a source text and a model pass, so they are
    not independent draws. DEFF converts the required independent-slot count
    into a required actual-slot count.
    """
    return 1.0 + (slots_per_encounter - 1.0) * icc


@dataclass(frozen=True, slots=True)
class ReplicatePlan:
    pi: float
    d_min: int
    slots_per_instance: int
    deff: float
    instances_required: int | None
    effective_slots: int | None
    achieved_power: float | None
    note: str


def instances_required(
    *,
    pi: float,
    slots_per_instance: int,
    icc: float = 0.20,
    slots_per_encounter: float = 3.2,
    alpha: float = ALPHA,
    power: float = POWER,
    max_instances: int = 200,
) -> ReplicatePlan:
    """Smallest K whose informative slots reach `power` under the exact test.

    `slots_per_instance` must be the count of slots that can actually produce a
    discordance — for exact_gold_span that is the present-value slots only. The
    NOT_MENTIONED slots have no gold span and can never flip, so counting them
    would silently overstate power.
    """
    d_min = min_discordant_one_directional(alpha)
    deff = design_effect(icc, slots_per_encounter)

    if pi <= 0.0:
        return ReplicatePlan(
            pi=pi,
            d_min=d_min,
            slots_per_instance=slots_per_instance,
            deff=deff,
            instances_required=None,
            effective_slots=None,
            achieved_power=0.0,
            note=(
                "pi = 0: the manipulation has no purchase on this model, so the "
                "contrast is not underpowered, it is UNIDENTIFIED. No K fixes it."
            ),
        )

    for k in range(1, max_instances + 1):
        raw_slots = k * slots_per_instance
        # Clustering costs power; spend it by discounting the effective N.
        eff = int(raw_slots / deff)
        if prob_at_least_d(eff, pi, d_min) >= power:
            return ReplicatePlan(
                pi=pi,
                d_min=d_min,
                slots_per_instance=slots_per_instance,
                deff=deff,
                instances_required=k,
                effective_slots=eff,
                achieved_power=round(prob_at_least_d(eff, pi, d_min), 4),
                note="powered",
            )
    return ReplicatePlan(
        pi=pi,
        d_min=d_min,
        slots_per_instance=slots_per_instance,
        deff=deff,
        instances_required=None,
        effective_slots=None,
        achieved_power=None,
        note=f"no K <= {max_instances} reaches power={power} at pi={pi}",
    )


def clopper_pearson_upper(successes: int, n: int, conf: float = 0.95) -> float:
    """One-sided upper confidence bound on a proportion.

    For successes = 0 this reduces to the rule of three's exact form,
    1 - (1 - conf)^(1/n) — the honest way to report "we saw none in n".
    """
    if n <= 0:
        return 1.0
    if successes == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    # Bisection on the binomial tail; adequate and dependency-free.
    lo, hi = successes / n, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(_binom_pmf(k, n, mid) for k in range(successes + 1))
        if tail > 1.0 - conf:
            lo = mid
        else:
            hi = mid
    return hi
