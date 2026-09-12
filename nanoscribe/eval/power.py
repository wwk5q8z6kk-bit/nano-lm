"""Sample-size and power analysis for evaluation design.

`~63 per axis` is NOT a universal law. It is the answer to one specific question
(two-proportion, unpaired, p~0.5, delta=0.25, alpha=0.05, power=0.80). Required n
moves by an order of magnitude with the effect size, the baseline rate, whether
the design is paired, and the axis prevalence.

Every declared threshold must therefore carry its assumptions. This module
computes n from stated assumptions and records them, so a number in a manifest
can always be traced back to the question it answers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

# Normal quantiles for the alpha/power levels used here; avoids a scipy dependency.
_Z_TWO_SIDED = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}
_Z_ONE_SIDED = {0.10: 1.2816, 0.05: 1.6449, 0.01: 2.3263}
_Z_POWER = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}


@dataclass(frozen=True)
class PowerAssumptions:
    """Everything required to reproduce a sample-size number."""

    metric: str
    unit_of_analysis: str
    baseline_rate: float
    minimum_relevant_effect: float
    alpha: float = 0.05
    power: float = 0.80
    paired: bool = False
    two_sided: bool = True
    axis_prevalence: float = 1.0
    note: str = ""


def wilson(k: int, n: int, z: float = 1.9600) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def required_n(assumptions: PowerAssumptions) -> dict[str, Any]:
    """Eligible cases needed per arm, from the stated assumptions.

    Unpaired two-proportion:
        n = (z_alpha + z_beta)^2 * 2 * pbar * (1 - pbar) / delta^2

    Paired (McNemar-style) designs need fewer cases because between-case variance
    cancels; approximated via the discordant-pair rate. Evaluating the same cases
    across models is paired, so treating it as unpaired overstates the requirement.
    """
    a = assumptions
    z_a = (_Z_TWO_SIDED if a.two_sided else _Z_ONE_SIDED).get(a.alpha, 1.9600)
    z_b = _Z_POWER.get(a.power, 0.8416)

    p1 = a.baseline_rate
    p2 = min(1.0, max(0.0, p1 + a.minimum_relevant_effect))
    delta = abs(p2 - p1)
    if delta == 0:
        return {"error": "minimum_relevant_effect must be non-zero", **asdict(a)}

    pbar = (p1 + p2) / 2
    n_unpaired = ((z_a + z_b) ** 2 * 2 * pbar * (1 - pbar)) / (delta**2)

    # Paired: assume discordance ~ the union of change in either direction.
    disc = max(1e-6, p1 * (1 - p2) + p2 * (1 - p1))
    n_paired = ((z_a * math.sqrt(disc) + z_b * math.sqrt(disc - delta**2)) ** 2) / (delta**2)
    n_eligible = n_paired if a.paired else n_unpaired

    # Axis prevalence: if only 30% of cases exercise this axis, the SUITE must be
    # correspondingly larger to yield that many eligible cases.
    n_suite = n_eligible / a.axis_prevalence if a.axis_prevalence > 0 else float("inf")

    return {
        **asdict(a),
        "z_alpha": z_a,
        "z_power": z_b,
        "baseline_p1": round(p1, 4),
        "target_p2": round(p2, 4),
        "delta": round(delta, 4),
        "required_eligible_cases_per_arm": math.ceil(n_eligible),
        "required_suite_cases": math.ceil(n_suite),
        "design": "paired" if a.paired else "unpaired",
    }


def detectable_effect(n: int, baseline: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Inverse question: with n eligible cases, what effect can be detected?

    Use this to state honestly what an EXISTING suite can and cannot resolve
    rather than implying it can resolve anything.
    """
    z_a = _Z_TWO_SIDED.get(alpha, 1.9600)
    z_b = _Z_POWER.get(power, 0.8416)
    if n <= 0:
        return 1.0
    return round((z_a + z_b) * math.sqrt(2 * baseline * (1 - baseline) / n), 4)


def report(assumption_list: list[PowerAssumptions]) -> dict[str, Any]:
    rows = [required_n(a) for a in assumption_list]
    return {
        "schema": "nano.eval.power_report.v1",
        "method": "normal-approximation two-proportion; paired via discordant-pair rate",
        "caveat": (
            "These are design targets under the stated assumptions, not universal "
            "minima. Changing the minimum relevant effect changes n quadratically."
        ),
        "rows": rows,
        "max_required_suite_cases": max((r.get("required_suite_cases", 0) for r in rows), default=0),
    }
