"""Exact rational diagnostics for one two-dimensional edit example.

The accompanying proof is elementary real arithmetic, not a new theorem about
neural-language-model capacity. Fractions make these examples exact; the code
does not constitute machine-checked verification of the general proof.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path


def margins(point):
    x, y = map(Fraction, point)
    return {"target_B_minus_A": y - x, "protected_C_minus_D": x - Fraction(2, 5) * y}


def l1_from_initial(point):
    x, y = map(Fraction, point)
    return abs(x - 2) + abs(y - 1)


def protected_optimum(mu):
    """Minimum L1 edit from (2,1) with B-A >=3 and C-D >=mu, mu>=0.

    With displacement (a,b), feasibility implies b-a>=4 and
    a-(2/5)b>=mu-8/5, hence a>=5mu/3 and b>=4+5mu/3.
    Both are nonnegative. Therefore |a|+|b|>=4+10mu/3, attained
    by the returned point. This derivation is specific to these coordinates.
    """
    if isinstance(mu, bool) or not isinstance(mu, (int, Fraction)):
        raise TypeError("mu must be an exact integer or Fraction")
    mu = Fraction(mu)
    if mu < 0:
        raise ValueError("This formula is scoped to mu >= 0")
    a = Fraction(5, 3) * mu
    return (2 + a, 5 + a)


def _row(point):
    return {
        "point_exact": [str(Fraction(z)) for z in point],
        "l1_edit_exact": str(l1_from_initial(point)),
        "margins_exact": {key: str(value) for key, value in margins(point).items()},
        "protected_strict_winner": margins(point)["protected_C_minus_D"] > 0,
        "protected_tied": margins(point)["protected_C_minus_D"] == 0,
    }


def run():
    return {
        "scope": "Exact diagnostics for one fixed linear two-relation, two-value example; no trained model, general capacity bound or optimization-method superiority result",
        "initial": _row((2, 1)),
        "ideal_target_gradient_hitting_point": _row((0, 3)),
        "target_only_minimum_l1": {
            "value_exact": "4",
            "all_minimizers": "(x,x+3), -2 <= x <= 2",
            "protected_margin_on_minimizers": "3*x/5 - 6/5 <= 0",
            "tie_endpoint": _row((2, 5)),
            "tie_is_argmax_error_independent_of_tie_breaking": False,
            "strict_preservation_at_minimum_possible": False,
        },
        "protected_optima": [
            {"required_margin_exact": str(mu), **_row(protected_optimum(mu))}
            for mu in [Fraction(0), Fraction(1, 10), Fraction(1), Fraction(8, 5)]
        ],
        "tradeoff_formula": "minimum L1 = 4 + (10/3)*mu for mu >= 0",
        "strict_only_problem": {"infimum_exact": "4", "minimum_attained": False},
        "limits": [
            "The ideal gradient hitting point is a geometric construction, not a recorded optimizer trajectory.",
            "A nonpositive protected margin excludes a strict winner; a zero margin alone does not determine a tie-broken argmax answer.",
            "The L1 objective and its minimizers depend on coordinates and metric.",
            "No assertion that every later update must replay the protected fact follows from this example.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Create a NEW receipt; existing paths are never overwritten")
    args = parser.parse_args()
    text = json.dumps(run(), indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
