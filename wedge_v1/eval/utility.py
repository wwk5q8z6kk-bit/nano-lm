"""Frozen draft U for Wedge v1 Phase 2."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Weights:
    alpha_Q: float = 1.0
    beta_E: float = 0.5
    gamma_R: float = 0.3
    lam_L: float = 0.02
    kappa_C: float = 0.05
    delta: float = 0.05


def utility(Q: float, E: float, R: float, L: float, C: float, w: Weights = Weights()) -> float:
    return w.alpha_Q * Q - w.beta_E * E - w.gamma_R * R - w.lam_L * L - w.kappa_C * C
