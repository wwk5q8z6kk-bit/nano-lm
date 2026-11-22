"""Bilinear margin-drift diagnostics in float64; no training or interval proof.

The inequalities are sufficient in exact real arithmetic. These numerical
examples do not use outward-rounded interval arithmetic and therefore are not
machine-verified certificates. Targets must retain the same semantic version.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _inputs(w0, u0, w1, u1, target):
    arrays = tuple(np.asarray(x, dtype=np.float64) for x in (w0, u0, w1, u1))
    w0, u0, w1, u1 = arrays
    if (w0.ndim != 2 or w0.shape[0] < 2 or w0.shape[1] < 1
            or w1.shape != w0.shape or u0.shape != (w0.shape[1],)
            or u1.shape != u0.shape):
        raise ValueError('Require two [values,width] decoders and two [width] states')
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError('Finite inputs required')
    if not isinstance(target, (int, np.integer)) or isinstance(target, (bool, np.bool_)):
        raise ValueError('Target must be an integer index')
    if not 0 <= target < len(w0):
        raise ValueError('Target outside value set')
    return arrays


def margin_drift(w0, u0, w1, u1, target):
    """Return exact-form drift terms and conservative norm estimates per rival.

    Scores are W @ u. A positive computed lower bound is a diagnostic, not a
    floating-point safety certificate. A nonpositive bound is inconclusive.
    """
    w0, u0, w1, u1 = _inputs(w0, u0, w1, u1, target)
    rivals = np.array([k for k in range(len(w0)) if k != target], dtype=int)
    with np.errstate(over='raise', invalid='raise'):
        try:
            a0 = w0[target] - w0[rivals]
            dw = w1 - w0
            da = dw[target] - dw[rivals]
            du = u1 - u0
            before = a0 @ u0
            after = (w1[target] - w1[rivals]) @ u1
            terms = np.stack((a0 @ du, da @ u0, da @ du), axis=-1)
            bounds = np.stack((np.linalg.norm(a0, axis=-1) * np.linalg.norm(du),
                               np.linalg.norm(da, axis=-1) * np.linalg.norm(u0),
                               np.linalg.norm(da, axis=-1) * np.linalg.norm(du)), axis=-1)
            total = np.sum(bounds, axis=-1)
            lower = before - total
            finite = all(np.isfinite(x).all() for x in (before, after, terms, bounds, lower))
            if not finite:
                raise ValueError('Nonfinite derived diagnostics')
        except FloatingPointError as exc:
            raise ValueError('Overflow in derived diagnostics') from exc
    return {
        'rivals': rivals.tolist(), 'margins_before': before.tolist(),
        'margins_after': after.tolist(),
        'signed_terms_state_decoder_interaction': terms.tolist(),
        'norm_bounds_state_decoder_interaction': bounds.tolist(),
        'total_bound': total.tolist(), 'margin_lower_bound': lower.tolist(),
        'decomposition_max_abs_residual': float(np.max(np.abs(after - before - terms.sum(-1)))),
        'positive_lower_bound': bool(np.all(lower > 0)),
        'actual_strict_target_winner_after': bool(np.all(after > 0)),
        'scope': 'Float64 evaluation of a sufficient real-arithmetic inequality; not outward-rounded certification, causal evidence or a learned LM result.',
    }


def trajectory_margin_drift(states, target):
    """Telescoping bound for a supplied finite trajectory of (W,u) snapshots.

    All snapshots must use the same target/value interpretation. New legitimate
    corrections need a new trajectory; this routine does not infer truth.
    """
    if len(states) < 2:
        raise ValueError('At least two states required')
    rows = [margin_drift(*states[i], *states[i+1], target) for i in range(len(states)-1)]
    initial = np.asarray(rows[0]['margins_before'])
    accumulated = np.cumsum(np.asarray([row['total_bound'] for row in rows]), axis=0)
    lower = initial[None, :] - accumulated
    actual = np.asarray([row['margins_after'] for row in rows])
    if not np.isfinite(lower).all():
        raise ValueError('Nonfinite accumulated bound')
    return {'steps': rows, 'cumulative_bounds': accumulated.tolist(),
            'margin_lower_bounds': lower.tolist(), 'actual_margins': actual.tolist(),
            'positive_lower_bounds_through_horizon': bool(np.all(lower > 0)),
            'all_actual_margins_positive': bool(np.all(actual > 0)),
            'scope': 'Finite supplied trajectory only; target version must be unchanged; nonpositive lower bounds are inconclusive.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='Optional new diagnostic JSON; refuses an existing path')
    args = parser.parse_args()
    w = np.array([[1.,0.],[-1.,0.]])
    u = np.array([1.,0.])
    result = {
        'large_initial_margin_can_be_overturned': margin_drift(10*w, u, -10*w, u, 0),
        'orthogonal_state_change_preserves_prediction': margin_drift(w, u, w, np.array([1.,100.]), 0),
        'coherent_transport_preserves_prediction_despite_loose_bound': margin_drift(w, u, w/100, 100*u, 0),
        'claim': 'Constructed counterexamples and inequalities; no training or architecture comparison.'}
    if args.out is not None:
        with args.out.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
