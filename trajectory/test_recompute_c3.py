# Fixture tests for the C-3 independent recompute (trajectory/recompute_c3.py).
# Pins the FROZEN decision-rule logic (verdict thresholds, seed-majority flip, contrast
# averaging) against synthetic recall tables built over the REAL committed manifest type
# labels — so the harness that will emit the independent verdict is trustworthy before
# any real result lands. No model, no GPU, no network.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_c3 as rc

CELL = [t for t in rc.LABEL if rc.MAN[t]["T"] in ("T-avail", "T-sep")
        and rc.MAN[t]["B"] in ("B-sub", "B-space") and rc.MAN[t]["L"] in ("short", "long")]

def rt_const(pred_flip):
    """synthetic (type->{seed->recall}) where flip state = pred_flip(type), all 3 seeds agree."""
    return {t: {s: (1.0 if pred_flip(t) else 0.0) for s in range(3)} for t in rc.LABEL}

def test_verdict_thresholds():
    assert rc.verdict(40) == "SUPPORTED" and rc.verdict(100) == "SUPPORTED"
    assert rc.verdict(15) == "REFUTED" and rc.verdict(0) == "REFUTED" and rc.verdict(-4) == "REFUTED"
    assert rc.verdict(25) == "UNRESOLVED" and rc.verdict(39.9) == "UNRESOLVED"

def test_majority_flip():
    rt = {"x": {0: 1.0, 1: 1.0, 2: 0.0}}   # 2/3 -> flip
    assert rc.majority_flip(rt, "x")[0] is True
    rt = {"y": {0: 1.0, 1: 0.0, 2: 0.0}}   # 1/3 -> no flip
    assert rc.majority_flip(rt, "y")[0] is False

def test_transition_contrast_supported():
    # all T-avail flip, no T-sep flips -> dT = +100 in every matched (B,L) stratum
    rt = rt_const(lambda t: rc.LABEL[t]["T"] == "T-avail")
    dT = rc.contrast(rt, CELL, "T", "T-avail", "T-sep")
    assert dT == 100.0 and rc.verdict(dT) == "SUPPORTED", dT

def test_transition_contrast_refuted():
    # flip state independent of T (depends on B) -> dT = 0 -> REFUTED
    rt = rt_const(lambda t: rc.LABEL[t]["B"] == "B-sub")
    dT = rc.contrast(rt, CELL, "T", "T-avail", "T-sep")
    assert abs(dT) <= 1e-9 and rc.verdict(dT) == "REFUTED", dT

def test_boundary_contrast_isolated_from_T():
    # flip depends only on B: dB should be +100, dT and dL ~ 0
    rt = rt_const(lambda t: rc.LABEL[t]["B"] == "B-sub")
    dB = rc.contrast(rt, CELL, "B", "B-sub", "B-space")
    dT = rc.contrast(rt, CELL, "T", "T-avail", "T-sep")
    dL = rc.contrast(rt, CELL, "L", "short", "long")
    assert dB == 100.0 and abs(dT) <= 1e-9 and abs(dL) <= 1e-9, (dB, dT, dL)

def test_seed_unstable_types_are_excluded():
    # Regression pin for the bug fixed in 823e1ca: the seed-unstable set was computed
    # for reporting but never applied as an exclusion before the frozen contrasts, which
    # PREREG_C3 requires ("3-way-unstable types are reported separately and excluded").
    avail = [t for t in CELL if rc.LABEL[t]["T"] == "T-avail"
             and rc.LABEL[t]["B"] == "B-sub" and rc.LABEL[t]["L"] == "short"]
    assert len(avail) >= 2
    # all T-avail-in-cell miss, EXCEPT one made seed-unstable but majority-flip [T,T,F]
    rt = {t: {0: 0.0, 1: 0.0, 2: 0.0} for t in rc.LABEL}
    u = avail[0]
    rt[u] = {0: 1.0, 1: 1.0, 2: 0.0}          # votes T,T,F -> majority flip, but unstable
    unstable = set(t for t in CELL if len(set(rc.majority_flip(rt, t)[1])) > 1)
    assert u in unstable, "unstable type not detected"
    rate_incl, n_incl = rc.cell_rate(avail, rt, lambda t: True)              # bug behaviour
    rate_excl, n_excl = rc.cell_rate(avail, rt, lambda t: True, unstable)    # frozen rule
    assert n_excl == n_incl - 1, "exactly the unstable type must be excluded"
    assert rate_incl != rate_excl, "exclusion must change the estimand here"
    assert rate_excl == 0.0, f"with u excluded, no cell type flips -> 0%, got {rate_excl}"

def test_cell_coverage():
    # every registered cell is non-empty in the frozen manifest (design constructible)
    import collections
    c = collections.Counter((rc.LABEL[t]["T"], rc.LABEL[t]["B"], rc.LABEL[t]["L"]) for t in CELL)
    assert len(c) == 8 and all(v >= 5 for v in c.values()), dict(c)

if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in fns:
        f(); print(f"  PASS {n}")
    print(f"recompute_c3 fixture tests: {len(fns)}/{len(fns)} PASS")
