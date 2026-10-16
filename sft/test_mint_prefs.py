import os, sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mint_prefs as mp

# We need the IMS token for testing leak penalty
IMS = mp.IMS

def test_empty_output():
    assert mp.rubric([], True) == 0.0
    assert mp.rubric([], False) == 0.0

def test_distinctness():
    # clean_stop = 1.0 (len=4, stopped=True)
    # distinct = 1.0 (4 unique tokens out of 4)
    # leak = 0.0
    # expected = 1.0 + 0.5 * 1.0 - 0.0 = 1.5
    out_unique = [1, 2, 3, 4]
    assert math.isclose(mp.rubric(out_unique, True), 1.5)

    # clean_stop = 1.0 (len=4, stopped=True)
    # distinct = 0.25 (1 unique token out of 4)
    # leak = 0.0
    # expected = 1.0 + 0.5 * 0.25 - 0.0 = 1.125
    out_repeated = [1, 1, 1, 1]
    assert math.isclose(mp.rubric(out_repeated, True), 1.125)

def test_clean_stop():
    # stopped=True, length in [3, 50]
    out = [1, 2, 3] # length 3
    # clean_stop = 1.0, distinct = 1.0, leak = 0.0 => 1.5
    assert math.isclose(mp.rubric(out, True), 1.5)

    # stopped=False, length in [3, 50]
    # clean_stop = 0.0, distinct = 1.0, leak = 0.0 => 0.5
    assert math.isclose(mp.rubric(out, False), 0.5)

    # stopped=True, length < 3
    out_short = [1, 2] # length 2
    # clean_stop = 0.0, distinct = 1.0, leak = 0.0 => 0.5
    assert math.isclose(mp.rubric(out_short, True), 0.5)

    # stopped=True, length > 50
    out_long = list(range(51)) # length 51
    # clean_stop = 0.0, distinct = 1.0, leak = 0.0 => 0.5
    assert math.isclose(mp.rubric(out_long, True), 0.5)

def test_leak_penalty():
    # clean_stop = 1.0, distinct = 1.0, leak = 0.5 (has IMS)
    # expected = 1.0 + 0.5 * 1.0 - 0.5 = 1.0
    out = [1, 2, 3, IMS] # length 4
    assert math.isclose(mp.rubric(out, True), 1.0)

    # distinctness affects score too
    # distinct = 2 / 4 = 0.5
    # clean_stop = 1.0
    # leak = 0.5
    # expected = 1.0 + 0.5 * 0.5 - 0.5 = 0.75
    out_repeated = [1, 1, 1, IMS]
    assert math.isclose(mp.rubric(out_repeated, True), 0.75)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in fns:
        f(); print(f"  PASS {n}")
    print(f"mint_prefs rubric tests: {len(fns)}/{len(fns)} PASS")
