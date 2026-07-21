import math
import pytest
from sft import lr_at, PEAK_LR, FLOOR

def test_lr_at_warmup():
    steps = 1000
    warm = 100

    # Test that learning rate increases linearly during warmup
    t1 = 0
    assert lr_at(t1, steps, warm) == 0.0

    t2 = 50
    expected_lr = PEAK_LR * t2 / warm
    assert math.isclose(lr_at(t2, steps, warm), expected_lr, rel_tol=1e-5)

def test_lr_at_peak():
    steps = 1000
    warm = 100

    # Test that learning rate peaks exactly at 'warm' steps
    t = warm
    assert math.isclose(lr_at(t, steps, warm), PEAK_LR, rel_tol=1e-5)

def test_lr_at_cosine_decay():
    steps = 1000
    warm = 100

    # Test various points in the cosine decay phase
    t = (steps + warm) // 2
    p = (t - warm) / (steps - warm)
    expected_lr = PEAK_LR * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * p)))
    assert math.isclose(lr_at(t, steps, warm), expected_lr, rel_tol=1e-5)

def test_lr_at_min_lr():
    steps = 1000
    warm = 100

    # Test that learning rate hits the floor at the end of steps
    t = steps
    expected_lr = PEAK_LR * FLOOR
    assert math.isclose(lr_at(t, steps, warm), expected_lr, rel_tol=1e-5)

    # And stays there (if we extrapolate beyond, the math extends the cosine wave,
    # but based on the code's math, it actually starts going back up. In practice,
    # t shouldn't exceed STEPS, but t == STEPS is the key check)
    assert math.isclose(lr_at(steps, steps, warm), expected_lr, rel_tol=1e-5)

def test_lr_at_no_warmup():
    steps = 1000
    warm = 0

    # If no warmup, at t=0, it should be the peak LR
    t = 0
    assert math.isclose(lr_at(t, steps, warm), PEAK_LR, rel_tol=1e-5)

def test_lr_at_all_warmup():
    steps = 100
    warm = 100

    # When steps == warm, testing to ensure no division by zero in the cosine decay calculation
    # (max(1, steps - warm) handles this)
    t = 100
    assert math.isclose(lr_at(t, steps, warm), PEAK_LR, rel_tol=1e-5)

    # Let's see what happens if t > warm in this case (t = 101)
    # p = (101 - 100) / max(1, 0) = 1
    t_beyond = 101
    expected_lr_beyond = PEAK_LR * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * 1)))
    assert math.isclose(lr_at(t_beyond, steps, warm), expected_lr_beyond, rel_tol=1e-5)
