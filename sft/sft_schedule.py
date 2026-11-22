"""Learning-rate schedule for Stage 1 supervised fine-tuning."""
import math


PEAK_LR = 3e-4
FLOOR = 0.1


def lr_at(t, steps, warm):
    if t < warm: return PEAK_LR * t / max(1, warm)
    p = (t - warm) / max(1, steps - warm)
    return PEAK_LR * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * p)))
