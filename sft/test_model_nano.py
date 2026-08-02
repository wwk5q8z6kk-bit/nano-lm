import math
import torch
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_nano as nano

def reference_rope(x):
    B, H, S, hd = x.shape
    out = torch.zeros_like(x)
    for m in range(S):
        for i in range(hd // 2):
            theta_i = 1.0 / (10000 ** (2 * i / hd))
            angle = m * theta_i
            cos = math.cos(angle)
            sin = math.sin(angle)
            out[:, :, m, 2*i]   = x[:, :, m, 2*i] * cos - x[:, :, m, 2*i+1] * sin
            out[:, :, m, 2*i+1] = x[:, :, m, 2*i] * sin + x[:, :, m, 2*i+1] * cos
    return out

def test_rope_values():
    torch.manual_seed(42)
    # Use small B, H, but full S and hd since rope uses global nano.S
    B, H, S, hd = 2, 2, nano.S, nano.hd
    q = torch.randn(B, H, S, hd, device=nano.dev)
    k = torch.randn(B, H, S, hd, device=nano.dev)

    q_rot, k_rot = nano.rope(q, k)

    q_ref = reference_rope(q.cpu()).to(nano.dev)
    k_ref = reference_rope(k.cpu()).to(nano.dev)

    torch.testing.assert_close(q_rot, q_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(k_rot, k_ref, rtol=1e-4, atol=1e-4)

def test_rope_shapes():
    B, H, S, hd = 1, 1, nano.S, nano.hd
    q = torch.randn(B, H, S, hd, device=nano.dev)
    k = torch.randn(B, H, S, hd, device=nano.dev)

    q_rot, k_rot = nano.rope(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
