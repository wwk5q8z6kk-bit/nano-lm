import torch
import pytest
import os
import sys

# Need to run from sft/ or handle paths.
# We'll just patch the path so `from model_nano import ...` works, assuming we run pytest from /app/sft
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# We must execute tests safely even if tokenizer.json is relative and executed from root.
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model_nano import rope, S, hd, dev

def reference_rope(q, k):
    """
    Reference implementation of Rotary Positional Embeddings (RoPE).
    Operates on last dimension `hd` and second-to-last dimension `S`.
    """
    q_out = torch.zeros_like(q)
    k_out = torch.zeros_like(k)

    for pos in range(S):
        for i in range(hd // 2):
            theta_i = 1.0 / (10000 ** (2 * i / hd))
            angle = pos * theta_i
            c = torch.cos(torch.tensor(angle, device=q.device))
            s = torch.sin(torch.tensor(angle, device=q.device))

            # Apply to q
            q_out[..., pos, 2*i] = q[..., pos, 2*i] * c - q[..., pos, 2*i+1] * s
            q_out[..., pos, 2*i+1] = q[..., pos, 2*i] * s + q[..., pos, 2*i+1] * c

            # Apply to k
            k_out[..., pos, 2*i] = k[..., pos, 2*i] * c - k[..., pos, 2*i+1] * s
            k_out[..., pos, 2*i+1] = k[..., pos, 2*i] * s + k[..., pos, 2*i+1] * c

    return q_out, k_out

def test_rope_standard():
    torch.manual_seed(42)
    B = 2
    H = 4
    KV = 2

    # Initialize random query and key tensors
    q = torch.randn(B, H, S, hd, device=dev)
    k = torch.randn(B, KV, S, hd, device=dev)

    # Run standard implementation
    q_opt, k_opt = rope(q, k)

    # Run reference implementation
    q_ref, k_ref = reference_rope(q, k)

    # Compare results
    # Use larger tolerance due to dtype=float32 precision limits
    torch.testing.assert_close(q_opt, q_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(k_opt, k_ref, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    test_rope_standard()
