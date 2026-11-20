import os
import sys
import torch
import pytest

# Add the project root to the path so we can import pretrain.train
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain.train import batch, dev, S

def test_batch():
    # Set a fixed seed for reproducibility
    torch.manual_seed(42)

    # Create a dummy 1D tensor of length 1000
    src_len = 1000
    src = torch.arange(src_len, dtype=torch.float32)

    # Batch size
    bs = 8

    # Call the batch function
    x, y = batch(src, bs=bs)

    # 1. Check types and devices
    assert x.device.type == torch.device(dev).type
    assert y.device.type == torch.device(dev).type

    # 2. Check shapes
    # Shape should be (batch_size, sequence_length)
    assert x.shape == (bs, S)
    assert y.shape == (bs, S)

    # 3. Check offsets (y should be offset by +1 token from x)
    # y[i, j] should be exactly x[i, j] + 1
    # We can check this directly because we created src as an arange
    assert torch.all(y == x + 1), "y should be offset by exactly 1 from x"

    # Check another property: elements should be contiguous in sequence dimension
    for i in range(bs):
        # x sequences should increment by 1
        diff_x = x[i, 1:] - x[i, :-1]
        assert torch.all(diff_x == 1)

        # y sequences should increment by 1
        diff_y = y[i, 1:] - y[i, :-1]
        assert torch.all(diff_y == 1)
