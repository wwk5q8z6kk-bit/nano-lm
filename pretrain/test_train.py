import pytest
import torch
from pretrain.train import batch, dev, S, BATCH

def test_batch_logic():
    # Use a dummy 1D tensor representing sequential token IDs
    # Ensure it's long enough to extract batches of size S
    dummy_src = torch.arange(2000, dtype=torch.long)

    bs = 4
    x, y = batch(dummy_src, bs=bs)

    # Check shapes
    assert x.shape == (bs, S), f"Expected x shape {(bs, S)}, got {x.shape}"
    assert y.shape == (bs, S), f"Expected y shape {(bs, S)}, got {y.shape}"

    # Check devices
    assert x.device.type == torch.device(dev).type, f"Expected x on {dev}, got {x.device}"
    assert y.device.type == torch.device(dev).type, f"Expected y on {dev}, got {y.device}"

    # Move back to CPU for easy values inspection if needed
    x = x.cpu()
    y = y.cpu()

    # Check logic: y should be exactly x shifted by 1
    # since dummy_src is sequential: src[i+1:i+S+1] is exactly src[i:i+S] + 1
    for i in range(bs):
        # Because dummy_src = [0, 1, 2, ...], x[i] should be a sequence of length S
        # and y[i] should be the same sequence shifted by +1
        assert torch.all(y[i] == x[i] + 1), "y should be x shifted by 1 offset in dummy sequential tensor"

        # Verify it's a contiguous sequence from original dummy_src
        start_val = x[i][0].item()
        expected_seq = torch.arange(start_val, start_val + S, dtype=torch.long)
        assert torch.all(x[i] == expected_seq), "x batch should contain contiguous elements from source"
