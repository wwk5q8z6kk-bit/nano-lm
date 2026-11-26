from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _frozen_checkpoint() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "checkpoints"
        / "anchors"
        / "nano_v01_scribe.pt"
    )


def test_native_config_and_model_retain_exact_frozen_geometry():
    from nano_ai.training.model import NANO_MODEL_CONFIG, NanoGPT

    model = NanoGPT()
    state_dict = model.state_dict()

    assert NANO_MODEL_CONFIG.parameter_count == 3_148_608
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_148_608
    assert tuple(state_dict["emb.weight"].shape) == (4098, 192)
    assert tuple(state_dict["nf.weight"].shape) == (192,)
    assert len(model.blocks) == 6
    for index in range(6):
        prefix = f"blocks.{index}."
        assert tuple(state_dict[f"{prefix}n1.weight"].shape) == (192,)
        assert tuple(state_dict[f"{prefix}n2.weight"].shape) == (192,)
        assert tuple(state_dict[f"{prefix}q.weight"].shape) == (192, 192)
        assert tuple(state_dict[f"{prefix}k.weight"].shape) == (64, 192)
        assert tuple(state_dict[f"{prefix}v.weight"].shape) == (64, 192)
        assert tuple(state_dict[f"{prefix}o.weight"].shape) == (192, 192)
        assert tuple(state_dict[f"{prefix}g.weight"].shape) == (512, 192)
        assert tuple(state_dict[f"{prefix}u.weight"].shape) == (512, 192)
        assert tuple(state_dict[f"{prefix}dn.weight"].shape) == (192, 512)
    assert not any("head" in key for key in state_dict)


@pytest.mark.parametrize("sequence_length", [1, 7, 31])
def test_forward_accepts_dynamic_causal_sequence_lengths(sequence_length):
    torch = pytest.importorskip("torch")
    from nano_ai.training.model import NANO_MODEL_CONFIG, NanoGPT

    model = NanoGPT()
    token_ids = torch.randint(
        NANO_MODEL_CONFIG.vocabulary_size,
        (2, sequence_length),
        dtype=torch.long,
    )

    logits = model(token_ids)

    assert tuple(logits.shape) == (
        2,
        sequence_length,
        NANO_MODEL_CONFIG.vocabulary_size,
    )
    logits[:, -1].mean().backward()
    assert model.emb.weight.grad is not None


def test_forward_rejects_invalid_sequence_shapes_and_context_overflow():
    torch = pytest.importorskip("torch")
    from nano_ai.training.model import NANO_MODEL_CONFIG, NanoGPT

    model = NanoGPT()

    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError, match="at least one token"):
        model(torch.zeros((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="exceeds Nano"):
        model(
            torch.zeros(
                (1, NANO_MODEL_CONFIG.sequence_length + 1),
                dtype=torch.long,
            )
        )


def test_exact_frozen_checkpoint_loads_strictly_when_available():
    pytest.importorskip("torch")
    from nano_ai.training.model import NanoGPT, load_frozen_nano_state_dict

    checkpoint = _frozen_checkpoint()
    if not checkpoint.is_file():
        pytest.skip("frozen release checkpoint is not present in this checkout")

    state_dict = load_frozen_nano_state_dict(checkpoint)
    model = NanoGPT()
    incompatible = model.load_state_dict(state_dict, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert sum(tensor.numel() for tensor in state_dict.values()) == 3_148_608


def test_frozen_checkpoint_loader_rejects_unregistered_bytes(tmp_path):
    from nano_ai.training.model import (
        FrozenNanoCheckpointError,
        load_frozen_nano_state_dict,
    )

    checkpoint = tmp_path / "not-the-frozen-checkpoint.pt"
    checkpoint.write_bytes(b"unregistered checkpoint bytes")

    with pytest.raises(FrozenNanoCheckpointError, match="SHA-256 mismatch"):
        load_frozen_nano_state_dict(checkpoint)


def test_verified_loader_accepts_an_explicit_hash_identified_candidate(tmp_path):
    torch = pytest.importorskip("torch")
    from nano_ai.training.model import (
        NanoGPT,
        load_verified_nano_state_dict,
    )

    checkpoint = tmp_path / "candidate.pt"
    torch.save(NanoGPT().state_dict(), checkpoint)
    expected_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    state_dict = load_verified_nano_state_dict(
        checkpoint,
        expected_sha256=expected_sha256,
    )

    assert sum(tensor.numel() for tensor in state_dict.values()) == 3_148_608


@pytest.mark.parametrize(
    "expected_sha256",
    ["", "0" * 63, "A" * 64, "g" * 64, object()],
)
def test_verified_loader_rejects_noncanonical_expected_digests(
    tmp_path, expected_sha256
):
    from nano_ai.training.model import load_verified_nano_state_dict

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"unused")

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_verified_nano_state_dict(
            checkpoint,
            expected_sha256=expected_sha256,
        )
