from __future__ import annotations

import math
from pathlib import Path

import pytest

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.adapters.state_checkpoint import build_state_span_prompt
from nano_ai.training import train_state_span
from nano_ai.training.state_span_data import generate_split, write_dataset
from nano_ai.training.train_state_span import (
    BATCH_SIZE,
    COSINE_FLOOR,
    PEAK_LEARNING_RATE,
    TRAINING_SEEDS,
    TrainingInputError,
    encode_training_example,
    grouped_batch_indices,
    learning_rate_at,
    load_frozen_tokenizer,
    load_training_bundle,
    masked_cross_entropy,
    tokenize_split,
    train_candidate,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tokenizer_path() -> Path:
    return _root() / "sft" / "tokenizer.json"


def test_chatml_encoding_uses_shared_prompt_and_assistant_only_mask() -> None:
    tokenizer = load_frozen_tokenizer(_tokenizer_path())
    example = generate_split("dev", worlds=5)[0]
    token_ids, loss_mask = encode_training_example(tokenizer, example)

    first_supervised = loss_mask.index(1)
    assert all(value == 0 for value in loss_mask[:first_supervised])
    assert all(value == 1 for value in loss_mask[first_supervised:])
    assert token_ids[-1] == tokenizer.token_to_id("<|im_end|>")
    assert tokenizer.decode(list(token_ids[first_supervised:-1])) == example.target

    prompt_ids = tokenizer.encode(
        build_state_span_prompt(example.transcript), add_special_tokens=False
    ).ids
    assert _contains_subsequence(token_ids[:first_supervised], prompt_ids)


def _contains_subsequence(values: tuple[int, ...], expected: list[int]) -> bool:
    width = len(expected)
    return any(
        list(values[index : index + width]) == expected for index in range(len(values))
    )


def test_tokenization_never_drops_or_overflows_frozen_examples() -> None:
    tokenizer = load_frozen_tokenizer(_tokenizer_path())
    examples = generate_split("dev")
    tokenized = tokenize_split(tokenizer, examples)

    assert tokenized.token_ids.shape == (1000, 512)
    assert tokenized.loss_mask.shape == (1000, 512)
    assert len(tokenized.lengths) == len(examples)
    assert max(tokenized.lengths) <= 512
    assert min(tokenized.loss_mask.sum(dim=1).tolist()) > 1


def test_grouped_batches_keep_all_four_world_variants_together() -> None:
    examples = generate_split("train", worlds=20)
    batches = grouped_batch_indices(
        examples, batch_size=8, seed=TRAINING_SEEDS[0], epoch=1
    )
    repeated = grouped_batch_indices(
        examples, batch_size=8, seed=TRAINING_SEEDS[0], epoch=1
    )
    next_epoch = grouped_batch_indices(
        examples, batch_size=8, seed=TRAINING_SEEDS[0], epoch=2
    )

    assert batches == repeated
    assert batches != next_epoch
    assert sorted(index for batch in batches for index in batch) == list(
        range(len(examples))
    )
    for batch in batches:
        worlds = [examples[index].world_id for index in batch]
        assert sorted(worlds.count(world) for world in set(worlds)) == [4, 4]


def test_learning_rate_matches_frozen_warmup_and_cosine_floor() -> None:
    total_steps = 1000
    warmup_steps = 30
    assert learning_rate_at(1, total_steps=total_steps) == pytest.approx(
        PEAK_LEARNING_RATE / warmup_steps
    )
    assert learning_rate_at(warmup_steps, total_steps=total_steps) == pytest.approx(
        PEAK_LEARNING_RATE
    )
    assert learning_rate_at(total_steps, total_steps=total_steps) == pytest.approx(
        PEAK_LEARNING_RATE * COSINE_FLOOR
    )


def test_masked_cross_entropy_ignores_unsupervised_positions() -> None:
    torch = pytest.importorskip("torch")
    vocabulary = FROZEN_NANO_V01.vocabulary_size
    token_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
    loss_mask = torch.tensor([[False, False, True]])
    logits = torch.zeros((1, 3, vocabulary))
    logits[0, 1, 7] = 4.0

    loss, supervised = masked_cross_entropy(logits, token_ids, loss_mask)
    logits[0, 0, 6] = -1000.0
    changed, changed_supervised = masked_cross_entropy(logits, token_ids, loss_mask)

    expected = math.log(math.exp(4.0) + vocabulary - 1) - 4.0
    assert loss.item() == pytest.approx(expected)
    assert changed.item() == pytest.approx(loss.item())
    assert supervised.item() == changed_supervised.item() == 1


def test_tokenizer_loader_rejects_unregistered_bytes(tmp_path: Path) -> None:
    fake = tmp_path / "tokenizer.json"
    fake.write_text("{}", encoding="utf-8")
    with pytest.raises(TrainingInputError, match="SHA-256 mismatch"):
        load_frozen_tokenizer(fake)


def test_production_bundle_is_exactly_regenerable(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    manifest = write_dataset(
        data_dir,
        tokenizer_sha256=FROZEN_NANO_V01.tokenizer_sha256,
        base_checkpoint_sha256=FROZEN_NANO_V01.checkpoint_sha256,
    )

    bundle = load_training_bundle(data_dir)

    assert len(bundle.train) == 12_000
    assert len(bundle.dev) == 1_000
    assert bundle.manifest == manifest


def test_training_refuses_nonrecipe_seed_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    with pytest.raises(TrainingInputError, match="seed"):
        train_candidate(
            data_dir=tmp_path / "missing-data",
            base_checkpoint=tmp_path / "missing-base.pt",
            tokenizer_path=tmp_path / "missing-tokenizer.json",
            output_dir=output,
            seed=1,
            device="cpu",
        )
    assert not output.exists()


def test_training_refuses_existing_output_before_reading_inputs(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    output.mkdir()
    marker = output / "owned.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(TrainingInputError, match="must not exist"):
        train_candidate(
            data_dir=tmp_path / "missing-data",
            base_checkpoint=tmp_path / "missing-base.pt",
            tokenizer_path=tmp_path / "missing-tokenizer.json",
            output_dir=output,
            seed=TRAINING_SEEDS[0],
            device="cpu",
        )
    assert marker.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("batch_size", [0, 2, 6])
def test_grouping_rejects_nonpaired_batch_sizes(batch_size: int) -> None:
    with pytest.raises(ValueError, match="multiple of four"):
        grouped_batch_indices(
            generate_split("train", worlds=5),
            batch_size=batch_size,
            seed=TRAINING_SEEDS[0],
            epoch=1,
        )


def test_frozen_batch_size_preserves_eight_worlds() -> None:
    assert BATCH_SIZE // 4 == 8


def test_training_detects_source_mutation_during_execution(monkeypatch) -> None:
    expected = {"training": "0" * 64}
    monkeypatch.setattr(
        train_state_span,
        "_source_hashes",
        lambda: {"training": "1" * 64},
    )

    with pytest.raises(TrainingInputError, match="source changed"):
        train_state_span._require_unchanged_sources(expected)
