from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FIELD_ORDER, FieldState
from nano_ai.training import state_span_data
from nano_ai.training.state_span_data import (
    DEV_WORLDS,
    FORBIDDEN_HISTORICAL_SENTINELS,
    TRAIN_WORLDS,
    assert_static_split_isolation,
    build_manifest,
    canonical_json_bytes,
    generate_split,
    load_records,
    supported_value_sets,
    write_dataset,
)

_TOKENIZER_SHA256 = "b" * 64
_CHECKPOINT_SHA256 = "c" * 64
_GENERATOR_SHA256 = "d" * 64


def _records_sha256(split: str, *, worlds: int) -> str:
    payload = b"".join(
        canonical_json_bytes(example.to_dict())
        for example in generate_split(split, worlds=worlds)
    )
    return hashlib.sha256(payload).hexdigest()


def test_generation_is_cross_process_deterministic() -> None:
    script = (
        "from nano_ai.tests.test_state_span_data import _records_sha256;"
        "print(_records_sha256('train', worlds=10))"
    )
    observed = []
    for hash_seed in ("1", "777"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        observed.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=Path(__file__).parents[2],
                env=env,
                text=True,
            ).strip()
        )
    assert (
        observed
        == ["6e6096381eef6f6d7b388a5aeeac660f2c7c23c11c4fef7cbf78b3d3871844e6"] * 2
    )


def test_full_recipe_counts_and_exact_state_field_quotas() -> None:
    expected_worlds = {"train": TRAIN_WORLDS, "dev": DEV_WORLDS}
    expected_quota = {"train": 600, "dev": 50}
    for split in ("train", "dev"):
        examples = generate_split(split)
        assert len(examples) == expected_worlds[split] * 4
        assert len({example.world_id for example in examples}) == expected_worlds[split]
        counts = Counter(
            (example.variant, example.target_field) for example in examples
        )
        assert set(counts.values()) == {expected_quota[split]}
        assert set(counts) == {
            (variant, field)
            for variant in ("normal", "missing", "uncertain", "conflicting")
            for field in FIELD_ORDER
        }


def test_every_generated_target_is_strictly_parseable_and_state_correct() -> None:
    for split in ("train", "dev"):
        for example in generate_split(split):
            proposals = parse_state_span_summary(example.target, example.transcript)
            assert len(proposals) == len(FIELD_ORDER)
            if example.target_state is not None:
                target_index = FIELD_ORDER.index(example.target_field)
                assert proposals[target_index].state is example.target_state


def test_static_split_and_historical_isolation() -> None:
    assert_static_split_isolation()
    train = supported_value_sets("train")
    dev = supported_value_sets("dev")
    open_values: set[str] = set()
    for field in FIELD_ORDER:
        if field.value != "severity":
            assert train[field].isdisjoint(dev[field])
            open_values.update(value.casefold() for value in train[field] | dev[field])
    assert open_values.isdisjoint(FORBIDDEN_HISTORICAL_SENTINELS)


def test_generator_has_no_benchmark_import() -> None:
    source = Path(state_span_data.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(
        module.startswith(("nano_ai.benchmark", "nano_ai.benchmarks"))
        for module in imported_modules
    )


def test_manifest_binds_exact_bytes_and_split_identity() -> None:
    train = generate_split("train", worlds=10)
    dev = generate_split("dev", worlds=10)
    manifest = build_manifest(
        train,
        dev,
        generator_sha256=_GENERATOR_SHA256,
        tokenizer_sha256=_TOKENIZER_SHA256,
        base_checkpoint_sha256=_CHECKPOINT_SHA256,
    )
    assert manifest["train"]["records"] == 40
    assert manifest["dev"]["records"] == 40
    assert manifest["train"]["sha256"] == _records_sha256("train", worlds=10)
    assert manifest["dev"]["sha256"] == _records_sha256("dev", worlds=10)
    assert manifest["isolation"] == {
        "worlds_disjoint": True,
        "transcripts_disjoint": True,
        "open_value_lexicons_disjoint": True,
        "question_templates_disjoint": True,
        "answer_templates_disjoint": True,
        "denial_phrases_disjoint": True,
        "uncertainty_phrases_disjoint": True,
        "fresh_v0_read_by_generator": False,
    }


def test_write_load_digest_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(state_span_data, "TRAIN_WORLDS", 10)
    monkeypatch.setattr(state_span_data, "DEV_WORLDS", 10)
    output = tmp_path / "state-span"
    manifest = write_dataset(
        output,
        tokenizer_sha256=_TOKENIZER_SHA256,
        base_checkpoint_sha256=_CHECKPOINT_SHA256,
    )
    train = load_records(
        output / "train.jsonl", expected_sha256=manifest["train"]["sha256"]
    )
    dev = load_records(output / "dev.jsonl", expected_sha256=manifest["dev"]["sha256"])
    assert len(train) == len(dev) == 40
    with pytest.raises(FileExistsError):
        write_dataset(
            output,
            tokenizer_sha256=_TOKENIZER_SHA256,
            base_checkpoint_sha256=_CHECKPOINT_SHA256,
        )
    (output / "train.jsonl").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_records(
            output / "train.jsonl",
            expected_sha256=manifest["train"]["sha256"],
        )


@pytest.mark.parametrize("split", ["bad", ""])
def test_rejects_unknown_split(split: str) -> None:
    with pytest.raises(ValueError, match="split"):
        generate_split(split)


@pytest.mark.parametrize("worlds", [True, 0, 3])
def test_rejects_invalid_world_count(worlds: int) -> None:
    with pytest.raises(ValueError, match="worlds"):
        generate_split("train", worlds=worlds)


def test_uncertainty_targets_are_not_collapsed_to_missing() -> None:
    examples = generate_split("dev", worlds=10)
    uncertain = [example for example in examples if example.variant == "uncertain"]
    assert uncertain
    for example in uncertain:
        proposal = parse_state_span_summary(example.target, example.transcript)[
            FIELD_ORDER.index(example.target_field)
        ]
        assert proposal.state is FieldState.UNCERTAIN
        assert proposal.spans
