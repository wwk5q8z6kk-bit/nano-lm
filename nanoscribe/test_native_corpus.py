"""Corpus factory pins.

These guard the properties that make a corpus trustworthy: no frozen-eval value
ever reaches training, partitions never share values, the loader actually reads
the file, and the label distribution is not degenerate.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.native.corpus import vocab
from nanoscribe.native.corpus.adversarial import generate_adversarial
from nanoscribe.native.corpus.generators import generate_mechanism
from nanoscribe.native.corpus.schema import CORPUS_SCHEMA, FIXTURE_SCHEMA, Partition
from nanoscribe.native.corpus.validate import (
    axis_coverage_floor,
    check_leakage,
    dedupe,
    statistics,
)


def _sample(partition: Partition, n_composed: int = 40):
    return list(generate_mechanism(partition, limit_composed=n_composed)) + list(
        generate_adversarial(partition, limit_composed=n_composed // 2)
    )


def test_frozen_eval_snapshot_matches_live_suite() -> None:
    """The reserved vocabulary must be DERIVED, never drift from the suite.

    A hand-authored version of this list once invented 14 values and omitted 14
    real ones, three of which are in the generation pools — a direct leak.
    """
    live = vocab._live_frozen_eval_values()
    if live is None:  # suite unavailable in a minimal env
        return
    snapshot = frozenset(v.lower() for v in vocab._FROZEN_EVAL_SNAPSHOT)
    assert live == snapshot, (
        "frozen-eval snapshot drifted from the live suite; "
        f"missing={sorted(live - snapshot)} extra={sorted(snapshot - live)}"
    )


def test_generation_never_uses_reserved_values() -> None:
    for partition in (Partition.TRAIN, Partition.DEV):
        for ex in _sample(partition):
            assert ex.raw_value.lower() not in vocab.forbidden_values(), (
                f"{ex.encounter_id} generated a value reserved to the frozen eval suite"
            )


def test_partitions_do_not_share_values() -> None:
    train = {e.raw_value.lower() for e in _sample(Partition.TRAIN)}
    dev = {e.raw_value.lower() for e in _sample(Partition.DEV)}
    test = {e.raw_value.lower() for e in _sample(Partition.INTERNAL_TEST)}
    assert not train & dev, f"train/dev value overlap: {sorted(train & dev)[:5]}"
    assert not train & test, f"train/test value overlap: {sorted(train & test)[:5]}"


def test_leakage_check_passes_on_generated_sample() -> None:
    examples, _ = dedupe(_sample(Partition.TRAIN) + _sample(Partition.DEV))
    result = check_leakage(examples)
    assert result["pass"], result


def test_prompts_use_canonical_builder_format() -> None:
    """Training prompts must match the evaluator's format exactly."""
    for ex in _sample(Partition.TRAIN)[:20]:
        assert ex.prompt.startswith("Transcript:\n")
        assert "\n\nQuestion: regarding " in ex.prompt
        assert "Answer on one line" in ex.prompt


def test_target_labels_are_not_degenerate() -> None:
    """No single label may dominate; an imbalanced corpus teaches always-abstain."""
    stats = statistics(_sample(Partition.TRAIN))
    hist = stats["target_label_histogram"]
    total = sum(hist.values())
    assert total > 0
    assert max(hist.values()) / total < 0.60, f"label distribution too skewed: {hist}"
    assert set(hist) >= {"ASSERTED", "DENIED", "UNCERTAIN", "NOT_MENTIONED"}


def test_corpus_schema_is_not_the_regenerating_fixture_schema() -> None:
    """load_train_examples ignores file contents for FIXTURE_SCHEMA.

    Emitting the corpus under that schema would silently discard it and train on
    the 96-row fixture instead.
    """
    assert CORPUS_SCHEMA != FIXTURE_SCHEMA


def test_dedupe_removes_repeats() -> None:
    sample = _sample(Partition.TRAIN)[:50]
    kept, stats = dedupe(sample + sample)
    assert len(kept) == len(sample)
    assert stats["removed_exact_duplicates"] == len(sample)


def test_axis_coverage_reports_shortfalls() -> None:
    result = axis_coverage_floor(_sample(Partition.TRAIN), minimum=10**6)
    assert not result["pass"]
    assert result["below_floor"], "an unreachable floor must report the shortfall"


def test_corpus_build_is_deterministic() -> None:
    """The corpus is not committed; determinism is what makes that safe.

    The generator and the manifest content_hash are the durable artifacts. If a
    rebuild did not reproduce the hash, the manifest would describe a corpus that
    no longer exists and the pod would train on something different from what was
    validated.
    """
    import subprocess
    import sys as _sys

    root = Path(__file__).resolve().parents[1]
    build = root / "scripts" / "build_native_corpus.py"
    proc = subprocess.run(
        [_sys.executable, str(build), "--stage", "smoke", "--manifest-only"],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stderr[-400:]
    import json as _json

    first = _json.loads(
        (root / "artifacts/campaign/native_corpus_smoke_v1_manifest.json").read_text()
    )["content_hash"]
    subprocess.run(
        [_sys.executable, str(build), "--stage", "smoke", "--manifest-only"],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    second = _json.loads(
        (root / "artifacts/campaign/native_corpus_smoke_v1_manifest.json").read_text()
    )["content_hash"]
    assert first == second, "corpus build is not deterministic; the manifest hash cannot be trusted"


def test_guard_blocks_unit_fixture_from_scientific_use() -> None:
    """The 4.5KB fixture must fail before launch, not after conclusions are drawn."""
    import pytest as _pytest

    from nanoscribe.native.corpus.guard import (
        CorpusGuardError,
        assert_usable,
        classify,
    )

    fixture = {"corpus_id": "native_unit_overfit_fixture_v1", "training_tokens_char_level": 4516}
    assert classify(fixture) == "NATIVE_UNIT_OVERFIT_FIXTURE"

    # allowed fixture purposes pass
    for purpose in ("trainer_smoke", "checkpoint_resume_test", "qlora_compatibility_canary"):
        assert_usable(fixture, purpose)

    # scientific purposes must raise
    for purpose in ("architecture_ranking", "promotion", "held_out_generalization"):
        with _pytest.raises(CorpusGuardError):
            assert_usable(fixture, purpose)


def test_guard_allows_real_corpus_for_science() -> None:
    from nanoscribe.native.corpus.guard import assert_usable, classify

    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/campaign/native_corpus_screen_v1_manifest.json"
    )
    if not manifest_path.is_file():
        return
    import json as _json

    manifest = _json.loads(manifest_path.read_text())
    assert classify(manifest) == "NATIVE_TRAINING_CORPUS"
    assert_usable(manifest, "architecture_ranking")


def test_guard_rejects_undersized_corpus_even_if_mislabelled() -> None:
    """A label is not evidence — size is checked independently."""
    import pytest as _pytest

    from nanoscribe.native.corpus.guard import CorpusGuardError, assert_usable

    lying = {
        "corpus_id": "totally_a_real_corpus",
        "corpus_class": "NATIVE_TRAINING_CORPUS",
        "training_tokens_char_level": 4516,
    }
    with _pytest.raises(CorpusGuardError):
        assert_usable(lying, "promotion")


def test_train_cli_does_not_override_registered_dataset() -> None:
    """`--dataset` must not silently replace a run's registered corpus.

    It previously defaulted to the 96-row fixture path and was applied
    unconditionally, so `--run-id reval30_*` had its architecture-screen corpus
    swapped for the unit fixture. The corpus launch guard then refused the run —
    correctly, but only after a paid pod was already up.
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts" / "train_native_nano.py").read_text()
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        if not (node.args and getattr(node.args[0], "value", "") == "--dataset"):
            continue
        default = next((kw.value for kw in node.keywords if kw.arg == "default"), None)
        assert default is not None, "--dataset must declare an explicit default"
        assert getattr(default, "value", None) in ("", None), (
            "--dataset must default to empty so the run's registered corpus wins; "
            "a fixture path default silently downgrades every run"
        )
        break
    else:  # pragma: no cover
        raise AssertionError("--dataset argument not found in train_native_nano.py")

    # and the override must be conditional
    assert "if args.dataset:" in src, "--dataset override must be conditional on the flag being set"
