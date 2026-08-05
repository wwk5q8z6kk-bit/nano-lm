from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.adapters.state_checkpoint import (
    STATE_PROMPT_TEMPLATE_ID,
    STATE_SPAN_PROMPT_INSTRUCTION,
)
from nano_ai.adapters.state_span import StateSpanSolver
from nano_ai.evaluation import evaluate_solver
from nano_ai.training import evaluate_state_span, state_span_data
from nano_ai.training.evaluate_state_span import (
    CandidateCheckpoint,
    DevelopmentEvaluationError,
    acceptance_diagnostics,
    batched_greedy_generate,
    encode_chatml_prompts,
    load_candidates_from_training_report,
    load_development_bundle,
    raw_state_span_diagnostics,
)
from nano_ai.training.model import NANO_MODEL_CONFIG
from nano_ai.training.state_span_data import (
    DATASET_SCHEMA_VERSION,
    TARGET_GRAMMAR_VERSION,
    build_manifest,
    canonical_json_bytes,
    generate_split,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _small_data_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train = generate_split("train", worlds=5)
    dev = generate_split("dev", worlds=5)
    generator_sha = _sha256(Path(state_span_data.__file__).read_bytes())
    manifest = build_manifest(
        train,
        dev,
        generator_sha256=generator_sha,
        tokenizer_sha256=FROZEN_NANO_V01.tokenizer_sha256,
        base_checkpoint_sha256=FROZEN_NANO_V01.checkpoint_sha256,
    )
    root = tmp_path / "data"
    root.mkdir()
    manifest_bytes = canonical_json_bytes(manifest)
    (root / "manifest.json").write_bytes(manifest_bytes)
    (root / "dev.jsonl").write_bytes(
        b"".join(canonical_json_bytes(example.to_dict()) for example in dev)
    )
    monkeypatch.setattr(evaluate_state_span, "DEV_WORLDS", 5)
    monkeypatch.setattr(evaluate_state_span, "TRAIN_WORLDS", 5)
    monkeypatch.setattr(
        evaluate_state_span,
        "generate_split",
        lambda split: generate_split(split, worlds=5),
    )
    return root, manifest, _sha256(manifest_bytes), dev


def _training_report(
    *, manifest_sha: str, dev_sha: str, seed: int = 20260805
) -> dict[str, object]:
    checkpoints = [
        {
            "filename": "epoch-1.pt",
            "sha256": "1" * 64,
            "bytes": 101,
        },
        {
            "filename": "epoch-2.pt",
            "sha256": "2" * 64,
            "bytes": 102,
        },
        {
            "filename": "candidate.pt",
            "sha256": "3" * 64,
            "bytes": 103,
        },
    ]
    epochs = [
        {
            "epoch": index,
            "train_loss": 1.0 / index,
            "dev_loss": 2.0 / index,
            "seconds": float(index),
            "checkpoint": checkpoint,
        }
        for index, checkpoint in enumerate(checkpoints, 1)
    ]
    return {
        "schema_version": "nano.state-span-training-report.v0",
        "recipe": "nano-native-state-span-sft-v0",
        "status": "complete",
        "seed": seed,
        "device": "cpu",
        "parameter_count": NANO_MODEL_CONFIG.parameter_count,
        "architecture_identity": FROZEN_NANO_V01.architecture_identity,
        "base_checkpoint_sha256": FROZEN_NANO_V01.checkpoint_sha256,
        "tokenizer_sha256": FROZEN_NANO_V01.tokenizer_sha256,
        "dataset_manifest_sha256": manifest_sha,
        "dataset": {
            "schema_version": DATASET_SCHEMA_VERSION,
            "target_grammar": TARGET_GRAMMAR_VERSION,
            "train_sha256": "a" * 64,
            "dev_sha256": dev_sha,
            "train_records": evaluate_state_span.TRAIN_WORLDS * 4,
            "dev_records": evaluate_state_span.DEV_WORLDS * 4,
        },
        "prompt": {
            "template_id": STATE_PROMPT_TEMPLATE_ID,
            "instruction_sha256": _sha256(
                STATE_SPAN_PROMPT_INSTRUCTION.encode("utf-8")
            ),
        },
        "hyperparameters": {},
        "epochs": epochs,
        "candidate": checkpoints[-1],
        "source_sha256": evaluate_state_span._training_source_hashes(),
        "runtime": {},
        "selection_note": "unselected",
    }


def test_development_loader_verifies_manifest_and_dev_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, manifest_sha, dev = _small_data_bundle(tmp_path, monkeypatch)

    bundle = load_development_bundle(root, expected_manifest_sha256=manifest_sha)

    assert bundle.examples == dev
    assert bundle.manifest_sha256 == manifest_sha
    assert bundle.dev_sha256 == manifest["dev"]["sha256"]
    assert not (root / "train.jsonl").exists()

    (root / "dev.jsonl").write_bytes(b"not json\n")
    with pytest.raises(DevelopmentEvaluationError, match="SHA-256 mismatch"):
        load_development_bundle(root, expected_manifest_sha256=manifest_sha)


def test_manifest_digest_is_checked_before_invalid_json_is_parsed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "manifest.json").write_bytes(b"not json")

    with pytest.raises(DevelopmentEvaluationError, match="SHA-256 mismatch"):
        load_development_bundle(root, expected_manifest_sha256="0" * 64)


def test_verified_training_report_yields_all_three_hash_identified_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, manifest, manifest_sha, _dev = _small_data_bundle(tmp_path, monkeypatch)
    report = _training_report(
        manifest_sha=manifest_sha,
        dev_sha=manifest["dev"]["sha256"],
    )
    report_path = tmp_path / "seed" / "training_report.json"
    report_path.parent.mkdir()
    snapshot = canonical_json_bytes(report)
    report_path.write_bytes(snapshot)

    candidates = load_candidates_from_training_report(
        report_path,
        expected_report_sha256=_sha256(snapshot),
        expected_manifest_sha256=manifest_sha,
        expected_dev_sha256=manifest["dev"]["sha256"],
    )

    assert [candidate.label for candidate in candidates] == [
        "seed-20260805-epoch-1",
        "seed-20260805-epoch-2",
        "seed-20260805-epoch-3",
    ]
    assert [candidate.sha256 for candidate in candidates] == [
        "1" * 64,
        "2" * 64,
        "3" * 64,
    ]
    assert candidates[-1].path == report_path.parent / "candidate.pt"
    assert candidates[0].provenance["training_report_sha256"] == _sha256(snapshot)


def test_training_report_hash_precedes_json_and_rejects_unsafe_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, manifest, manifest_sha, _dev = _small_data_bundle(tmp_path, monkeypatch)
    report = _training_report(
        manifest_sha=manifest_sha,
        dev_sha=manifest["dev"]["sha256"],
    )
    report["epochs"][0]["checkpoint"]["filename"] = "../epoch-1.pt"
    report_path = tmp_path / "training_report.json"
    snapshot = canonical_json_bytes(report)
    report_path.write_bytes(snapshot)

    with pytest.raises(DevelopmentEvaluationError, match="SHA-256 mismatch"):
        load_candidates_from_training_report(
            report_path,
            expected_report_sha256="0" * 64,
            expected_manifest_sha256=manifest_sha,
            expected_dev_sha256=manifest["dev"]["sha256"],
        )
    with pytest.raises(DevelopmentEvaluationError, match="filename is unsafe"):
        load_candidates_from_training_report(
            report_path,
            expected_report_sha256=_sha256(snapshot),
            expected_manifest_sha256=manifest_sha,
            expected_dev_sha256=manifest["dev"]["sha256"],
        )


def test_training_report_must_match_current_training_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, manifest, manifest_sha, _dev = _small_data_bundle(tmp_path, monkeypatch)
    report = _training_report(
        manifest_sha=manifest_sha,
        dev_sha=manifest["dev"]["sha256"],
    )
    report["source_sha256"]["training"] = "f" * 64
    report_path = tmp_path / "training_report.json"
    snapshot = canonical_json_bytes(report)
    report_path.write_bytes(snapshot)

    with pytest.raises(DevelopmentEvaluationError, match="executable recipe"):
        load_candidates_from_training_report(
            report_path,
            expected_report_sha256=_sha256(snapshot),
            expected_manifest_sha256=manifest_sha,
            expected_dev_sha256=manifest["dev"]["sha256"],
        )


def test_candidate_requires_safe_label_and_explicit_digest(tmp_path: Path) -> None:
    with pytest.raises(DevelopmentEvaluationError, match="candidate label"):
        CandidateCheckpoint(label="Bad Label", path=tmp_path, sha256="0" * 64)
    with pytest.raises(DevelopmentEvaluationError, match="SHA-256"):
        CandidateCheckpoint(label="good", path=tmp_path, sha256="unchecked")


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    def token_to_id(self, token: str) -> int | None:
        return {"<|im_start|>": 1, "<|im_end|>": 2}.get(token)

    def encode(self, text: str, *, add_special_tokens: bool) -> _FakeEncoding:
        assert add_special_tokens is False
        if text == "user\n":
            return _FakeEncoding([3])
        if text == "assistant\n":
            return _FakeEncoding([4])
        return _FakeEncoding([5] * len(text))

    def decode(self, ids: list[int]) -> str:
        return "ok" if ids == [7] else "?"


class _FakeModel:
    def __init__(self, torch_module) -> None:
        self.torch = torch_module
        self.shapes: list[tuple[int, int]] = []

    def eval(self):
        return self

    def __call__(self, token_ids):
        self.shapes.append(tuple(token_ids.shape))
        batch, sequence = token_ids.shape
        logits = self.torch.zeros((batch, sequence, 8), device=token_ids.device)
        next_ids = self.torch.where(
            token_ids[:, -1].eq(7),
            self.torch.tensor(2, device=token_ids.device),
            self.torch.tensor(7, device=token_ids.device),
        )
        logits[self.torch.arange(batch), sequence - 1, next_ids] = 1
        return logits


def test_batched_generation_groups_equal_prompt_lengths_and_restores_order() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = _FakeTokenizer()
    model = _FakeModel(torch)
    prompts = ((1, 3), (1, 4, 3), (1, 5))

    summaries = batched_greedy_generate(
        model,
        tokenizer,
        prompts,
        device="cpu",
        batch_size=8,
        max_new_tokens=3,
    )

    assert summaries == ("ok", "ok", "ok")
    assert model.shapes == [(2, 2), (2, 3), (1, 3), (1, 4)]


def test_prompt_encoder_uses_exact_chatml_prefix_without_padding() -> None:
    tokenizer = _FakeTokenizer()

    encoded = encode_chatml_prompts(tokenizer, ("ab", "c"))

    assert encoded == ((1, 3, 5, 5, 2, 1, 4), (1, 3, 5, 2, 1, 4))


def test_raw_diagnostics_separate_parse_state_span_and_wrong_presentation() -> None:
    examples = generate_split("dev", worlds=5)
    cases = evaluate_state_span._fixture_cases(examples)
    summaries = [example.target for example in examples]

    perfect = raw_state_span_diagnostics(examples, cases, summaries)

    assert perfect["malformed_items"] == 0
    assert perfect["fields"]["exact_accuracy"] == 1.0
    assert perfect["wrong_presented_field_count"] == 0
    assert perfect["target_challenge"]["missing"]["total"] == 5
    assert perfect["target_challenge"]["missing"]["exact"] == 5
    assert perfect["target_challenge"]["conflicting"]["total"] == 5
    assert perfect["target_challenge"]["conflicting"]["exact"] == 5

    summaries[0] = "malformed"
    malformed = raw_state_span_diagnostics(examples, cases, summaries)
    assert malformed["malformed_items"] == 1
    assert malformed["malformed_rate"] == 1 / 20
    assert malformed["fields"]["parsed"] == 95
    assert malformed["item_diagnostics"][0]["parse_status"] == "malformed"


def test_acceptance_metrics_pin_exact_final_denominators() -> None:
    examples = generate_split("dev", worlds=5)
    cases = evaluate_state_span._fixture_cases(examples)
    predictions = {example.transcript: example.target for example in examples}
    solver = StateSpanSolver(
        predictions.__getitem__, solver_id="perfect-native-state-span"
    )
    report = evaluate_solver(solver, cases)

    acceptance = acceptance_diagnostics(report, examples)
    metrics = acceptance["metrics"]

    assert metrics["overall"] == {"numerator": 100, "denominator": 100, "rate": 1.0}
    assert metrics["held_value"]["numerator"] == metrics["held_value"]["denominator"]
    assert metrics["held_value"]["denominator"] > 0
    assert metrics["missing_target"] == {
        "numerator": 5,
        "denominator": 5,
        "rate": 1.0,
    }
    assert metrics["conflict_target"] == {
        "numerator": 5,
        "denominator": 5,
        "rate": 1.0,
    }
    assert metrics["failures"] == {"numerator": 0, "denominator": 20, "rate": 0.0}
    assert metrics["false_presented"]["numerator"] == 0


def test_no_clobber_report_writer_is_canonical(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    value = {"z": 1, "a": {"b": True}}

    evaluate_state_span._write_json_no_clobber(path, value)

    assert path.read_bytes() == b'{"a":{"b":true},"z":1}\n'
    with pytest.raises(DevelopmentEvaluationError, match="already exists"):
        evaluate_state_span._write_json_no_clobber(path, value)


def test_evaluator_has_no_benchmark_import_or_discovery() -> None:
    source = Path(evaluate_state_span.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert not any(module.startswith("nano_ai.benchmark") for module in imports)
    assert ".rglob(" not in source
    assert ".glob(" not in source

    sensitive_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and ("fresh_v0" in node.value.lower() or "benchmark" in node.value.lower())
    }
    module_docstring = tree.body[0]
    assert isinstance(module_docstring, ast.Expr)
    assert isinstance(module_docstring.value, ast.Constant)
    assert sensitive_literals == {
        module_docstring.value.value,
        "fresh_v0_read_by_generator",
        "historical_benchmark_read",
    }

    prohibited_keys = {"fresh_v0_read_by_generator", "historical_benchmark_read"}
    guarded_false_keys = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant)
        and key.value in prohibited_keys
        and isinstance(value, ast.Constant)
        and value.value is False
    }
    assert guarded_false_keys == prohibited_keys

    file_access_names = {
        "open",
        "read_bytes",
        "read_text",
        "_read_regular_file",
        "_read_verified_file",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else None
        )
        if name not in file_access_names:
            continue
        string_arguments = {
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        }
        assert not any(
            "fresh_v0" in value.lower() or "benchmark" in value.lower()
            for value in string_arguments
        )


def test_cli_exposes_explicit_candidates_and_verified_reports() -> None:
    help_text = evaluate_state_span._parser().format_help()

    assert "--manifest-sha256" in help_text
    assert "--candidate LABEL CHECKPOINT SHA256" in help_text
    assert "--training-report REPORT SHA256" in help_text
    assert "--output" in help_text
