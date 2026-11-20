from __future__ import annotations

import ast
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.adapters.state_span import (
    StateSpanProposal,
    StateSpanSolver,
    parse_state_span_summary,
)
from nano_ai.contract import FIELD_ORDER, EvidenceSpan, FieldState, normalize_value
from nano_ai.evaluation import evaluate_solver
from nano_ai.training import evaluate_pointer
from nano_ai.training.evaluate_pointer import (
    PointerCandidateCheckpoint,
    PointerEvaluationError,
    PointerInferenceInput,
    PointerPrediction,
    batched_pointer_inference,
    decode_pointer_logits,
    load_authenticated_frozen_base,
    load_candidates_from_training_report,
    raw_pointer_diagnostics,
)
from nano_ai.training.evaluate_state_span import (
    DEVELOPMENT_EVALUATION_SCHEMA_VERSION,
    DEVELOPMENT_PARTITION_ID,
    _fixture_cases,
    acceptance_diagnostics,
    final_state_diagnostics,
)
from nano_ai.training.model import NANO_MODEL_CONFIG
from nano_ai.training.pointer_data import (
    POINTER_PROMPT_TEMPLATE_ID,
    POINTER_SUPERVISION_VERSION,
    STATE_ORDER,
)
from nano_ai.training.pointer_model import (
    NANO_POINTER_PARAMETER_COUNT,
    NANO_TRUNK_PARAMETER_COUNT,
    POINTER_HEAD_PARAMETER_COUNT,
)
from nano_ai.training.state_span_data import (
    DATASET_SCHEMA_VERSION,
    TARGET_GRAMMAR_VERSION,
    canonical_json_bytes,
    generate_split,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manual_input(transcript: str, words: tuple[str, ...]) -> PointerInferenceInput:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        start = transcript.index(word, cursor)
        end = start + len(word)
        cursor = end
        offsets.append((start, end))
    return PointerInferenceInput(
        example_id="manual",
        transcript=transcript,
        token_ids=tuple(range(1, len(words) + 1)),
        attention_mask=(True,) * len(words),
        pointer_mask=(True,) * len(words),
        token_offsets=tuple(offsets),
    )


def _state_logits(states: tuple[FieldState, ...]) -> torch.Tensor:
    logits = torch.full((len(FIELD_ORDER), len(STATE_ORDER)), -20.0)
    for field_index, state in enumerate(states):
        logits[field_index, STATE_ORDER.index(state)] = 20.0
    return logits


def test_decoder_maps_states_to_exact_pointer_counts_and_patient_spans() -> None:
    words = ("alpha", "beta", "gamma", "delta", "epsilon")
    item = _manual_input("Doctor: Q?\nPatient: alpha beta gamma delta epsilon", words)
    states = (
        FieldState.SUPPORTED,
        FieldState.ABSENT,
        FieldState.MISSING,
        FieldState.UNCERTAIN,
        FieldState.CONFLICTING,
    )
    start = torch.full((len(words), len(FIELD_ORDER), 2), -20.0)
    end = torch.full_like(start, -20.0)
    assignments = {
        (0, 0): (0, 0),
        (1, 0): (1, 1),
        (3, 0): (2, 2),
        (4, 0): (3, 3),
        (4, 1): (4, 4),
    }
    for (field_index, slot), (first, last) in assignments.items():
        start[first, field_index, slot] = 20.0
        end[last, field_index, slot] = 20.0

    proposals = decode_pointer_logits(item, _state_logits(states), start, end)

    assert tuple(proposal.state for proposal in proposals) == states
    assert tuple(len(proposal.spans) for proposal in proposals) == (1, 1, 0, 1, 2)
    assert [span.text for proposal in proposals for span in proposal.spans] == list(
        words
    )
    assert all(
        span.speaker == "patient" for proposal in proposals for span in proposal.spans
    )


def test_decoder_forbids_cross_turn_span_even_when_its_boundary_score_is_best() -> None:
    words = ("alpha", "beta", "gamma", "delta")
    item = _manual_input(
        "Doctor: Q1?\nPatient: alpha beta\nDoctor: Q2?\nPatient: gamma delta",
        words,
    )
    states = (FieldState.SUPPORTED,) + (FieldState.MISSING,) * 4
    start = torch.full((len(words), len(FIELD_ORDER), 2), -100.0)
    end = torch.full_like(start, -100.0)
    # beta -> gamma would win without the same-Patient-turn constraint.
    start[1, 0, 0] = 100.0
    end[2, 0, 0] = 100.0
    start[0, 0, 0] = 10.0
    end[0, 0, 0] = 10.0

    proposals = decode_pointer_logits(item, _state_logits(states), start, end)

    assert proposals[0].spans[0].text == "alpha"
    assert "beta\nDoctor" not in proposals[0].spans[0].text


def test_decoder_accepts_patient_boundary_token_that_absorbs_outer_space() -> None:
    transcript = "Patient: No allergies."
    item = PointerInferenceInput(
        example_id="leading-space-token",
        transcript=transcript,
        token_ids=(1, 2),
        attention_mask=(True, True),
        pointer_mask=(True, True),
        token_offsets=((8, 11), (11, 22)),
    )
    states = (FieldState.SUPPORTED,) + (FieldState.MISSING,) * 4
    start = torch.full((2, len(FIELD_ORDER), 2), -20.0)
    end = torch.full_like(start, -20.0)
    start[0, 0, 0] = 20.0
    end[1, 0, 0] = 20.0

    proposals = decode_pointer_logits(item, _state_logits(states), start, end)

    assert proposals[0].spans == (
        EvidenceSpan(start=9, end=22, text="No allergies.", speaker="patient"),
    )


def test_decoder_allows_one_patient_span_to_support_multiple_fields() -> None:
    item = _manual_input("Patient: alpha", ("alpha",))
    states = (FieldState.SUPPORTED, FieldState.SUPPORTED) + (FieldState.MISSING,) * 3
    start = torch.zeros((1, len(FIELD_ORDER), 2))
    end = torch.zeros_like(start)

    proposals = decode_pointer_logits(item, _state_logits(states), start, end)

    assert proposals[0].spans == proposals[1].spans
    assert proposals[0].spans[0].text == "alpha"


class _MissingModel:
    def __init__(self) -> None:
        self.batch_shapes: list[tuple[int, int]] = []

    def eval(self) -> _MissingModel:
        return self

    def __call__(
        self, token_ids: torch.Tensor, *, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        self.batch_shapes.append(tuple(token_ids.shape))
        batch, tokens = token_ids.shape
        states = torch.full((batch, len(FIELD_ORDER), len(STATE_ORDER)), -1.0)
        states[:, :, STATE_ORDER.index(FieldState.MISSING)] = 1.0
        pointers = torch.zeros((batch, tokens, len(FIELD_ORDER), 2))
        assert attention_mask.dtype is torch.bool
        return SimpleNamespace(
            state_logits=states,
            start_logits=pointers,
            end_logits=pointers.clone(),
        )


def test_batched_direct_inference_right_pads_and_restores_input_order() -> None:
    first = _manual_input("Patient: alpha beta", ("alpha", "beta"))
    second = replace(_manual_input("Patient: gamma", ("gamma",)), example_id="manual-2")
    model = _MissingModel()

    predictions = batched_pointer_inference(
        model, (first, second), device="cpu", batch_size=2
    )

    assert model.batch_shapes == [(2, 2)]
    assert len(predictions) == 2
    assert all(prediction.error is None for prediction in predictions)
    assert all(
        proposal.state is FieldState.MISSING
        for prediction in predictions
        for proposal in prediction.proposals or ()
    )


def test_raw_scores_precede_pointer_verifier_gating() -> None:
    examples = generate_split("dev", worlds=5)
    cases = _fixture_cases(examples)
    perfect = tuple(
        PointerPrediction(
            proposals=parse_state_span_summary(example.target, example.transcript)
        )
        for example in examples
    )

    raw = raw_pointer_diagnostics(examples, cases, perfect)

    assert raw["decode_failure_items"] == 0
    assert raw["fields"]["state_accuracy"] == 1.0
    assert raw["fields"]["span_exact_accuracy"] == 1.0
    assert raw["fields"]["joint_exact_accuracy"] == 1.0
    assert raw["acceptance"]["metrics"]["overall"] == {
        "numerator": 100,
        "denominator": 100,
        "rate": 1.0,
    }
    assert raw["acceptance"]["metrics"]["uncertain_target"] == {
        "numerator": 5,
        "denominator": 5,
        "rate": 1.0,
    }

    first = perfect[0].proposals
    assert first is not None
    original = first[0].spans[0]
    wrong = EvidenceSpan(
        start=original.start,
        end=original.end - 1,
        text=examples[0].transcript[original.start : original.end - 1],
        speaker="patient",
    )
    changed = StateSpanProposal(
        field=first[0].field,
        state_code=first[0].state_code,
        state=first[0].state,
        spans=(wrong,),
    )
    predictions = (PointerPrediction(proposals=(changed, *first[1:])), *perfect[1:])
    inputs = tuple(
        PointerInferenceInput(
            example_id=example.example_id,
            transcript=example.transcript,
            token_ids=(1,),
            attention_mask=(True,),
            pointer_mask=(False,),
            token_offsets=(None,),
        )
        for example in examples
    )
    candidate = PointerCandidateCheckpoint(
        label="test", path=Path("unused.pt"), sha256="a" * 64
    )

    gated = evaluate_pointer._evaluate_predictions(
        inputs=inputs,
        predictions=predictions,
        cases=cases,
        candidate=candidate,
    )

    assert gated.items[0]["status"] == "ok"
    assert gated.items[0]["field_results"][0]["predicted_state"] == "uncertain"
    assert gated.items[0]["field_results"][0]["grounded_exact"] is False


def test_full_sealed_oracle_labels_are_realizable_under_final_contract() -> None:
    examples = generate_split("dev")
    cases = _fixture_cases(examples)
    predictions = tuple(
        PointerPrediction(
            proposals=parse_state_span_summary(example.target, example.transcript)
        )
        for example in examples
    )
    inputs = tuple(
        PointerInferenceInput(
            example_id=example.example_id,
            transcript=example.transcript,
            token_ids=(1,),
            attention_mask=(True,),
            pointer_mask=(False,),
            token_offsets=(None,),
        )
        for example in examples
    )
    candidate = PointerCandidateCheckpoint(
        label="sealed-oracle", path=Path("unused.pt"), sha256="f" * 64
    )
    casing_differences = sum(
        1
        for prediction, case in zip(predictions, cases, strict=True)
        for proposal in prediction.proposals or ()
        if proposal.state is FieldState.SUPPORTED
        and proposal.spans[0].text != case.gold.field(proposal.field).value
        and normalize_value(proposal.spans[0].text)
        == case.gold.field(proposal.field).value
    )

    raw = raw_pointer_diagnostics(examples, cases, predictions)
    final = evaluate_pointer._evaluate_predictions(
        inputs=inputs,
        predictions=predictions,
        cases=cases,
        candidate=candidate,
    )

    assert casing_differences == 83
    assert raw["acceptance"]["metrics"]["overall"] == {
        "numerator": 5_000,
        "denominator": 5_000,
        "rate": 1.0,
    }
    assert raw["wrong_presented_field_count"] == 0
    assert final.quality["grounded_exact_field_count"] == 5_000
    assert final.quality["grounded_exact_field_accuracy"] == 1.0
    assert final.quality["false_presented_count"] == 0


def _pointer_training_report(
    *, manifest_sha: str, train_sha: str, dev_sha: str, seed: int = 20260805
) -> dict[str, object]:
    checkpoints = [
        {"filename": "epoch-1.pt", "sha256": "1" * 64, "bytes": 101},
        {"filename": "epoch-2.pt", "sha256": "2" * 64, "bytes": 102},
        {"filename": "candidate.pt", "sha256": "3" * 64, "bytes": 103},
    ]
    epochs = [
        {
            "epoch": epoch,
            "train_loss": 2.0 / epoch,
            "state_loss": 1.0 / epoch,
            "pointer_loss": 1.0 / epoch,
            "dev_loss": 2.0 / epoch,
            "dev_state_loss": 1.0 / epoch,
            "dev_pointer_loss": 1.0 / epoch,
            "seconds": float(epoch),
            "checkpoint": checkpoint,
        }
        for epoch, checkpoint in enumerate(checkpoints, 1)
    ]
    steps = (evaluate_pointer.TRAIN_WORLDS * 4 + 31) // 32
    return {
        "schema_version": "nano.pointer-span-training-report.v0",
        "recipe": "nano-pointer-span-supervision-v0",
        "status": "complete",
        "seed": seed,
        "device": "cpu",
        "parameter_count": NANO_POINTER_PARAMETER_COUNT,
        "trunk_parameter_count": NANO_TRUNK_PARAMETER_COUNT,
        "pointer_head_parameter_count": POINTER_HEAD_PARAMETER_COUNT,
        "architecture_identity": FROZEN_NANO_V01.architecture_identity,
        "base_checkpoint_sha256": FROZEN_NANO_V01.checkpoint_sha256,
        "tokenizer_sha256": FROZEN_NANO_V01.tokenizer_sha256,
        "dataset_manifest_sha256": manifest_sha,
        "dataset": {
            "schema_version": DATASET_SCHEMA_VERSION,
            "target_grammar": TARGET_GRAMMAR_VERSION,
            "train_sha256": train_sha,
            "dev_sha256": dev_sha,
            "train_records": evaluate_pointer.TRAIN_WORLDS * 4,
            "dev_records": evaluate_pointer.DEV_WORLDS * 4,
        },
        "hyperparameters": {
            "epochs": 3,
            "batch_size": 32,
            "paired_variants_per_world": 4,
            "peak_learning_rate": 1.5e-4,
            "warmup_fraction": 0.03,
            "cosine_floor": 0.1,
            "weight_decay": 0.1,
            "gradient_clip": 1.0,
            "adam_betas": [0.9, 0.95],
            "adam_epsilon": 1e-8,
            "steps_per_epoch": steps,
            "total_steps": steps * 3,
            "state_class_order": [state.value for state in STATE_ORDER],
            "state_class_counts": {
                "supported": 46_050,
                "absent": 4_950,
                "missing": 3_000,
                "uncertain": 3_000,
                "conflicting": 3_000,
            },
            "state_class_weights": [
                0.26058631921824105,
                2.4242424242424243,
                4.0,
                4.0,
                4.0,
            ],
            "state_loss_weight": 1.0,
            "state_loss_definition": (
                "train_inverse_frequency_weighted_cross_entropy_mean_by_weight_mass"
            ),
            "pointer_loss_weight": 1.0,
            "pointer_loss_definition": (
                "mean_of_start_and_end_cross_entropy_active_slots"
            ),
            "patient_token_masked": True,
            "prompt_template_id": POINTER_PROMPT_TEMPLATE_ID,
            "supervision_version": POINTER_SUPERVISION_VERSION,
            "uncertain_pointer_count": 1,
            "causal_pointer_heads": True,
            "deterministic_algorithms": True,
            "full_trunk_trainable": True,
            "world_grouped_batches": True,
        },
        "epochs": epochs,
        "candidate": checkpoints[-1],
        "source_sha256": evaluate_pointer._training_source_hashes(),
        "runtime": {
            "python": "3.test",
            "torch": "test",
            "tokenizers": "test",
            "cuda": None,
            "gpu": None,
            "cublas_workspace_config": None,
            "platform": "test",
            "seconds": 1.0,
        },
        "selection_note": (
            "This is an unselected H2 development candidate. Historical fresh-v0 "
            "and the sealed fresh-v1 confirmation partition were not read. Causal "
            "pointer logits at earlier transcript tokens can use prefix context only."
        ),
    }


def test_training_report_authenticates_recipe_sources_and_all_epochs(
    tmp_path: Path,
) -> None:
    manifest_sha, train_sha, dev_sha = "a" * 64, "b" * 64, "c" * 64
    report = _pointer_training_report(
        manifest_sha=manifest_sha, train_sha=train_sha, dev_sha=dev_sha
    )
    path = tmp_path / "seed" / "training_report.json"
    path.parent.mkdir()
    snapshot = canonical_json_bytes(report)
    path.write_bytes(snapshot)

    candidates = load_candidates_from_training_report(
        path,
        expected_report_sha256=_sha256(snapshot),
        expected_manifest_sha256=manifest_sha,
        expected_train_sha256=train_sha,
        expected_dev_sha256=dev_sha,
    )

    assert [candidate.label for candidate in candidates] == [
        "seed-20260805-epoch-1",
        "seed-20260805-epoch-2",
        "seed-20260805-epoch-3",
    ]
    assert candidates[-1].path == path.parent / "candidate.pt"
    assert candidates[0].provenance["training_source_sha256"] == report["source_sha256"]

    with pytest.raises(PointerEvaluationError, match="SHA-256 mismatch"):
        load_candidates_from_training_report(
            path,
            expected_report_sha256="0" * 64,
            expected_manifest_sha256=manifest_sha,
            expected_train_sha256=train_sha,
            expected_dev_sha256=dev_sha,
        )


def _frozen_h1_report(examples):
    cases = _fixture_cases(examples)
    summaries = {example.transcript: example.target for example in examples}
    standard = evaluate_solver(
        StateSpanSolver(
            summaries.__getitem__,
            solver_id="frozen-base-test",
            parameter_count=NANO_MODEL_CONFIG.parameter_count,
        ),
        cases,
        measure_latency=False,
    )
    return {
        "schema_version": DEVELOPMENT_EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "partition": {
            "partition_id": DEVELOPMENT_PARTITION_ID,
            "manifest_sha256": "d" * 64,
            "development_sha256": "e" * 64,
            "records": len(examples),
            "worlds": len({example.world_id for example in examples}),
            "target_grammar": TARGET_GRAMMAR_VERSION,
            "historical_benchmark_read": False,
        },
        "artifacts": {
            "frozen_base_checkpoint_sha256": FROZEN_NANO_V01.checkpoint_sha256,
            "tokenizer_sha256": FROZEN_NANO_V01.tokenizer_sha256,
            "architecture_identity": FROZEN_NANO_V01.architecture_identity,
            "parameter_count": NANO_MODEL_CONFIG.parameter_count,
        },
        "protocol": {},
        "runtime": {},
        "source_sha256": {},
        "frozen_base": {
            "checkpoint_sha256": FROZEN_NANO_V01.checkpoint_sha256,
            "evaluation": standard.to_dict(),
            "final_state": final_state_diagnostics(standard, examples),
            "acceptance": acceptance_diagnostics(standard, examples),
        },
        "candidates": [],
        "selection_boundary": "metric-only authenticated reference",
    }


def test_frozen_base_is_digest_pinned_and_metrics_are_recomputed(
    tmp_path: Path,
) -> None:
    examples = generate_split("dev", worlds=5)
    report = _frozen_h1_report(examples)
    path = tmp_path / "development_evaluation.json"
    snapshot = canonical_json_bytes(report)
    path.write_bytes(snapshot)

    authenticated = load_authenticated_frozen_base(
        path,
        expected_report_sha256=_sha256(snapshot),
        manifest_sha256="d" * 64,
        development_sha256="e" * 64,
        examples=examples,
    )

    assert authenticated.report_sha256 == _sha256(snapshot)
    assert authenticated.evaluation["quality"]["grounded_exact_field_accuracy"] == 1.0

    cases = _fixture_cases(examples)
    predictions = tuple(
        PointerPrediction(
            proposals=parse_state_span_summary(example.target, example.transcript)
        )
        for example in examples
    )
    inputs = tuple(
        PointerInferenceInput(
            example_id=example.example_id,
            transcript=example.transcript,
            token_ids=(1,),
            attention_mask=(True,),
            pointer_mask=(False,),
            token_offsets=(None,),
        )
        for example in examples
    )
    candidate = PointerCandidateCheckpoint(
        label="perfect", path=Path("unused.pt"), sha256="f" * 64
    )
    standard = evaluate_pointer._evaluate_predictions(
        inputs=inputs,
        predictions=predictions,
        cases=cases,
        candidate=candidate,
    )
    raw = raw_pointer_diagnostics(examples, cases, predictions)
    comparison = evaluate_pointer._comparison_to_base(
        standard,
        raw,
        final_state_diagnostics(standard, examples),
        acceptance_diagnostics(standard, examples),
        authenticated,
    )
    raw_overall = comparison["raw_eligibility"]["gate_evidence"][
        "raw_overall_gain_at_least_5pp"
    ]
    assert raw_overall["candidate"] == {
        "numerator": 100,
        "denominator": 100,
        "rate": 1.0,
    }
    assert raw_overall["threshold_rate"] == 1.05
    assert (
        comparison["raw_eligibility"]["gates"]["all_raw_eligibility_gates_pass"]
        is False
    )
    assert comparison["quality_eligibility"]["all_quality_gates_pass"] is False

    report["frozen_base"]["final_state"]["by_gold_state"]["supported"][
        "state_accuracy"
    ] = 0.0
    tampered = canonical_json_bytes(report)
    path.write_bytes(tampered)
    with pytest.raises(PointerEvaluationError, match="final-state metrics"):
        load_authenticated_frozen_base(
            path,
            expected_report_sha256=_sha256(tampered),
            manifest_sha256="d" * 64,
            development_sha256="e" * 64,
            examples=examples,
        )


def test_report_writer_is_canonical_and_never_clobbers(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    evaluate_pointer._write_json_no_clobber(path, {"z": 1, "a": {"b": True}})

    assert path.read_bytes() == b'{"a":{"b":true},"z":1}\n'
    with pytest.raises(PointerEvaluationError, match="already exists"):
        evaluate_pointer._write_json_no_clobber(path, {"different": True})


def test_evaluator_has_no_benchmark_import_or_partition_discovery() -> None:
    source = Path(evaluate_pointer.__file__).read_text(encoding="utf-8")
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
    assert ".glob(" not in source
    assert ".rglob(" not in source
    help_text = evaluate_pointer._parser().format_help()
    assert "--frozen-base-report-sha256" in help_text
    assert "--training-report REPORT SHA256" in help_text
    assert "--output" in help_text
