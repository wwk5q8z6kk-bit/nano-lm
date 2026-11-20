from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import pytest

from nano_ai.contract import FieldName, FieldState, NanoInput
from nano_ai.solver import SolverKind, run_inference

TRANSCRIPT = """Doctor: Good morning, what brings you in today?
Patient: I've been having chest pain.
Doctor: How long has this been going on?
Patient: For about 3 days now.
Doctor: How bad would you say it is?
Patient: I'd call it moderate.
Doctor: Have you taken anything for it?
Patient: No, nothing yet.
Doctor: Any allergies I should know about?
Patient: I'm allergic to penicillin."""

SUMMARY = "CC: chest pain | DUR: 3 days | SEV: moderate | MED: none | ALG: penicillin"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fake_spec(checkpoint: bytes, tokenizer: bytes):
    from nano_ai.adapters.anchor_checkpoint import AnchorArtifactSpec

    return AnchorArtifactSpec(
        name="test-anchor",
        release="test-v1",
        checkpoint_filename="test.pt",
        checkpoint_sha256=_sha(checkpoint),
        tokenizer_sha256=_sha(tokenizer),
        model_width=8,
        layer_count=1,
        attention_heads=2,
        kv_heads=1,
        head_width=4,
        feed_forward_width=16,
        vocabulary_size=32,
        sequence_length=64,
    )


def _artifacts(
    tmp_path: Path, checkpoint: bytes = b"checkpoint", tokenizer: bytes = b"tokenizer"
):
    checkpoint_path = tmp_path / "test.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path.write_bytes(checkpoint)
    tokenizer_path.write_bytes(tokenizer)
    return _fake_spec(checkpoint, tokenizer), checkpoint_path, tokenizer_path


class _FakeRuntime:
    def __init__(self, summary: str = SUMMARY) -> None:
        self.summary = summary
        self.received: list[str] = []

    def generate(self, transcript: str) -> str:
        self.received.append(transcript)
        return self.summary


def test_import_does_not_import_historical_script_or_ml_runtimes(monkeypatch):
    module_name = "nano_ai.adapters.anchor_checkpoint"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delitem(sys.modules, "trajectory.rescore_anchors", raising=False)
    torch_before = sys.modules.get("torch")
    tokenizers_before = sys.modules.get("tokenizers")

    importlib.import_module(module_name)

    assert "trajectory.rescore_anchors" not in sys.modules
    assert sys.modules.get("torch") is torch_before
    assert sys.modules.get("tokenizers") is tokenizers_before


def test_registered_artifact_identities_use_full_recorded_hashes():
    from nano_ai.adapters.anchor_checkpoint import (
        FROZEN_NANO_V01,
        FROZEN_SCALE_V01,
        TOKENIZER_SHA256,
    )

    assert FROZEN_NANO_V01.checkpoint_sha256 == (
        "0e4f348eea00c660236cfd9e5bc2d9a71274adfc4d738db6f664664c9a06725b"
    )
    assert FROZEN_SCALE_V01.checkpoint_sha256 == (
        "f5aca5f04bd1045cc158d46a27b84024bb94baa349ed330933631c8b8d5acf0d"
    )
    assert TOKENIZER_SHA256 == (
        "bae49648bfcc4904c50e2f006ee184bd26e74454ee170663e30a8e71640ce3c9"
    )
    assert FROZEN_NANO_V01.checkpoint_filename == "nano_v01_scribe.pt"
    assert FROZEN_NANO_V01.checkpoint_filename != "scribe.pt"
    assert FROZEN_NANO_V01.solver_id != FROZEN_SCALE_V01.solver_id
    assert FROZEN_NANO_V01.checkpoint_sha256 in FROZEN_NANO_V01.solver_id
    assert TOKENIZER_SHA256 in FROZEN_NANO_V01.solver_id
    assert FROZEN_NANO_V01.parameter_count == 3_148_608
    assert FROZEN_SCALE_V01.parameter_count == 10_000_320


def test_generator_verifies_both_snapshots_before_lazy_runtime_load(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import AnchorSummaryGenerator

    spec, checkpoint_path, tokenizer_path = _artifacts(tmp_path)
    calls = []
    runtime = _FakeRuntime()

    def loader(artifacts, *, device, max_new_tokens):
        calls.append((artifacts, device, max_new_tokens))
        return runtime

    generator = AnchorSummaryGenerator(
        spec,
        checkpoint_path,
        tokenizer_path,
        runtime_loader=loader,
        runtime_id="fake-summary-runtime-v1",
        device="cpu",
        max_new_tokens=17,
    )

    assert not generator.loaded
    assert calls == []
    assert generator.generate(TRANSCRIPT) == SUMMARY
    assert generator.loaded
    assert len(calls) == 1
    verified, device, max_new_tokens = calls[0]
    assert verified.checkpoint_bytes == b"checkpoint"
    assert verified.tokenizer_bytes == b"tokenizer"
    assert verified.artifact_identity == spec.artifact_identity
    assert verified.artifact_bytes == len(b"checkpointtokenizer")
    assert device == "cpu"
    assert max_new_tokens == 17
    assert runtime.received == [TRANSCRIPT]

    assert generator.generate(TRANSCRIPT) == SUMMARY
    assert len(calls) == 1


@pytest.mark.parametrize("changed_role", ["checkpoint", "tokenizer"])
def test_hash_mismatch_fails_before_runtime_loader(tmp_path, changed_role):
    from nano_ai.adapters.anchor_checkpoint import (
        AnchorSummaryGenerator,
        ArtifactIntegrityError,
    )

    spec, checkpoint_path, tokenizer_path = _artifacts(tmp_path)
    target = checkpoint_path if changed_role == "checkpoint" else tokenizer_path
    target.write_bytes(b"different bytes")
    loader_called = False

    def loader(artifacts, *, device, max_new_tokens):
        nonlocal loader_called
        loader_called = True
        return _FakeRuntime()

    generator = AnchorSummaryGenerator(
        spec,
        checkpoint_path,
        tokenizer_path,
        runtime_loader=loader,
        runtime_id="fake-summary-runtime-v1",
    )

    with pytest.raises(
        ArtifactIntegrityError, match=f"{changed_role} SHA-256 mismatch"
    ):
        generator.generate(TRANSCRIPT)
    assert not generator.loaded
    assert not loader_called


def test_missing_dependency_fails_closed_after_verification(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import (
        AnchorRuntimeUnavailableError,
        AnchorSummaryGenerator,
    )

    spec, checkpoint_path, tokenizer_path = _artifacts(tmp_path)

    def loader(artifacts, *, device, max_new_tokens):
        raise ModuleNotFoundError("fake optional dependency")

    generator = AnchorSummaryGenerator(
        spec,
        checkpoint_path,
        tokenizer_path,
        runtime_loader=loader,
        runtime_id="missing-dependency-runtime-v1",
    )

    with pytest.raises(
        AnchorRuntimeUnavailableError, match="dependencies are unavailable"
    ):
        generator.generate(TRANSCRIPT)
    assert not generator.loaded


def test_native_anchor_identity_covers_the_full_hybrid_pipeline(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import (
        ANCHOR_PIPELINE_VERSION,
        DECODE_POLICY_ID,
        GROUNDING_VERIFIER_ID,
        NATIVE_RUNTIME_ID,
        PROMPT_TEMPLATE_ID,
        AnchorCheckpointSolver,
    )

    spec, checkpoint_path, tokenizer_path = _artifacts(tmp_path)
    solver = AnchorCheckpointSolver(
        spec,
        checkpoint_path,
        tokenizer_path,
        device="cpu",
        max_new_tokens=17,
    )

    assert solver.descriptor.kind is SolverKind.HYBRID
    assert solver.descriptor.version == ANCHOR_PIPELINE_VERSION
    assert solver.descriptor.parameter_count == spec.parameter_count
    assert solver.descriptor.artifact_bytes is None
    assert not solver.generator.loaded
    for component in (
        spec.solver_id,
        f"/arch-{spec.architecture_identity}",
        f"/pipeline-{ANCHOR_PIPELINE_VERSION}",
        f"/runtime-{NATIVE_RUNTIME_ID}",
        f"/prompt-{PROMPT_TEMPLATE_ID}",
        f"/decode-{DECODE_POLICY_ID}",
        "/max-new-17",
        f"/grounding-{GROUNDING_VERIFIER_ID}",
        "/device-cpu",
    ):
        assert component in solver.descriptor.solver_id


def test_generation_limit_and_device_are_identity_bearing(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import AnchorCheckpointSolver

    spec, checkpoint_path, tokenizer_path = _artifacts(tmp_path)
    base = AnchorCheckpointSolver(
        spec,
        checkpoint_path,
        tokenizer_path,
        device="cpu",
        max_new_tokens=17,
    )
    different_limit = AnchorCheckpointSolver(
        spec,
        checkpoint_path,
        tokenizer_path,
        device="cpu",
        max_new_tokens=18,
    )
    different_device = AnchorCheckpointSolver(
        spec,
        checkpoint_path,
        tokenizer_path,
        device="cuda:0",
        max_new_tokens=17,
    )

    assert base.descriptor.solver_id != different_limit.descriptor.solver_id
    assert base.descriptor.solver_id != different_device.descriptor.solver_id
    assert "/max-new-18" in different_limit.descriptor.solver_id
    assert "/device-cuda%3A0" in different_device.descriptor.solver_id


def test_device_defaults_to_cpu_and_rejects_ambiguous_selection(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import AnchorCheckpointSolver

    spec, checkpoint_path, tokenizer_path = _artifacts(tmp_path)
    default_solver = AnchorCheckpointSolver(spec, checkpoint_path, tokenizer_path)

    assert "/device-cpu" in default_solver.descriptor.solver_id
    assert "/device-auto" not in default_solver.descriptor.solver_id

    for ambiguous_device in (None, "", " auto", "auto", "AUTO", "cpu "):
        with pytest.raises(ValueError, match="device must be an explicit"):
            AnchorCheckpointSolver(
                spec,
                checkpoint_path,
                tokenizer_path,
                device=ambiguous_device,  # type: ignore[arg-type]
            )


def test_injected_runtime_is_a_distinct_legacy_adapter_and_grounds_summary(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import AnchorCheckpointSolver
    from nano_ai.adapters.legacy_summary import LEGACY_DIAGNOSTICS_VERSION

    spec, checkpoint_path, tokenizer_path = _artifacts(tmp_path)
    runtime = _FakeRuntime()
    seen_artifacts = []

    def loader(artifacts, *, device, max_new_tokens):
        seen_artifacts.append(artifacts.artifact_identity)
        return runtime

    solver = AnchorCheckpointSolver(
        spec,
        checkpoint_path,
        tokenizer_path,
        runtime_loader=loader,
        runtime_id="fixture-summary-runtime-v1",
    )
    native_solver = AnchorCheckpointSolver(spec, checkpoint_path, tokenizer_path)
    request = NanoInput(item_id="anchor-case", transcript=TRANSCRIPT)
    descriptor_before = solver.descriptor

    result = run_inference(solver, request)

    assert result.ok
    assert result.diagnostics is not None
    assert result.diagnostics["protocol_version"] == LEGACY_DIAGNOSTICS_VERSION
    assert result.diagnostics["raw_summary"] == SUMMARY
    assert len(result.diagnostics["fields"]) == 5
    assert solver.descriptor == descriptor_before
    assert result.output is not None
    assert result.output.solver_id == descriptor_before.solver_id
    direct_output = solver.infer(request)
    assert direct_output == result.output
    assert solver.descriptor == descriptor_before
    assert solver.descriptor.kind is SolverKind.LEGACY_ADAPTER
    assert solver.descriptor.kind is not SolverKind.TRAINED
    assert solver.descriptor.solver_id != spec.solver_id
    assert solver.descriptor.solver_id != native_solver.descriptor.solver_id
    assert "/runtime-injected-fixture-summary-runtime-v1" in (
        solver.descriptor.solver_id
    )
    assert solver.descriptor.parameter_count is None
    assert solver.descriptor.artifact_bytes is None
    assert seen_artifacts == [spec.artifact_identity]
    assert runtime.received == [TRANSCRIPT, TRANSCRIPT]
    fields = {field.field: field for field in result.output.fields}
    assert fields[FieldName.CHIEF_COMPLAINT].value == "chest pain"
    assert fields[FieldName.MEDICATION].state is FieldState.ABSENT
    assert fields[FieldName.ALLERGY].value == "penicillin"


def test_injected_runtime_requires_an_explicit_safe_identity(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import AnchorCheckpointSolver

    spec, checkpoint_path, tokenizer_path = _artifacts(tmp_path)

    def loader(artifacts, *, device, max_new_tokens):
        return _FakeRuntime()

    with pytest.raises(ValueError, match="runtime_id is required"):
        AnchorCheckpointSolver(
            spec,
            checkpoint_path,
            tokenizer_path,
            runtime_loader=loader,
        )

    for unsafe_id in ("", "Native Runtime", "../native", "runtime:latest", "UPPER"):
        with pytest.raises(ValueError, match="runtime_id must be"):
            AnchorCheckpointSolver(
                spec,
                checkpoint_path,
                tokenizer_path,
                runtime_loader=loader,
                runtime_id=unsafe_id,
            )


def test_native_loader_rejects_a_custom_runtime_identity(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import AnchorCheckpointSolver

    spec, checkpoint_path, tokenizer_path = _artifacts(tmp_path)

    with pytest.raises(ValueError, match="reserved for an explicitly injected"):
        AnchorCheckpointSolver(
            spec,
            checkpoint_path,
            tokenizer_path,
            runtime_id="misleading-native-v1",
        )


def test_verified_artifact_constructor_rejects_mutable_or_mismatched_bytes(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import (
        ArtifactIntegrityError,
        VerifiedAnchorArtifacts,
    )

    checkpoint = b"checkpoint"
    tokenizer = b"tokenizer"
    spec, checkpoint_path, tokenizer_path = _artifacts(
        tmp_path, checkpoint=checkpoint, tokenizer=tokenizer
    )
    valid = VerifiedAnchorArtifacts(
        spec=spec,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        checkpoint_bytes=checkpoint,
        tokenizer_bytes=tokenizer,
    )
    assert valid.artifact_bytes == len(checkpoint) + len(tokenizer)

    with pytest.raises(TypeError, match="checkpoint_bytes must be immutable"):
        VerifiedAnchorArtifacts(
            spec=spec,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            checkpoint_bytes=bytearray(checkpoint),  # type: ignore[arg-type]
            tokenizer_bytes=tokenizer,
        )

    with pytest.raises(ArtifactIntegrityError, match="checkpoint SHA-256 mismatch"):
        VerifiedAnchorArtifacts(
            spec=spec,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            checkpoint_bytes=b"forged-checkpoint",
            tokenizer_bytes=tokenizer,
        )


def test_repository_nano_solver_rejects_a_nonfrozen_scribe_file(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import (
        FROZEN_NANO_V01,
        AnchorCheckpointSolver,
        ArtifactIntegrityError,
    )

    anchor_dir = tmp_path / "checkpoints" / "anchors"
    tokenizer_dir = tmp_path / "sft"
    anchor_dir.mkdir(parents=True)
    tokenizer_dir.mkdir()
    (anchor_dir / FROZEN_NANO_V01.checkpoint_filename).write_bytes(
        b"not the frozen nano anchor"
    )
    # Checkpoint verification happens first, so this file is intentionally not
    # made to match the registered tokenizer in this mismatch regression.
    (tokenizer_dir / "tokenizer.json").write_bytes(b"irrelevant")

    solver = AnchorCheckpointSolver.from_repository("nano", repository_root=tmp_path)

    with pytest.raises(ArtifactIntegrityError) as exc_info:
        solver.generator.generate(TRANSCRIPT)
    assert FROZEN_NANO_V01.checkpoint_sha256 in str(exc_info.value)
    assert not solver.generator.loaded


def test_runtime_must_return_a_nonempty_summary(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import (
        AnchorGenerationError,
        AnchorSummaryGenerator,
    )

    spec, checkpoint_path, tokenizer_path = _artifacts(tmp_path)

    def loader(artifacts, *, device, max_new_tokens):
        return _FakeRuntime("   ")

    generator = AnchorSummaryGenerator(
        spec,
        checkpoint_path,
        tokenizer_path,
        runtime_loader=loader,
        runtime_id="empty-summary-runtime-v1",
    )

    with pytest.raises(AnchorGenerationError, match="returned no summary"):
        generator.generate(TRANSCRIPT)
