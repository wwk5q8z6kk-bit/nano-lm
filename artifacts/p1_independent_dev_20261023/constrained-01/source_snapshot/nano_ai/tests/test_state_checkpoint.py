from __future__ import annotations

import hashlib
import importlib
import sys

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

STATE_SUMMARY = (
    "CC:S[chest pain]|DUR:S[3 days]|SEV:S[moderate]|"
    "MED:A[No, nothing yet.]|ALG:S[penicillin]"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _candidate_artifacts(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import AnchorArtifactSpec

    checkpoint = b"candidate checkpoint"
    tokenizer = b"candidate tokenizer"
    checkpoint_path = tmp_path / "candidate.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path.write_bytes(checkpoint)
    tokenizer_path.write_bytes(tokenizer)
    spec = AnchorArtifactSpec(
        name="nano-native-state-candidate",
        release="h1-test",
        checkpoint_filename="candidate.pt",
        checkpoint_sha256=_sha(checkpoint),
        tokenizer_sha256=_sha(tokenizer),
    )
    return spec, checkpoint_path, tokenizer_path


class _FakeRuntime:
    def __init__(self, summary: str = STATE_SUMMARY) -> None:
        self.summary = summary
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.summary


def test_import_keeps_torch_and_tokenizers_lazy(monkeypatch):
    module_name = "nano_ai.adapters.state_checkpoint"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    torch_before = sys.modules.get("torch")
    tokenizers_before = sys.modules.get("tokenizers")

    importlib.import_module(module_name)

    assert sys.modules.get("torch") is torch_before
    assert sys.modules.get("tokenizers") is tokenizers_before


def test_native_candidate_identity_is_distinct_and_retains_nano_size(tmp_path):
    from nano_ai.adapters.state_checkpoint import (
        STATE_CHECKPOINT_PIPELINE_VERSION,
        STATE_GROUNDING_VERIFIER_ID,
        STATE_PROMPT_TEMPLATE_ID,
        StateCheckpointSolver,
    )

    spec, checkpoint_path, tokenizer_path = _candidate_artifacts(tmp_path)
    solver = StateCheckpointSolver(spec, checkpoint_path, tokenizer_path)

    assert solver.descriptor.kind is SolverKind.HYBRID
    assert solver.descriptor.parameter_count == 3_148_608
    assert solver.descriptor.version == STATE_CHECKPOINT_PIPELINE_VERSION
    assert not solver.generator.loaded
    for component in (
        spec.solver_id,
        f"/arch-{spec.architecture_identity}",
        f"/pipeline-{STATE_CHECKPOINT_PIPELINE_VERSION}",
        f"/prompt-{STATE_PROMPT_TEMPLATE_ID}",
        f"/grounding-{STATE_GROUNDING_VERIFIER_ID}",
        "/max-new-128",
        "/device-cpu",
    ):
        assert component in solver.descriptor.solver_id


def test_candidate_hashes_are_verified_before_loading_and_new_prompt_is_used(
    tmp_path,
):
    from nano_ai.adapters.state_checkpoint import (
        StateCheckpointSolver,
        build_state_span_prompt,
    )

    spec, checkpoint_path, tokenizer_path = _candidate_artifacts(tmp_path)
    runtime = _FakeRuntime()
    calls = []

    def loader(artifacts, *, device, max_new_tokens):
        calls.append((artifacts, device, max_new_tokens))
        return runtime

    solver = StateCheckpointSolver(
        spec,
        checkpoint_path,
        tokenizer_path,
        runtime_loader=loader,
        runtime_id="fixture-native-state-runtime-v0",
        max_new_tokens=96,
    )
    request = NanoInput(item_id="candidate-case", transcript=TRANSCRIPT)

    result = run_inference(solver, request)

    assert result.ok
    assert result.diagnostics is None
    assert len(calls) == 1
    artifacts, device, max_new_tokens = calls[0]
    assert artifacts.checkpoint_bytes == b"candidate checkpoint"
    assert artifacts.tokenizer_bytes == b"candidate tokenizer"
    assert device == "cpu"
    assert max_new_tokens == 96
    assert len(runtime.prompts) == 1
    prompt = runtime.prompts[0]
    assert prompt == build_state_span_prompt(TRANSCRIPT)
    assert prompt.startswith(TRANSCRIPT)
    assert "Use only S[exact Patient span]" in prompt
    assert "Copy every span verbatim" in prompt
    assert prompt.endswith("Summarize the visit.")

    fields = {field.field: field for field in result.output.fields}
    assert fields[FieldName.CHIEF_COMPLAINT].value == "chest pain"
    assert fields[FieldName.MEDICATION].state is FieldState.ABSENT

    _, diagnostics = solver.infer_with_state_diagnostics(request)
    assert diagnostics["protocol_version"] == "state-span-diagnostics-v0"
    assert len(calls) == 1


def test_candidate_hash_mismatch_fails_before_runtime_loader(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import ArtifactIntegrityError
    from nano_ai.adapters.state_checkpoint import StateCheckpointSolver

    spec, checkpoint_path, tokenizer_path = _candidate_artifacts(tmp_path)
    checkpoint_path.write_bytes(b"changed after registration")
    loader_called = False

    def loader(artifacts, *, device, max_new_tokens):
        nonlocal loader_called
        loader_called = True
        return _FakeRuntime()

    solver = StateCheckpointSolver(
        spec,
        checkpoint_path,
        tokenizer_path,
        runtime_loader=loader,
        runtime_id="fixture-native-state-runtime-v0",
    )

    with pytest.raises(ArtifactIntegrityError, match="checkpoint SHA-256 mismatch"):
        solver.infer(NanoInput(item_id="hash-failure", transcript=TRANSCRIPT))
    assert not loader_called
    assert not solver.generator.loaded


def test_state_candidate_rejects_architecture_drift(tmp_path):
    from nano_ai.adapters.anchor_checkpoint import AnchorArtifactSpec
    from nano_ai.adapters.state_checkpoint import StateCheckpointSolver

    checkpoint = b"small checkpoint"
    tokenizer = b"tokenizer"
    checkpoint_path = tmp_path / "small.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path.write_bytes(checkpoint)
    tokenizer_path.write_bytes(tokenizer)
    small = AnchorArtifactSpec(
        name="wrong-size-candidate",
        release="test",
        checkpoint_filename="small.pt",
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

    with pytest.raises(ValueError, match="3,148,608-parameter architecture"):
        StateCheckpointSolver(small, checkpoint_path, tokenizer_path)
