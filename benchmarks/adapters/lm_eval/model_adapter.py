"""Model and deterministic solver adapters for Program 0 smoke (no GPU)."""

from __future__ import annotations

import re
from typing import Any, Protocol

from .schemas import SCHEMA_VERSION, ModelManifest, SolverManifest, sha256_hex
from .task_adapter import format_target


class Predictor(Protocol):
    def predict(self, doc: dict[str, Any]) -> str: ...


class MockedModel:
    """Always emits empty string — proves failure logging path."""

    model_id = "mock_empty_v1"

    def predict(self, doc: dict[str, Any]) -> str:
        return ""


class DeterministicTemplateSolver:
    """Classical baseline: rebuild target line from gold tuple fields.

    For Program 0 infrastructure smoke this is intentionally perfect on the
    bound fixture (information-parity with gold labels). It exists to prove
    classical-baseline wiring, not to claim a new scientific result.
    """

    solver_id = "deterministic_template_from_tuple_v1"

    def predict(self, doc: dict[str, Any]) -> str:
        return format_target(doc["tuple"])


_SLOT_FROM_DIALOGUE = {
    "cc": re.compile(
        r"Patient: (?:Honestly,\s+a\s+|It started as\s+)(.+?)(?:\s+has been troubling me\.|\s+and hasn't stopped\.)"
    ),
    "dur": re.compile(r"Patient: I'd say it's been (.+?)\."),
    "sev": re.compile(r"Patient: Definitely (.+?)\."),
    "med": re.compile(r"Patient: Only (.+?) so far\."),
    "alg": re.compile(
        r"Patient: (?:None whatsoever\.|I do — (.+?)\.)"
    ),
}


class DeterministicDialogueExtractor:
    """Extract slots from dialogue text (no gold tuple)."""

    solver_id = "deterministic_dialogue_extractor_v1"

    def predict(self, doc: dict[str, Any]) -> str:
        text = doc["doc_to_text"]
        fields: dict[str, str] = {}
        for key, pat in _SLOT_FROM_DIALOGUE.items():
            m = pat.search(text)
            if not m:
                fields[key] = ""
                continue
            if key == "alg":
                fields[key] = m.group(1) if m.lastindex and m.group(1) else "none"
            else:
                fields[key] = m.group(1).strip()
        return format_target(fields)


def mocked_model_manifest() -> ModelManifest:
    return ModelManifest(
        schema_version=SCHEMA_VERSION,
        model_id=MockedModel.model_id,
        model_family="mock",
        architecture="none",
        parameter_count=0,
        checkpoint_sha256=None,
        tokenizer_sha256=None,
        training_token_metadata={"known": False},
        adapter_or_finetune_state="none",
        quantization="none",
        backend="python_mock",
        decoding_configuration={"temperature": 0.0, "max_new_tokens": 0},
        resource_class_ids=["rc_micro"],
    )


def deterministic_solver_manifest() -> SolverManifest:
    impl_src = (
        "DeterministicTemplateSolver.v1:"
        "return format_target(doc['tuple'])"
    )
    return SolverManifest(
        schema_version=SCHEMA_VERSION,
        solver_id=DeterministicTemplateSolver.solver_id,
        solver_family="classical_template",
        method="format_target_from_tuple",
        implementation_hash=sha256_hex(impl_src),
        resource_class_ids=["rc_classical_baseline"],
        notes=(
            "Program 0 classical baseline wiring. Not a scientific claim; "
            "uses gold tuple fields for exact reconstruction."
        ),
    )
