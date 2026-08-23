"""Model adapter implementations — proposals only, never trusted evidence.

Fixture adapters simulate Qwen2.5-1.5B-class span-port one-liners for CI.
Qwen25BaselineAdapter is a weight-loading skeleton; no weights ship in git.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from nanoscribe.adapt import (
    CandidateAtom,
    ModelCandidateBatch,
    ModelInput,
    candidate_from_span_port_line,
)
from nanoscribe.encounter import AtomType, Experiencer, Speaker, TemporalState
from nanoscribe.qwen_inference import DEFAULT_QWEN_MODEL, generate_span_port_lines, resolve_weights_path


@dataclass(frozen=True, slots=True)
class AtomSpec:
    """One clinical atom slot the adapter should propose for an encounter."""

    atom_id: str
    atom_type: AtomType
    raw_value: str
    speaker: Speaker = Speaker.PATIENT
    experiencer: Experiencer = Experiencer.PATIENT
    temporality: TemporalState | None = None


@runtime_checkable
class ModelAdapter(Protocol):
    """Model-independent adapter protocol: source in, typed candidates out."""

    model_id: str

    def propose(
        self,
        model_input: ModelInput,
        atom_specs: Sequence[AtomSpec],
    ) -> ModelCandidateBatch:
        """Return quote-only candidates. Must not emit offsets or evidence IDs."""


@dataclass(frozen=True, slots=True)
class FixtureSpanPortAdapter:
    """Deterministic span-port baseline using pre-recorded one-line answers."""

    model_id: str = "fixture/qwen2.5-1.5b-span-port"
    lines: Mapping[str, str] | None = None

    def propose(
        self,
        model_input: ModelInput,
        atom_specs: Sequence[AtomSpec],
    ) -> ModelCandidateBatch:
        del model_input  # fixture ignores prompt; source is bound downstream
        lines = self.lines or {}
        atoms: list[CandidateAtom] = []
        for spec in atom_specs:
            raw_line = lines.get(spec.atom_id, "NOT_MENTIONED")
            atoms.append(
                candidate_from_span_port_line(
                    atom_id=spec.atom_id,
                    atom_type=spec.atom_type,
                    raw_value=spec.raw_value,
                    raw_line=raw_line,
                    speaker=spec.speaker,
                    experiencer=spec.experiencer,
                    temporality=spec.temporality,
                )
            )
        return ModelCandidateBatch(atoms=tuple(atoms))


@dataclass(frozen=True, slots=True)
class Qwen25BaselineAdapter:
    """Qwen2.5-1.5B-class baseline skeleton.

    Production path: build prompt from ``ModelInput``, run generation, parse
  span-port lines. CI uses ``fixture_lines`` when ``weights_path`` is unset.
    """

    model_id: str = "qwen2.5-1.5b-instruct-span-port"
    weights_path: str | None = None
    fixture_lines: Mapping[str, str] | None = None
    max_new_tokens: int = 48

    def propose(
        self,
        model_input: ModelInput,
        atom_specs: Sequence[AtomSpec],
    ) -> ModelCandidateBatch:
        resolved = resolve_weights_path(self.weights_path)
        if resolved is None:
            return FixtureSpanPortAdapter(
                model_id=self.model_id,
                lines=self.fixture_lines,
            ).propose(model_input, atom_specs)

        lines, latency_s, memory_bytes = generate_span_port_lines(
            model_input,
            atom_specs,
            weights_path=resolved,
            max_new_tokens=self.max_new_tokens,
        )
        atoms: list[CandidateAtom] = []
        for spec in atom_specs:
            raw_line = lines.get(spec.atom_id, "NOT_MENTIONED")
            atoms.append(
                candidate_from_span_port_line(
                    atom_id=spec.atom_id,
                    atom_type=spec.atom_type,
                    raw_value=spec.raw_value,
                    raw_line=raw_line,
                    speaker=spec.speaker,
                    experiencer=spec.experiencer,
                    temporality=spec.temporality,
                )
            )
        return ModelCandidateBatch(
            atoms=tuple(atoms),
            latency_s=latency_s,
            memory_bytes=memory_bytes,
        )


def default_baseline_specs() -> tuple[AtomSpec, ...]:
    """Shared atom slots for the deterministic encounter fixture."""
    return (
        AtomSpec("atom-neck", AtomType.SYMPTOM, "neck"),
        AtomSpec("atom-alg", AtomType.ALLERGY, "allergies"),
        AtomSpec("atom-hist", AtomType.SYMPTOM, "migraines"),
        AtomSpec(
            "atom-assess",
            AtomType.ASSESSMENT,
            "cervical strain",
            speaker=Speaker.CLINICIAN,
            experiencer=Experiencer.PATIENT,
        ),
        AtomSpec("medication", AtomType.MEDICATION, "medication"),
    )


DEFAULT_BASELINE_LINES: dict[str, str] = {
    "atom-neck": 'STATED: "neck"',
    "atom-alg": 'DENIED: "No allergies."',
    "atom-hist": 'STATED: "migraines"',
    "atom-assess": 'STATED: "cervical strain"',
    "medication": "NOT_MENTIONED",
}


def default_qwen_fixture_adapter() -> Qwen25BaselineAdapter:
    return Qwen25BaselineAdapter(fixture_lines=DEFAULT_BASELINE_LINES)
