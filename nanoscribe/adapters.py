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
    # Role-based slot identifier that shares no surface form with raw_value.
    # Lets the question specify the slot without naming the answer (Q_SURFACE).
    concept_label: str = ""


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
    raw_line_sink: dict[str, str] | None = None

    def propose(
        self,
        model_input: ModelInput,
        atom_specs: Sequence[AtomSpec],
    ) -> ModelCandidateBatch:
        # The fixture ignores the prompt, but E-DELIMIT's menu/offsets arms need
        # the source to resolve an index or an offset pair into a quote.
        lines = self.lines or {}
        atoms: list[CandidateAtom] = []
        for spec in atom_specs:
            raw_line = lines.get(spec.atom_id, "NOT_MENTIONED")
            if self.raw_line_sink is not None:
                self.raw_line_sink[spec.atom_id] = raw_line
            atoms.append(
                candidate_from_span_port_line(
                    atom_id=spec.atom_id,
                    atom_type=spec.atom_type,
                    raw_value=spec.raw_value,
                    raw_line=raw_line,
                    speaker=spec.speaker,
                    experiencer=spec.experiencer,
                    temporality=spec.temporality,
                    source=model_input.source,
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
    raw_line_sink: dict[str, str] | None = None

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
                raw_line_sink=self.raw_line_sink,
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
            if self.raw_line_sink is not None:
                self.raw_line_sink[spec.atom_id] = raw_line
            atoms.append(
                candidate_from_span_port_line(
                    atom_id=spec.atom_id,
                    atom_type=spec.atom_type,
                    raw_value=spec.raw_value,
                    raw_line=raw_line,
                    speaker=spec.speaker,
                    experiencer=spec.experiencer,
                    temporality=spec.temporality,
                    source=model_input.source,
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


@dataclass(frozen=True, slots=True)
class ApiTeacherAdapter:
    """Hosted API span-port adapter (legacy name — prefer SmallApiReferenceAdapter)."""

    model_id: str = "api/gpt-4o-mini-span-port"
    api_model: str = "gpt-4o-mini"

    def propose(
        self,
        model_input: ModelInput,
        atom_specs: Sequence[AtomSpec],
    ) -> ModelCandidateBatch:
        return SmallApiReferenceAdapter(
            model_id=self.model_id,
            api_model=self.api_model,
        ).propose(model_input, atom_specs)


@dataclass(frozen=True, slots=True)
class KimiK3SpanPortAdapter:
    """Track A frontier teacher — Kimi K3 RunPod Public Endpoint (span-port baseline)."""

    model_id: str = "public/kimi-k3-span-port"

    def propose(
        self,
        model_input: ModelInput,
        atom_specs: Sequence[AtomSpec],
    ) -> ModelCandidateBatch:
        from nanoscribe.kimi_teacher import (
            generate_kimi_span_port_lines,
            kimi_span_port_batch_to_candidates,
        )

        lines, latency_s, memory_bytes = generate_kimi_span_port_lines(model_input, atom_specs)
        batch = kimi_span_port_batch_to_candidates(model_input, atom_specs, lines)
        return ModelCandidateBatch(
            atoms=batch.atoms,
            latency_s=latency_s,
            memory_bytes=memory_bytes,
        )


@dataclass(frozen=True, slots=True)
class KimiK3StructuredAdapter:
    """Track A frontier teacher — Kimi K3 structured CandidateAtom JSON."""

    model_id: str = "public/kimi-k3-structured"

    def propose(
        self,
        model_input: ModelInput,
        atom_specs: Sequence[AtomSpec],
    ) -> ModelCandidateBatch:
        from nanoscribe.kimi_teacher import generate_kimi_structured_candidates

        batch, latency_s, memory_bytes = generate_kimi_structured_candidates(
            model_input, atom_specs
        )
        return ModelCandidateBatch(
            atoms=batch.atoms,
            latency_s=latency_s,
            memory_bytes=memory_bytes,
        )


@dataclass(frozen=True, slots=True)
class ServerlessQwen38ToolAdapter:
    """Qwen3.8-27B CandidateAtom extraction via OpenAI tool calling."""

    model_id: str = "serverless/qwen3.8-27b-tool"
    api_model: str = "Qwen/Qwen3.8-27B"
    endpoint_id: str | None = None
    base_url: str | None = None
    max_tokens: int = 1024
    include_coding_stub: bool = False

    def propose(
        self,
        model_input: ModelInput,
        atom_specs: Sequence[AtomSpec],
    ) -> ModelCandidateBatch:
        from nanoscribe.serverless_inference import generate_serverless_tool_candidates

        batch, latency_s, memory_bytes = generate_serverless_tool_candidates(
            model_input,
            atom_specs,
            model=self.api_model,
            endpoint_id=self.endpoint_id,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            include_coding_stub=self.include_coding_stub,
        )
        return ModelCandidateBatch(
            atoms=batch.atoms,
            latency_s=latency_s,
            memory_bytes=memory_bytes,
        )


@dataclass(frozen=True, slots=True)
class ServerlessQwen38StructuredAdapter:
    """Qwen3.8-27B structured CandidateAtom JSON via RunPod Serverless."""

    model_id: str = "serverless/qwen3.8-27b-structured"
    api_model: str = "Qwen/Qwen3.8-27B"
    endpoint_id: str | None = None
    base_url: str | None = None
    max_tokens: int = 1024

    def propose(
        self,
        model_input: ModelInput,
        atom_specs: Sequence[AtomSpec],
    ) -> ModelCandidateBatch:
        from nanoscribe.serverless_inference import generate_serverless_structured_candidates

        batch, latency_s, memory_bytes = generate_serverless_structured_candidates(
            model_input,
            atom_specs,
            model=self.api_model,
            endpoint_id=self.endpoint_id,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
        )
        return ModelCandidateBatch(
            atoms=batch.atoms,
            latency_s=latency_s,
            memory_bytes=memory_bytes,
        )


@dataclass(frozen=True, slots=True)
class SmallApiReferenceAdapter:
    """Small hosted API reference — not a frontier capability ceiling."""

    model_id: str = "api/gpt-4o-mini-span-port"
    api_model: str = "gpt-4o-mini"

    def propose(
        self,
        model_input: ModelInput,
        atom_specs: Sequence[AtomSpec],
    ) -> ModelCandidateBatch:
        from nanoscribe.api_teacher import generate_api_span_port_lines

        lines, latency_s, memory_bytes = generate_api_span_port_lines(
            model_input,
            atom_specs,
            model=self.api_model,
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
                    source=model_input.source,
                )
            )
        return ModelCandidateBatch(
            atoms=tuple(atoms),
            latency_s=latency_s,
            memory_bytes=memory_bytes,
        )


@dataclass(frozen=True, slots=True)
class ServerlessQwen38Adapter:
    """Strong control lane — Qwen3.8-27B via RunPod Serverless OpenAI API."""

    model_id: str = "serverless/qwen3.8-27b-strong-control"
    api_model: str = "Qwen/Qwen3.8-27B"
    endpoint_id: str | None = None
    base_url: str | None = None
    max_tokens: int = 64

    def propose(
        self,
        model_input: ModelInput,
        atom_specs: Sequence[AtomSpec],
    ) -> ModelCandidateBatch:
        from nanoscribe.serverless_inference import generate_serverless_span_port_lines

        lines, latency_s, memory_bytes = generate_serverless_span_port_lines(
            model_input,
            atom_specs,
            model=self.api_model,
            endpoint_id=self.endpoint_id,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
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
                    source=model_input.source,
                )
            )
        return ModelCandidateBatch(
            atoms=tuple(atoms),
            latency_s=latency_s,
            memory_bytes=memory_bytes,
        )
