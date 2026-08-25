"""P1 campaign tracks — fixture, Kimi frontier, serverless strong control, references."""

from __future__ import annotations

from nanoscribe.adapt import ModelInput
from nanoscribe.adapters import (
    AtomSpec,
    KimiK3SpanPortAdapter,
    KimiK3StructuredAdapter,
    Qwen25BaselineAdapter,
    ServerlessQwen38Adapter,
    ServerlessQwen38StructuredAdapter,
    ServerlessQwen38ToolAdapter,
    SmallApiReferenceAdapter,
    default_baseline_specs,
    default_qwen_fixture_adapter,
)
from nanoscribe.harness import HarnessCase, ModelTrack, P1TestSet, TrackConfig
from nanoscribe.test_adapt import _gold, _model_input

from nanoscribe.serverless_endpoint import (
    DELETED_QWEN_SERVERLESS_ENDPOINT_ID,
    resolve_serverless_endpoint_id,
)

COMPACT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
STUDENT_MODEL = "Qwen/Qwen2.5-32B-Instruct"
FRONTIER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
FRONTIER_ALT = "meta-llama/Llama-3.3-70B-Instruct"
SMALL_API_MODEL = "gpt-4o-mini"
SERVERLESS_STRONG_MODEL = "Qwen/Qwen3.8-27B"
# Do not call live APIs with this constant — endpoint was deleted; set RUNPOD_SERVERLESS_ENDPOINT_ID.
SERVERLESS_ENDPOINT_ID = DELETED_QWEN_SERVERLESS_ENDPOINT_ID


def tiny_fixture_case() -> HarnessCase:
    gold = _gold()
    return HarnessCase(
        test_set=P1TestSet.TINY_FIXTURE,
        encounter_id="enc-1",
        gold=gold,
        model_input=_model_input(gold.sources[0]),
        atom_specs=default_baseline_specs(),
    )


def fixture_track() -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.FIXTURE,
        model_id="fixture/qwen2.5-1.5b-span-port",
        adapter_factory=default_qwen_fixture_adapter,
        cost_class="zero_local",
        notes="Deterministic CI fixture lines",
    )


def compact_track(weights_path: str = COMPACT_MODEL) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.COMPACT,
        model_id=weights_path,
        adapter_factory=lambda: Qwen25BaselineAdapter(
            model_id=weights_path,
            weights_path=weights_path,
        ),
        cost_class="routine_runpod_4090",
        notes="Qwen2.5-1.5B span-port historical continuity",
    )


def serverless_strong_control_track(
    endpoint_id: str | None = None,
    api_model: str = SERVERLESS_STRONG_MODEL,
) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.SERVERLESS,
        model_id="serverless/qwen3.8-27b-span-port",
        adapter_factory=lambda: ServerlessQwen38Adapter(
            endpoint_id=endpoint_id,
            api_model=api_model,
        ),
        cost_class="serverless_strong_control",
        notes="Lane B: Qwen3.8-27B span-port strong control",
    )


def serverless_strong_structured_track(
    endpoint_id: str | None = None,
    api_model: str = SERVERLESS_STRONG_MODEL,
) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.SERVERLESS,
        model_id="serverless/qwen3.8-27b-structured",
        adapter_factory=lambda: ServerlessQwen38StructuredAdapter(
            endpoint_id=endpoint_id,
            api_model=api_model,
        ),
        cost_class="serverless_strong_control",
        notes="Lane B: Qwen3.8-27B structured CandidateAtom JSON",
    )


def serverless_strong_tool_track(
    endpoint_id: str | None = None,
    api_model: str = SERVERLESS_STRONG_MODEL,
) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.SERVERLESS,
        model_id="serverless/qwen3.8-27b-tool",
        adapter_factory=lambda: ServerlessQwen38ToolAdapter(
            endpoint_id=endpoint_id,
            api_model=api_model,
        ),
        cost_class="serverless_strong_control",
        notes="Lane B: Qwen3.8-27B OpenAI tool-calling CandidateAtom path",
    )


def kimi_frontier_span_port_track() -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.KIMI_FRONTIER,
        model_id="public/kimi-k3-span-port",
        adapter_factory=KimiK3SpanPortAdapter,
        cost_class="frontier_public_endpoint",
        notes="Track A: Kimi K3 public endpoint span-port baseline",
    )


def kimi_frontier_structured_track() -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.KIMI_FRONTIER,
        model_id="public/kimi-k3-structured",
        adapter_factory=KimiK3StructuredAdapter,
        cost_class="frontier_public_endpoint",
        notes="Track A: Kimi K3 structured CandidateAtom JSON",
    )


def small_api_reference_track(api_model: str = SMALL_API_MODEL) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.SMALL_API,
        model_id=f"api/{api_model}-span-port",
        adapter_factory=lambda: SmallApiReferenceAdapter(api_model=api_model),
        cost_class="api_teacher_low",
        notes="Small API reference — not capability ceiling",
    )


def api_teacher_track(api_model: str = SMALL_API_MODEL) -> TrackConfig:
    """Legacy alias — routes to small API reference, not frontier ceiling."""
    return small_api_reference_track(api_model)


def student_track(weights_path: str = STUDENT_MODEL) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.FRONTIER,
        model_id=weights_path,
        adapter_factory=lambda: Qwen25BaselineAdapter(
            model_id=weights_path,
            weights_path=weights_path,
            max_new_tokens=96,
        ),
        cost_class="experiment_scoped_a100_80gb",
        notes="Large student probe — do not launch without baseline gate",
    )


def frontier_track(weights_path: str = FRONTIER_MODEL) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.FRONTIER,
        model_id=weights_path,
        adapter_factory=lambda: Qwen25BaselineAdapter(
            model_id=weights_path,
            weights_path=weights_path,
            max_new_tokens=96,
        ),
        cost_class="experiment_scoped_a100_80gb",
        notes="Self-hosted open-weight frontier probe",
    )
