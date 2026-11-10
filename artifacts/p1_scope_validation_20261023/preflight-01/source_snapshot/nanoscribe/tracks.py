"""P1 model track registry and tiny fixture cases."""

from __future__ import annotations

from nanoscribe.adapt import ModelInput
from nanoscribe.adapters import (
    AtomSpec,
    ApiTeacherAdapter,
    Qwen25BaselineAdapter,
    ServerlessQwen38Adapter,
    default_baseline_specs,
    default_qwen_fixture_adapter,
)
from nanoscribe.harness import HarnessCase, ModelTrack, P1TestSet, TrackConfig
from nanoscribe.test_adapt import _gold, _model_input

# 2026-02 research: practical self-hostable open-weight instruct models for P1 probing.
# COMPACT = single 24GB GPU; FRONTIER = 48–80GB or quantized 70B on one A100/H100.
COMPACT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
STUDENT_MODEL = "Qwen/Qwen2.5-32B-Instruct"
FRONTIER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
FRONTIER_ALT = "meta-llama/Llama-3.3-70B-Instruct"
API_TEACHER_MODEL = "gpt-4o-mini"
SERVERLESS_STRONG_MODEL = "Qwen/Qwen3.8-27B"
SERVERLESS_ENDPOINT_ID = "tbnur4mac60i70"


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


def compact_track(
    weights_path: str = COMPACT_MODEL,
) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.COMPACT,
        model_id=weights_path,
        adapter_factory=lambda: Qwen25BaselineAdapter(
            model_id=weights_path,
            weights_path=weights_path,
        ),
        cost_class="routine_runpod_4090",
        notes="Qwen2.5-1.5B span-port baseline on ~24GB GPU",
    )


def serverless_strong_control_track(
    endpoint_id: str = SERVERLESS_ENDPOINT_ID,
    api_model: str = SERVERLESS_STRONG_MODEL,
) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.FRONTIER,
        model_id="serverless/qwen3.8-27b-strong-control",
        adapter_factory=lambda: ServerlessQwen38Adapter(
            endpoint_id=endpoint_id,
            api_model=api_model,
        ),
        cost_class="serverless_strong_control",
        notes="Lane 1: Qwen3.8-27B RunPod Serverless strong control / specialist base",
    )


def api_teacher_track(api_model: str = API_TEACHER_MODEL) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.FRONTIER,
        model_id=f"api/{api_model}-span-port",
        adapter_factory=lambda: ApiTeacherAdapter(api_model=api_model),
        cost_class="api_teacher_low",
        notes="Hosted frontier teacher — intelligence per dollar over self-host giants",
    )


def student_track(
    weights_path: str = STUDENT_MODEL,
) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.FRONTIER,
        model_id=weights_path,
        adapter_factory=lambda: Qwen25BaselineAdapter(
            model_id=weights_path,
            weights_path=weights_path,
            max_new_tokens=96,
        ),
        cost_class="experiment_scoped_a100_80gb",
        notes="30B-100B specialist student probe (Qwen2.5-32B on A100/L40S)",
    )


def frontier_track(
    weights_path: str = FRONTIER_MODEL,
) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.FRONTIER,
        model_id=weights_path,
        adapter_factory=lambda: Qwen25BaselineAdapter(
            model_id=weights_path,
            weights_path=weights_path,
            max_new_tokens=96,
        ),
        cost_class="experiment_scoped_a100_80gb",
        notes="Largest practical single-GPU open instruct for P1 ceiling probe",
    )
