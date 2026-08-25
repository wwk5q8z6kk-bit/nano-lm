"""P1 structured eval for native Nano checkpoints — smoke and screening suites."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoscribe.adapt import ModelCandidate, ModelInput, candidate_from_span_port_line, run_pipeline
from nanoscribe.adapters import AtomSpec
from nanoscribe.campaign_datasets import SMOKE_SUITE_REVISION, campaign_cases
from nanoscribe.harness import HarnessCase, aggregate_suite_metrics
from nanoscribe.native.checkpoint import load_checkpoint
from nanoscribe.native.config import checkpoint_root, NativeTrainConfig, config_for_run
from nanoscribe.native.inference import generate_target_line
from nanoscribe.native.model import build_native_model
from nanoscribe.prompt import build_canonical_span_port_prompt



_TAUTOLOGICAL_SPAN_KEYS = (
    "exact_gold_span_count",
    "exact_gold_span_eligible",
    "exact_gold_span_rate",
    "span_character_f1",
)


def _suppress_tautological_spans(
    metrics: dict[str, Any], *, constrained: bool
) -> dict[str, Any]:
    """Blank span metrics that are exact by construction.

    Under constrained candidate selection the gold raw_value is injected into
    the candidate set, so exact_gold_span is 1.0 whenever the model commits at
    all -- it measures the candidate set, not evidence transport. Emitting the
    number with a `span_metrics_are_tautological` flag beside it is not enough:
    a metric that reads 6/6 gets quoted by whoever missed the flag, which is how
    dc3b310's "0% -> 83%" propagated past its own conditions.

    The values are replaced with None and an explicit reason key, rather than
    deleted, so consumers that index these keys degrade to null instead of
    raising. Free-generation mode is untouched -- there exact_gold_span IS the
    transport measurement.

    Convergent evidence that span EXTENT is the fragile quantity, from two
    independent measurements: the span-port thread found exact-extent scores
    swinging 8x under a question-phrasing change while turn-selection did not
    move at all, and this track's constrained mode makes extent exact by
    construction. Extent is the metric to distrust in both tracks; selection is
    the stabler signal.
    """
    if not constrained:
        return metrics
    out = dict(metrics)
    suppressed = [k for k in _TAUTOLOGICAL_SPAN_KEYS if k in out]
    for k in suppressed:
        out[k] = None
    if suppressed:
        out["span_metrics_suppressed_reason"] = (
            "tautological under constrained_candidate_selection: the gold span is "
            "in the candidate set by construction; use free_generation to measure "
            "evidence transport"
        )
    return out

@dataclass(frozen=True, slots=True)
class NativeP1CaseResult:
    encounter_id: str
    aggregate: dict[str, Any]
    per_atom: dict[str, dict[str, Any]]
    raw_lines: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "encounter_id": self.encounter_id,
            "aggregate": self.aggregate,
            "per_atom": self.per_atom,
            "raw_lines": self.raw_lines,
        }


@dataclass(frozen=True, slots=True)
class NativeP1EvalResult:
    run_id: str
    suite: str
    checkpoint: str
    n_cases: int
    suite_metrics: dict[str, Any]
    cases: tuple[NativeP1CaseResult, ...]
    constrained: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "nano.native.p1_eval.v1",
            "run_id": self.run_id,
            "suite": self.suite,
            "checkpoint": self.checkpoint,
            "n_cases": self.n_cases,
            "decoding_mode": "constrained_candidate_selection" if self.constrained else "free_generation",
            # Under constrained selection the gold raw_value is injected into the
            # candidate set, so span metrics are exact by construction. Only
            # free_generation makes exact_gold_span a transport measurement.
            "span_metrics_are_tautological": self.constrained,
            "suite_metrics": _suppress_tautological_spans(
                self.suite_metrics, constrained=self.constrained
            ),
            "cases": [case.to_dict() for case in self.cases],
        }


def _resolve_device(cpu: bool) -> str:
    import torch

    if cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def load_native_model_for_eval(
    run_id: str,
    *,
    step: int | None = None,
    cpu: bool = True,
) -> tuple[Any, NativeTrainConfig, str]:
    cfg = config_for_run(run_id, cpu_smoke=False)
    build = build_native_model(cfg)
    device = _resolve_device(cpu)
    build.model.to(device)
    ckpt_path = checkpoint_root(cfg) / cfg.run_id / (
        "latest.pt" if step is None else f"step_{step:06d}.pt"
    )
    load_checkpoint(cfg, build.model, step=step)
    return build.model, cfg, str(ckpt_path)


def _propose_case(
    model: Any,
    cfg: NativeTrainConfig,
    case: HarnessCase,
    *,
    constrained: bool = True,
) -> tuple[ModelCandidate, dict[str, str]]:
    atoms = []
    raw_lines: dict[str, str] = {}
    for spec in case.atom_specs:
        prompt = build_canonical_span_port_prompt(case.model_input.source, spec)
        raw_line = generate_target_line(
            model,
            prompt,
            cfg,
            raw_value=spec.raw_value,
            source=case.model_input.source,
            constrained=constrained,
        )
        raw_lines[spec.atom_id] = raw_line
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
    return ModelCandidate(atoms=tuple(atoms)), raw_lines


def evaluate_native_p1_suite(
    model: Any,
    cfg: NativeTrainConfig,
    *,
    suite: str = SMOKE_SUITE_REVISION,
    checkpoint_path: str = "",
    constrained: bool = True,
) -> NativeP1EvalResult:
    from nanoscribe.harness import FailureTaxonomy, HarnessResult, ModelTrack, _per_atom, _report_aggregate

    cases = campaign_cases(suite)
    case_results: list[NativeP1CaseResult] = []
    harness_results: list[HarnessResult] = []

    for case in cases:
        t0 = time.perf_counter()
        batch, raw_lines = _propose_case(model, cfg, case, constrained=constrained)
        latency = time.perf_counter() - t0
        batch = ModelCandidate(
            atoms=batch.atoms,
            latency_s=latency,
            memory_bytes=0,
        )
        _predicted, report = run_pipeline(case.model_input, batch, gold=case.gold)
        assert report is not None
        aggregate = _report_aggregate(report)
        case_results.append(
            NativeP1CaseResult(
                encounter_id=case.encounter_id,
                aggregate=aggregate,
                per_atom=_per_atom(report),
                raw_lines=raw_lines,
            )
        )
        harness_results.append(
            HarnessResult(
                track=ModelTrack.FIXTURE,
                model_id=f"native/{cfg.run_id}",
                test_set=case.test_set,
                encounter_id=case.encounter_id,
                cost_class="native_cpu",
                aggregate=aggregate,
                failures=FailureTaxonomy.from_report(report),
                per_atom=_per_atom(report),
                latency_s=latency,
                memory_bytes=0,
                raw_lines=raw_lines,
            )
        )

    return NativeP1EvalResult(
        run_id=cfg.run_id,
        suite=suite,
        checkpoint=checkpoint_path,
        n_cases=len(cases),
        suite_metrics=aggregate_suite_metrics(harness_results),
        cases=tuple(case_results),
        constrained=constrained,
    )
