"""Campaign fan-out helpers — job specs, status, disagreement."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nanoscribe.adapt import (
    ModelCandidate,
    ModelCandidateBatch,
    candidate_from_span_port_line,
    run_pipeline,
)
from nanoscribe.campaign import CampaignLedger, DEFAULT_LEDGER
from nanoscribe.harness import HarnessCase, HarnessResult, TrackConfig, FailureTaxonomy
from nanoscribe.harness import _per_atom, _report_aggregate
from nanoscribe.prompt import (
    build_span_port_prompt,
    build_structured_candidate_prompt,
    span_port_system_prompt,
    structured_candidate_system_prompt,
    tool_candidate_system_prompt,
)
from nanoscribe.inference.tool_registry import (
    resolve_inference_tool_choice,
    scribe_only_tool_definitions,
)
from nanoscribe.tool_calling import build_openai_chat_kwargs, resolve_vllm_tool_env
from nanoscribe.serverless_fanout import (
    CONTRACT_VERSION,
    FanoutJobRecord,
    FanoutJobSpec,
    SERVERLESS_RATE_PER_HR,
    estimate_worker_cost_usd,
)
from nanoscribe.tracks import SERVERLESS_ENDPOINT_ID, SERVERLESS_STRONG_MODEL

CAMPAIGN_STATUS_PATH = Path("artifacts/campaign/campaign_status.json")
DISAGREEMENT_PATH = Path("artifacts/campaign/disagreement_live.json")
NATIVE_MANIFEST_PATH = Path("artifacts/campaign/native_ab_manifests.json")
VERIFIER_DATASET_PATH = Path("artifacts/campaign/verifier_dataset.json")
DISTILL_PATH = Path("artifacts/campaign/p1_distill_train_v1.json")
TEACHER_DATA_V0_PATH = Path("artifacts/campaign/teacher_data_v0.json")
FANOUT_ARTIFACT_GLOB = "fanout_*_qwen_structured_*.json"


def ensure_campaign_spend_start(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    status_path: Path = CAMPAIGN_STATUS_PATH,
) -> float:
    """Pin campaign_spend_start on first v2 orchestrator pass (persisted immediately)."""
    ledger = CampaignLedger.load(ledger_path)
    if status_path.is_file():
        existing = json.loads(status_path.read_text())
        if "campaign_spend_start" in existing:
            return float(existing["campaign_spend_start"])
    start = ledger.campaign_spend_actual
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if status_path.is_file():
        payload = json.loads(status_path.read_text())
    else:
        payload = {"schema": "nano.campaign.status.v1"}
    payload["campaign_spend_start"] = start
    payload["timestamp"] = datetime.now(UTC).isoformat()
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return start


def spend_delta_since_start(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    status_path: Path = CAMPAIGN_STATUS_PATH,
) -> dict[str, float]:
    ledger = CampaignLedger.load(ledger_path)
    start = ensure_campaign_spend_start(ledger_path=ledger_path, status_path=status_path)
    new_actual = round(ledger.campaign_spend_actual - start, 4)
    new_committed = round(ledger.campaign_spend_committed, 4)
    remaining = round(185.0 - new_actual - new_committed, 4)
    return {
        "campaign_spend_start": start,
        "new_actual_spend": new_actual,
        "new_committed_spend": new_committed,
        "remaining_new_budget": remaining,
    }


def hourly_exposure_usd(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    window_hours: float = 1.0,
) -> float:
    """Sum actual spend entries in the last window_hours."""
    ledger = CampaignLedger.load(ledger_path)
    cutoff = datetime.now(UTC).timestamp() - window_hours * 3600.0
    total = 0.0
    for entry in ledger.entries:
        if entry.status != "actual" or not entry.ended_at:
            continue
        try:
            ended = datetime.fromisoformat(entry.ended_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ended.timestamp() >= cutoff:
            total += entry.amount_usd
    return round(total, 4)


def origin_master_sha(short: bool = True) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "origin/master"],
        capture_output=True,
        text=True,
        check=True,
    )
    sha = proc.stdout.strip()
    return sha[:7] if short else sha


def experiment_id_for(lane: str, suite: str, mode: str) -> str:
    return f"{lane}:{suite}:{mode}:{CONTRACT_VERSION}"


def build_serverless_job_specs(
    cases: Sequence[HarnessCase],
    *,
    experiment_id: str,
    mode: str,
    model: str = SERVERLESS_STRONG_MODEL,
) -> list[FanoutJobSpec]:
    specs: list[FanoutJobSpec] = []
    for case in cases:
        if mode == "structured":
            user_prompt = build_structured_candidate_prompt(case.model_input.source, case.atom_specs)
            openai_input = {
                "model": model,
                "messages": [
                    {"role": "system", "content": structured_candidate_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"},
            }
            specs.append(
                FanoutJobSpec(
                    case_id=case.encounter_id,
                    experiment_id=experiment_id,
                    contract_version=CONTRACT_VERSION,
                    mode=mode,
                    atom_id=None,
                    openai_input=openai_input,
                )
            )
        elif mode == "tool":
            user_prompt = build_structured_candidate_prompt(case.model_input.source, case.atom_specs)
            vllm_env = resolve_vllm_tool_env()
            tools = scribe_only_tool_definitions()
            tool_choice = resolve_inference_tool_choice(None, vllm_env, scribe_only=True)
            openai_input = build_openai_chat_kwargs(
                model=model,
                system_prompt=tool_candidate_system_prompt(),
                user_prompt=user_prompt,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=1024,
            )
            specs.append(
                FanoutJobSpec(
                    case_id=case.encounter_id,
                    experiment_id=experiment_id,
                    contract_version=CONTRACT_VERSION,
                    mode=mode,
                    atom_id=None,
                    openai_input=openai_input,
                )
            )
        elif mode == "span_port":
            for atom_spec in case.atom_specs:
                user_prompt = build_span_port_prompt(case.model_input.source, atom_spec)
                openai_input = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": span_port_system_prompt()},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 64,
                }
                specs.append(
                    FanoutJobSpec(
                        case_id=case.encounter_id,
                        experiment_id=experiment_id,
                        contract_version=CONTRACT_VERSION,
                        mode=mode,
                        atom_id=atom_spec.atom_id,
                        openai_input=openai_input,
                    )
                )
        else:
            raise ValueError(f"unknown mode: {mode}")
    return specs


def _parse_tool_mode_response(raw: Any) -> ModelCandidateBatch:
    """Parse fanout tool-mode response into ModelCandidateBatch."""
    from nanoscribe.capabilities import CapabilityId, CapabilityToolParser
    from nanoscribe.tool_calling import ToolCallParser

    if raw is None:
        return ModelCandidateBatch(atoms=())
    text = str(raw).strip()
    if not text:
        return ModelCandidateBatch(atoms=())
    parser = CapabilityToolParser(allowed_capabilities=(CapabilityId.SCRIBE,))
    fake_call = {
        "id": "fanout_tool",
        "function": {"name": "submit_candidate_atoms", "arguments": text},
    }
    result = parser.parse_tool_call(fake_call)
    if result.candidate is not None:
        return ModelCandidateBatch(atoms=result.candidate.atoms)
    legacy = ToolCallParser().parse_text_json(text)
    return ModelCandidateBatch(atoms=ToolCallParser().to_model_candidate(legacy).atoms)


def records_to_candidate_batch(
    case: HarnessCase,
    records: Sequence[FanoutJobRecord],
    *,
    mode: str,
) -> ModelCandidateBatch:
    if mode in {"structured", "tool"}:
        record = next((item for item in records if item.case_id == case.encounter_id), None)
        if record is None or not record.response:
            return ModelCandidateBatch(atoms=())
        try:
            if mode == "tool":
                return _parse_tool_mode_response(record.response)
            from nanoscribe.tool_calling import ToolCallParser

            parser = ToolCallParser()
            result = parser.parse_text_json(str(record.response))
            return ModelCandidateBatch(atoms=parser.to_model_candidate(result).atoms)
        except Exception:
            return ModelCandidateBatch(atoms=())
    atoms = []
    by_atom = {
        item.atom_id: item.response
        for item in records
        if item.case_id == case.encounter_id and item.atom_id
    }
    for spec in case.atom_specs:
        raw_line = str(by_atom.get(spec.atom_id) or "NOT_MENTIONED")
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


def evaluate_fanout_case(
    track: TrackConfig,
    case: HarnessCase,
    records: Sequence[FanoutJobRecord],
    *,
    mode: str,
) -> HarnessResult:
    batch = records_to_candidate_batch(case, records, mode=mode)
    predicted, report = run_pipeline(case.model_input, batch, gold=case.gold)
    assert report is not None
    latency = 0.0
    case_records = [item for item in records if item.case_id == case.encounter_id]
    if case_records:
        exec_ms = [item.execution_time_ms for item in case_records if item.execution_time_ms]
        if exec_ms:
            latency = max(exec_ms) / 1000.0
    return HarnessResult(
        track=track.track,
        model_id=track.model_id,
        test_set=case.test_set,
        encounter_id=case.encounter_id,
        cost_class=track.cost_class,
        aggregate=_report_aggregate(report),
        failures=FailureTaxonomy.from_report(report),
        per_atom=_per_atom(report),
        latency_s=latency,
        memory_bytes=0,
    )


def actualize_serverless_spend(
    records: Sequence[FanoutJobRecord],
    *,
    lane: str,
    description: str,
    ledger_path: Path = DEFAULT_LEDGER,
) -> dict[str, Any]:
    ledger = CampaignLedger.load(ledger_path)
    exec_ms = [item.execution_time_ms for item in records if item.execution_time_ms]
    if exec_ms:
        worker_seconds = sum(exec_ms) / 1000.0
        amount = round((worker_seconds / 3600.0) * SERVERLESS_RATE_PER_HR, 4)
    else:
        amount = estimate_worker_cost_usd(len(records))
    allowed, reason = ledger.budget_gate(amount)
    if not allowed:
        return {"recorded": False, "reason": reason, "amount_usd": amount}
    entry = ledger.commit(
        lane,
        description,
        amount,
        gpu="A100-80GB-serverless",
        rate_per_hr=SERVERLESS_RATE_PER_HR,
        notes=f"fanout jobs={len(records)} worker_s={sum(exec_ms)/1000.0 if exec_ms else 0:.1f}",
    )
    ledger.actualize(entry, amount)
    ledger.save(ledger_path)
    return {"recorded": True, "amount_usd": amount, "summary": ledger.summary()}


def write_campaign_status(
    *,
    active_tracks: Sequence[str],
    serverless: Mapping[str, Any],
    model_statuses: Mapping[str, Any],
    leading_failures: Sequence[str],
    next_reallocations: Sequence[str],
    lanes: Mapping[str, Any] | None = None,
    structured_contract: Mapping[str, Any] | None = None,
    students: Mapping[str, Any] | None = None,
    ledger_path: Path = DEFAULT_LEDGER,
    path: Path = CAMPAIGN_STATUS_PATH,
) -> dict[str, Any]:
    ledger = CampaignLedger.load(ledger_path)
    spend_delta = spend_delta_since_start(ledger_path=ledger_path, status_path=path)
    payload = {
        "schema": "nano.campaign.status.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "origin_master": origin_master_sha(),
        "mandate_baseline_usd": 185.0,
        "campaign_spend_start": spend_delta["campaign_spend_start"],
        "new_actual_spend": spend_delta["new_actual_spend"],
        "new_committed_spend": spend_delta["new_committed_spend"],
        "remaining_new_budget": spend_delta["remaining_new_budget"],
        "hourly_exposure": hourly_exposure_usd(ledger_path=ledger_path),
        "actual_spend": ledger.campaign_spend_actual,
        "remaining_budget": round(185.0 - ledger.campaign_spend_actual, 4),
        "ledger_remaining": ledger.campaign_spend_remaining,
        "posture": ledger.posture(),
        "active_tracks": list(active_tracks),
        "lanes": dict(lanes or {}),
        "serverless": dict(serverless),
        "raw_pods": [],
        "model_statuses": dict(model_statuses),
        "structured_contract": dict(structured_contract or {}),
        "students": dict(students or {}),
        "leading_failures": list(leading_failures),
        "next_reallocations": list(next_reallocations),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def update_disagreement_matrix(
    kimi_results: Sequence[HarnessResult],
    qwen_results: Sequence[HarnessResult],
    *,
    path: Path = DISAGREEMENT_PATH,
) -> dict[str, Any]:
    kimi_by_case = {item.encounter_id: item for item in kimi_results}
    qwen_by_case = {item.encounter_id: item for item in qwen_results}
    cases = sorted(set(kimi_by_case) | set(qwen_by_case))
    rows: list[dict[str, Any]] = []
    for case_id in cases:
        kimi = kimi_by_case.get(case_id)
        qwen = qwen_by_case.get(case_id)
        row: dict[str, Any] = {"case_id": case_id}
        if kimi:
            row["kimi"] = {
                "model_id": kimi.model_id,
                "aggregate": kimi.aggregate,
                "failures": kimi.failures.to_dict(),
            }
        if qwen:
            row["qwen"] = {
                "model_id": qwen.model_id,
                "aggregate": qwen.aggregate,
                "failures": qwen.failures.to_dict(),
            }
        if kimi and qwen:
            row["disagreement"] = {
                "exact_gold_span_delta": (
                    kimi.aggregate.get("exact_gold_span", 0)
                    - qwen.aggregate.get("exact_gold_span", 0)
                ),
                "support_direct_exact_delta": (
                    kimi.aggregate.get("support_direct_exact", 0)
                    - qwen.aggregate.get("support_direct_exact", 0)
                ),
                "malformed_delta": (
                    kimi.failures.malformed - qwen.failures.malformed
                ),
            }
        rows.append(row)
    payload = {
        "schema": "nano.campaign.disagreement.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "cases": rows,
        "summary": {
            "n_cases": len(rows),
            "n_with_both": sum(1 for row in rows if "disagreement" in row),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_native_ab_manifests() -> dict[str, Any]:
    """Lane 4 — factorial 2×2 native screen manifests (A/B/C/D × 2 seeds @ 30M)."""
    from nanoscribe.native.factorial import factorial_manifest

    payload = factorial_manifest()
    NATIVE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    NATIVE_MANIFEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def export_screening_p1_distill(
    *,
    artifact_dir: Path = Path("artifacts/p1_runs"),
    out_path: Path = DISTILL_PATH,
) -> dict[str, Any]:
    """Export teacher structured outputs from latest Qwen fanout artifacts.

    FORBIDDEN: export fanout eval artifacts as training data — use p1_distill_train_v1.
    """
    from nanoscribe.native.data import export_distill_train_json

    return {
        "exported": False,
        "reason": "screening_p1_distill deprecated — use p1_distill_train_v1.json",
        "train_export": export_distill_train_json(out_path),
    }


def build_teacher_data_v0(
    *,
    disagreement_path: Path = DISAGREEMENT_PATH,
    max_cases: int = 32,
    out_path: Path = TEACHER_DATA_V0_PATH,
) -> dict[str, Any]:
    """Build Teacher Data V0 from disagreement_live.json (first 16–32 paired cases)."""
    if not disagreement_path.is_file():
        return {"built": False, "reason": "disagreement_live.json missing"}
    data = json.loads(disagreement_path.read_text())
    rows = [row for row in data.get("cases", []) if "qwen" in row][:max_cases]
    payload = {
        "schema": "nano.campaign.teacher_data.v0",
        "timestamp": datetime.now(UTC).isoformat(),
        "n_cases": len(rows),
        "source": str(disagreement_path),
        "cases": rows,
        "note": "Kimi lane absent — Qwen-only rows until Kimi recovers",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"built": True, "n_cases": len(rows), "path": str(out_path)}


def load_verifier_dataset(path: Path = VERIFIER_DATASET_PATH) -> dict[str, Any]:
    if not path.is_file():
        return {"n_cases": 0, "entries": []}
    return json.loads(path.read_text())


def evaluate_verifier_dataset_coverage(
    cases: Sequence[HarnessCase],
    *,
    path: Path = VERIFIER_DATASET_PATH,
) -> dict[str, Any]:
    """Wire verifier_dataset.json into eval — coverage check vs harness cases."""
    dataset = load_verifier_dataset(path)
    ds_ids = {entry["encounter_id"] for entry in dataset.get("entries", [])}
    case_ids = {case.encounter_id for case in cases}
    covered = ds_ids & case_ids
    return {
        "verifier_dataset_cases": len(ds_ids),
        "harness_cases": len(case_ids),
        "covered": len(covered),
        "coverage_pct": round(100.0 * len(covered) / max(1, len(case_ids)), 2),
        "ready_for_training": len(covered) >= 16,
    }


def build_verifier_dataset(cases: Sequence[HarnessCase]) -> dict[str, Any]:
    """Lane F — local verifier dataset assembly from screening cases."""
    entries = []
    for case in cases:
        entries.append(
            {
                "encounter_id": case.encounter_id,
                "test_set": case.test_set.value,
                "n_atoms": len(case.atom_specs),
                "gold_atoms": [
                    {
                        "atom_id": atom.atom_id,
                        "atom_type": atom.atom_type.value,
                        "assertion_state": atom.assertion_state.value,
                        "evidence_ids": list(atom.evidence_ids),
                    }
                    for atom in case.gold.atoms
                ],
                "transcript_turns": len(case.model_input.source.turns),
            }
        )
    payload = {
        "schema": "nano.campaign.verifier_dataset.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "n_cases": len(entries),
        "entries": entries,
    }
    VERIFIER_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERIFIER_DATASET_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
