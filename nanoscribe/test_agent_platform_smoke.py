# Agent platform offline smoke — p1_contract_smoke_v1 + unified tool inference.
# Run: python3 -m pytest nanoscribe/test_agent_platform_smoke.py -q
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.adapt import CANDIDATE_SCHEMA_VERSION, CandidateAtom, ModelCandidate
from nanoscribe.campaign_datasets import campaign_cases, SMOKE_SUITE_REVISION
from nanoscribe.encounter import AssertionState, AtomType, Certainty, Experiencer, Speaker, TemporalState, Temporality
from nanoscribe.inference.tool_registry import agent_tool_definitions
from nanoscribe.structured_inference import generate_structured_candidates
from nanoscribe.tools import SUBMIT_CANDIDATE_ATOMS_TOOL


def _minimal_candidate(atom_specs) -> ModelCandidate:
    atoms = []
    for spec in atom_specs:
        atoms.append(
            CandidateAtom(
                atom_id=spec.atom_id,
                atom_type=spec.atom_type,
                raw_value=spec.raw_value,
                assertion_state=AssertionState.ASSERTED,
                speaker=spec.speaker,
                experiencer=spec.experiencer,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.STATED,
                quotes=(spec.raw_value,),
            )
        )
    return ModelCandidate(atoms=tuple(atoms))


def _mock_tool_response(candidate: ModelCandidate) -> MagicMock:
    fn = MagicMock()
    fn.name = SUBMIT_CANDIDATE_ATOMS_TOOL
    fn.arguments = json.dumps(candidate.to_dict())
    tool_call = MagicMock()
    tool_call.id = "smoke_call"
    tool_call.function = fn
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=None, tool_calls=[tool_call]))]
    return mock_response


def test_agent_tool_definitions_count() -> None:
    names = {tool.name for tool in agent_tool_definitions()}
    assert names == {"submit_candidate_atoms", "submit_summary", "submit_table"}


def test_smoke_contract_suite_tool_inference_offline() -> None:
    cases = campaign_cases(SMOKE_SUITE_REVISION)
    assert len(cases) == 3
    encounter_ids = {case.encounter_id for case in cases}
    assert encounter_ids == {"enc-1", "enc-2", "enc-3"}

    mock_client = MagicMock()
    results: list[tuple[str, int]] = []

    for case in cases:
        candidate = _minimal_candidate(case.atom_specs)
        mock_client.chat.completions.create.return_value = _mock_tool_response(candidate)
        batch, latency_s, memory = generate_structured_candidates(
            case.model_input,
            case.atom_specs,
            client=mock_client,
            model="offline-smoke",
            use_tools=True,
            include_agent_tools=True,
            vllm_env={"TOOL_CALL_PARSER": "qwen3_coder", "ENABLE_AUTO_TOOL_CHOICE": "true"},
        )
        assert latency_s >= 0
        assert memory == 0
        assert len(batch.atoms) == len(case.atom_specs)
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" in kwargs
        assert kwargs["tool_choice"] == "auto"
        tool_names = {t["function"]["name"] for t in kwargs["tools"]}
        assert tool_names == {"submit_candidate_atoms", "submit_summary", "submit_table"}
        assert "response_format" not in kwargs
        results.append((case.encounter_id, len(batch.atoms)))

    for case, (enc_id, atom_count) in zip(cases, results, strict=True):
        assert enc_id == case.encounter_id
        assert atom_count == len(case.atom_specs)


def run_smoke_report() -> dict[str, object]:
    """Execute offline smoke and return JSON-serializable report (no pytest)."""
    cases = campaign_cases(SMOKE_SUITE_REVISION)
    mock_client = MagicMock()
    passed = 0
    details: list[dict[str, object]] = []
    for case in cases:
        candidate = _minimal_candidate(case.atom_specs)
        mock_client.chat.completions.create.return_value = _mock_tool_response(candidate)
        try:
            batch, latency_s, _ = generate_structured_candidates(
                case.model_input,
                case.atom_specs,
                client=mock_client,
                model="offline-smoke",
                use_tools=True,
                include_agent_tools=True,
                vllm_env={"TOOL_CALL_PARSER": "qwen3_coder", "ENABLE_AUTO_TOOL_CHOICE": "true"},
            )
            ok = len(batch.atoms) == len(case.atom_specs)
            if ok:
                passed += 1
            details.append(
                {
                    "encounter_id": case.encounter_id,
                    "atom_slots": len(case.atom_specs),
                    "atoms_parsed": len(batch.atoms),
                    "latency_s": round(latency_s, 4),
                    "pass": ok,
                }
            )
        except Exception as exc:  # noqa: BLE001 — smoke report must capture failures
            details.append(
                {
                    "encounter_id": case.encounter_id,
                    "pass": False,
                    "error": str(exc),
                }
            )
    return {
        "suite": SMOKE_SUITE_REVISION,
        "mode": "offline_mock",
        "use_tools": True,
        "include_agent_tools": True,
        "cases_total": len(cases),
        "cases_passed": passed,
        "verdict": "PASS" if passed == len(cases) else "FAIL",
        "details": details,
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke_report(), indent=2))
