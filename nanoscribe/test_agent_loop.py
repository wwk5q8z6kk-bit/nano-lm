# Unified agent loop tests — offline mock, sandbox, capabilities.
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.adapt import CANDIDATE_SCHEMA_VERSION, CandidateAtom, ModelCandidate
from nanoscribe.agent.loop import AgentConfig, NanoAgent
from nanoscribe.artifacts.summary_spec import SUMMARY_SCHEMA_VERSION
from nanoscribe.capabilities.registry import SUBMIT_SUMMARY_TOOL
from nanoscribe.coding_tools import LIST_DIRECTORY_TOOL, READ_FILE_TOOL
from nanoscribe.encounter import AssertionState, AtomType, Certainty, Experiencer, Speaker, TemporalState, Temporality
from nanoscribe.tools import SUBMIT_CANDIDATE_ATOMS_TOOL


def _candidate_payload() -> dict[str, object]:
    atom = CandidateAtom(
        atom_id="atom-1",
        atom_type=AtomType.SYMPTOM,
        raw_value="headache",
        assertion_state=AssertionState.ASSERTED,
        speaker=Speaker.PATIENT,
        experiencer=Experiencer.PATIENT,
        temporality=TemporalState(kind=Temporality.CURRENT),
        certainty=Certainty.STATED,
        quotes=("headache",),
    )
    return ModelCandidate(atoms=(atom,)).to_dict()


def _mock_client_sequence(steps: list[dict[str, object]]) -> MagicMock:
    calls = {"index": 0}

    def _create(**kwargs: object) -> MagicMock:
        idx = calls["index"]
        calls["index"] += 1
        step = steps[min(idx, len(steps) - 1)]
        if step.get("tool_calls"):
            tool_calls = []
            for item in step["tool_calls"]:
                fn = MagicMock()
                fn.name = item["name"]
                fn.arguments = json.dumps(item["arguments"])
                tc = MagicMock()
                tc.id = item.get("id", f"call_{idx}")
                tc.function = fn
                tool_calls.append(tc)
            message = MagicMock(content=None, tool_calls=tool_calls)
        else:
            message = MagicMock(content=step.get("content", "done"), tool_calls=None)
        response = MagicMock()
        response.choices = [MagicMock(message=message, finish_reason=step.get("finish_reason", "stop"))]
        return response

    client = MagicMock()
    client.chat.completions.create.side_effect = _create
    return client


def test_agent_completes_after_coding_tool() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "README.md").write_text("hello nano\n", encoding="utf-8")
        client = _mock_client_sequence(
            [
                {"tool_calls": [{"name": READ_FILE_TOOL, "arguments": {"path": "README.md"}}]},
                {"content": "The README says hello nano."},
            ]
        )
        agent = NanoAgent(
            client,
            AgentConfig(model="test", sandbox_root=root, max_steps=4, timeout_s=30),
        )
        result = agent.run("What does README.md contain?")
        assert result.stop_reason == "completed"
        assert result.final_content == "The README says hello nano."
        assert len(result.steps) == 2
        assert result.steps[0].tool_results[0]["ok"] is True


def test_agent_capability_submit_summary_validates() -> None:
    client = _mock_client_sequence(
        [
            {
                "tool_calls": [
                    {
                        "name": SUBMIT_SUMMARY_TOOL,
                        "arguments": {
                            "schema_version": SUMMARY_SCHEMA_VERSION,
                            "title": "Visit summary",
                            "sections": [{"heading": "Chief complaint", "bullets": ["Headache"]}],
                        },
                    }
                ]
            },
            {"content": "Summary submitted."},
        ]
    )
    agent = NanoAgent(client, AgentConfig(model="test", max_steps=4))
    result = agent.run("Summarize the visit.")
    assert result.stop_reason == "completed"
    assert len(result.artifacts) == 1
    assert result.artifacts[0].metadata.capability_id == "summarize"


def test_agent_capability_invalid_args_returns_error() -> None:
    client = _mock_client_sequence(
        [
            {
                "tool_calls": [
                    {
                        "name": SUBMIT_CANDIDATE_ATOMS_TOOL,
                        "arguments": {"schema_version": "wrong", "atoms": []},
                    }
                ]
            },
            {"content": "Could not submit candidates."},
        ]
    )
    agent = NanoAgent(client, AgentConfig(model="test", max_steps=4, scribe_only=True))
    result = agent.run("Extract atoms.")
    assert result.steps[0].tool_results[0]["ok"] is False


def test_agent_max_steps_stop() -> None:
    client = _mock_client_sequence(
        [
            {"tool_calls": [{"name": LIST_DIRECTORY_TOOL, "arguments": {"path": "."}}]},
        ]
        * 5
    )
    agent = NanoAgent(client, AgentConfig(model="test", max_steps=3, timeout_s=30))
    result = agent.run("Keep listing.")
    assert result.stop_reason == "max_steps"
    assert len(result.steps) == 3


def test_agent_tool_definitions_include_coding_and_capabilities() -> None:
    agent = NanoAgent(MagicMock(), AgentConfig())
    names = {tool.name for tool in agent.tool_definitions()}
    assert READ_FILE_TOOL in names
    assert SUBMIT_CANDIDATE_ATOMS_TOOL in names
    assert SUBMIT_SUMMARY_TOOL in names


def test_agent_scribe_payload_roundtrip() -> None:
    payload = _candidate_payload()
    assert payload["schema_version"] == CANDIDATE_SCHEMA_VERSION
    client = _mock_client_sequence(
        [
            {"tool_calls": [{"name": SUBMIT_CANDIDATE_ATOMS_TOOL, "arguments": payload}]},
            {"content": "Atoms submitted."},
        ]
    )
    agent = NanoAgent(client, AgentConfig(model="test", scribe_only=True, max_steps=4))
    result = agent.run("Scribe the encounter.")
    assert result.steps[0].tool_results[0]["ok"] is True
    assert len(result.artifacts) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
