# Tool calling pins — no live RunPod calls in CI.
# Run: python3 -m pytest nanoscribe/test_tool_calling.py -q
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.adapt import CANDIDATE_SCHEMA_VERSION, CandidateAtom, ModelCandidate
from nanoscribe.adapters import ModelAdapter, ServerlessQwen38ToolAdapter, default_baseline_specs
from nanoscribe.candidate_schema import candidate_batch_parameters_schema
from nanoscribe.coding_tools import (
    APPLY_PATCH_TOOL,
    CodingToolExecutor,
    READ_FILE_TOOL,
    coding_tool_definitions,
)
from nanoscribe.encounter import AssertionState, AtomType, Certainty, Experiencer, Speaker, TemporalState, Temporality
from nanoscribe.serverless_fanout import extract_openai_content, extract_openai_message
from nanoscribe.structured_inference import generate_structured_candidates
from nanoscribe.test_adapt import _model_input
from nanoscribe.tool_calling import (
    ToolCallParser,
    ToolCallParseOutcome,
    build_openai_chat_kwargs,
    resolve_vllm_tool_env,
)
from nanoscribe.tools import (
    SUBMIT_CANDIDATE_ATOMS_TOOL,
    parse_openai_tool_calls,
    parse_tool_arguments,
    scribing_tools,
    submit_candidate_atoms_tool,
)
from nanoscribe.tool_inference import generate_tool_candidates

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tool_calls"


def _valid_candidate() -> ModelCandidate:
    return ModelCandidate(
        atoms=(
            CandidateAtom(
                atom_id="atom-neck",
                atom_type=AtomType.SYMPTOM,
                raw_value="neck",
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.STATED,
                quotes=("neck",),
            ),
        )
    )


def test_schema_generated_from_candidate_contract() -> None:
    schema = candidate_batch_parameters_schema()
    assert schema["properties"]["schema_version"]["enum"] == [CANDIDATE_SCHEMA_VERSION]
    atom_schema = schema["properties"]["atoms"]["items"]
    assert "atom_id" in atom_schema["properties"]
    assert "start" not in atom_schema["properties"]


def test_submit_tool_openai_format() -> None:
    tool = submit_candidate_atoms_tool()
    assert tool["type"] == "function"
    assert tool["function"]["name"] == SUBMIT_CANDIDATE_ATOMS_TOOL


def test_parse_valid_tool_arguments_fixture() -> None:
    raw = json.loads((FIXTURES / "valid_submit_arguments.json").read_text())
    candidate = parse_tool_arguments(raw)
    assert len(candidate.atoms) == 1
    assert candidate.atoms[0].atom_id == "atom-neck"


def test_parse_valid_tool_calls_fixture() -> None:
    calls = json.loads((FIXTURES / "valid_tool_calls.json").read_text())
    candidate = parse_openai_tool_calls(calls)
    assert candidate is not None
    assert candidate.atoms[0].quotes == ("neck",)


def test_unknown_tool_never_silent_candidate() -> None:
    calls = json.loads((FIXTURES / "unknown_tool_calls.json").read_text())
    parser = ToolCallParser()
    result = parser.parse_tool_calls(calls)
    assert result.outcome is ToolCallParseOutcome.UNKNOWN_TOOL
    assert parser.to_model_candidate(result).atoms == ()


def test_invalid_schema_args() -> None:
    calls = json.loads((FIXTURES / "invalid_schema_tool_calls.json").read_text())
    parser = ToolCallParser()
    result = parser.parse_tool_calls(calls)
    assert result.outcome is ToolCallParseOutcome.INVALID_ARGS
    assert parser.to_model_candidate(result).atoms == ()


def test_text_fallback_not_silent() -> None:
    text = (FIXTURES / "text_fallback.txt").read_text()
    parser = ToolCallParser()
    result = parser.parse_text_json(text)
    assert result.outcome is ToolCallParseOutcome.INVALID_ARGS
    assert parser.to_model_candidate(result).atoms == ()


def test_structured_json_backward_compat() -> None:
    payload = _valid_candidate().to_dict()
    parser = ToolCallParser()
    result = parser.parse_text_json(json.dumps(payload))
    assert result.outcome is ToolCallParseOutcome.JSON_FALLBACK
    assert len(result.candidate.atoms) == 1


def test_extract_openai_content_prefers_tool_arguments() -> None:
    output = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": json.loads((FIXTURES / "valid_tool_calls.json").read_text()),
                }
            }
        ]
    }
    raw = extract_openai_content(output)
    candidate = parse_tool_arguments(json.loads(raw))
    assert candidate.atoms[0].atom_id == "atom-neck"


def test_extract_openai_message() -> None:
    output = {"choices": [{"message": {"content": "hello"}}]}
    assert extract_openai_message(output)["content"] == "hello"


def test_build_openai_chat_kwargs_tool_path() -> None:
    from nanoscribe.tools import scribing_tool_definitions, tool_choice_submit_candidates

    kwargs = build_openai_chat_kwargs(
        model="Qwen/Qwen3.8-27B",
        system_prompt="sys",
        user_prompt="user",
        tools=scribing_tool_definitions(),
        tool_choice=tool_choice_submit_candidates(),
    )
    assert "tools" in kwargs
    assert kwargs["tool_choice"]["function"]["name"] == SUBMIT_CANDIDATE_ATOMS_TOOL
    assert "response_format" not in kwargs


def test_build_openai_chat_kwargs_structured_default() -> None:
    kwargs = build_openai_chat_kwargs(
        model="m",
        system_prompt="sys",
        user_prompt="user",
        tools=[],
        tool_choice=None,
        use_json_object=True,
    )
    assert kwargs["response_format"] == {"type": "json_object"}


def test_resolve_vllm_tool_env_precedence() -> None:
    with patch.dict("os.environ", {"TOOL_CALL_PARSER": "override"}, clear=False):
        env = resolve_vllm_tool_env(
            manifest_env={"TOOL_CALL_PARSER": "manifest"},
            deploy_env={"TOOL_CALL_PARSER": "deploy"},
        )
    assert env["TOOL_CALL_PARSER"] == "override"


def test_tool_adapter_implements_protocol() -> None:
    adapter = ServerlessQwen38ToolAdapter(endpoint_id="test-endpoint")
    assert isinstance(adapter, ModelAdapter)


def test_generate_tool_candidates_mocked() -> None:
    valid = _valid_candidate()
    fn = MagicMock()
    fn.name = SUBMIT_CANDIDATE_ATOMS_TOOL
    fn.arguments = json.dumps(valid.to_dict())
    tool_call = MagicMock()
    tool_call.id = "c1"
    tool_call.function = fn
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=None, tool_calls=[tool_call]))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    batch, latency_s, memory = generate_tool_candidates(
        _model_input(),
        (default_baseline_specs()[0],),
        client=mock_client,
        model="Qwen/Qwen3.8-27B",
    )
    assert len(batch.atoms) == 1
    assert latency_s >= 0
    assert memory == 0


def test_structured_inference_still_parses_json() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps(_valid_candidate().to_dict())))
    ]
    mock_client.chat.completions.create.return_value = mock_response
    batch, _, _ = generate_structured_candidates(
        _model_input(),
        (default_baseline_specs()[0],),
        client=mock_client,
        model="m",
    )
    assert len(batch.atoms) == 1


def test_coding_tools_sandbox_read() -> None:
    root = Path(__file__).resolve().parents[1]
    executor = CodingToolExecutor(sandbox_root=root)
    result = executor.execute(READ_FILE_TOOL, {"path": "nanoscribe/tools.py", "limit": 20})
    assert result.ok
    assert "CandidateAtom" in result.data["content"]


def test_coding_tools_reject_path_escape() -> None:
    root = Path(__file__).resolve().parents[1]
    executor = CodingToolExecutor(sandbox_root=root)
    result = executor.execute(READ_FILE_TOOL, {"path": "../../../etc/passwd"})
    assert not result.ok
    assert "sandbox" in (result.error or "").lower()


def test_coding_tool_definitions_count() -> None:
    assert len(coding_tool_definitions()) == 5


def test_scribing_tools_include_submit_only_by_default() -> None:
    tools = scribing_tools()
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == SUBMIT_CANDIDATE_ATOMS_TOOL


def test_apply_patch_dry_run() -> None:
    root = Path(__file__).resolve().parents[1]
    executor = CodingToolExecutor(sandbox_root=root)
    result = executor.execute(APPLY_PATCH_TOOL, {"patch": "--- a\n+++ b\n", "apply": False})
    assert result.ok
    assert result.data["dry_run"] is True


def test_resolve_inference_tool_choice_auto() -> None:
    from nanoscribe.inference.tool_registry import resolve_inference_tool_choice

    choice = resolve_inference_tool_choice(env={"ENABLE_AUTO_TOOL_CHOICE": "true"}, scribe_only=True)
    assert choice == "auto"


def test_resolve_inference_tool_choice_required_scribe() -> None:
    from nanoscribe.inference.tool_registry import resolve_inference_tool_choice

    choice = resolve_inference_tool_choice(env={"ENABLE_AUTO_TOOL_CHOICE": "false"}, scribe_only=True)
    assert choice == {"type": "function", "function": {"name": SUBMIT_CANDIDATE_ATOMS_TOOL}}


def test_resolve_inference_tool_choice_explicit_override() -> None:
    from nanoscribe.inference.tool_registry import resolve_inference_tool_choice

    assert resolve_inference_tool_choice("none", scribe_only=True) == "none"
    forced = {"type": "function", "function": {"name": "submit_table"}}
    assert resolve_inference_tool_choice(forced, scribe_only=True) == forced


def test_agent_tool_definitions_include_summarize_table() -> None:
    from nanoscribe.inference.tool_registry import agent_tool_definitions

    names = {tool.name for tool in agent_tool_definitions()}
    assert names == {"submit_candidate_atoms", "submit_summary", "submit_table"}


def test_structured_inference_use_tools_path() -> None:
    valid = _valid_candidate()
    fn = MagicMock()
    fn.name = SUBMIT_CANDIDATE_ATOMS_TOOL
    fn.arguments = json.dumps(valid.to_dict())
    tool_call = MagicMock()
    tool_call.id = "c1"
    tool_call.function = fn
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=None, tool_calls=[tool_call]))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    batch, latency_s, memory = generate_structured_candidates(
        _model_input(),
        (default_baseline_specs()[0],),
        client=mock_client,
        model="Qwen/Qwen3.8-27B",
        use_tools=True,
    )
    assert len(batch.atoms) == 1
    assert latency_s >= 0
    assert memory == 0
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "tools" in kwargs
    assert "response_format" not in kwargs
