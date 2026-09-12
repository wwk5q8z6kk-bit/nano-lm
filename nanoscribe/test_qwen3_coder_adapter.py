# Qwen3-Coder inference adapter tests.
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.capabilities import CapabilityId
from nanoscribe.inference import Qwen3CoderInferenceAdapter, normalize_openai_message
from nanoscribe.render_table import render_table_markdown
from nanoscribe.artifacts import TableSpec
from nanoscribe.tool_calling import ToolCallParseOutcome


def test_normalize_embedded_qwen_tool_calls() -> None:
    args = {
        "schema_version": "nano.summary.v0",
        "title": "T",
        "sections": [{"heading": "H", "bullets": ["b"]}],
    }
    payload = json.dumps({"name": "submit_summary", "arguments": args})
    content = f"thinking\n<|tool_call|>{payload}<|/tool_call|>"
    normalized = normalize_openai_message({"role": "assistant", "content": content})
    assert normalized.get("tool_calls")
    assert normalized["tool_calls"][0]["function"]["name"] == "submit_summary"


def test_qwen_adapter_parses_embedded_summary() -> None:
    args = {
        "schema_version": "nano.summary.v0",
        "title": "Visit",
        "sections": [{"heading": "Symptoms", "bullets": ["Pain"]}],
    }
    payload = json.dumps({"name": "submit_summary", "arguments": args})
    message = {
        "role": "assistant",
        "content": f"<|tool_call|>{payload}<|/tool_call|>",
    }
    adapter = Qwen3CoderInferenceAdapter()
    result = adapter.parse_message(message)
    assert result.ok
    assert result.capability_id is CapabilityId.SUMMARIZE
    assert result.artifact is not None
    assert result.artifact.metadata.capability_id == "summarize"


def test_qwen_adapter_auto_tool_choice_flag() -> None:
    adapter = Qwen3CoderInferenceAdapter(env={"ENABLE_AUTO_TOOL_CHOICE": "false"})
    assert not adapter.auto_tool_choice
    kwargs = adapter.chat_completion_kwargs(
        model="Qwen/Qwen3-Coder-30B",
        system_prompt="sys",
        user_prompt="user",
        tools=[],
    )
    assert kwargs["tool_choice"] == "required"


def test_table_markdown_renderer() -> None:
    spec = TableSpec.from_dict(
        {
            "schema_version": "nano.table.v0",
            "title": "Meds",
            "columns": [
                {"key": "name", "label": "Name"},
                {"key": "dose", "label": "Dose"},
            ],
            "rows": [["ibuprofen", "400mg"]],
        }
    )
    md = render_table_markdown(spec)
    assert "## Meds" in md
    assert "| Name | Dose |" in md
    assert "| ibuprofen | 400mg |" in md


def test_openai_tool_calls_path() -> None:
    adapter = Qwen3CoderInferenceAdapter()
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "submit_table",
                                "arguments": json.dumps(
                                    {
                                        "schema_version": "nano.table.v0",
                                        "columns": [{"key": "k", "label": "K"}],
                                        "rows": [["v"]],
                                    }
                                ),
                            },
                        }
                    ]
                }
            }
        ]
    }
    result = adapter.parse_openai_response(response)
    assert result.outcome is ToolCallParseOutcome.SUCCESS
    assert result.capability_id is CapabilityId.TABLE
