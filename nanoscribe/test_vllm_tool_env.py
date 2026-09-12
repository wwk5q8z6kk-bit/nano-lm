# vLLM tool env resolution tests.
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.inference import auto_tool_choice_enabled, build_tool_choice, tool_call_parser_name
from nanoscribe.tool_calling import DEFAULT_VLLM_TOOL_ENV, resolve_vllm_tool_env


def test_default_vllm_tool_env() -> None:
    env = resolve_vllm_tool_env()
    assert env["TOOL_CALL_PARSER"] == "qwen3_coder"
    assert env["ENABLE_AUTO_TOOL_CHOICE"] == "true"


def test_resolve_precedence_deploy_over_manifest() -> None:
    env = resolve_vllm_tool_env(
        manifest_env={"TOOL_CALL_PARSER": "hermes"},
        deploy_env={"TOOL_CALL_PARSER": "qwen3_coder"},
    )
    assert env["TOOL_CALL_PARSER"] == "qwen3_coder"


def test_process_env_wins() -> None:
    os.environ["TOOL_CALL_PARSER"] = "override_parser"
    try:
        env = resolve_vllm_tool_env()
        assert env["TOOL_CALL_PARSER"] == "override_parser"
    finally:
        del os.environ["TOOL_CALL_PARSER"]


def test_auto_tool_choice_behavior() -> None:
    assert auto_tool_choice_enabled({"ENABLE_AUTO_TOOL_CHOICE": "true"})
    assert not auto_tool_choice_enabled({"ENABLE_AUTO_TOOL_CHOICE": "false"})
    choice = build_tool_choice(env={"ENABLE_AUTO_TOOL_CHOICE": "true"})
    assert choice == "auto"
    forced = build_tool_choice(env={"ENABLE_AUTO_TOOL_CHOICE": "true"}, forced_tool="submit_table")
    assert forced == {"type": "function", "function": {"name": "submit_table"}}
    required = build_tool_choice(env={"ENABLE_AUTO_TOOL_CHOICE": "false"})
    assert required == "required"


def test_parser_name_default() -> None:
    assert tool_call_parser_name() == DEFAULT_VLLM_TOOL_ENV["TOOL_CALL_PARSER"]
