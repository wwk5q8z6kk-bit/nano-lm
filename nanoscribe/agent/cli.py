"""Nano agent CLI — inspect repo, answer questions, run tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from nanoscribe.agent.loop import AgentConfig, NanoAgent
from nanoscribe.coding_tools import LIST_DIRECTORY_TOOL, READ_FILE_TOOL, RUN_COMMAND_TOOL


def _repo_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return Path.cwd().resolve()


def _offline_client() -> Any:
    """Deterministic mock client for offline inspect/ask flows."""
    from unittest.mock import MagicMock

    state = {"turn": 0}

    def _create(**kwargs: Any) -> MagicMock:
        messages = kwargs.get("messages") or []
        last_user = next(
            (message["content"] for message in reversed(messages) if message.get("role") == "user"),
            "",
        )
        state["turn"] += 1
        if state["turn"] == 1:
            if "test" in last_user.lower():
                fn = MagicMock()
                fn.name = RUN_COMMAND_TOOL
                fn.arguments = json.dumps(
                    {
                        "argv": ["python3", "-m", "pytest", "nanoscribe/test_agent_loop.py", "-q"],
                        "cwd": ".",
                    }
                )
            else:
                fn = MagicMock()
                fn.name = LIST_DIRECTORY_TOOL
                fn.arguments = json.dumps({"path": "nanoscribe"})
            tool_call = MagicMock()
            tool_call.id = "offline_call_1"
            tool_call.function = fn
            message = MagicMock(content=None, tool_calls=[tool_call])
        else:
            message = MagicMock(
                content=f"Offline Nano agent answer for: {last_user[:200]}",
                tool_calls=None,
            )
        response = MagicMock()
        response.choices = [MagicMock(message=message, finish_reason="stop")]
        return response

    client = MagicMock()
    client.chat.completions.create.side_effect = _create
    return client


def _resolve_client(args: argparse.Namespace) -> tuple[Any, str]:
    if args.offline:
        return _offline_client(), "offline-mock"
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set OPENAI_API_KEY or RUNPOD_API_KEY, or pass --offline for deterministic mock runs."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("openai package required for live agent runs") from exc
    base_url = args.base_url
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url), args.model
    return OpenAI(api_key=api_key), args.model


def _build_config(args: argparse.Namespace, sandbox: Path) -> AgentConfig:
    return AgentConfig(
        model=args.model,
        max_steps=args.max_steps,
        timeout_s=args.timeout_s,
        sandbox_root=sandbox,
        include_coding_tools=not args.no_coding_tools,
        include_capabilities=not args.no_capabilities,
        scribe_only=args.scribe_only,
        max_tokens=args.max_tokens,
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    sandbox = _repo_root(args.repo_root)
    client, model = _resolve_client(args)
    agent = NanoAgent(client, _build_config(args, sandbox))
    prompt = (
        f"Inspect the repository at {sandbox.name}. "
        f"List the top-level entries under '{args.path}' and summarize what you find."
    )
    result = agent.run(prompt)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.stop_reason == "completed" else 1


def cmd_ask(args: argparse.Namespace) -> int:
    sandbox = _repo_root(args.repo_root)
    client, model = _resolve_client(args)
    config = _build_config(args, sandbox)
    if config.model != model:
        config = AgentConfig(
            model=model,
            max_steps=config.max_steps,
            timeout_s=config.timeout_s,
            sandbox_root=config.sandbox_root,
            include_coding_tools=config.include_coding_tools,
            include_capabilities=config.include_capabilities,
            scribe_only=config.scribe_only,
            capability_ids=config.capability_ids,
            vllm_env=config.vllm_env,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system_prompt=config.system_prompt,
        )
    agent = NanoAgent(client, config)
    result = agent.run(args.question)
    payload = result.to_dict()
    payload["answer"] = result.final_content
    print(json.dumps(payload, indent=2))
    return 0 if result.stop_reason == "completed" else 1


def cmd_test(args: argparse.Namespace) -> int:
    sandbox = _repo_root(args.repo_root)
    client, model = _resolve_client(args)
    agent = NanoAgent(client, _build_config(args, sandbox))
    argv = args.argv or ["python3", "-m", "pytest", "nanoscribe/", "-q", "--tb=no"]
    prompt = (
        "Run the project's test suite using run_command with this argv: "
        f"{json.dumps(argv)}. Report pass/fail from stdout."
    )
    result = agent.run(prompt)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.stop_reason == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nano coding + tool-calling agent")
    parser.add_argument("--repo-root", default=None, help="Repository sandbox root (default: cwd)")
    parser.add_argument("--model", default=os.environ.get("NANO_AGENT_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--offline", action="store_true", help="Use deterministic mock client")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--no-coding-tools", action="store_true")
    parser.add_argument("--no-capabilities", action="store_true")
    parser.add_argument("--scribe-only", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="List and summarize a repo path")
    inspect.add_argument("--path", default=".", help="Relative path under repo root")
    inspect.set_defaults(handler=cmd_inspect)

    ask = sub.add_parser("ask", help="Answer a question about the repo")
    ask.add_argument("question", help="Question for the agent")
    ask.set_defaults(handler=cmd_ask)

    test = sub.add_parser("test", help="Run tests via agent run_command tool")
    test.add_argument("argv", nargs="*", help="Command argv (default: pytest nanoscribe/)")
    test.set_defaults(handler=cmd_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
