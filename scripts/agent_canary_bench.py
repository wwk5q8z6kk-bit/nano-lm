#!/usr/bin/env python3
"""Agent canary bench — ephemeral serverless or managed-ref teacher probe."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.agent_canary import SCORE_AXES, TaskFamily, load_agent_canary_tasks
from nanoscribe.campaign import CampaignLedger, DEFAULT_LEDGER
from nanoscribe.inference.tool_registry import agent_tool_definitions
from nanoscribe.serverless_config import configure_pause
from nanoscribe.serverless_inference import _openai_client

MANIFEST_PATH = ROOT / "artifacts/campaign/manifests/nano_agent_canary_v1.json"
OUT_PATH = ROOT / "artifacts/campaign/agent_canary_v1_results.json"
HUB_VLLM = "cm8h09d9n000008jvh2rqdsmb"

MANAGED_TEACHERS = {
    "gpt_oss_120b": {
        "base_url": "https://api.runpod.ai/v2/gpt-oss-120b/openai/v1",
        "model": "openai/gpt-oss-120b",
        "surface": "managed_reference",
        "role": "structured_output",
    },
    # Generic-Qwen control (briefing SS XXII). Answers the baseline half of
    # SS LXII Q13 -- "does Qwen3-Coder-Next materially outperform generic Qwen on
    # tool/recovery behavior" -- at zero cost, before any paid Coder-Next probe.
    "qwen3_32b_awq": {
        "base_url": "https://api.runpod.ai/v2/qwen3-32b-awq/openai/v1",
        "model": "Qwen/Qwen3-32B-AWQ",
        "surface": "managed_reference",
        "role": "generic_qwen_control",
        # Qwen3 emits <think>...</think> inline in content. Left on, a 48-task
        # sweep spends most of its budget and wall-clock on reasoning the action
        # task does not need; vLLM honours this chat-template flag to skip it.
        "disable_thinking": True,
    },
}

EPHEMERAL_TEACHERS = {
    "glm_45_air_fp8": {
        "model": "THUDM/GLM-4.5-Air-FP8",
        "gpu": "NVIDIA A100 80GB PCIe",
        "probe_cost_est": 2.0,
    },
}

ACTIONS = (
    "TOOL_NONE",
    "READ",
    "SEARCH",
    "VERIFY",
    "QUERY",
    "ACT",
    "ABSTAIN",
    "STOP",
    "RECOVERY",
)


def _subset_tasks(n_tasks: int) -> list:
    """Round-robin across families so n>12 deepens every axis instead of capping at one per family.

    Ranking teachers per capability needs n>1 per axis; the previous one-per-family
    pick made every per-capability mean an n=1 coin flip.
    """
    tasks = load_agent_canary_tasks()
    by_family: dict[TaskFamily, list] = defaultdict(list)
    for t in tasks:
        by_family[t.family].append(t)
    picked = []
    depth = 0
    while len(picked) < n_tasks:
        added = False
        for family in TaskFamily:
            bucket = by_family[family]
            if depth < len(bucket):
                picked.append(bucket[depth])
                added = True
                if len(picked) >= n_tasks:
                    break
        if not added:
            break
        depth += 1
    return picked[:n_tasks]


def _system_prompt(*, include_agent_tools: bool) -> str:
    tools_note = (
        "You may also use registered tools (submit_candidate_atoms, submit_summary, submit_table) when appropriate."
        if include_agent_tools
        else ""
    )
    # The action space must be DEFINED, not just enumerated. Listing nine bare
    # tokens left STOP / TOOL_NONE / ABSTAIN mutually indistinguishable, so a
    # teacher answering TOOL_NONE where gold was STOP was graded wrong for a
    # distinction it was never told about — measuring the prompt, not the model.
    definitions = "\n".join(
        [
            "TOOL_NONE - no tool call is needed right now; proceed from what you already have",
            "READ - read a specific document or note you already have a pointer to",
            "SEARCH - look for records you do not yet have a pointer to",
            "VERIFY - check a claim or span against its supporting evidence",
            "QUERY - retrieve structured values such as labs or orders",
            "ACT - perform a state-changing operation such as placing an order",
            "ABSTAIN - decline to answer because the available information is insufficient",
            "STOP - end the episode: the task is complete, or no further action can change the outcome",
            "RECOVERY - the previous action failed; take a corrective or alternative step",
        ]
    )
    return (
        "You are an agent policy model for clinical workflows.\n"
        "Choose the single best next action.\n\n"
        f"{definitions}\n\n"
        f"Respond with exactly one action token on the first line from: {', '.join(ACTIONS)}. "
        f"{tools_note}"
    ).strip()


def _user_prompt(task) -> str:
    tools = ", ".join(task.tools_available) if task.tools_available else "(none)"
    return (
        f"Task: {task.prompt}\n"
        f"Available tools: {tools}\n"
        f"Requires verifier: {task.requires_verifier}\n"
        f"Expect abstain if insufficient: {task.expect_abstain}\n"
        "First line: action token only."
    )


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove inline reasoning blocks before parsing an action.

    Qwen3 and GLM emit their chain of thought inline in `content` as
    <think>...</think> rather than in a separate reasoning_content field. Parsing
    the first line would otherwise see "<think>" and find no action token,
    scoring a correct answer as a parse failure. An unclosed <think> (the block
    was cut off by max_tokens) is dropped entirely.
    """
    if not text:
        return ""
    text = _THINK_RE.sub(" ", text)
    text = _OPEN_THINK_RE.sub(" ", text)
    return text.strip()


def _parse_action(text: str) -> str:
    text = strip_thinking(text)
    if not text:
        return ""
    first = text.strip().splitlines()[0].strip().upper()
    for action in ACTIONS:
        if action in first:
            return action
    m = re.search(r"\b(" + "|".join(ACTIONS) + r")\b", text.upper())
    if m:
        return m.group(1)
    # No known action anywhere in the response. Previously this fell back to
    # first.split()[0], which returned whatever token happened to start the line
    # — a model emitting JSON scored the action '{', which _score_task then
    # graded as a wrong answer instead of an unparsed response. Returning ""
    # marks the row unparsed so it is excluded from per-capability means.
    return ""


def _score_task(task, predicted: str) -> dict[str, float | bool]:
    """Score a parsed action. Callers MUST NOT invoke this on an unparsed response.

    An empty prediction is 'we failed to read the teacher', not 'the teacher chose
    badly'. Scoring it conflates transport failure with capability: e.g.
    unnecessary_call_rate = float(pred not in {TOOL_NONE, ABSTAIN}) awards the
    max-badness 1.0 to a non-response. Unparsed rows are excluded from aggregation
    instead (see _aggregate / row['parsed']).
    """
    gold = task.gold_action.upper()
    pred = predicted.upper()
    if not pred:
        return {}
    match = pred == gold or (gold == "RECOVERY" and pred in {"RECOVERY", "SEARCH", "READ"})
    scores: dict[str, float | bool] = {"outcome": float(match)}
    if task.expect_abstain:
        scores["abstention"] = float(pred == "ABSTAIN")
    if task.requires_verifier:
        scores["verifier_invocation"] = float(pred == "VERIFY")
    if task.family in {TaskFamily.TIMEOUT_ERROR, TaskFamily.STATE_UPDATE_RECOVERY}:
        scores["recovery"] = float(pred in {"RECOVERY", gold})
    if task.family in {TaskFamily.REPEATED_CALLS, TaskFamily.PREMATURE_STOP}:
        scores["stop_accuracy"] = float(pred in {"STOP", gold})
    if task.family in {TaskFamily.TOOL_NEEDED, TaskFamily.SIMILAR_TOOLS, TaskFamily.MULTI_TOOL}:
        scores["tool_selection"] = float(match and pred not in {"TOOL_NONE", "ABSTAIN"})
    if task.family == TaskFamily.INVALID_TOOL_ARGS:
        scores["arg_validity"] = float(pred == "ABSTAIN")
    if task.family == TaskFamily.TOOL_NOT_NEEDED:
        scores["unnecessary_call_rate"] = float(pred not in {"TOOL_NONE", "ABSTAIN"})
    return scores


def _deploy_glm(name: str) -> dict[str, Any]:
    cfg = EPHEMERAL_TEACHERS["glm_45_air_fp8"]
    proc = subprocess.run(
        [
            "runpodctl",
            "serverless",
            "create",
            "--hub-id",
            HUB_VLLM,
            "--gpu-id",
            cfg["gpu"],
            "--model-reference",
            f"https://huggingface.co/{cfg['model']}:main",
            "--name",
            name,
            "--workers-min",
            "0",
            "--workers-max",
            "1",
            "--idle-timeout",
            "120",
            "--env",
            f"MODEL_NAME={cfg['model']}",
            "--env",
            "TRUST_REMOTE_CODE=true",
            "--env",
            "DTYPE=bfloat16",
            "--env",
            "TOOL_CALL_PARSER=qwen3_coder",
            "--env",
            "ENABLE_AUTO_TOOL_CHOICE=true",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip() or proc.stdout.strip()}
    data = json.loads(proc.stdout)
    return {"ok": True, "endpoint_id": data.get("id"), "model": cfg["model"], "cost_per_hr": data.get("costPerHr")}


def _delete_endpoint(endpoint_id: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["runpodctl", "serverless", "delete", endpoint_id, "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"deleted": False, "error": proc.stderr.strip() or proc.stdout.strip()}
    return {"deleted": True, "endpoint_id": endpoint_id}


def _wait_ready(endpoint_id: str, timeout_s: int = 300) -> bool:
    from nanoscribe.serverless_config import fetch_health

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            health = fetch_health(endpoint_id)
            if health.get("jobs", {}).get("completed", 0) >= 0 and health.get("workers", {}).get("ready", 0) >= 1:
                return True
            if health.get("workers", {}).get("ready", 0) >= 1:
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def run_bench(
    *,
    tasks,
    teacher_id: str,
    base_url: str | None,
    endpoint_id: str | None,
    model: str,
    include_agent_tools: bool,
    max_tokens: int,
    reasoning_effort: str | None = None,
    disable_thinking: bool = False,
) -> tuple[list[dict[str, Any]], float]:
    client = _openai_client(endpoint_id=endpoint_id, base_url=base_url)
    tools = agent_tool_definitions() if include_agent_tools else None
    results: list[dict[str, Any]] = []
    total_latency = 0.0
    for task in tasks:
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": _system_prompt(include_agent_tools=include_agent_tools)},
                {"role": "user", "content": _user_prompt(task)},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if disable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"
        finish_reason: str | None = None
        completion_tokens: int | None = None
        source = "none"
        try:
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                # Not every teacher's OpenAI shim accepts reasoning_effort; a
                # rejected parameter would otherwise fail all tasks and be
                # misread as a capability score of zero.
                if "reasoning_effort" in kwargs and "reasoning_effort" in str(exc):
                    kwargs.pop("reasoning_effort")
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise
            choice = response.choices[0]
            msg = choice.message
            finish_reason = choice.finish_reason
            completion_tokens = getattr(response.usage, "completion_tokens", None)
            raw = (msg.content or "").strip()
            if raw:
                source = "content"
            if not raw and msg.tool_calls:
                fn = msg.tool_calls[0].function
                raw = (fn.name or "") or (fn.arguments or "")
                if raw:
                    source = "tool_call"
            if not raw:
                # Reasoning models (gpt-oss, GLM thinking mode) emit the action in
                # reasoning_content and leave content empty when the token budget is
                # exhausted mid-thought (finish_reason == "length").
                reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
                raw = reasoning.strip()
                if raw:
                    source = "reasoning_content"
            predicted = _parse_action(raw)
            err = None
        except Exception as exc:
            predicted = ""
            raw = ""
            err = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started
        total_latency += latency
        parsed = bool(predicted) and not err
        # Unparsed rows carry no scores; they are excluded from per-capability means
        # so a transport failure never masquerades as a capability deficit.
        scores = _score_task(task, predicted) if parsed else {}
        results.append(
            {
                "task_id": task.task_id,
                "family": task.family.value,
                "gold_action": task.gold_action,
                "predicted_action": predicted,
                "raw_snippet": raw[:200],
                "error": err,
                "parsed": parsed,
                "action_source": source,
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "latency_s": round(latency, 3),
                "scores": scores,
            }
        )
    return results, total_latency


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate over PARSED rows only, carrying per-axis n so means are interpretable.

    A per-capability mean computed over a different n than its peers is not
    comparable across teachers; every axis therefore reports {mean, n}. Unparsed
    rows are counted separately as a transport-health signal (parse_rate), never
    folded into capability scores.
    """
    parsed_rows = [r for r in results if r.get("parsed")]
    axis_vals: dict[str, list[float]] = defaultdict(list)
    family_vals: dict[str, list[float]] = defaultdict(list)
    for row in parsed_rows:
        family_vals[row["family"]].append(float(row["scores"].get("outcome", 0.0)))
        for axis, val in row["scores"].items():
            if isinstance(val, (int, float)):
                axis_vals[axis].append(float(val))
    per_capability = {
        axis: (
            {"mean": round(sum(vals) / len(vals), 4), "n": len(vals)}
            if (vals := axis_vals.get(axis, []))
            else {"mean": None, "n": 0}
        )
        for axis in SCORE_AXES
    }
    per_family = {
        fam: {"mean": round(sum(v) / len(v), 4), "n": len(v)} for fam, v in family_vals.items()
    }
    outcomes = [float(r["scores"].get("outcome", 0.0)) for r in parsed_rows]
    n_unparsed = sum(1 for r in results if not r.get("parsed"))
    return {
        "per_capability": per_capability,
        "per_family_outcome": per_family,
        "outcome_mean": round(sum(outcomes) / len(outcomes), 4) if outcomes else None,
        "outcome_n": len(outcomes),
        "n_tasks": len(results),
        "n_parsed": len(parsed_rows),
        "n_unparsed": n_unparsed,
        "parse_rate": round(len(parsed_rows) / len(results), 4) if results else None,
        "n_errors": sum(1 for r in results if r.get("error")),
        "unparsed_finish_reasons": dict(
            Counter(r.get("finish_reason") or "none" for r in results if not r.get("parsed"))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent canary teacher probe")
    parser.add_argument("--tasks", type=int, default=48, help="tasks, round-robin across families")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="reasoning models need headroom; 128 truncated gpt-oss mid-thought (finish_reason=length)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        help="'low' keeps reasoning models from spending the whole budget before emitting content",
    )
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--teacher", choices=["auto", "glm", "gpt_oss", "qwen3_32b_awq"], default="auto"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="results path (default artifacts/campaign/agent_canary_v1_results.json); "
        "use a per-teacher path to keep scoreboards side by side",
    )
    # store_true with default=True could never be switched off. The P1 scribing
    # tools (submit_candidate_atoms/summary/table) also invite a structured-output
    # model to answer the SCRIBING task instead of the action-policy one: gpt-oss
    # returned {"atoms":[...]} on 5/48 tasks and scored unparsed. Default off for
    # a clean action-policy measurement; opt in to test tool-surface interference.
    parser.add_argument(
        "--include-agent-tools",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="expose the P1 scribing tools to the teacher (default: off)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = _subset_tasks(args.tasks)
    ledger = CampaignLedger.load(args.ledger)
    allowed, reason = ledger.budget_gate(min(args.budget, EPHEMERAL_TEACHERS["glm_45_air_fp8"]["probe_cost_est"]))
    if not allowed and not args.dry_run:
        print(json.dumps({"error": "budget_gate_failed", "reason": reason}))
        return 1

    endpoint_id: str | None = None
    deleted_endpoints: list[str] = []
    teacher_id = ""
    base_url: str | None = None
    model = ""
    deploy_cost = 0.0
    deploy_error: str | None = None

    if args.teacher in {"auto", "glm"} and not args.dry_run:
        ts = datetime.now(UTC).strftime("%Y%m%d%H%M")
        deploy = _deploy_glm(f"agent-canary-glm-{ts}")
        if deploy.get("ok"):
            endpoint_id = str(deploy["endpoint_id"])
            teacher_id = "glm_45_air_fp8"
            model = deploy["model"]
            deploy_cost = float(deploy.get("cost_per_hr") or 2.0) * 0.25
            ledger.commit("agent_canary", "GLM-4.5-Air-FP8 ephemeral probe", deploy_cost, notes="est 15min")
            ledger.save(args.ledger)
            if not _wait_ready(endpoint_id, timeout_s=90):
                deploy_error = "endpoint_not_ready_within_90s"
                configure_pause(endpoint_id)
                deleted = _delete_endpoint(endpoint_id)
                if deleted.get("deleted"):
                    deleted_endpoints.append(endpoint_id)
                endpoint_id = None
        else:
            deploy_error = deploy.get("error")

    if not endpoint_id:
        teacher_id = args.teacher if args.teacher in MANAGED_TEACHERS else "gpt_oss_120b"
        cfg = MANAGED_TEACHERS[teacher_id]
        base_url = cfg["base_url"]
        model = cfg["model"]
        if deploy_error:
            print(json.dumps({"glm_deploy_failed": deploy_error, "fallback": teacher_id}), file=sys.stderr)

    if args.dry_run:
        payload = {
            "dry_run": True,
            "n_tasks": len(tasks),
            "teacher_planned": teacher_id or "glm_then_gpt_oss",
            "task_ids": [t.task_id for t in tasks],
        }
        print(json.dumps(payload, indent=2))
        return 0

    results, total_latency = run_bench(
        tasks=tasks,
        teacher_id=teacher_id,
        base_url=base_url,
        endpoint_id=endpoint_id,
        model=model,
        include_agent_tools=args.include_agent_tools,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort or None,
        disable_thinking=bool(MANAGED_TEACHERS.get(teacher_id, {}).get("disable_thinking")),
    )

    if endpoint_id:
        configure_pause(endpoint_id)
        deleted = _delete_endpoint(endpoint_id)
        if deleted.get("deleted"):
            deleted_endpoints.append(endpoint_id)

    summary = _aggregate(results)
    winners_by_family = {
        fam: max(
            (r for r in results if r["family"] == fam and r.get("parsed")),
            key=lambda r: float(r["scores"].get("outcome", 0.0)),
            default={"task_id": None},
        )["task_id"]
        for fam in {r["family"] for r in results}
    }

    payload = {
        "schema": "nano.campaign.agent_canary_results.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "experiment_id": "nano_agent_canary_v1",
        "teacher_id": teacher_id,
        "model": model,
        "endpoint_id": endpoint_id,
        "endpoint_deleted": deleted_endpoints,
        "include_agent_tools": args.include_agent_tools,
        "deploy_error": deploy_error,
        "spend_usd_est": round(deploy_cost, 4),
        "total_latency_s": round(total_latency, 3),
        "summary": summary,
        "winners_by_family_task": winners_by_family,
        "results": results,
    }
    out_path = args.out or OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ledger.save(args.ledger)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
