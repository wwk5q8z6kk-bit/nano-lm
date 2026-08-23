"""P1 agent canary task suite — multi-teacher trajectory benchmark stub."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

AGENT_CANARY_REVISION = "p1_agent_canary_v1"
AGENT_CANARY_PATH = Path(__file__).resolve().parents[1] / "data" / f"{AGENT_CANARY_REVISION}.json"


class TaskFamily(str, Enum):
    TOOL_NEEDED = "tool_needed"
    TOOL_NOT_NEEDED = "tool_not_needed"
    SIMILAR_TOOLS = "similar_tools"
    INVALID_TOOL_ARGS = "invalid_tool_args"
    TIMEOUT_ERROR = "timeout_error"
    AMBIGUOUS_RESULT = "ambiguous_result"
    MULTI_TOOL = "multi_tool"
    VERIFY_NEEDED = "verify_needed"
    INSUFFICIENT_INFO = "insufficient_info"
    PREMATURE_STOP = "premature_stop"
    REPEATED_CALLS = "repeated_calls"
    STATE_UPDATE_RECOVERY = "state_update_recovery"


SCORE_AXES: tuple[str, ...] = (
    "tool_selection",
    "arg_validity",
    "unnecessary_call_rate",
    "recovery",
    "steps_to_resolution",
    "observation_use",
    "state_fidelity",
    "verifier_invocation",
    "abstention",
    "stop_accuracy",
    "outcome",
    "cost",
)


class NativeAgentAction(str, Enum):
    """Future P2+ native agent head — not a P1 scribing blocker."""

    TOOL_NONE = "TOOL_NONE"
    READ = "READ"
    SEARCH = "SEARCH"
    VERIFY = "VERIFY"
    QUERY = "QUERY"
    ACT = "ACT"
    ABSTAIN = "ABSTAIN"
    STOP = "STOP"


@dataclass(frozen=True, slots=True)
class AgentCanaryTask:
    task_id: str
    family: TaskFamily
    prompt: str
    tools_available: tuple[str, ...]
    gold_action: str
    requires_verifier: bool
    expect_abstain: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family.value,
            "prompt": self.prompt,
            "tools_available": list(self.tools_available),
            "gold_action": self.gold_action,
            "requires_verifier": self.requires_verifier,
            "expect_abstain": self.expect_abstain,
            "notes": self.notes,
        }


def _family_tasks(family: TaskFamily, base_idx: int) -> list[AgentCanaryTask]:
    templates: dict[TaskFamily, tuple[tuple[str, str, tuple[str, ...], bool, bool], ...]] = {
        TaskFamily.TOOL_NEEDED: (
            ("lookup lab value", "SEARCH", ("search_records", "read_note"), False, False),
            ("fetch prior imaging", "READ", ("read_note", "search_records"), False, False),
            ("confirm medication dose", "QUERY", ("query_pharmacy", "read_note"), False, False),
            ("check allergy list", "READ", ("read_note",), False, False),
        ),
        TaskFamily.TOOL_NOT_NEEDED: (
            ("summarize stated symptoms only", "TOOL_NONE", (), False, False),
            ("answer from given context", "TOOL_NONE", (), False, False),
            ("format existing facts", "TOOL_NONE", (), False, False),
            ("clarify patient quote", "TOOL_NONE", (), False, False),
        ),
        TaskFamily.SIMILAR_TOOLS: (
            ("search vs read for one line", "READ", ("read_note", "search_records"), False, False),
            ("query vs search for labs", "QUERY", ("query_labs", "search_records"), False, False),
            ("verify vs read for span", "VERIFY", ("verify_span", "read_note"), True, False),
            ("act vs query for order", "ACT", ("act_order", "query_orders"), False, False),
        ),
        TaskFamily.INVALID_TOOL_ARGS: (
            ("search with empty query", "ABSTAIN", ("search_records",), False, True),
            ("read missing note id", "ABSTAIN", ("read_note",), False, True),
            ("query malformed date", "ABSTAIN", ("query_labs",), False, True),
            ("act without required fields", "ABSTAIN", ("act_order",), False, True),
        ),
        TaskFamily.TIMEOUT_ERROR: (
            ("search times out", "RECOVERY", ("search_records",), False, False),
            ("read transient 503", "RECOVERY", ("read_note",), False, False),
            ("query slow then retry", "RECOVERY", ("query_labs",), False, False),
            ("verify endpoint error", "RECOVERY", ("verify_span",), True, False),
        ),
        TaskFamily.AMBIGUOUS_RESULT: (
            ("two matching notes", "VERIFY", ("search_records", "verify_span"), True, False),
            ("conflicting lab units", "QUERY", ("query_labs", "verify_span"), True, False),
            ("unclear speaker attribution", "VERIFY", ("read_note", "verify_span"), True, False),
            ("partial match span", "VERIFY", ("search_records", "verify_span"), True, False),
        ),
        TaskFamily.MULTI_TOOL: (
            ("read then verify span", "VERIFY", ("read_note", "verify_span"), True, False),
            ("search then query labs", "QUERY", ("search_records", "query_labs"), False, False),
            ("read query act chain", "ACT", ("read_note", "query_orders", "act_order"), False, False),
            ("search verify stop", "STOP", ("search_records", "verify_span"), True, False),
        ),
        TaskFamily.VERIFY_NEEDED: (
            ("assertion without support", "VERIFY", ("verify_span", "read_note"), True, False),
            ("transport mismatch", "VERIFY", ("verify_span",), True, False),
            ("spurious atom risk", "VERIFY", ("verify_span",), True, False),
            ("state contradiction", "VERIFY", ("verify_span", "read_note"), True, False),
        ),
        TaskFamily.INSUFFICIENT_INFO: (
            ("missing source text", "ABSTAIN", ("read_note",), False, True),
            ("no labs ordered", "ABSTAIN", ("query_labs",), False, True),
            ("unknown medication", "ABSTAIN", ("query_pharmacy",), False, True),
            ("cannot attribute speaker", "ABSTAIN", ("read_note",), False, True),
        ),
        TaskFamily.PREMATURE_STOP: (
            ("stop before verify", "VERIFY", ("verify_span", "read_note"), True, False),
            ("stop after search only", "READ", ("search_records", "read_note"), False, False),
            ("stop with open question", "QUERY", ("query_labs", "read_note"), False, False),
            ("stop without abstain", "ABSTAIN", (), False, True),
        ),
        TaskFamily.REPEATED_CALLS: (
            ("duplicate search", "STOP", ("search_records",), False, False),
            ("re-read same note", "STOP", ("read_note",), False, False),
            ("retry loop on verify", "STOP", ("verify_span",), True, False),
            ("ping-pong search read", "STOP", ("search_records", "read_note"), False, False),
        ),
        TaskFamily.STATE_UPDATE_RECOVERY: (
            ("update after new observation", "ACT", ("read_note", "act_order"), False, False),
            ("recover from wrong tool", "RECOVERY", ("search_records", "read_note"), False, False),
            ("rollback bad assertion", "VERIFY", ("verify_span", "read_note"), True, False),
            ("resume after timeout", "RECOVERY", ("search_records", "query_labs"), False, False),
        ),
    }
    out: list[AgentCanaryTask] = []
    for i, (prompt, gold, tools, req_ver, abstain) in enumerate(templates[family], start=1):
        tid = f"agent-{family.value}-{base_idx + i:02d}"
        out.append(
            AgentCanaryTask(
                task_id=tid,
                family=family,
                prompt=prompt,
                tools_available=tools,
                gold_action=gold,
                requires_verifier=req_ver,
                expect_abstain=abstain,
                notes=f"family={family.value}",
            )
        )
    return out


def build_agent_canary_tasks() -> tuple[AgentCanaryTask, ...]:
    tasks: list[AgentCanaryTask] = []
    idx = 0
    for family in TaskFamily:
        batch = _family_tasks(family, idx)
        tasks.extend(batch)
        idx += len(batch)
    return tuple(tasks)


def agent_canary_manifest() -> dict[str, Any]:
    tasks = build_agent_canary_tasks()
    families = {f.value: sum(1 for t in tasks if t.family == f) for f in TaskFamily}
    return {
        "schema": "nano.campaign.agent_canary.v1",
        "revision": AGENT_CANARY_REVISION,
        "n_tasks": len(tasks),
        "families": families,
        "score_axes": list(SCORE_AXES),
        "native_agent_head": [a.value for a in NativeAgentAction],
        "native_agent_head_note": "P2+ agent policy head; not P1 scribing blocker",
        "transfer_types": ["behavior", "outcome", "mechanism"],
        "rule": "no teacher copied wholesale; verified+ablated behavior only",
    }


def export_agent_canary_json(path: Path | None = None) -> Path:
    dest = path or AGENT_CANARY_PATH
    tasks = build_agent_canary_tasks()
    payload = {
        **agent_canary_manifest(),
        "tasks": [t.to_dict() for t in tasks],
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return dest


def load_agent_canary_tasks(path: Path | None = None) -> tuple[AgentCanaryTask, ...]:
    dest = path or AGENT_CANARY_PATH
    if not dest.is_file():
        export_agent_canary_json(dest)
    data = json.loads(dest.read_text())
    return tuple(
        AgentCanaryTask(
            task_id=row["task_id"],
            family=TaskFamily(row["family"]),
            prompt=row["prompt"],
            tools_available=tuple(row["tools_available"]),
            gold_action=row["gold_action"],
            requires_verifier=bool(row["requires_verifier"]),
            expect_abstain=bool(row["expect_abstain"]),
            notes=row.get("notes", ""),
        )
        for row in data["tasks"]
    )
