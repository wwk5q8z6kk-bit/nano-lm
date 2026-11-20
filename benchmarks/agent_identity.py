"""Agent-run identity — content-addressed identity for agentic evaluation runs.

Why this module exists
----------------------
`benchmarks/adapters/lm_eval/schemas.py::compute_run_id` binds four inputs:
benchmark manifest, model/solver manifest, config, code commit. That is sufficient
for a checkpoint-scored task and insufficient for an agentic one, where the
evaluated unit is the whole system that produces the engineering result.

Identity binding in this repository is by **explicit enumeration**. The one
adjacent component that is already computed — `environment_hash` — is deliberately
excluded from `compute_run_id` (computed `runner.py:217`, stored on the manifest
`runner.py:302`, not passed at `runner.py:219-224`), which shows that environment
is not currently treated as identity-bearing. Nothing in the contract requires
agent policy, tool inventory, permissions, budget, or retry policy to appear at
all. Two runs of one checkpoint under different scaffolds therefore share a
`run_id`.

This module is **strictly additive**. It does not modify `compute_run_id`, whose
output is the on-disk directory name for every existing run. Legacy runs keep
their identity; agentic runs get a superset identity, with the legacy id carried
as lineage.

Scope: infrastructure for identifiability and reproducibility only. Registering
suites, authorizing execution, and collecting scores remain out of scope
(`benchmarks/BENCHMARK_CONSTITUTION.md`; `PROGRAM1 = NOT_AUTHORIZED`).

Layering debt: `canonical_json` / `sha256_hex` are imported from the lm_eval
adapter because that is where they live today. They are two pure functions with
no adapter coupling; extract them to a shared module when a third consumer
appears, rather than duplicating them here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.adapters.lm_eval.schemas import canonical_json, sha256_hex

AGENT_IDENTITY_SCHEMA_VERSION = "benchmark_lab.agent_run_identity.v1"

#: The eight identity-bearing components of an agentic evaluation run, in the
#: order they are enumerated by the owner directive. Every one is required; the
#: contract has no optional identity component by design.
IDENTITY_COMPONENTS: tuple[str, ...] = (
    "model_pin_sha256",
    "agent_policy_pin_sha256",
    "tool_pin_sha256",
    "environment_pin_sha256",
    "task_spec_pin_sha256",
    "budget_pin_sha256",
    "evaluator_pin_sha256",
    "code_pin_sha256",
)

_ALLOWED_NETWORK_POLICIES = frozenset({"none", "allowlist", "open"})


class IdentityContractError(ValueError):
    """Raised when an identity component is absent, empty, or malformed.

    Fail-closed by design: an unpinned component must stop the run rather than
    silently produce an identity that cannot distinguish it from another run.
    """


def _require_digest(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise IdentityContractError(f"{name} must be a non-empty string, got {value!r}")
    stripped = value.strip().lower()
    if len(stripped) != 64 or any(c not in "0123456789abcdef" for c in stripped):
        raise IdentityContractError(f"{name} must be a 64-char sha256 hex digest, got {value!r}")
    if stripped == "0" * 64:
        raise IdentityContractError(f"{name} is the null digest — placeholder pins are not valid")
    return stripped


@dataclass(frozen=True)
class AgentPolicyPin:
    """The scaffold: what decides the next action.

    `policy_source_sha256` is authoritative; `policy_id` is a human label and is
    included in the digest so that renaming a policy is a visible identity change.
    """

    policy_id: str
    policy_source_sha256: str
    system_prompt_sha256: str
    context_policy: str = "unspecified"

    def digest(self) -> str:
        if not self.policy_id:
            raise IdentityContractError("policy_id must be non-empty")
        _require_digest("policy_source_sha256", self.policy_source_sha256)
        _require_digest("system_prompt_sha256", self.system_prompt_sha256)
        return sha256_hex(canonical_json(asdict(self)))


@dataclass(frozen=True)
class ToolPin:
    """The tool set *and its permissions*.

    Permissions are part of identity, not configuration: the same tool inventory
    under read-only versus write permission is a different system.
    """

    tool_manifest_sha256: str
    tool_ids: list[str]
    permissions: dict[str, str]

    def digest(self) -> str:
        _require_digest("tool_manifest_sha256", self.tool_manifest_sha256)
        if sorted(self.tool_ids) != sorted(set(self.tool_ids)):
            raise IdentityContractError("tool_ids must be unique")
        missing = [t for t in self.tool_ids if t not in self.permissions]
        if missing:
            raise IdentityContractError(f"tools without a declared permission: {sorted(missing)}")
        payload = {
            "tool_manifest_sha256": self.tool_manifest_sha256,
            "tool_ids": sorted(self.tool_ids),
            "permissions": dict(sorted(self.permissions.items())),
        }
        return sha256_hex(canonical_json(payload))


@dataclass(frozen=True)
class EnvironmentPin:
    """Container and world state.

    `image_digest` must be a digest, never a mutable tag: `:latest` is not an
    identity. `seed` may be null only when `deterministic` is false.
    """

    image_digest: str
    network_policy: str
    deterministic: bool
    seed: int | None = None

    def digest(self) -> str:
        if not self.image_digest or ":latest" in self.image_digest:
            raise IdentityContractError("image_digest must be an immutable digest, not a tag")
        if self.network_policy not in _ALLOWED_NETWORK_POLICIES:
            raise IdentityContractError(
                f"network_policy must be one of {sorted(_ALLOWED_NETWORK_POLICIES)}"
            )
        if self.deterministic and self.seed is None:
            raise IdentityContractError("deterministic runs must pin a seed")
        return sha256_hex(canonical_json(asdict(self)))


@dataclass(frozen=True)
class BudgetPin:
    """Budget *and retry policy*.

    `retry_policy` and `repeats` are identity components rather than result
    metadata: unreported retries are a benchmark-integrity failure, and the only
    structural defense is making the policy part of what identifies the run.
    `repeats` is required because pass^k is undefined without it.
    """

    retry_policy: str
    repeats: int
    max_steps: int | None = None
    max_wall_clock_s: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None

    def digest(self) -> str:
        if not self.retry_policy:
            raise IdentityContractError("retry_policy must be explicit; use 'none' if there are no retries")
        if not isinstance(self.repeats, int) or self.repeats < 1:
            raise IdentityContractError("repeats must be an integer >= 1")
        return sha256_hex(canonical_json(asdict(self)))


@dataclass(frozen=True)
class EvaluatorPin:
    """The grader: harness version plus the scoring implementation.

    `judge_calibration_sha256` is null unless an LLM judge contributes to the
    score. When a judge is used, an uncalibrated judge must not be identity-
    bearing-but-unmeasured — see the calibration protocol in
    `papers/SWE_CAPABILITY_EVALUATION_PROPOSAL.md` §2.2.
    """

    evaluator_id: str
    evaluator_version: str
    scorer_sha256: str
    judge_calibration_sha256: str | None = None

    def digest(self) -> str:
        if not self.evaluator_id or not self.evaluator_version:
            raise IdentityContractError("evaluator_id and evaluator_version must be non-empty")
        _require_digest("scorer_sha256", self.scorer_sha256)
        if self.judge_calibration_sha256 is not None:
            _require_digest("judge_calibration_sha256", self.judge_calibration_sha256)
        return sha256_hex(canonical_json(asdict(self)))


@dataclass(frozen=True)
class CodePin:
    """Code commit and resolved configuration."""

    code_git_commit: str
    config_sha256: str

    def digest(self) -> str:
        commit = (self.code_git_commit or "").strip().lower()
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise IdentityContractError("code_git_commit must be a full 40-char sha")
        _require_digest("config_sha256", self.config_sha256)
        return sha256_hex(canonical_json({"code_git_commit": commit, "config_sha256": self.config_sha256}))


@dataclass(frozen=True)
class AgentRunIdentity:
    """The eight-component identity of an agentic evaluation run.

    Equation:
        agent_run_id = SHA256(canonical_json({component_name: digest, ...}))
        over exactly the eight names in ``IDENTITY_COMPONENTS``.

    Purpose:
        Make two runs comparable if and only if they agree on model, agent
        policy, tool set and permissions, environment, task specification,
        budget and retry policy, evaluator, and code/configuration.

    Output range:
        64-char lowercase hex.

    Why this form:
        A flat hash over a canonical mapping is the smallest construction that
        (a) changes when any component changes, (b) is order-independent, and
        (c) is auditable — each component digest is retained on the record, so a
        mismatch localizes to a component rather than to the whole run.

    Why not extend ``compute_run_id``:
        Its output is the on-disk directory name for existing runs; adding an
        input would rewrite historical identities. This is additive; the legacy
        id is retained as ``base_run_id`` lineage.

    Failure modes:
        A component digest computed over an under-specified manifest still
        produces a well-formed id. Identity is only as strong as the manifests
        beneath it — the fail-closed checks here catch absent and placeholder
        pins, not dishonest ones.
    """

    model_pin_sha256: str
    agent_policy_pin_sha256: str
    tool_pin_sha256: str
    environment_pin_sha256: str
    task_spec_pin_sha256: str
    budget_pin_sha256: str
    evaluator_pin_sha256: str
    code_pin_sha256: str
    schema_version: str = AGENT_IDENTITY_SCHEMA_VERSION
    base_run_id: str | None = None
    notes: str = ""

    def identity_payload(self) -> dict[str, str]:
        return {name: _require_digest(name, getattr(self, name)) for name in IDENTITY_COMPONENTS}

    def agent_run_id(self) -> str:
        return sha256_hex(canonical_json(self.identity_payload()))

    def to_dict(self) -> dict[str, Any]:
        """Emit the record with **normalized** component digests.

        `identity_payload` strips and lowercases each digest before hashing, so
        emitting the raw dataclass fields would let a record carry an upper-case
        digest that hashes to a correct `agent_run_id` while failing the schema's
        `^[0-9a-f]{64}$` pattern. The record and the id must agree.
        """
        d = asdict(self)
        d.update(self.identity_payload())
        d["agent_run_id"] = self.agent_run_id()
        return d


def compute_agent_run_id(
    *,
    model_pin_sha256: str,
    agent_policy_pin_sha256: str,
    tool_pin_sha256: str,
    environment_pin_sha256: str,
    task_spec_pin_sha256: str,
    budget_pin_sha256: str,
    evaluator_pin_sha256: str,
    code_pin_sha256: str,
) -> str:
    """Content-addressed id over the eight identity components. Keyword-only."""
    return AgentRunIdentity(
        model_pin_sha256=model_pin_sha256,
        agent_policy_pin_sha256=agent_policy_pin_sha256,
        tool_pin_sha256=tool_pin_sha256,
        environment_pin_sha256=environment_pin_sha256,
        task_spec_pin_sha256=task_spec_pin_sha256,
        budget_pin_sha256=budget_pin_sha256,
        evaluator_pin_sha256=evaluator_pin_sha256,
        code_pin_sha256=code_pin_sha256,
    ).agent_run_id()
