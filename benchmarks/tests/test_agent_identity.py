"""Agent-run identity tests — distinguishability, determinism, fail-closed pins.

Run explicitly (``benchmarks`` is not in ``pytest.ini`` testpaths, matching the
existing convention for ``benchmarks/adapters/lm_eval/tests/test_program0.py``):

    pytest benchmarks/ -q
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import jsonschema
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.agent_identity import (  # noqa: E402
    AGENT_IDENTITY_SCHEMA_VERSION,
    IDENTITY_COMPONENTS,
    AgentPolicyPin,
    AgentRunIdentity,
    BudgetPin,
    CodePin,
    EnvironmentPin,
    EvaluatorPin,
    IdentityContractError,
    ToolPin,
    compute_agent_run_id,
)
from benchmarks.adapters.lm_eval.schemas import (  # noqa: E402
    canonical_json,
    compute_run_id,
    sha256_hex,
)

SCHEMA_PATH = REPO / "benchmarks/AGENT_RUN_IDENTITY.schema.json"


def _digest(tag: str) -> str:
    """Distinct well-formed sha256 digests for identity tests.

    Uses sha256 rather than ``hash()``: builtin string hashing is salted per
    process, which would make these fixtures non-reproducible across runs.
    """
    return hashlib.sha256(tag.encode()).hexdigest()


def _components(**overrides: str) -> dict[str, str]:
    base = {name: _digest(name) for name in IDENTITY_COMPONENTS}
    base.update(overrides)
    return base


def test_all_eight_components_are_required():
    assert len(IDENTITY_COMPONENTS) == 8
    assert set(IDENTITY_COMPONENTS) == {
        "model_pin_sha256",
        "agent_policy_pin_sha256",
        "tool_pin_sha256",
        "environment_pin_sha256",
        "task_spec_pin_sha256",
        "budget_pin_sha256",
        "evaluator_pin_sha256",
        "code_pin_sha256",
    }


def test_identity_is_deterministic():
    c = _components()
    assert compute_agent_run_id(**c) == compute_agent_run_id(**c)
    assert len(compute_agent_run_id(**c)) == 64


@pytest.mark.parametrize("component", IDENTITY_COMPONENTS)
def test_varying_each_component_changes_identity(component):
    """The property that makes the contract worth having.

    Two runs that differ in exactly one identity-bearing component must not
    share an id. This is checked once per component so a future edit cannot
    silently drop one from the payload.
    """
    base = compute_agent_run_id(**_components())
    varied = compute_agent_run_id(**_components(**{component: _digest("varied-" + component)}))
    assert varied != base, f"{component} is not bound into agent_run_id"


def test_identity_hash_is_insensitive_to_payload_insertion_order():
    """Pin the hashing surface, not the call.

    Keyword arguments resolve by name, so passing a reordered dict to a
    keyword-only function is the same call and proves nothing. The property
    worth pinning is that the digest is taken over *sorted* canonical JSON, so
    a payload built in a different insertion order hashes identically.
    """
    ident = AgentRunIdentity(**_components())
    payload = ident.identity_payload()
    reversed_payload = {k: payload[k] for k in reversed(list(payload))}
    assert list(reversed_payload) != list(payload)
    assert sha256_hex(canonical_json(reversed_payload)) == ident.agent_run_id()


@pytest.mark.parametrize(
    "bad",
    ["", "not-a-digest", "ABC", "0" * 64, "g" * 64, "a" * 63],
)
def test_malformed_or_placeholder_pins_fail_closed(bad):
    with pytest.raises(IdentityContractError):
        compute_agent_run_id(**_components(tool_pin_sha256=bad))


def test_legacy_run_id_contract_is_untouched():
    """This module must not alter Program 0 identity.

    ``compute_run_id`` output is the on-disk directory name of every existing
    run; a changed signature or value would rewrite historical identities.
    """
    import inspect

    assert list(inspect.signature(compute_run_id).parameters) == [
        "benchmark_manifest_sha256",
        "model_or_solver_manifest_sha256",
        "config_sha256",
        "code_git_commit",
    ]
    assert (
        compute_run_id(
            benchmark_manifest_sha256="aa",
            model_or_solver_manifest_sha256="bb",
            config_sha256="cc",
            code_git_commit="dd",
        )
        == compute_run_id(
            benchmark_manifest_sha256="aa",
            model_or_solver_manifest_sha256="bb",
            config_sha256="cc",
            code_git_commit="dd",
        )
    )


def test_base_run_id_is_lineage_not_identity():
    c = _components()
    with_lineage = AgentRunIdentity(**c, base_run_id=_digest("legacy"))
    without = AgentRunIdentity(**c)
    assert with_lineage.agent_run_id() == without.agent_run_id()
    assert with_lineage.to_dict()["base_run_id"] == _digest("legacy")


# --- component pins ---------------------------------------------------------


def test_tool_pin_requires_permission_for_every_tool():
    ok = ToolPin(
        tool_manifest_sha256=_digest("tm"),
        tool_ids=["read", "write"],
        permissions={"read": "ro", "write": "rw"},
    )
    assert len(ok.digest()) == 64
    with pytest.raises(IdentityContractError):
        ToolPin(
            tool_manifest_sha256=_digest("tm"),
            tool_ids=["read", "write"],
            permissions={"read": "ro"},
        ).digest()


def test_tool_pin_permissions_are_identity_bearing():
    ro = ToolPin(_digest("tm"), ["write"], {"write": "ro"}).digest()
    rw = ToolPin(_digest("tm"), ["write"], {"write": "rw"}).digest()
    assert ro != rw


def test_environment_pin_rejects_mutable_tag_and_unseeded_determinism():
    with pytest.raises(IdentityContractError):
        EnvironmentPin("repo/img:latest", "none", True, 0).digest()
    with pytest.raises(IdentityContractError):
        EnvironmentPin("sha256:abc", "none", True, None).digest()
    with pytest.raises(IdentityContractError):
        EnvironmentPin("sha256:abc", "sometimes", False).digest()
    assert len(EnvironmentPin("sha256:abc", "allowlist", True, 7).digest()) == 64


def test_budget_pin_requires_explicit_retry_policy_and_repeats():
    with pytest.raises(IdentityContractError):
        BudgetPin(retry_policy="", repeats=1).digest()
    with pytest.raises(IdentityContractError):
        BudgetPin(retry_policy="none", repeats=0).digest()
    assert len(BudgetPin(retry_policy="none", repeats=1).digest()) == 64


def test_budget_pin_distinguishes_retry_policy_and_repeats():
    a = BudgetPin(retry_policy="none", repeats=1).digest()
    b = BudgetPin(retry_policy="up_to_3", repeats=1).digest()
    c = BudgetPin(retry_policy="none", repeats=5).digest()
    assert len({a, b, c}) == 3


def test_agent_policy_and_code_pins_validate():
    assert len(AgentPolicyPin("react_v3", _digest("p"), _digest("sp")).digest()) == 64
    with pytest.raises(IdentityContractError):
        AgentPolicyPin("", _digest("p"), _digest("sp")).digest()
    assert len(CodePin("c" * 40, _digest("cfg")).digest()) == 64
    with pytest.raises(IdentityContractError):
        CodePin("c" * 7, _digest("cfg")).digest()


def test_evaluator_pin_judge_calibration_optional_but_validated():
    assert len(EvaluatorPin("harness", "1.0", _digest("s")).digest()) == 64
    with pytest.raises(IdentityContractError):
        EvaluatorPin("harness", "1.0", _digest("s"), judge_calibration_sha256="nope").digest()


# --- schema -----------------------------------------------------------------


def test_schema_and_python_component_lists_cannot_drift():
    """Regression guard: the JSON schema and the module define one contract.

    Passes today. It exists so that adding, renaming, or dropping a component on
    one side fails loudly instead of forking the contract silently.
    """
    schema = json.loads(SCHEMA_PATH.read_text())
    components = schema["properties"]["components"]
    assert sorted(components["required"]) == sorted(IDENTITY_COMPONENTS)
    assert sorted(components["properties"]) == sorted(IDENTITY_COMPONENTS)
    assert components["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == AGENT_IDENTITY_SCHEMA_VERSION


def test_to_dict_emits_normalized_digests_that_validate():
    """The emitted record and the id it carries must agree.

    `identity_payload` lowercases before hashing, so an upper-case input yields a
    correct id; if `to_dict` emitted the raw field it would fail the schema's
    lowercase-hex pattern while claiming that id.
    """
    schema = json.loads(SCHEMA_PATH.read_text())
    upper = _components(model_pin_sha256=_digest("model_pin_sha256").upper())
    ident = AgentRunIdentity(**upper)
    assert ident.agent_run_id() == compute_agent_run_id(**_components())
    record = ident.to_dict()
    assert record["model_pin_sha256"] == _digest("model_pin_sha256")
    jsonschema.validate(
        instance={
            "schema_version": AGENT_IDENTITY_SCHEMA_VERSION,
            "agent_run_id": record["agent_run_id"],
            "components": {k: record[k] for k in IDENTITY_COMPONENTS},
        },
        schema=schema,
    )


def test_schema_validates_a_full_record_whose_sources_are_the_real_preimages():
    """Exemplar in which `component_sources` actually hashes to `components`.

    The schema documents `component_sources` as the pre-image of each digest but
    cannot recompute it — that relation is a producer obligation, so it is
    demonstrated here rather than asserted and left unchecked.
    """
    schema = json.loads(SCHEMA_PATH.read_text())
    policy = AgentPolicyPin("react_v3", _digest("policy-src"), _digest("sys-prompt"))
    tools = ToolPin(_digest("tool-manifest"), ["read", "bash"], {"read": "ro", "bash": "sandboxed"})
    env = EnvironmentPin("sha256:" + _digest("image"), "none", True, 0)
    budget = BudgetPin(retry_policy="none", repeats=3, max_steps=100)
    evaluator = EvaluatorPin("swe_harness", "0.1.0", _digest("scorer"))
    code = CodePin("c" * 40, _digest("config"))

    components = {
        # model and task-spec digests come from the existing Program 0 manifests
        "model_pin_sha256": _digest("model-manifest"),
        "task_spec_pin_sha256": _digest("benchmark-manifest"),
        "agent_policy_pin_sha256": policy.digest(),
        "tool_pin_sha256": tools.digest(),
        "environment_pin_sha256": env.digest(),
        "budget_pin_sha256": budget.digest(),
        "evaluator_pin_sha256": evaluator.digest(),
        "code_pin_sha256": code.digest(),
    }
    record = {
        "schema_version": AGENT_IDENTITY_SCHEMA_VERSION,
        "agent_run_id": compute_agent_run_id(**components),
        "base_run_id": None,
        "components": components,
        "component_sources": {
            "agent_policy": asdict(policy),
            "tools": {
                "tool_manifest_sha256": tools.tool_manifest_sha256,
                "tool_ids": tools.tool_ids,
                "permissions": tools.permissions,
            },
            "environment": asdict(env),
            "budget": asdict(budget),
            "evaluator": asdict(evaluator),
            "code": asdict(code),
        },
    }
    jsonschema.validate(instance=record, schema=schema)

    # The relation the schema describes, exercised rather than assumed.
    sources = record["component_sources"]
    assert AgentPolicyPin(**sources["agent_policy"]).digest() == components["agent_policy_pin_sha256"]
    assert ToolPin(**sources["tools"]).digest() == components["tool_pin_sha256"]
    assert EnvironmentPin(**sources["environment"]).digest() == components["environment_pin_sha256"]
    assert BudgetPin(**sources["budget"]).digest() == components["budget_pin_sha256"]
    assert EvaluatorPin(**sources["evaluator"]).digest() == components["evaluator_pin_sha256"]
    assert CodePin(**sources["code"]).digest() == components["code_pin_sha256"]


def test_schema_rejects_missing_component_and_null_digest():
    schema = json.loads(SCHEMA_PATH.read_text())
    c = _components()
    incomplete = {k: v for k, v in c.items() if k != "budget_pin_sha256"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance={
                "schema_version": AGENT_IDENTITY_SCHEMA_VERSION,
                "agent_run_id": _digest("x"),
                "components": incomplete,
            },
            schema=schema,
        )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance={
                "schema_version": AGENT_IDENTITY_SCHEMA_VERSION,
                "agent_run_id": _digest("x"),
                "components": _components(model_pin_sha256="0" * 64),
            },
            schema=schema,
        )


def test_boundary_governed_files_untouched_by_import():
    """Infrastructure must not amend Layer-1 evidence or the registry."""
    ledger = (REPO / "papers/EVIDENCE_LEDGER.md").read_bytes()
    registry = (REPO / "benchmarks/REGISTRY.yaml").read_bytes()
    compute_agent_run_id(**_components())
    assert (REPO / "papers/EVIDENCE_LEDGER.md").read_bytes() == ledger
    assert (REPO / "benchmarks/REGISTRY.yaml").read_bytes() == registry
