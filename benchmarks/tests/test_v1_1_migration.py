"""v1.1 migration tests — backward-compat identity (T3), migration invariants
(T4), cost identity (T7), and the acceptance gate on the checked-in fixtures.

Run explicitly (``benchmarks`` is not in ``pytest.ini`` testpaths):

    pytest benchmarks/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.process_metrics import class_a_profile as profile_v10  # noqa: E402
from benchmarks.transcript_emitter import emit_file as emit_v10  # noqa: E402
from benchmarks.v1_1.emitter import emit_file as emit_v11  # noqa: E402
from benchmarks.v1_1.metrics import class_a_profile as profile_v11  # noqa: E402
from benchmarks.v1_1.migration import V10_KEYS, compare_file  # noqa: E402

V10_FIX = REPO / "benchmarks/fixtures/transcripts"
COMPOUND_FREE = ["session_small_reads", "session_small_errors", "session_trivial"]
ALL_V10 = ["synthetic_minimal", "session_large_varied", *COMPOUND_FREE]


# --- T3: on a compound-free log every v1.0 key is identical ---------------------------


@pytest.mark.parametrize("name", COMPOUND_FREE)
def test_t3_backward_compat_identity_on_compound_free_fixtures(name):
    path = V10_FIX / f"{name}.jsonl"
    p10 = profile_v10(emit_v10(path).log)
    p11 = profile_v11(emit_v11(path).log)
    for key in V10_KEYS:
        assert p11[key] == p10[key], key
    assert p11["call_count"] == p11["action_count"]


# --- T4 / T7: migration invariants on the large fixture -------------------------------


@pytest.fixture(scope="module")
def large():
    return compare_file(V10_FIX / "session_large_varied.jsonl")


def test_t4_invariants_hold_on_large_fixture(large):
    failed = {k: d for k, (ok, d) in large.invariants.items() if not ok}
    assert not failed, failed


def test_t4_new_mutate_targets_are_exactly_the_edits_hidden_inside_v1_0_verify_calls(large):
    assert large.new_mutate_targets, "the audit found 17 such calls on this fixture"
    assert set(large.new_mutate_targets.values()) == {"verify"}


def test_t7_cost_totals_identical(large):
    assert large.p10["cost_totals"] == large.p11["cost_totals"]


def test_verify_presence_is_per_call_and_surplus_is_interleaving(large):
    """The spec's T4 said 'verify sub-actions == v1.0 verify actions'. That was
    too strong: the first side-by-side run found 43 vs 38 on this fixture,
    because five v1.0 verify calls contain two test segments separated by
    another effect. The invariant that actually holds is per-call presence,
    with the sub-action surplus equal to the interleaving count."""
    assert large.p11["call_count"] == large.p10["action_count"]
    assert large.invariants["calls containing a verify == v1.0 verify calls"][0]
    ok, detail = large.invariants["verify sub-action surplus == interleaved test segments"]
    assert ok, detail


# --- acceptance gate on every checked-in fixture ----------------------------------------


@pytest.mark.parametrize("name", ALL_V10)
def test_gate_every_delta_is_explained(name):
    c = compare_file(V10_FIX / f"{name}.jsonl")
    assert not c.unexplained, [(d.key, d.category, d.explanation) for d in c.unexplained]


def test_gate_with_scopes_declared_on_large_fixture():
    c = compare_file(V10_FIX / "session_large_varied.jsonl",
                     declared_scope=["."], source_scope=["benchmarks/", "nanoscribe/", "scripts/", "fabric/"])
    assert not c.unexplained, [(d.key, d.category, d.explanation) for d in c.unexplained]
    assert c.p11["verification_attempted_in_scope"] is not None


def test_per_call_rates_never_differ_from_v1_0(large):
    """D4 by construction: a delta here is an unexplained category, so it must not occur."""
    for key in ("invalid_tool_call_rate", "destructive_action_rate"):
        assert large.p10[key] == large.p11[key], key


# --- T9: v1.1 characterization pins ------------------------------------------------
#
# Pinned from the first side-by-side run whose every delta was explained
# (artifacts/.../v1.1-migration/fixture-delta.md). Not correctness claims: they
# make a later classifier change visible. Re-derive only after re-auditing.

EXPECTED_V11_BY_RULE = {
    "session_large_varied": {
        "bash:fallback": 78, "bash:file-write": 57, "bash:git-index": 43, "bash:read-only": 285,
        "bash:test-runner": 43, "tool:Edit": 15, "tool:ListAgents": 2, "tool:Read": 5, "tool:Skill": 3,
        "tool:ToolSearch": 3, "tool:Write": 12, "tool:message-passing": 7, "tool:peer-discovery": 2,
    },
    "session_small_errors": {"bash:read-only": 4, "tool:Glob": 1, "tool:Grep": 2, "tool:Read": 5, "tool:StructuredOutput": 2},
    "session_small_reads": {"bash:read-only": 1, "tool:Read": 8, "tool:StructuredOutput": 1},
    "session_trivial": {"tool:StructuredOutput": 1},
}


@pytest.mark.parametrize("name", sorted(EXPECTED_V11_BY_RULE))
def test_t9_characterization_by_rule_v1_1(name):
    got = emit_v11(V10_FIX / f"{name}.jsonl").stats["by_rule"]
    assert got == EXPECTED_V11_BY_RULE[name], f"v1.1 classifier behaviour changed on {name}; re-audit before updating this pin"


def test_t9_large_fixture_decomposition_shape():
    """The transition the fixture-delta table documents: 269 calls → 555 sub-actions."""
    r = emit_v11(V10_FIX / "session_large_varied.jsonl")
    assert r.stats["calls"] == 269
    assert r.stats["actions"] == sum(EXPECTED_V11_BY_RULE["session_large_varied"].values())


# --- explanation categories found by the rerun, pinned on synthetic fixtures -------------

V11_FIX = REPO / "benchmarks/v1_1/fixtures/synthetic"


def test_f03_hidden_read_before_edit_is_an_explained_premature_delta():
    c = compare_file(V11_FIX / "F03_read_then_sed.jsonl")
    assert c.p10["premature_edit_rate"] == 1.0 and c.p11["premature_edit_rate"] == 0.0
    d = next(x for x in c.deltas if x.key == "premature_edit_rate")
    assert d.category == "hidden-read-before-edit" and d.explained, d
    assert c.unblinded == ["src/a.py"] and not c.newly_blind
    assert not c.unexplained


def test_f15_v1_0_over_attribution_is_explained_and_not_a_loss():
    c = compare_file(V11_FIX / "F15_mixed_evidence_over_attribution.jsonl")
    assert c.over_attributed == {"out/manifest.json": "t0"}
    assert not c.lost_mutate_targets
    assert c.p10["files_changed"] == 2 and c.p11["files_changed"] == 1
    d = next(x for x in c.deltas if x.key == "files_changed")
    assert d.category == "over-attributed-mutate" and d.explained, d
    assert not c.unexplained


@pytest.mark.parametrize("path", sorted((REPO / "benchmarks/v1_1/fixtures/synthetic").glob("*.jsonl")), ids=lambda p: p.stem)
def test_gate_every_delta_is_explained_on_v1_1_fixtures(path):
    c = compare_file(path)
    assert not c.unexplained, [(d.key, d.category, d.explanation) for d in c.unexplained]
