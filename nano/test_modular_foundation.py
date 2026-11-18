"""Tests for domain-agnostic modular foundation: units/denominations, dynamic hierarchy, and epistemic engine."""
from __future__ import annotations

import pytest

from nano.contracts import DerivationMode, DomainAssertion, EpistemicStatus
from nano.grounded_engine import GroundedState, SourceSpan, resolve_world_state
from nano.schema import NodeRelation, OntologyGraph, OntologyNode, ValueKind
from nano.units import ConversionEdge, DenomRelation, DenominationRegistry


def test_denomination_scale_equivalence_and_explicit_conversion() -> None:
    reg = DenominationRegistry()
    q1 = reg.parse("$1.5M")
    q2 = reg.parse("$1,500,000")
    q3 = reg.parse("1500k USD")
    assert q1 is not None and q2 is not None and q3 is not None
    assert reg.compare(q1, q2).relation is DenomRelation.EQUIVALENT
    assert reg.compare(q1, q3).relation is DenomRelation.EQUIVALENT

    # Cross-currency comparison MUST fail closed unless an explicit ConversionEdge exists
    q_eur = reg.parse("1500k EUR")
    assert q_eur is not None
    cmp_no_fx = reg.compare(q1, q_eur)
    assert cmp_no_fx.relation is DenomRelation.INCOMPARABLE_DENOMINATION

    # Same currency via subunit (cents) compares identically
    q_cents = reg.parse("250 cents")
    q_usd = reg.parse("$2.50")
    assert q_cents is not None and q_usd is not None
    assert reg.compare(q_cents, q_usd).relation is DenomRelation.EQUIVALENT

    # Register an evidence-backed FX edge (1 EUR = 1.1 USD) -> traceable lineage & exact conflict
    reg.register_conversion(ConversionEdge("EUR", "USD", factor=1.1, evidence_id="fx:ev-1"))
    cmp_with_fx = reg.compare(q1, q_eur)
    assert cmp_with_fx.relation is DenomRelation.CONFLICTING  # 1.5M USD vs 1.65M USD
    q_eur_eq = reg.parse("1M EUR")
    assert q_eur_eq is not None
    reg.register_conversion(ConversionEdge("USD", "EUR", factor=1.0))  # test exact custom lineage lookup on fresh reg
    reg2 = DenominationRegistry()
    reg2.register_conversion(ConversionEdge("EUR", "USD", factor=1.5, evidence_id="fx:spot"))
    conv = reg2.convert(q_eur_eq, "USD")
    assert conv is not None and abs(conv.midpoint - 1.5e6) < 1e-3 and conv.conversion_lineage == ("fx:spot",)


def test_physical_iec_decimal_ratio_and_temporal_roles() -> None:
    reg = DenominationRegistry()
    # Physical mass across scales
    assert reg.compare(reg.parse("500 mg"), reg.parse("0.5 g")).relation is DenomRelation.EQUIVALENT
    # Binary IEC (GiB) vs decimal SI (GB): NOT equal!
    q_gib = reg.parse("2 GiB")
    q_gb = reg.parse("2 GB")
    q_mib = reg.parse("2048 MiB")
    assert reg.compare(q_gib, q_mib).relation is DenomRelation.EQUIVALENT
    assert reg.compare(q_gib, q_gb).relation is DenomRelation.CONFLICTING
    # Basis points vs percentage
    assert reg.compare(reg.parse("150 bps"), reg.parse("1.5%")).relation is DenomRelation.EQUIVALENT
    # Temporal point equivalence ("Yesterday" vs "since yesterday") and dimension/role protection
    assert reg.compare(reg.parse("Yesterday"), reg.parse("since yesterday")).relation is DenomRelation.EQUIVALENT
    assert reg.compare(reg.parse("3 days"), reg.parse("Yesterday")).relation is DenomRelation.INCOMPARABLE_DIMENSION
    # Stock vs flow mismatch
    stock = reg.parse("$1.2M", role="stock")
    flow = reg.parse("$1.2M", role="flow")
    assert reg.compare(stock, flow).relation is DenomRelation.ROLE_MISMATCH


def test_hierarchical_ontology_gauge_and_mixed_denom_rollup() -> None:
    schema = OntologyGraph([
        OntologyNode("assets", value_kind=ValueKind.QUANTITY, dimension="currency", default_denomination="USD", rollup_op="sum"),
        OntologyNode("assets.current", parent_id="assets", relation_to_parent=NodeRelation.PART_OF, value_kind=ValueKind.QUANTITY, dimension="currency", default_denomination="USD", rollup_op="sum"),
        OntologyNode("assets.fixed", parent_id="assets", relation_to_parent=NodeRelation.PART_OF, value_kind=ValueKind.QUANTITY, dimension="currency", default_denomination="USD"),
        OntologyNode("assets.current.cash", parent_id="assets.current", relation_to_parent=NodeRelation.PART_OF, value_kind=ValueKind.QUANTITY, dimension="currency", default_denomination="USD"),
        OntologyNode("assets.current.receivables", parent_id="assets.current", relation_to_parent=NodeRelation.PART_OF, value_kind=ValueKind.QUANTITY, dimension="currency", default_denomination="USD"),
    ])
    # Dynamic runtime node extension without modifying core library
    schema.register(OntologyNode("assets.fixed.equipment", parent_id="assets.fixed", relation_to_parent=NodeRelation.PART_OF, value_kind=ValueKind.QUANTITY, dimension="currency"))
    assert schema.depth_of("assets.current.cash") == 2
    assert schema.relative_address("assets.current.cash", "assets.fixed.equipment") == (2, (1, 0))

    reg = DenominationRegistry()
    ok_checks = {c.parent_id: c for c in schema.verify_rollups({
        "assets.current.cash": reg.parse("$400k"),
        "assets.current.receivables": reg.parse("600000 USD"),
        "assets.current": reg.parse("$1.0M"),
        "assets.fixed": reg.parse("$2.5M"),
        "assets": reg.parse("$3.5M"),
    }, reg)}
    assert ok_checks["assets.current"].status == "consistent"
    assert ok_checks["assets"].status == "consistent"

    # Mixed currency without FX fails closed as incomparable
    bad_fx = {c.parent_id: c for c in schema.verify_rollups({
        "assets.current.cash": reg.parse("$400k"),
        "assets.current.receivables": reg.parse("600k EUR"),
        "assets.current": reg.parse("$1.0M"),
    }, reg)}
    assert bad_fx["assets.current"].status == "incomparable"

    # Algebraic conservation conflict detected when sum != parent
    mismatch = {c.parent_id: c for c in schema.verify_rollups({
        "assets.current.cash": reg.parse("$400k"),
        "assets.current.receivables": reg.parse("$700k"),
        "assets.current": reg.parse("$1.0M"),
    }, reg)}
    assert mismatch["assets.current"].status == "conflict"


def test_grounded_engine_cross_domain_finance_and_infra() -> None:
    # 1) Finance schema: currency denominations, hierarchy rollup, self-correction, third-party rejection
    fin_schema = OntologyGraph([
        OntologyNode("current_assets", dimension="currency", value_kind=ValueKind.QUANTITY, default_denomination="USD", rollup_op="sum",
                     topic_cues=(r"\bcurrent assets\b",), predicate_cues=(r"\bcurrent assets\s+(?:are|total)\b",)),
        OntologyNode("cash", parent_id="current_assets", relation_to_parent=NodeRelation.PART_OF, dimension="currency", value_kind=ValueKind.QUANTITY,
                     topic_cues=(r"\bcash\b",), predicate_cues=(r"\bcash\s+(?:on hand|balance|is)\b",)),
        OntologyNode("receivables", parent_id="current_assets", relation_to_parent=NodeRelation.PART_OF, dimension="currency", value_kind=ValueKind.QUANTITY,
                     topic_cues=(r"\breceivables\b",), predicate_cues=(r"\breceivables\s+(?:are|total)\b",)),
        OntologyNode("debt", value_kind=ValueKind.QUANTITY, dimension="currency", supports_absence=True,
                     topic_cues=(r"\bdebt\b|\bloans?\b",), denial_cues=(r"\bno debt\b|\bdebt-free\b|\bzero debt\b",)),
    ])
    transcript_ok = (
        "Auditor: Walk through Q3 liquid position and debt.\n"
        "Client: Cash balance is $400k. Receivables total 600k USD. Current assets total $1.0M.\n"
        "Client: Our competitor has $9M debt; we have no debt."
    )
    # Wait: last turn mentions competitor -> entire clause has third_party word "competitor" before semicolon, whereas "we have no debt." is its own clause!
    rep_ok = resolve_world_state(transcript_ok, fin_schema)
    assert rep_ok.resolutions["cash"].state is GroundedState.SUPPORTED
    assert rep_ok.resolutions["receivables"].state is GroundedState.SUPPORTED
    assert rep_ok.resolutions["current_assets"].state is GroundedState.SUPPORTED
    assert rep_ok.resolutions["debt"].state is GroundedState.ABSENT
    assert not rep_ok.conflicts

    # Financial contradiction across denominations ($1.5M vs $2.1M cash? Or rollup mismatch!)
    transcript_conflict = (
        "Auditor: Report Q3 cash and current assets.\n"
        "Client: Cash balance is $400k. Receivables total $700k. Current assets total $1.0M."
    )
    rep_bad = resolve_world_state(transcript_conflict, fin_schema)
    assert rep_bad.resolutions["current_assets"].state is GroundedState.CONFLICTING
    assert any(c.kind == "hierarchy_conflict" for c in rep_bad.conflicts["current_assets"])

    # 2) DomainAssertion contract integration
    da = DomainAssertion(
        subject_id="acme-q3", node_id="current_assets", predicate="equals", obj="$1.0M",
        original_wording="Current assets total $1.0M", epistemic_status=EpistemicStatus.DIRECT_DOCUMENTATION,
        evidence_span_ids=("ev:1",), denomination="USD", unit="USD",
        quantity=rep_ok.resolutions["current_assets"].quantity,
        derivation=DerivationMode.OBSERVED,
    )
    assert da.denomination == "USD" and da.quantity.base_min == 1_000_000.0

    # 3) Engineering/SRE SLA hierarchy + priority ordinal + binary MiB vs GB units + legal fee rate
    sre_schema = OntologyGraph([
        OntologyNode("e2e_latency", dimension="time_duration", value_kind=ValueKind.QUANTITY, rollup_op="sum",
                     topic_cues=(r"\bend-to-end latency\b|\be2e latency\b",)),
        OntologyNode("db_latency", parent_id="e2e_latency", relation_to_parent=NodeRelation.PART_OF, dimension="time_duration", value_kind=ValueKind.QUANTITY,
                     topic_cues=(r"\bdb latency\b|\bdatabase latency\b",)),
        OntologyNode("api_latency", parent_id="e2e_latency", relation_to_parent=NodeRelation.PART_OF, dimension="time_duration", value_kind=ValueKind.QUANTITY,
                     topic_cues=(r"\bapi latency\b",)),
        OntologyNode("mem_budget", dimension="data_size", value_kind=ValueKind.QUANTITY,
                     topic_cues=(r"\bmemory budget\b|\bram\b",)),
        OntologyNode("sla_penalty", dimension="ratio", value_kind=ValueKind.QUANTITY,
                     topic_cues=(r"\bpenalty\b|\bsla fee\b",)),
        OntologyNode("severity", dimension="ordinal", value_kind=ValueKind.ORDINAL,
                     ordinal_scale=((r"p3|low", 0.0, 0.25), (r"p2|medium", 0.26, 0.5), (r"p1|high", 0.51, 0.75), (r"p0|critical", 0.76, 1.0)),
                     topic_cues=(r"\bseverity\b|\bincident level\b",)),
    ])
    sre_doc = (
        "Engineer: DB latency is 450 ms. API latency is 0.55 seconds. End-to-end latency is 1 second.\n"
        "Engineer: Memory budget is 2 GiB. Incident level is P1. SLA fee is 150 bps.\n"
        "Engineer: Actually, memory budget is 2048 MiB. SLA fee is 1.5%."
    )
    rep_sre = resolve_world_state(sre_doc, sre_schema)
    assert rep_sre.resolutions["e2e_latency"].state is GroundedState.SUPPORTED
    assert not rep_sre.conflicts  # 450ms + 0.55s == 1s; 2 GiB == 2048 MiB; 150 bps == 1.5%!
