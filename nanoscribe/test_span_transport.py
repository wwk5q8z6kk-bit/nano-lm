"""Tests for campaign B2 span transport benchmark."""

from __future__ import annotations

from nanoscribe.span_transport import run_b2_local_benchmark, write_span_transport_v2


def test_b2_local_benchmark_v2_beats_v1_on_degraded_fixture() -> None:
    payload = run_b2_local_benchmark()
    gate = payload["interim_gate"]
    assert gate["local_smoke_passes_oracle_fixture"]
    assert gate["local_smoke_v2_beats_v1_on_degraded"]
    v1 = payload["arms"]["selector_v1_degraded_fixture"]["exact_gold_span_rate"]
    v2 = payload["arms"]["selector_v2_degraded_fixture"]["exact_gold_span_rate"]
    assert v2 > v1


def test_write_span_transport_v2_artifact(tmp_path) -> None:
    out = tmp_path / "span_transport_v2.json"
    path = write_span_transport_v2(out)
    assert path.is_file()
    assert path.read_text(encoding="utf-8").startswith("{")
