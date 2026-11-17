"""Tests for the offline epistemic density and UMA benchmark script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_info_density import build_report_bundle, main


def test_offline_info_density_bundle_authenticates_h6_and_computes_ratios(tmp_path: Path):
    out_file = tmp_path / "bundle.json"
    rc = main(["--output-json", str(out_file)])
    assert rc == 0
    data = json.loads(out_file.read_text(encoding="utf-8"))
    h6 = data["epistemic_density"]["native_nano_h6_5000_fields"]
    pilot = data["epistemic_density"]["compact_qwen_pilot_100_fields"]
    c14 = data["epistemic_density"]["compact_qwen_c14_fresh_120_fields"]

    assert h6["field_count"] == 5000
    assert h6["grounded_exact_count"] == 3901
    assert h6["mutual_information_bits_per_field"] > 0.7
    assert pilot["grounded_exact_count"] == 95
    assert c14["field_count"] == 120
    assert data["epistemic_density"]["native_vs_compact_pilot_mi_density_ratio_x"] > 200.0
    assert data["live_apple_silicon_benchmark"] is None


def test_build_report_bundle_deterministic():
    first = build_report_bundle()
    second = build_report_bundle()
    assert first["epistemic_density"] == second["epistemic_density"]
    assert first["devx_uma_tier_routing"] == second["devx_uma_tier_routing"]
