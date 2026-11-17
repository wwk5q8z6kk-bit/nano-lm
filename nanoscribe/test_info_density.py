"""Tests for epistemic information density, commit-gated KV eviction, and UMA bridge."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from nanoscribe.info_density import (
    APPLE_M3_MAX_40GPU_64GB,
    CommitGatedKVCacheLedger,
    NANO_H6_FP16,
    NANO_V01_FP16,
    QWEN25_1_5B_FP16,
    QWEN25_1_5B_Q4,
    UMA_BRIDGE_SCHEMA_VERSION,
    UncommittedSpanErasureError,
    compute_epistemic_density,
    compute_uma_roofline,
    fano_error_lower_bound,
    inspect_safetensors_zero_copy,
    recommend_uma_tier,
    repack_safetensors_page_aligned,
    simulate_sliding_window_erasure,
)


def test_noiseless_channel_maximizes_mi_and_zeros_fano_bound():
    states = ["supported", "absent", "missing", "uncertain", "conflicting"] * 20
    report = compute_epistemic_density(
        states,
        states,
        grounded_exact_count=len(states),
        geometry=NANO_H6_FP16,
    )
    assert report.conditional_entropy_bits == pytest.approx(0.0, abs=1e-9)
    assert report.mutual_information_bits_per_field == pytest.approx(report.target_entropy_bits)
    assert report.normalized_mutual_information == pytest.approx(1.0)
    assert report.fano_error_lower_bound == pytest.approx(0.0, abs=1e-9)
    assert report.observed_state_error_rate == 0.0
    assert report.grounded_exact_rate == 1.0


def test_independent_noise_collapses_mi_and_fano_respects_observed_error():
    gold = ["supported"] * 25 + ["absent"] * 25 + ["missing"] * 25 + ["uncertain"] * 25
    pred = ["missing"] * 100
    report = compute_epistemic_density(
        gold,
        pred,
        grounded_exact_count=25,
        geometry=QWEN25_1_5B_FP16,
    )
    assert report.mutual_information_bits_per_field == pytest.approx(0.0, abs=1e-9)
    assert report.fano_error_lower_bound <= report.observed_state_error_rate + 1e-6
    assert report.observed_state_error_rate == pytest.approx(0.75)


def test_native_nano_exceeds_compact_llm_parameter_density_at_matched_accuracy():
    states = ["supported", "absent", "missing", "uncertain", "conflicting"] * 10
    nano_report = compute_epistemic_density(
        states, states, grounded_exact_count=45, geometry=NANO_H6_FP16
    )
    qwen_report = compute_epistemic_density(
        states, states, grounded_exact_count=45, geometry=QWEN25_1_5B_FP16
    )
    density_ratio = (
        nano_report.mi_bits_per_field_per_megabyte
        / qwen_report.mi_bits_per_field_per_megabyte
    )
    assert density_ratio == pytest.approx(
        QWEN25_1_5B_FP16.weight_bytes / NANO_H6_FP16.weight_bytes, rel=1e-6
    )
    assert density_ratio > 450.0


def test_commit_gated_kv_eviction_blocks_uncommitted_erasure_and_rehydrates_spans():
    transcript = (
        "Doctor: What brings you in today?\n"
        "Patient: I've been having stomach cramps.\n"
        "Doctor: Any allergies?\n"
        "Patient: I'm allergic to cobalt."
    )
    ledger = CommitGatedKVCacheLedger(transcript, QWEN25_1_5B_FP16)
    ledger.register_turn_kv(0, 12)
    ledger.register_turn_kv(1, 14)
    ledger.register_turn_kv(2, 10)
    ledger.register_turn_kv(3, 11)

    with pytest.raises(UncommittedSpanErasureError):
        ledger.evict_turn_kv(1)

    cc_start = transcript.index("stomach cramps")
    alg_start = transcript.index("cobalt")
    ledger.commit_turn(0)
    ledger.commit_turn(1, [("chief_complaint", "supported", cc_start, cc_start + len("stomach cramps"))])
    ledger.commit_turn(2)
    ledger.commit_turn(3, [("allergy", "supported", alg_start, alg_start + len("cobalt"))])

    reclaimed = ledger.evict_all_committed()
    assert reclaimed == QWEN25_1_5B_FP16.kv_bytes(47)
    assert ledger.active_kv_bytes == 0
    assert ledger.rehydrate_field("chief_complaint") == (("supported", "stomach cramps"),)
    assert ledger.rehydrate_field("allergy") == (("supported", "cobalt"),)
    summary = ledger.summary()
    assert summary["memory_reduction_ratio"] > 0.99
    assert summary["uncommitted_erasure_events"] == 0


def test_sliding_window_erasure_quantifies_lost_early_evidence():
    result = simulate_sliding_window_erasure(
        turn_token_counts=[40, 40, 40, 40, 40],
        evidence_turn_indices=[0, 1, 4],
        window_tokens=100,
    )
    assert result["evidence_spans_erased"] == 2
    assert result["erased_epistemic_bits"] > 4.6
    assert result["fano_min_field_error_rate"] == pytest.approx(2 / 3, rel=1e-5)


def test_safetensors_zero_copy_mmap_inspector_checks_alignment(tmp_path: Path):
    header_obj = {
        "__metadata__": {"format": "pt"},
        "emb.weight": {"dtype": "F16", "shape": [4, 8], "data_offsets": [0, 64]},
        "nf.weight": {"dtype": "F16", "shape": [8], "data_offsets": [64, 80]},
    }
    raw_header = json.dumps(header_obj).encode("utf-8")
    # Pad JSON header with spaces so 8 + len(raw_header) == 64 (Metal 64B aligned).
    pad = (64 - ((8 + len(raw_header)) % 64)) % 64
    padded_header = raw_header + (b" " * pad)
    payload = b"\x00" * 80
    target = tmp_path / "toy.safetensors"
    target.write_bytes(struct.pack("<Q", len(padded_header)) + padded_header + payload)

    info = inspect_safetensors_zero_copy(target)
    assert info.tensor_count == 2
    assert info.total_tensor_bytes == 80
    assert info.data_section_offset % 64 == 0
    assert info.metal_64b_aligned is True

    # Reproduce upstream HuggingFace misalignment: header_len % 64 == 0 leaves data_offset == 8 (mod 64).
    hf_pad = (64 - (len(raw_header) % 64)) % 64
    hf_header = raw_header + (b" " * hf_pad)
    misaligned = tmp_path / "hf_unaligned.safetensors"
    misaligned.write_bytes(struct.pack("<Q", len(hf_header)) + hf_header + payload)
    bad_info = inspect_safetensors_zero_copy(misaligned)
    assert bad_info.data_section_offset % 64 == 8
    assert bad_info.metal_64b_aligned is False
    assert bad_info.vm_page_16k_aligned is False

    repacked = tmp_path / "uma_page_aligned.safetensors"
    fixed_info = repack_safetensors_page_aligned(misaligned, repacked)
    assert fixed_info.data_section_offset == 16_384
    assert fixed_info.metal_64b_aligned is True
    assert fixed_info.vm_page_16k_aligned is True
    assert fixed_info.total_tensor_bytes == 80


def test_uma_roofline_and_devx_tier_recommendations():
    nano_roof = compute_uma_roofline(NANO_V01_FP16, hardware=APPLE_M3_MAX_40GPU_64GB)
    assert nano_roof["slc_l2_cache_resident"] is True

    qwen_roof = compute_uma_roofline(
        QWEN25_1_5B_FP16,
        hardware=APPLE_M3_MAX_40GPU_64GB,
        uncached_forward_calls=35,
        cached_forward_calls=7,
    )
    assert qwen_roof["slc_l2_cache_resident"] is False
    assert qwen_roof["dram_traffic_reduction_ratio"] == pytest.approx(0.8, abs=1e-4)

    assert recommend_uma_tier(thermal="nominal", available_memory_bytes=24 * 1024**3)[
        "recommended_solver_tier"
    ] == "compact_1_5b_fp16_prefix_kv"
    assert recommend_uma_tier(thermal="nominal", available_memory_bytes=4 * 1024**3)[
        "recommended_solver_tier"
    ] == "compact_1_5b_q4_prefix_kv"
    assert recommend_uma_tier(thermal="serious", available_memory_bytes=24 * 1024**3)[
        "recommended_solver_tier"
    ] == "native_nano_3_3m"
    critical = recommend_uma_tier(thermal="critical", available_memory_bytes=24 * 1024**3)
    assert critical["schema"] == UMA_BRIDGE_SCHEMA_VERSION
    assert critical["recommended_solver_tier"] == "deterministic_reference"


def test_prefix_cached_constrained_decoder_coalesces_unary_chains_and_evicts_kv(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from nanoscribe.compact import LocalCompactPredictor

    class _CharTokenizer:
        eos_token_id = 1

        def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, continue_final_message=False):
            assert not tokenize
            parts = [f"<{m['role']}>{m['content']}" for m in messages]
            if add_generation_prompt:
                parts.append("<assistant>")
            elif not continue_final_message:
                parts.append("<eos>")
            return "".join(parts)

        def encode(self, text, add_special_tokens=False):
            return [ord(ch) + 10 for ch in text]

        def __call__(self, text, return_tensors="pt"):
            return {"input_ids": torch.tensor([self.encode(text)], dtype=torch.long)}

        def decode(self, ids, skip_special_tokens=True):
            vals = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            return "".join(chr(v - 10) for v in vals if v != self.eos_token_id)

    class _DeterministicLogitModel:
        def __init__(self):
            self.calls = []

        def __call__(self, *, input_ids, past_key_values=None, use_cache=True):
            seq_len = int(input_ids.shape[1])
            self.calls.append(seq_len)
            if past_key_values is not None:
                k = torch.zeros((1, 1, seq_len, 4))
                v = torch.zeros((1, 1, seq_len, 4))
                past_key_values.update(k, v, 0)
            logits = torch.zeros((1, seq_len, 512))
            # Prefer ':S[' then supported value tokens over ':M' / ':U['.
            for ch, score in (("S", 9.0), ("t", 8.0), ("d", 8.0), ("M", 5.0), ("U", 1.0)):
                logits[0, -1, ord(ch) + 10] = score
            return type("Out", (), {"logits": logits})()

    source = (
        "Doctor: What symptoms?\nPatient: I've been having toe pain.\n"
        "Doctor: Any medicine?\nPatient: I've been taking diclofenac."
    )
    predictor = LocalCompactPredictor(str(tmp_path), decoding="constrained", prefix_cache=True)
    fake_model = _DeterministicLogitModel()
    predictor._loaded = (_CharTokenizer(), fake_model, "cpu")

    summary = predictor(source)
    assert summary == "CC:S[toe pain]|DUR:M|SEV:M|MED:S[diclofenac]|ALG:M"
    assert predictor._prefix_kv is None  # Commit-gated KV eviction after encounter completion!
    record = predictor.records[source]
    assert record["decoding"] == "typed-context-source-trie-v9-prefix-kv"
    # 1 base prefill + 5 single-branch steps = 6 forward calls total (all long quote tails coalesced!).
    assert record["forward_passes"] == 6
    assert record["coalesced_zero_entropy_tokens"] >= len("toe pain]") + len("diclofenac]")
