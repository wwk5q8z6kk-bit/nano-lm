"""Information density (bits/param), epistemic KV eviction, and Apple Silicon UMA bridge.

Implements three deterministic, stdlib-only primitives linking ``nano-lm``
epistemic representation contracts to physical unified-memory compute limits:

1. **Epistemic Operator Information Density** — Computes empirical Shannon mutual
   information ``I(Y_state; Y_hat | X)`` (bits/field and bits/MB of weights),
   conditional residual entropy ``H(Y | Y_hat)`` (Fano bound residual), and
   grounded-exact efficiency.  Open-vocabulary patient values stay in verbatim
   source pointer spans (0 weight bits), isolating weight capacity to the
   epistemic operator ``T: Source -> State x Span``.
2. **Commit-Gated Two-Tier KV Eviction** — Enforces ``NANO_INVARIANT`` ("a latent
   state is not evidence"): transient ``KV`` cache pages may be evicted at zero
   epistemic erasure cost if and only if every covered turn's observations are
   committed to the symbolic ledger with exact UTF-8 byte/char offsets.
3. **Apple Silicon UMA Roofline & Zero-Copy Bridge (``nano.devx.uma_bridge.v1``)** —
   Inspects ``.safetensors`` tensor alignment via read-only POSIX ``mmap`` for
   zero-copy ``MTLBuffer(bytesNoCopy:)`` / multi-worker physical page sharing,
   computes unified-memory roofline bounds, and maps host thermal + available
   memory headroom to solver tiers compatible with ``DevX`` ``host_resource_budget_v1``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct
from typing import Any

UMA_BRIDGE_SCHEMA_VERSION = "nano.devx.uma_bridge.v1"
APPLE_SILICON_VM_PAGE_BYTES = 16_384
METAL_BUFFER_ALIGNMENT_BYTES = 64


class UncommittedSpanErasureError(RuntimeError):
    """Attempted to evict KV cache tokens whose epistemic state is not committed."""


@dataclass(frozen=True, slots=True)
class ModelMemoryGeometry:
    """Physical weight and KV-cache footprint for one transformer geometry."""

    model_id: str
    parameter_count: int
    layer_count: int
    kv_heads: int
    head_width: int
    bytes_per_weight: float = 2.0
    bytes_per_kv_element: int = 2

    def __post_init__(self) -> None:
        if self.parameter_count <= 0 or self.layer_count <= 0:
            raise ValueError("parameter_count and layer_count must be positive")
        if self.kv_heads <= 0 or self.head_width <= 0:
            raise ValueError("kv_heads and head_width must be positive")
        if self.bytes_per_weight <= 0 or self.bytes_per_kv_element <= 0:
            raise ValueError("byte widths must be positive")

    @property
    def weight_bytes(self) -> int:
        return int(round(self.parameter_count * self.bytes_per_weight))

    @property
    def weight_megabytes(self) -> float:
        return self.weight_bytes / (1024.0 * 1024.0)

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.layer_count * self.kv_heads * self.head_width * self.bytes_per_kv_element

    def kv_bytes(self, token_count: int) -> int:
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        return self.kv_bytes_per_token * token_count


NANO_V01_FP16 = ModelMemoryGeometry(
    model_id="nano/v0.1-3.15m-fp16",
    parameter_count=3_148_608,
    layer_count=6,
    kv_heads=2,
    head_width=32,
    bytes_per_weight=2.0,
    bytes_per_kv_element=2,
)

NANO_H6_FP16 = ModelMemoryGeometry(
    model_id="nano/h6-evidence-query-3.29m-fp16",
    parameter_count=3_287_109,
    layer_count=6,
    kv_heads=2,
    head_width=32,
    bytes_per_weight=2.0,
    bytes_per_kv_element=2,
)

QWEN25_1_5B_FP16 = ModelMemoryGeometry(
    model_id="compact/qwen2.5-1.5b-fp16",
    parameter_count=1_543_714_816,
    layer_count=28,
    kv_heads=2,
    head_width=128,
    bytes_per_weight=2.0,
    bytes_per_kv_element=2,
)

QWEN25_1_5B_Q4 = ModelMemoryGeometry(
    model_id="compact/qwen2.5-1.5b-q4",
    parameter_count=1_543_714_816,
    layer_count=28,
    kv_heads=2,
    head_width=128,
    bytes_per_weight=0.55,
    bytes_per_kv_element=2,
)


@dataclass(frozen=True, slots=True)
class HardwareBandwidthProfile:
    """Unified memory bandwidth and cache capacity for an Apple Silicon host."""

    chip_id: str
    unified_memory_bytes: int
    peak_bandwidth_gbps: float
    slc_and_l2_bytes: int

    @property
    def peak_bandwidth_bytes_per_second(self) -> float:
        return self.peak_bandwidth_gbps * 1e9


APPLE_M3_MAX_40GPU_64GB = HardwareBandwidthProfile(
    chip_id="Apple M3 Max (40-core GPU, 64 GB UMA)",
    unified_memory_bytes=68_719_476_736,
    peak_bandwidth_gbps=400.0,
    slc_and_l2_bytes=48 * 1024 * 1024,
)


@dataclass(frozen=True, slots=True)
class EpistemicDensityReport:
    """Information-theoretic operator efficiency for an evaluated solver."""

    model_id: str
    parameter_count: int
    weight_bytes: int
    field_count: int
    target_entropy_bits: float
    conditional_entropy_bits: float
    mutual_information_bits_per_field: float
    normalized_mutual_information: float
    fano_error_lower_bound: float
    observed_state_error_rate: float
    grounded_exact_count: int
    grounded_exact_rate: float
    total_epistemic_bits: float
    mi_bits_per_field_per_megabyte: float
    grounded_exact_fields_per_megabyte: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "parameter_count": self.parameter_count,
            "weight_bytes": self.weight_bytes,
            "weight_megabytes": round(self.weight_bytes / (1024.0 * 1024.0), 6),
            "field_count": self.field_count,
            "target_entropy_bits": round(self.target_entropy_bits, 6),
            "conditional_entropy_bits": round(self.conditional_entropy_bits, 6),
            "mutual_information_bits_per_field": round(self.mutual_information_bits_per_field, 6),
            "normalized_mutual_information": round(self.normalized_mutual_information, 6),
            "fano_error_lower_bound": round(self.fano_error_lower_bound, 6),
            "observed_state_error_rate": round(self.observed_state_error_rate, 6),
            "grounded_exact_count": self.grounded_exact_count,
            "grounded_exact_rate": round(self.grounded_exact_rate, 6),
            "total_epistemic_bits": round(self.total_epistemic_bits, 6),
            "mi_bits_per_field_per_megabyte": round(self.mi_bits_per_field_per_megabyte, 6),
            "grounded_exact_fields_per_megabyte": round(self.grounded_exact_fields_per_megabyte, 6),
        }


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def fano_error_lower_bound(conditional_entropy_bits: float, alphabet_size: int = 5) -> float:
    """Invert Fano's inequality ``H(Y|Y_hat) <= H_b(P_e) + P_e * log2(|Y| - 1)``."""

    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least 2")
    if conditional_entropy_bits <= 0.0:
        return 0.0
    max_error = 1.0 - (1.0 / alphabet_size)
    penalty = math.log2(alphabet_size - 1) if alphabet_size > 2 else 0.0
    max_rhs = _binary_entropy(max_error) + max_error * penalty
    if conditional_entropy_bits >= max_rhs:
        return max_error

    low, high = 0.0, max_error
    for _ in range(60):
        mid = 0.5 * (low + high)
        rhs = _binary_entropy(mid) + mid * penalty
        if rhs < conditional_entropy_bits:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def compute_epistemic_density(
    gold_states: Sequence[str],
    predicted_states: Sequence[str],
    *,
    grounded_exact_count: int,
    geometry: ModelMemoryGeometry,
    alphabet_size: int = 5,
) -> EpistemicDensityReport:
    """Compute epistemic state MI and grounded-exact density per MB of weights."""

    n = len(gold_states)
    if n == 0 or n != len(predicted_states):
        raise ValueError("gold_states and predicted_states must be non-empty and equal length")
    if not 0 <= grounded_exact_count <= n:
        raise ValueError("grounded_exact_count must lie in [0, len(gold_states)]")

    joint = Counter(zip(gold_states, predicted_states, strict=True))
    gold_counts = Counter(gold_states)
    pred_counts = Counter(predicted_states)

    h_y = -sum((c / n) * math.log2(c / n) for c in gold_counts.values())
    mi = 0.0
    for (y, y_hat), count in joint.items():
        p_xy = count / n
        p_x = gold_counts[y] / n
        p_y = pred_counts[y_hat] / n
        mi += p_xy * math.log2(p_xy / (p_x * p_y))
    mi = max(0.0, mi)
    h_cond = max(0.0, h_y - mi)
    nmi = (mi / h_y) if h_y > 0.0 else 0.0

    errors = sum(1 for y, y_hat in zip(gold_states, predicted_states, strict=True) if y != y_hat)
    err_rate = errors / n
    fano_lb = fano_error_lower_bound(h_cond, alphabet_size=alphabet_size)
    mb = geometry.weight_megabytes
    grounded_rate = grounded_exact_count / n

    return EpistemicDensityReport(
        model_id=geometry.model_id,
        parameter_count=geometry.parameter_count,
        weight_bytes=geometry.weight_bytes,
        field_count=n,
        target_entropy_bits=h_y,
        conditional_entropy_bits=h_cond,
        mutual_information_bits_per_field=mi,
        normalized_mutual_information=nmi,
        fano_error_lower_bound=fano_lb,
        observed_state_error_rate=err_rate,
        grounded_exact_count=grounded_exact_count,
        grounded_exact_rate=grounded_rate,
        total_epistemic_bits=grounded_exact_count * h_y,
        mi_bits_per_field_per_megabyte=mi / mb,
        grounded_exact_fields_per_megabyte=grounded_exact_count / mb,
    )


@dataclass(frozen=True, slots=True)
class CommittedSpanEntry:
    """Immutable symbolic ledger entry backed by exact source character offsets."""

    turn_index: int
    field: str
    state: str
    start: int
    end: int

    def extract_text(self, immutable_source: str) -> str:
        if not (0 <= self.start <= self.end <= len(immutable_source)):
            raise ValueError("committed span offsets out of bounds for source")
        return immutable_source[self.start : self.end]


class CommitGatedKVCacheLedger:
    """Two-tier KV eviction manager enforcing zero epistemic erasure.

    Turn KV blocks remain pinned in Tier 0 (active working KV) until caller
    explicitly commits every epistemic observation (or a verified no-observation
    marker) for that turn into Tier 1 (symbolic span ledger + immutable UTF-8
    source).  Once committed, KV tensors are evicted safely while exact source
    spans remain deterministically rehydratable.
    """

    def __init__(self, immutable_source: str, geometry: ModelMemoryGeometry) -> None:
        self.immutable_source = immutable_source
        self.geometry = geometry
        self._turn_tokens: dict[int, int] = {}
        self._active_kv_turns: set[int] = set()
        self._committed_turns: set[int] = set()
        self._entries: list[CommittedSpanEntry] = []
        self._evicted_tokens: int = 0

    def register_turn_kv(self, turn_index: int, token_count: int) -> None:
        if turn_index < 0 or token_count <= 0:
            raise ValueError("turn_index must be >= 0 and token_count > 0")
        if turn_index in self._turn_tokens:
            raise ValueError(f"turn {turn_index} already registered")
        self._turn_tokens[turn_index] = token_count
        self._active_kv_turns.add(turn_index)

    def commit_turn(
        self,
        turn_index: int,
        spans: Sequence[tuple[str, str, int, int]] = (),
    ) -> None:
        if turn_index not in self._turn_tokens:
            raise ValueError(f"unknown turn {turn_index}")
        for field, state, start, end in spans:
            if not (0 <= start <= end <= len(self.immutable_source)):
                raise ValueError("span offsets exceed immutable_source")
            self._entries.append(
                CommittedSpanEntry(
                    turn_index=turn_index,
                    field=field,
                    state=state,
                    start=start,
                    end=end,
                )
            )
        self._committed_turns.add(turn_index)

    def evict_turn_kv(self, turn_index: int) -> int:
        if turn_index not in self._active_kv_turns:
            return 0
        if turn_index not in self._committed_turns:
            raise UncommittedSpanErasureError(
                f"refusing irreversible KV eviction of uncommitted turn {turn_index}"
            )
        self._active_kv_turns.remove(turn_index)
        tokens = self._turn_tokens[turn_index]
        self._evicted_tokens += tokens
        return self.geometry.kv_bytes(tokens)

    def evict_all_committed(self) -> int:
        reclaimed = 0
        for turn_index in sorted(list(self._active_kv_turns)):
            if turn_index in self._committed_turns:
                reclaimed += self.evict_turn_kv(turn_index)
        return reclaimed

    def rehydrate_field(self, field: str) -> tuple[tuple[str, str], ...]:
        return tuple(
            (entry.state, entry.extract_text(self.immutable_source))
            for entry in self._entries
            if entry.field == field
        )

    @property
    def active_kv_bytes(self) -> int:
        return sum(self.geometry.kv_bytes(self._turn_tokens[t]) for t in self._active_kv_turns)

    @property
    def symbolic_ledger_bytes(self) -> int:
        return len(self.immutable_source.encode("utf-8")) + 32 * len(self._entries)

    def summary(self) -> dict[str, int | float]:
        total_tokens = sum(self._turn_tokens.values())
        unbounded_kv = self.geometry.kv_bytes(total_tokens)
        retained = self.active_kv_bytes + self.symbolic_ledger_bytes
        return {
            "total_turns": len(self._turn_tokens),
            "committed_turns": len(self._committed_turns),
            "active_kv_turns": len(self._active_kv_turns),
            "evicted_tokens": self._evicted_tokens,
            "unbounded_kv_bytes": unbounded_kv,
            "active_kv_bytes": self.active_kv_bytes,
            "symbolic_ledger_bytes": self.symbolic_ledger_bytes,
            "memory_reduction_ratio": (
                1.0 - (retained / unbounded_kv) if unbounded_kv > 0 else 0.0
            ),
            "uncommitted_erasure_events": 0,
        }


def simulate_sliding_window_erasure(
    turn_token_counts: Sequence[int],
    evidence_turn_indices: Sequence[int],
    *,
    window_tokens: int,
    bits_per_evidence_span: float = 2.321928,
) -> dict[str, float | int]:
    """Contrast naive sliding-window KV eviction against commit-gated retention."""

    if window_tokens <= 0:
        raise ValueError("window_tokens must be positive")
    total = sum(turn_token_counts)
    kept_turns: set[int] = set()
    budget = window_tokens
    for idx in range(len(turn_token_counts) - 1, -1, -1):
        cost = turn_token_counts[idx]
        if cost <= budget:
            kept_turns.add(idx)
            budget -= cost
        else:
            break
    lost = [t for t in evidence_turn_indices if t not in kept_turns]
    erased_bits = len(lost) * bits_per_evidence_span
    return {
        "total_tokens": total,
        "window_tokens": window_tokens,
        "evidence_spans_total": len(evidence_turn_indices),
        "evidence_spans_erased": len(lost),
        "erased_epistemic_bits": round(erased_bits, 6),
        "fano_min_field_error_rate": round(
            len(lost) / len(evidence_turn_indices) if evidence_turn_indices else 0.0, 6
        ),
    }


@dataclass(frozen=True, slots=True)
class SafetensorsZeroCopyInspection:
    """Read-only POSIX ``mmap`` audit of a ``.safetensors`` file for UMA sharing."""

    path: str
    file_size_bytes: int
    header_size_bytes: int
    data_section_offset: int
    tensor_count: int
    total_tensor_bytes: int
    metal_64b_aligned: bool
    vm_page_16k_aligned: bool
    header_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_size_bytes": self.file_size_bytes,
            "header_size_bytes": self.header_size_bytes,
            "data_section_offset": self.data_section_offset,
            "tensor_count": self.tensor_count,
            "total_tensor_bytes": self.total_tensor_bytes,
            "metal_64b_aligned": self.metal_64b_aligned,
            "vm_page_16k_aligned": self.vm_page_16k_aligned,
            "header_sha256": self.header_sha256,
        }


def inspect_safetensors_zero_copy(path: str | Path) -> SafetensorsZeroCopyInspection:
    """Inspect ``.safetensors`` layout via read-only ``mmap`` without copying weights."""

    resolved = Path(path).resolve(strict=True)
    size = resolved.stat().st_size
    if size < 8:
        raise ValueError("safetensors file too small for 8-byte header length")

    with resolved.open("rb") as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            (header_len,) = struct.unpack("<Q", mm[:8])
            data_offset = 8 + header_len
            if data_offset > size:
                raise ValueError("corrupt safetensors header length exceeds file size")
            header_bytes = mm[8:data_offset]
            header_digest = hashlib.sha256(header_bytes).hexdigest()
            header = json.loads(header_bytes.decode("utf-8"))
            if not isinstance(header, Mapping):
                raise ValueError("safetensors header must be a JSON object")

            tensor_count = 0
            total_bytes = 0
            metal_aligned = True
            for key, spec in header.items():
                if key == "__metadata__":
                    continue
                if not isinstance(spec, Mapping) or "data_offsets" not in spec:
                    raise ValueError(f"invalid tensor entry: {key}")
                start, end = spec["data_offsets"]
                if not (0 <= start <= end <= size - data_offset):
                    raise ValueError(f"tensor {key} data_offsets out of bounds")
                abs_offset = data_offset + start
                if abs_offset % METAL_BUFFER_ALIGNMENT_BYTES != 0:
                    metal_aligned = False
                tensor_count += 1
                total_bytes += end - start

    return SafetensorsZeroCopyInspection(
        path=resolved.name,
        file_size_bytes=size,
        header_size_bytes=header_len,
        data_section_offset=data_offset,
        tensor_count=tensor_count,
        total_tensor_bytes=total_bytes,
        metal_64b_aligned=metal_aligned,
        vm_page_16k_aligned=(data_offset % APPLE_SILICON_VM_PAGE_BYTES == 0),
        header_sha256=header_digest,
    )


def repack_safetensors_page_aligned(
    src: str | Path,
    dst: str | Path,
    *,
    alignment: int = APPLE_SILICON_VM_PAGE_BYTES,
) -> SafetensorsZeroCopyInspection:
    """Pad a ``.safetensors`` JSON header so tensor data starts on an Apple Silicon VM page.

    Standard HuggingFace ``.safetensors`` files often pad ``header_len`` to 64
    bytes without accounting for the 8-byte ``uint64`` length prefix, leaving
    ``data_section_offset == 8 (mod 64)`` and preventing ``MTLBuffer(bytesNoCopy:)``
    zero-copy page mapping.  Padding the JSON header with trailing ASCII spaces
    preserves exact tensor bytes and relative ``data_offsets`` while aligning
    ``8 + len(header)`` to ``alignment`` (16 KiB by default on Apple Silicon).
    """

    if alignment <= 0 or alignment % METAL_BUFFER_ALIGNMENT_BYTES != 0:
        raise ValueError("alignment must be a positive multiple of 64 bytes")
    src_path = Path(src).resolve(strict=True)
    dst_path = Path(dst)
    with src_path.open("rb") as reader:
        prefix = reader.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors file too small")
        (header_len,) = struct.unpack("<Q", prefix)
        header_bytes = reader.read(header_len).rstrip(b" ")
        json.loads(header_bytes.decode("utf-8"))
        pad = (alignment - ((8 + len(header_bytes)) % alignment)) % alignment
        aligned_header = header_bytes + (b" " * pad)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with dst_path.open("wb") as writer:
            writer.write(struct.pack("<Q", len(aligned_header)))
            writer.write(aligned_header)
            while chunk := reader.read(4 * 1024 * 1024):
                writer.write(chunk)
    return inspect_safetensors_zero_copy(dst_path)


def compute_uma_roofline(
    geometry: ModelMemoryGeometry,
    *,
    hardware: HardwareBandwidthProfile = APPLE_M3_MAX_40GPU_64GB,
    active_kv_tokens: int = 512,
    uncached_forward_calls: int = 1,
    cached_forward_calls: int = 1,
) -> dict[str, Any]:
    """Compute Apple Silicon unified-memory bandwidth roofline and cache savings."""

    if uncached_forward_calls < 1 or cached_forward_calls < 1:
        raise ValueError("forward call counts must be >= 1")
    cache_resident = (geometry.weight_bytes + geometry.kv_bytes(active_kv_tokens)) <= hardware.slc_and_l2_bytes
    per_step_bytes = geometry.weight_bytes + geometry.kv_bytes(active_kv_tokens)
    min_step_seconds = per_step_bytes / hardware.peak_bandwidth_bytes_per_second

    uncached_dram_bytes = per_step_bytes * uncached_forward_calls
    cached_dram_bytes = per_step_bytes * cached_forward_calls
    saved_bytes = max(0, uncached_dram_bytes - cached_dram_bytes)

    return {
        "chip_id": hardware.chip_id,
        "model_id": geometry.model_id,
        "slc_l2_cache_resident": cache_resident,
        "per_step_memory_bytes": per_step_bytes,
        "theoretical_min_step_ms": round(min_step_seconds * 1000.0, 4),
        "theoretical_max_tokens_per_sec": round(1.0 / min_step_seconds, 2),
        "uncached_dram_read_gb": round(uncached_dram_bytes / 1e9, 4),
        "cached_dram_read_gb": round(cached_dram_bytes / 1e9, 4),
        "dram_bandwidth_saved_gb": round(saved_bytes / 1e9, 4),
        "dram_traffic_reduction_ratio": round(
            saved_bytes / uncached_dram_bytes if uncached_dram_bytes else 0.0, 4
        ),
    }


def recommend_uma_tier(
    *,
    thermal: str = "nominal",
    low_power_mode: bool = False,
    available_memory_bytes: int = 16 * 1024**3,
) -> dict[str, Any]:
    """Bridge ``DevX`` ``host_resource_budget_v1`` + UMA headroom to a Nano solver tier."""

    thermal_norm = str(thermal or "nominal").strip().lower()
    if thermal_norm not in {"nominal", "fair", "serious", "critical"}:
        thermal_norm = "nominal"
    avail = max(0, int(available_memory_bytes))

    # Reserve at least 2.5x weight footprint headroom to prevent macOS jetsam/swap
    # during concurrent Xcode / SwiftPM / rustc builds in DevX.
    fp16_floor = int(QWEN25_1_5B_FP16.weight_bytes * 2.5)
    q4_floor = int(QWEN25_1_5B_Q4.weight_bytes * 2.5)
    nano_floor = 128 * 1024 * 1024

    if thermal_norm == "critical" or avail < nano_floor:
        solver_tier = "deterministic_reference"
        transport = "in_process"
    elif thermal_norm == "serious" or avail < q4_floor:
        solver_tier = "native_nano_3_3m"
        transport = "posix_mmap_readonly"
    elif low_power_mode or thermal_norm == "fair" or avail < fp16_floor:
        solver_tier = "compact_1_5b_q4_prefix_kv"
        transport = "posix_mmap_readonly"
    else:
        solver_tier = "compact_1_5b_fp16_prefix_kv"
        transport = "posix_mmap_readonly"

    return {
        "schema": UMA_BRIDGE_SCHEMA_VERSION,
        "thermal": thermal_norm,
        "low_power_mode": bool(low_power_mode),
        "available_memory_bytes": avail,
        "recommended_solver_tier": solver_tier,
        "weight_sharing_transport": transport,
        "kv_eviction_policy": "commit_gated_symbolic_ledger",
    }


__all__ = [
    "APPLE_M3_MAX_40GPU_64GB",
    "APPLE_SILICON_VM_PAGE_BYTES",
    "CommittedSpanEntry",
    "CommitGatedKVCacheLedger",
    "EpistemicDensityReport",
    "HardwareBandwidthProfile",
    "METAL_BUFFER_ALIGNMENT_BYTES",
    "ModelMemoryGeometry",
    "NANO_H6_FP16",
    "NANO_V01_FP16",
    "QWEN25_1_5B_FP16",
    "QWEN25_1_5B_Q4",
    "SafetensorsZeroCopyInspection",
    "UMA_BRIDGE_SCHEMA_VERSION",
    "UncommittedSpanErasureError",
    "compute_epistemic_density",
    "compute_uma_roofline",
    "fano_error_lower_bound",
    "inspect_safetensors_zero_copy",
    "recommend_uma_tier",
    "repack_safetensors_page_aligned",
    "simulate_sliding_window_erasure",
]
