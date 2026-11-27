"""Exact Sub-4B Parameter, KV-Cache, Roofline, and Empirical Scaling Audit.

Answers Blueprint Section 5 ("Implement exact counting before selecting a target"),
Section 8.7 (hardware roofline), Section 18 (H-13 Block AttnRes & H-14 hybrid ratios),
and fits empirical token/parameter/recipe scaling from the authenticated 2026-11-22
uniform-sampler NanoLM runs (14.3M, 40.4M, 100.7M, 210.8M on Corpus v1/v2).
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass


@dataclass
class ArchSpec:
    name: str
    layers: int
    d: int
    d_ff: int
    vocab: int
    tied_embed: bool
    attn_layers: int
    q_heads: int
    kv_heads: int
    head_dim: int
    qk_norm: bool
    attn_output_gate: bool  # e.g. Qwen3.5 gated attention
    rec_layers: int
    rec_kind: str  # "none" | "gdn_qwen" | "mamba3" | "kda"
    rec_d_inner: int = 0
    rec_state_dim: int = 0
    rec_n_heads: int = 0
    rec_conv_kernel: int = 4
    attn_res_blocks: int = 0  # H-13 Block AttnRes: 0 = standard PreNorm
    copy_head: bool = False   # H-01 lightweight conditional span-copy gate/proj
    loop_ties: int = 1        # >1 shares parameters across loop_ties applications


def exact_param_breakdown(s: ArchSpec) -> dict:
    assert s.attn_layers + s.rec_layers == s.layers, "Layer count mismatch"
    assert s.layers % s.loop_ties == 0, "Layers must divide by loop_ties"
    scale_unique = 1.0 / s.loop_ties

    # 1. Embeddings + Final Norm + LM Head
    emb = s.vocab * s.d
    lm_head = 0 if s.tied_embed else s.vocab * s.d
    final_norm = s.d

    # 2. Per-layer SwiGLU FFN + FFN Pre-RMSNorm: w_gate, w_up (d -> d_ff), w_down (d_ff -> d)
    ffn_per_layer = 3 * s.d * s.d_ff + s.d
    ffn_total = int(round(s.layers * ffn_per_layer * scale_unique))

    # 3. Full GQA Attention mixer + Pre-RMSNorm
    q_dim = s.q_heads * s.head_dim
    kv_dim = s.kv_heads * s.head_dim
    attn_proj = s.d * q_dim + 2 * s.d * kv_dim + q_dim * s.d
    if s.attn_output_gate:
        attn_proj += s.d * q_dim  # Qwen3.5 gated attention output gate
    qk_norm_p = (2 * s.head_dim) if s.qk_norm else 0
    attn_per_layer = attn_proj + qk_norm_p + s.d  # + pre-norm
    attn_total = int(round(s.attn_layers * attn_per_layer * scale_unique))

    # 4. Recurrent / Linear Attention mixer + Pre-RMSNorm
    rec_per_layer = 0
    if s.rec_layers > 0:
        if s.rec_kind == "gdn_qwen":
            # Qwen3.5 Gated DeltaNet style:
            # q, k: d -> rec_n_heads * rec_state_dim each; v, gate(z): d -> rec_d_inner each;
            # out: rec_d_inner -> d; alpha/beta gates: d -> 2 * rec_n_heads; short depthwise conv1d + norm + A_log/dt_bias
            qk_rec = 2 * s.d * (s.rec_n_heads * s.rec_state_dim)
            vz_rec = 2 * s.d * s.rec_d_inner
            out_rec = s.rec_d_inner * s.d
            ab_gates = s.d * (2 * s.rec_n_heads) + 2 * s.rec_n_heads
            conv_dim = 2 * (s.rec_n_heads * s.rec_state_dim) + s.rec_d_inner
            conv_p = conv_dim * s.rec_conv_kernel
            rec_per_layer = qk_rec + vz_rec + out_rec + ab_gates + conv_p + s.rec_d_inner + s.d
        elif s.rec_kind == "mamba3":
            # Mamba-3 MIMO/complex SSM block: in_proj (z, x, B, C, dt), conv1d, D/A, out_proj + pre-norm
            bc_dim = 2 * s.rec_n_heads * s.rec_state_dim
            in_proj = s.d * (2 * s.rec_d_inner + bc_dim + s.rec_n_heads)
            conv_p = (s.rec_d_inner + bc_dim) * s.rec_conv_kernel
            out_proj = s.rec_d_inner * s.d
            ssm_scalars = 3 * s.rec_n_heads + s.rec_d_inner
            rec_per_layer = in_proj + conv_p + out_proj + ssm_scalars + s.d
        else:
            raise ValueError(s.rec_kind)
    rec_total = int(round(s.rec_layers * rec_per_layer * scale_unique))

    # 5. Optional H-13 Block AttnRes (content-dependent pseudo-query + RMSNorm per layer over B blocks)
    attn_res_total = 0
    if s.attn_res_blocks > 0:
        # Per Moonshot Block AttnRes: 1 learned query vector (d) + RMSNorm (d) per sublayer (2 * layers)
        attn_res_total = 2 * s.layers * (2 * s.d)

    # 6. Optional H-01 Conditional Span-Copy Gate/Pointer Head (query/key low-rank d->256 + 3-way mode gate)
    copy_total = 0
    if s.copy_head:
        d_ptr = 256
        copy_total = 2 * s.d * d_ptr + s.d * 3 + 3 + s.d

    total = emb + lm_head + final_norm + ffn_total + attn_total + rec_total + attn_res_total + copy_total

    # KV + Recurrent state bytes per sequence in BF16 (2 bytes/elem) at 4k and 32k context
    def state_bytes_bf16(seq_len: int) -> dict:
        kv_bytes = s.attn_layers * 2 * s.kv_heads * s.head_dim * seq_len * 2
        rec_state_bytes = 0
        if s.rec_layers > 0:
            # Matrix state (n_heads * state_dim * head_v_dim) + conv buffer
            head_v = s.rec_d_inner // max(1, s.rec_n_heads)
            mat_elems = s.rec_n_heads * s.rec_state_dim * head_v
            conv_elems = (s.rec_d_inner + 2 * s.rec_n_heads * s.rec_state_dim) * s.rec_conv_kernel
            rec_state_bytes = s.rec_layers * (mat_elems + conv_elems) * 2
        return {
            "kv_mib": round(kv_bytes / (1024 ** 2), 2),
            "rec_mib": round(rec_state_bytes / (1024 ** 2), 2),
            "total_mib": round((kv_bytes + rec_state_bytes) / (1024 ** 2), 2),
        }

    # Decode bytes moved per generated token at batch=1 in BF16 (active weights + state read)
    # Note: looped models read shared weights once if cached in SRAM? No, 1B+ exceeds L2/SRAM, so batch=1 DRAM reads EVERY application unless batched!
    active_params_per_token = (
        emb + (0 if s.tied_embed else lm_head) + final_norm
        + s.layers * ffn_per_layer
        + s.attn_layers * attn_per_layer
        + s.rec_layers * rec_per_layer
        + attn_res_total + copy_total
    )
    return {
        "name": s.name,
        "total_params": total,
        "total_B": round(total / 1e9, 4),
        "strict_under_4B": total < 4_000_000_000,
        "headroom_to_4B_M": round((4_000_000_000 - total) / 1e6, 2),
        "breakdown_M": {
            "embed": round(emb / 1e6, 2),
            "lm_head": round(lm_head / 1e6, 2),
            "ffn": round(ffn_total / 1e6, 2),
            "full_attn": round(attn_total / 1e6, 2),
            "recurrent": round(rec_total / 1e6, 2),
            "attn_res": round(attn_res_total / 1e6, 2),
            "copy_head": round(copy_total / 1e6, 2),
        },
        "active_params_per_tok_B": round(active_params_per_token / 1e9, 4),
        "state_4k": state_bytes_bf16(4096),
        "state_32k": state_bytes_bf16(32768),
        "decode_dram_gb_per_tok_bf16_32k": round(
            (active_params_per_token * 2 + state_bytes_bf16(32768)["total_mib"] * (1024 ** 2)) / 1e9, 3
        ),
    }


candidates = [
    # 1. Blueprint Section 5 Illustrative Hybrid (65,536 vocab, standard Qwen-like GDN dimensions)
    ArchSpec(
        name="Blueprint H1 (32L, d=2560, V=65536, 24 GDN + 8 GQA)",
        layers=32, d=2560, d_ff=9216, vocab=65536, tied_embed=True,
        attn_layers=8, q_heads=16, kv_heads=4, head_dim=160, qk_norm=True, attn_output_gate=True,
        rec_layers=24, rec_kind="gdn_qwen", rec_d_inner=4096, rec_state_dim=128, rec_n_heads=32,
        attn_res_blocks=8, copy_head=True,
    ),
    # 2. Same H1 with Qwen3.5 248,320 vocab (demonstrating exact 4B ceiling breach!)
    ArchSpec(
        name="Qwen3.5-4B-like LM (32L, d=2560, V=248320, Qwen GDN sizes)",
        layers=32, d=2560, d_ff=9216, vocab=248320, tied_embed=True,
        attn_layers=8, q_heads=16, kv_heads=4, head_dim=256, qk_norm=True, attn_output_gate=True,
        rec_layers=24, rec_kind="gdn_qwen", rec_d_inner=4096, rec_state_dim=128, rec_n_heads=32,
    ),
    # 3. Matched Dense D0 Reference (32L, d=2560, V=65536, 32 GQA)
    ArchSpec(
        name="Dense D0 Reference (32L, d=2560, V=65536, 32 GQA)",
        layers=32, d=2560, d_ff=9216, vocab=65536, tied_embed=True,
        attn_layers=32, q_heads=20, kv_heads=4, head_dim=128, qk_norm=True, attn_output_gate=False,
        rec_layers=0, rec_kind="none",
    ),
    # 4. Mamba-3 5:1 Hybrid R1 (30L, d=2560, V=65536, 25 Mamba-3 + 5 GQA)
    ArchSpec(
        name="Mamba-3 5:1 Hybrid R1 (30L, d=2560, V=65536, 25 SSM + 5 GQA)",
        layers=30, d=2560, d_ff=9216, vocab=65536, tied_embed=True,
        attn_layers=5, q_heads=20, kv_heads=4, head_dim=128, qk_norm=True, attn_output_gate=False,
        rec_layers=25, rec_kind="mamba3", rec_d_inner=5120, rec_state_dim=64, rec_n_heads=40,
    ),
    # 5. Looped/Shared-Depth A2 (32 effective applications = 16 unique layers x 2 loops)
    ArchSpec(
        name="Looped Dense A2 (16 unique x 2 loops = 32L compute, d=2560, V=65536)",
        layers=32, d=2560, d_ff=9216, vocab=65536, tied_embed=True,
        attn_layers=32, q_heads=20, kv_heads=4, head_dim=128, qk_norm=True, attn_output_gate=False,
        rec_layers=0, rec_kind="none", loop_ties=2,
    ),
]

rows = [exact_param_breakdown(c) for c in candidates]
print(json.dumps(rows, indent=2))

# Empirical Chinchilla-like power-law fit L(N, D) = E + A * N^(-alpha) + B * D^(-beta)
# Using authenticated uniform-sampler default-AdamW points:
# (14.29M, 0.8B -> 1.1451), (40.38M, 0.8B -> 1.0466), (100.68M, 0.8B -> 0.9775),
# (100.68M, 2.0B -> 0.9399), (210.80M, 4.2B -> 0.8763)
pts = [
    (14.292224e6, 0.799539e9, 1.14513),
    (40.379904e6, 0.799539e9, 1.04660),
    (100.684032e6, 0.799539e9, 0.97750),
    (100.684032e6, 1.998848e9, 0.93990),
    (210.799616e6, 4.199219e9, 0.87630),
]
# Simple grid-refined least squares for E + A*N^-a + B*D^-b
best = None
for E_i in range(45, 75):
    E = E_i / 100.0
    for a_i in range(20, 50):
        a = a_i / 100.0
        for b_i in range(15, 45):
            b = b_i / 100.0
            # Linear least squares for A, B >= 0
            xs = [[n ** (-a), d ** (-b)] for n, d, _ in pts]
            ys = [y - E for _, _, y in pts]
            s11 = sum(x[0] ** 2 for x in xs)
            s12 = sum(x[0] * x[1] for x in xs)
            s22 = sum(x[1] ** 2 for x in xs)
            t1 = sum(x[0] * y for x, y in zip(xs, ys))
            t2 = sum(x[1] * y for x, y in zip(xs, ys))
            det = s11 * s22 - s12 * s12
            if det <= 1e-30:
                continue
            A = (t1 * s22 - t2 * s12) / det
            B = (s11 * t2 - s12 * t1) / det
            if A <= 0 or B <= 0:
                continue
            rss = sum((E + A * x[0] + B * x[1] - (y + E)) ** 2 for x, y in zip(xs, ys))
            if best is None or rss < best[0]:
                best = (rss, E, A, a, B, b)

rss, E, A, a, B, b = best
print("\nEMPIRICAL SCALING FIT (5 uniform-sampler NanoLM points, default AdamW):")
print(f"mix_bpb(N, D) ≈ {E:.2f} + {A:.2f}*N^(-{a:.2f}) + {B:.2f}*D^(-{b:.2f}) | RMSE={math.sqrt(rss/5):.5f}")
for n, d, y in pts:
    pred = E + A * (n ** (-a)) + B * (d ** (-b))
    print(f"  N={n/1e6:6.1f}M, D={d/1e9:4.1f}B: actual={y:.4f}, fit={pred:.4f}, resid={y-pred:+.4f}")
for n_m, d_b in [(210.8, 10.0), (500.0, 10.0), (1000.0, 20.0), (1000.0, 100.0), (3200.0, 100.0)]:
    pred = E + A * ((n_m * 1e6) ** (-a)) + B * ((d_b * 1e9) ** (-b))
    print(f"  Extrapolated (illustrative only, default recipe): N={n_m:6.1f}M, D={d_b:5.1f}B -> {pred:.4f} (with -0.027 Muon+floor offset: ~{pred-0.027:.4f})")
