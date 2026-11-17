"""Architecture arms for LAB-ARCH-02. All arms are matched on non-embedding parameter count
(FFN width auto-tuned to within ~1% of T0) unless the arm is explicitly an iso-param/extra-compute
arm (looped depth), in which case unique params match and applied params are reported separately."""

from __future__ import annotations

from lab.arch.model import LM, Cfg

BASE = dict(d=512, n_head=8, n_kv=4, ffn=1408, window=256)
N_LAYERS = 10

ARMS: dict[str, dict] = {
    # --- conventional baselines
    "T0_dense": {"layout": "A" * 10, "why": "Llama-style dense baseline (GQA, QK-norm, RoPE, SwiGLU)"},
    "T1_vres": {"layout": "A" * 10, "value_residual": True, "why": "ResFormer value residual"},
    "T2_swa": {"layout": "SSSASSSASA", "why": "3:1 sliding-window/global hybrid (W=256)"},
    # --- hybrids with linear-time recurrent mixers
    "H1_gla": {"layout": "LLLALLLALA", "why": "gated linear attention (SSD-like) 3:1 hybrid"},
    "H2_delta": {"layout": "DDDADDDADA", "why": "gated delta-rule 3:1 hybrid (Qwen3-Next/Kimi-Linear family)"},
    "H3_delta_vres": {"layout": "DDDADDDADA", "value_residual": True, "why": "H2 + value residual"},
    # --- pure recurrent (O(1) decode state)
    "P1_gla": {"layout": "L" * 10, "why": "pure gated linear attention"},
    "P2_delta": {"layout": "D" * 10, "why": "pure gated delta rule"},
    # --- adaptive / recurrent depth (iso unique params, more compute)
    "R1_loop": {"layout": "A" * 10, "loop": (2, 6, 2, 2), "why": "weight-shared core x2: 16 eff layers"},
    "R2_delta_loop": {"layout": "DDDADDDADA", "loop": (2, 6, 2, 2), "why": "H2 with looped core"},
}


def _cfg(arm: str, vocab: int, seq: int, ffn: int) -> Cfg:
    s = ARMS[arm]
    return Cfg(vocab=vocab, seq=seq, layout=s["layout"], value_residual=s.get("value_residual", False),
               loop=s.get("loop"), **{**BASE, "ffn": ffn})


def _non_emb(c: Cfg) -> int:
    m = LM(c)
    return m.param_report()["non_embedding"]


_TARGET: dict[tuple[int, int], int] = {}


def build(arm: str, vocab: int, seq: int) -> LM:
    key = (vocab, seq)
    if key not in _TARGET:
        _TARGET[key] = _non_emb(_cfg("T0_dense", vocab, seq, BASE["ffn"]))
    target = _TARGET[key]
    lo, hi = 256, 4096
    while hi - lo > 32:  # binary-search FFN width (multiple of 32) to match T0 non-embedding params
        mid = (lo + hi) // 2 // 32 * 32
        if _non_emb(_cfg(arm, vocab, seq, mid)) < target:
            lo = mid
        else:
            hi = mid
    best = min((lo, hi), key=lambda f: abs(_non_emb(_cfg(arm, vocab, seq, f)) - target))
    return LM(_cfg(arm, vocab, seq, best))
