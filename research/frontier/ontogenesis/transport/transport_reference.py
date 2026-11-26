"""CPU reference operators for causal information transport (research only).

Independent implementation of limited LoopRes/ALF-inspired primitives. Not a
full Transformer, upstream parity claim, training recipe or adopted backbone.
See pinned upstream source identities in the adjacent review. No model/data I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class LoopLayout:
    layers: int
    first: int
    last: int
    steps: int
    blocks: int
    history: int

    def __post_init__(self):
        if not (0 <= self.first < self.last <= self.layers and self.steps >= 1
                and self.blocks >= 1 and self.history >= 0
                and (self.last - self.first) % self.blocks == 0):
            raise ValueError("invalid loop layout")

    @property
    def cache_layers(self):
        return self.layers + (self.steps - 1) * (self.last - self.first)

    def cache_slot(self, layer: int, step: int = 0):
        if not 0 <= layer < self.layers or not 0 <= step < self.steps:
            raise ValueError("invalid layer or step")
        if self.first <= layer < self.last:
            return layer + step * (self.last - self.first)
        if step:
            raise ValueError("prelude/coda execute only once")
        return layer if layer < self.first else layer + (self.steps - 1) * (self.last - self.first)

    def sources(self, step: int, layer: int, sublayer: str):
        if not (0 <= step < self.steps and self.first <= layer < self.last
                and sublayer in ("attn", "mlp")):
            raise ValueError("invalid read location")
        block_size = (self.last - self.first) // self.blocks
        block, offset = divmod(layer - self.first, block_size)
        keys = [("prelude", -1, -1)]
        keys += [("block", s, b) for s in range(max(0, step - self.history), step)
                 for b in range(self.blocks)]
        keys += [("block", step, b) for b in range(block)]
        if offset or sublayer == "mlp":
            keys.append(("partial", step, block))
        return tuple(keys)

    def readout_sources(self):
        step = self.steps - 1
        return (("prelude", -1, -1),) + tuple(
            ("block", s, b) for s in range(max(0, step - self.history), step + 1)
            for b in range(self.blocks))


def _work_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def loop_mix(values, keys, query, value_query, norm_weight, eps=1e-6):
    """[S,B,T,D] sources; [S,D] keys; per-position, source-axis softmax.

    The result is a weighted sum of *raw* sources. No cross-token mixing.
    Reference math, not fused kernel or GPU performance evidence.
    """
    if values.ndim != 4 or not values.is_floating_point() or values.shape[0] == 0:
        raise ValueError("expected nonempty floating [S,B,T,D] values")
    s, _, _, d = values.shape
    if keys.shape != (s, d) or any(x.shape != (d,) for x in (query, value_query, norm_weight)):
        raise ValueError("invalid source/key/query shape")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("invalid epsilon")
    v = values.to(_work_dtype(values))
    key_bias = keys.to(v.dtype) @ query.to(v.dtype)
    rms = torch.sqrt(v.square().mean(dim=-1) + eps)
    logits = (v @ (norm_weight.to(v.dtype) * value_query.to(v.dtype))) / rms
    weights = torch.softmax(logits + key_bias[:, None, None], dim=0)
    return (weights[..., None] * v).sum(dim=0).to(values.dtype)


@dataclass(frozen=True)
class FeedbackCarry:
    request_ids: tuple[str, ...]
    slots: torch.Tensor   # [B, 2L, D], last position only
    top: torch.Tensor     # [B, D]
    segment: torch.Tensor # [B]
    valid: torch.Tensor   # [B]


def previous_token_states(slots, top, segment_ids, valid, bos, request_ids, carry=None):
    """Shift supplied activations, with explicit segment/BOS/validity guards.

    For chunked calls, a carry is accepted only for identical row identities.
    The caller owns ordering, session/reset policy and the upstream causal model.
    Returns prior slots/top, validity mask, and the next one-position carry.
    """
    if slots.ndim != 4 or not slots.is_floating_point():
        raise ValueError("expected floating [B,T,2L,D] slots")
    b, t, n, d = slots.shape
    ids = tuple(request_ids)
    if b < 1 or t < 1 or n < 2 or n % 2 or d < 1 or len(ids) != b or len(set(ids)) != b:
        raise ValueError("invalid batch, sublayer count or request identities")
    if top.shape != (b, t, d) or top.dtype != slots.dtype or top.device != slots.device:
        raise ValueError("invalid top-state shape/dtype/device")
    if any(x.shape != (b, t) or x.device != slots.device for x in (segment_ids, valid, bos)):
        raise ValueError("invalid metadata shape/device")
    if valid.dtype != torch.bool or bos.dtype != torch.bool or segment_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("invalid metadata dtype")
    if carry is None:
        head_slots, head_top = torch.zeros_like(slots[:, 0]), torch.zeros_like(top[:, 0])
        head_segment = segment_ids[:, 0]
        head_valid = torch.zeros_like(valid[:, 0])
    else:
        if carry.request_ids != ids:
            raise ValueError("stale or reordered request carry; reset explicitly")
        expected = ((b, n, d), (b, d), (b,), (b,))
        tensors = (carry.slots, carry.top, carry.segment, carry.valid)
        if any(x.shape != shape or x.device != slots.device for x, shape in zip(tensors, expected)):
            raise ValueError("invalid carry shape/device")
        if (carry.slots.dtype != slots.dtype or carry.top.dtype != top.dtype
                or carry.segment.dtype != segment_ids.dtype or carry.valid.dtype != torch.bool):
            raise ValueError("invalid carry dtype")
        head_slots, head_top, head_segment, head_valid = tensors
    pslots = torch.cat((head_slots[:, None], slots[:, :-1]), dim=1)
    ptop = torch.cat((head_top[:, None], top[:, :-1]), dim=1)
    pseg = torch.cat((head_segment[:, None], segment_ids[:, :-1]), dim=1)
    pvalid = torch.cat((head_valid[:, None], valid[:, :-1]), dim=1)
    allowed = valid & pvalid & ~bos & (segment_ids == pseg)
    pslots = torch.where(allowed[..., None, None], pslots, torch.zeros_like(pslots))
    ptop = torch.where(allowed[..., None], ptop, torch.zeros_like(ptop))
    # Clone the final position so the carry does not retain a full chunk view.
    next_carry = FeedbackCarry(ids, slots[:, -1].clone(), top[:, -1].clone(),
                               segment_ids[:, -1].clone(), valid[:, -1].clone())
    return pslots, ptop, allowed, next_carry


def fixed_feedback(x, previous, allowed, sublayer: int, window: int, share_logit, eps=1e-6):
    """Fixed nearest-sublayer mean, RMS-matched to residual, sigmoid share.

    [B,T,D] residual and [B,T,2L,D] previous states. The source window starts
    at the receiver's sublayer; no learned source maps or depth softmax here.
    """
    if (x.ndim != 3 or not x.is_floating_point() or previous.ndim != 4
            or previous.shape[:2] != x.shape[:2] or previous.shape[-1] != x.shape[-1]
            or previous.shape[2] < 2 or previous.shape[2] % 2
            or allowed.shape != x.shape[:2] or allowed.dtype != torch.bool
            or not 0 <= sublayer < previous.shape[2] or window < 1):
        raise ValueError("invalid feedback shape/window")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("invalid epsilon")
    wd = _work_dtype(x)
    a = x.to(wd)
    p = torch.where(allowed[..., None, None], previous, torch.zeros_like(previous))
    m = p[:, :, sublayer:sublayer + window].to(wd).mean(dim=2)
    rho = torch.sigmoid(torch.as_tensor(share_logit, dtype=wd, device=x.device))
    if rho.ndim != 0:
        raise ValueError("share_logit must be a scalar")
    scale = torch.sqrt((a.square().mean(-1, keepdim=True) + eps)
                       / (m.square().mean(-1, keepdim=True) + eps))
    mixed = a * (1 - rho) + m * (rho * scale)
    return torch.where(allowed[..., None], mixed.to(x.dtype), x)


def inject_previous_top(x, previous_top, allowed, projection, scale=1.0, eps=1e-6):
    """Separate input path: normalized previous top state, projected and added."""
    if (x.ndim != 3 or not x.is_floating_point() or previous_top.shape != x.shape
            or allowed.shape != x.shape[:2] or allowed.dtype != torch.bool
            or projection.shape != (x.shape[-1], x.shape[-1])):
        raise ValueError("invalid injection shape")
    if not math.isfinite(eps) or eps <= 0 or not math.isfinite(scale):
        raise ValueError("invalid injection scale/epsilon")
    wd = _work_dtype(x)
    prev = torch.where(allowed[..., None], previous_top, torch.zeros_like(previous_top)).to(wd)
    normed = prev / torch.sqrt(prev.square().mean(-1, keepdim=True) + eps)
    delta = scale * (normed @ projection.to(wd))
    return torch.where(allowed[..., None], (x.to(wd) + delta).to(x.dtype), x)
