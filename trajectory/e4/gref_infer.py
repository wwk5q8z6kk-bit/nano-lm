"""G-ref inference wrapper (nano R★ SFT)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from trajectory.e1.common import FIELDS, FieldPred, ItemPred, pred_from_values

REPO = Path(__file__).resolve().parents[2]
E4 = Path(__file__).resolve().parent
DEV = os.environ.get("NANO_DEV", "mps" if torch.backends.mps.is_available() else "cpu")
V, S = 4098, 512
d, L, H, KV, hd, ff = 192, 6, 6, 2, 32, 512


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1, s.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        s.q = nn.Linear(d, H * hd, bias=False)
        s.k = nn.Linear(d, KV * hd, bias=False)
        s.v = nn.Linear(d, KV * hd, bias=False)
        s.o = nn.Linear(H * hd, d, bias=False)
        s.g = nn.Linear(d, ff, bias=False)
        s.u = nn.Linear(d, ff, bias=False)
        s.dn = nn.Linear(ff, d, bias=False)

    def forward(s, x, cos, sin):
        B = x.shape[0]
        h = s.n1(x)
        q = s.q(h).view(B, S, H, hd).transpose(1, 2)
        k = s.k(h).view(B, S, KV, hd).transpose(1, 2)
        v = s.v(h).view(B, S, KV, hd).transpose(1, 2)

        def rot(t):
            t1, t2 = t[..., 0::2], t[..., 1::2]
            return torch.stack([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1).flatten(-2)

        q, k = rot(q), rot(k)
        k, v = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(B, S, H * hd))
        h = s.n2(x)
        return x + s.dn(F.silu(s.g(h)) * s.u(h))


class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, d)
        s.blocks = nn.ModuleList(Block() for _ in range(L))
        s.nf = nn.RMSNorm(d)

    def _rope(s, device):
        t = torch.arange(S, device=device, dtype=torch.float32)
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=device).float() / hd))
        f = torch.outer(t, inv)
        return f.cos()[None, None], f.sin()[None, None]

    def forward(s, x):
        cos, sin = s._rope(x.device)
        h = s.emb(x)
        for b in s.blocks:
            h = b(h, cos, sin)
        return F.linear(s.nf(h), s.emb.weight)


_RE = __import__("re").compile(
    r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$"
)


def make_gref_predict(ckpt: Path | None = None):
    ckpt = ckpt or (E4 / "checkpoints" / "gref_nano_rstar_sft_v1.pt")
    tok = Tokenizer.from_file(str(REPO / "sft" / "tokenizer.json"))
    ims, ime = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
    m = GPT()
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.to(DEV).eval()

    def prompt_ids(convo_user: str):
        ids = [ims] + tok.encode("user\n", add_special_tokens=False).ids
        ids += tok.encode(convo_user, add_special_tokens=False).ids + [ime]
        ids += [ims] + tok.encode("assistant\n", add_special_tokens=False).ids
        return ids

    @torch.no_grad()
    def generate(ids, max_new=64):
        ids = list(ids)
        for _ in range(max_new):
            if len(ids) >= S:
                break
            x = torch.tensor([ids + [0] * (S - len(ids))], device=DEV)
            nxt = int(m(x)[0, len(ids) - 1].argmax())
            if nxt == ime:
                break
            ids.append(nxt)
        return ids

    def predict(item: dict, source_id: str) -> ItemPred:
        t0 = time.perf_counter()
        content = item["convo"][0]["content"]
        ids = prompt_ids(content)
        out = generate(ids)
        text = tok.decode(out[len(ids) :]).strip()
        mm = _RE.match(text)
        if not mm:
            return ItemPred(
                fields={f: FieldPred("none") for f in FIELDS},
                latency_s=time.perf_counter() - t0,
                raw=text,
                parsed=False,
            )
        vals = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        return pred_from_values(vals, latency_s=time.perf_counter() - t0)

    predict.venue = DEV  # type: ignore
    predict.ckpt = str(ckpt)  # type: ignore
    return predict
