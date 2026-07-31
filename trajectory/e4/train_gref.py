#!/usr/bin/env python3
"""G-ref-nano-rstar-sft-v1 — FT nano anchors/scribe.pt on R★ train (local MPS)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
E4 = Path(__file__).resolve().parent
DATA = E4 / "data"
RECIPE = json.loads((E4 / "recipe_freeze.json").read_text())
SEED = RECIPE["seeds"]["gref_train"]
torch.manual_seed(SEED)
np.random.seed(SEED)

DEV = os.environ.get("NANO_DEV", "mps" if torch.backends.mps.is_available() else "cpu")
V, S = 4098, 512
d, L, H, KV, hd, ff = 192, 6, 6, 2, 32, 512
EPOCHS, PEAK_LR, WARM_FRAC, FLOOR, WD, CLIP, BATCH = 2, 1.0e-4, 0.05, 0.1, 0.1, 1.0, 16
OUT_CKPT = E4 / "checkpoints" / "gref_nano_rstar_sft_v1.pt"
OUT_META = E4 / "checkpoints" / "gref_nano_rstar_sft_v1.meta.json"


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


def render_ids(tok, ims, ime, convo):
    ids, mask = [], []
    for m in convo:
        head = tok.encode(f"{m['role']}\n", add_special_tokens=False).ids
        body = tok.encode(m["content"], add_special_tokens=False).ids
        seg = [ims] + head + body + [ime]
        tf = 1 if m["role"] == "assistant" else 0
        for j, tkn in enumerate(seg):
            ids.append(tkn)
            mask.append(1 if (tf and j >= 1 + len(head)) else 0)
    return ids, mask


def main():
    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    base = REPO / RECIPE["g_ref_recipe"]["base_checkpoint"]
    base_sha = hashlib.sha256(base.read_bytes()).hexdigest()
    expect = RECIPE["g_ref_recipe"].get("base_checkpoint_sha256")
    if expect and base_sha != expect:
        raise SystemExit(f"base SHA mismatch: {base_sha} != {expect}")

    tok = Tokenizer.from_file(str(REPO / "sft" / "tokenizer.json"))
    ims, ime = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
    train = json.loads((DATA / "rstar_train.json").read_text())

    X, M, dropped = [], [], 0
    for it in train:
        ids, mask = render_ids(tok, ims, ime, it["convo"])
        if len(ids) > S:
            dropped += 1
            continue
        X.append(ids + [0] * (S - len(ids)))
        M.append(mask + [0] * (S - len(mask)))
    X = torch.tensor(np.array(X, dtype=np.int64))
    Mk = torch.tensor(np.array(M, dtype=np.int64))
    N = X.shape[0]
    print(f"device={DEV} train_n={N} dropped={dropped} base_sha={base_sha[:16]}", flush=True)

    m = GPT()
    m.load_state_dict(torch.load(base, map_location="cpu", weights_only=True))
    m.to(DEV)
    decay = [p for p in m.parameters() if p.dim() >= 2]
    nodecay = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}],
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    STEPS = max(1, (N // BATCH) * EPOCHS)
    WARM = int(WARM_FRAC * STEPS)

    def lr_at(t):
        if t < WARM:
            return PEAK_LR * t / max(1, WARM)
        p = (t - WARM) / max(1, STEPS - WARM)
        return PEAK_LR * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * p)))

    perm = torch.randperm(N)
    t0 = time.time()
    m.train()
    for step in range(1, STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        i0 = ((step - 1) * BATCH) % max(1, N - BATCH)
        idx = perm[i0 : i0 + BATCH]
        if len(idx) < BATCH:
            idx = perm[:BATCH]
        x = X[idx].to(DEV)
        msk = Mk[idx].to(DEV)
        logits = m(x)[:, :-1]
        tgt = x[:, 1:]
        mtgt = msk[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
        loss = (ce * mtgt).sum() / mtgt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
        opt.step()
        if step == 1 or step % 50 == 0 or step == STEPS:
            print(f"step {step}/{STEPS} loss={loss.item():.4f} lr={lr_at(step):.2e}", flush=True)

    m.cpu()
    torch.save(m.state_dict(), OUT_CKPT)
    out_sha = hashlib.sha256(OUT_CKPT.read_bytes()).hexdigest()
    meta = {
        "recipe_id": RECIPE["g_ref_recipe"]["id"],
        "base_checkpoint": str(RECIPE["g_ref_recipe"]["base_checkpoint"]),
        "base_sha256": base_sha,
        "out_checkpoint": str(OUT_CKPT.relative_to(REPO)),
        "out_sha256": out_sha,
        "venue": DEV,
        "hardware_class": "apple-mps" if DEV == "mps" else DEV,
        "epochs": EPOCHS,
        "steps": STEPS,
        "batch": BATCH,
        "peak_lr": PEAK_LR,
        "seed": SEED,
        "train_n": N,
        "dropped": dropped,
        "train_split_sha256": hashlib.sha256((DATA / "rstar_train.json").read_bytes()).hexdigest(),
        "wall_s": time.time() - t0,
        "note": "G-ref for E4 only; not NanoScribe; not old-task M0",
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
