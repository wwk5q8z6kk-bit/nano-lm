"""Nano LM dense baseline on the full v1 corpus (800M tokens, one pass). Generated: model code is a verbatim copy of research/frontier/m21_dense_code/dense.py, followed by the training loop."""
"""GQA attention control from the NanoLM discovery lab at 3314170a (MIT)."""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    vocab: int = 32768
    d: int = 512
    n_head: int = 8
    n_kv: int = 4
    ffn: int = 1408
    layout: str = "AAAAAAAA"
    window: int = 256
    seq: int = 1024
    rope_theta: float = 10000.0
    value_residual: bool = False
    loop: tuple[int, int, int, int] | None = None  # (prelude, core, coda, r) -- layout length = p+c+k
    conv: int = 4
    chunk: int = 64
    tie: bool = True
    extra: dict = field(default_factory=dict)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.w, self.eps)


def rope_cache(T, hd, theta, device):
    inv = 1.0 / theta ** (torch.arange(0, hd, 2, device=device).float() / hd)
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # x: B,H,T,hd
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = cos[None, None, : x.shape[2]], sin[None, None, : x.shape[2]]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: Cfg, window: int | None):
        super().__init__()
        self.h, self.kv, self.hd = c.n_head, c.n_kv, c.d // c.n_head
        self.q = nn.Linear(c.d, self.h * self.hd, bias=False)
        self.k = nn.Linear(c.d, self.kv * self.hd, bias=False)
        self.v = nn.Linear(c.d, self.kv * self.hd, bias=False)
        self.o = nn.Linear(self.h * self.hd, c.d, bias=False)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)
        self.window = window
        self.vr = c.value_residual
        if self.vr:
            self.lam = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, rope, v0):
        B, T, _ = x.shape
        q = self.qn(self.q(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        k = self.kn(self.k(x).view(B, T, self.kv, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        if self.vr:
            if v0 is None:
                v0 = v
            else:
                v = self.lam * v + (1 - self.lam) * v0
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        rep = self.h // self.kv
        k, v_ = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        if self.window is None or self.window >= T:
            y = F.scaled_dot_product_attention(q, k, v_, is_causal=True)
        else:
            i = torch.arange(T, device=x.device)
            m = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < self.window)
            y = F.scaled_dot_product_attention(q, k, v_, attn_mask=m)
        return self.o(y.transpose(1, 2).reshape(B, T, -1)), v0


class MLP(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.gu = nn.Linear(c.d, 2 * c.ffn, bias=False)
        self.dn = nn.Linear(c.ffn, c.d, bias=False)

    def forward(self, x):
        g, u = self.gu(x).chunk(2, -1)
        return self.dn(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, c: Cfg, kind: str):
        super().__init__()
        self.kind = kind
        self.n1, self.n2 = RMSNorm(c.d), RMSNorm(c.d)
        if kind != "A":
            raise ValueError("This experiment uses full self-attention")
        self.mix = Attention(c, None)
        self.mlp = MLP(c)

    def forward(self, x, rope, v0):
        h, v0 = self.mix(self.n1(x), rope, v0)
        x = x + h
        return x + self.mlp(self.n2(x)), v0


class LM(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList(Block(c, k) for k in c.layout)
        self.nf = RMSNorm(c.d)
        self.head = None if c.tie else nn.Linear(c.d, c.vocab, bias=False)
        if c.loop:
            p, co, k, r = c.loop
            assert p + co + k == len(c.layout)
            self.inject = nn.Linear(2 * c.d, c.d, bias=False)
        self._init()

    def _init(self):
        n_eff = self.effective_depth()
        for name, p in self.named_parameters():
            if p.dim() == 2 and "emb" not in name:
                nn.init.normal_(p, std=0.02)
                if name.endswith(("o.weight", "dn.weight")):
                    p.data.mul_(1 / math.sqrt(2 * n_eff))
        nn.init.normal_(self.emb.weight, std=0.02)
        if self.c.loop:
            nn.init.zeros_(self.inject.weight)
            with torch.no_grad():
                self.inject.weight[:, : self.c.d].copy_(torch.eye(self.c.d))

    def effective_depth(self):
        if not self.c.loop:
            return len(self.c.layout)
        p, co, k, r = self.c.loop
        return p + co * r + k

    def schedule(self):
        if not self.c.loop:
            return list(range(len(self.blocks)))
        p, co, k, r = self.c.loop
        return list(range(p)) + list(range(p, p + co)) * r + list(range(p + co, p + co + k))

    def hidden(self, idx, loops: int | None = None):
        B, T = idx.shape
        x = self.emb(idx)
        hd = self.c.d // self.c.n_head
        rope = rope_cache(T, hd, self.c.rope_theta, idx.device)
        v0 = None
        if not self.c.loop:
            for b in self.blocks:
                x, v0 = b(x, rope, v0)
        else:
            p, co, k, r = self.c.loop
            r = loops or r
            for i in range(p):
                x, v0 = self.blocks[i](x, rope, v0)
            e = x
            for _ in range(r):
                x = self.inject(torch.cat([x, e], -1))
                for i in range(p, p + co):
                    x, v0 = self.blocks[i](x, rope, v0)
            for i in range(p + co, p + co + k):
                x, v0 = self.blocks[i](x, rope, v0)
        return self.nf(x)

    def logits(self, h):
        return h @ self.emb.weight.T if self.head is None else self.head(h)

    def forward(self, idx, tgt=None, loops=None):
        h = self.hidden(idx, loops)
        lg = self.logits(h).float()
        if tgt is None:
            return lg
        return F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.reshape(-1), reduction="mean")


# ----------------------------------------------------------------------------- training
import json, time, glob, os
import numpy as np

ROOT = next(p for p in glob.glob("/kaggle/input/**/manifest.json", recursive=True)).rsplit("/", 1)[0]
OUT = "/kaggle/working"
D, LAYERS, FFN, HEADS, KV = int(os.environ.get("NL_D", 512)), int(os.environ.get("NL_LAYERS", 8)), int(os.environ.get("NL_FFN", 1408)), 8, 4
SEQ, BS, LR, WARM, SEED = 512, 64, 1e-3, 500, 1000
MICRO = int(os.environ.get("NL_MICRO", 2))
MAX_HOURS = float(os.environ.get("NL_HOURS", 10.5))


def get(data, n, rng, dev):
    ix = rng.integers(0, len(data) - SEQ - 1, n)
    t = torch.from_numpy(np.stack([data[i:i + SEQ + 1] for i in ix]).astype(np.int64)).to(dev, non_blocking=True)
    return t[:, :-1], t[:, 1:]


@torch.no_grad()
def evaluate(model, manifest, dev):
    out = {}; model.eval()
    for dom in ("web", "textbook", "math", "code"):
        d = np.memmap(f"{ROOT}/val_{dom}.bin", dtype=np.uint16, mode="r"); rng = np.random.default_rng(0); nll = 0.0
        for _ in range(20):
            x, y = get(d, 16, rng, dev)
            with torch.autocast("cuda", dtype=torch.float16): nll += float(model(x, y))
        out[dom] = nll / 20 / math.log(2) / manifest["domains"][dom]["val"]["bytes_per_token"]
    out["mix"] = sum(manifest["mix"][k] * out[k] for k in manifest["mix"]); model.train()
    return {k: round(v, 4) for k, v in out.items()}


if __name__ == "__main__":
    dev = "cuda"; manifest = json.load(open(f"{ROOT}/manifest.json"))
    data = np.memmap(f"{ROOT}/train.bin", dtype=np.uint16, mode="r")
    steps = len(data) // (BS * SEQ)                                          # one pass in token count
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    model = LM(Cfg(d=D, n_head=HEADS, n_kv=KV, ffn=FFN, layout="A" * LAYERS, seq=SEQ, loop=None)).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    scaler = torch.amp.GradScaler("cuda"); t0 = time.time(); curve = []
    print(json.dumps({"params": params, "steps": steps, "gpu": torch.cuda.get_device_name(0), "tokens": int(len(data))}), flush=True)
    for step in range(1, steps + 1):
        lr = LR * min(1.0, step / WARM) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups: g["lr"] = lr
        x, y = get(data, BS, rng, dev)
        opt.zero_grad(set_to_none=True); loss = 0.0
        for xm, ym in zip(x.chunk(MICRO), y.chunk(MICRO)):      # same update as one batch of BS, in MICRO smaller passes (fits a 16 GB GPU)
            with torch.autocast("cuda", dtype=torch.float16): lm = model(xm, ym) / MICRO
            scaler.scale(lm).backward(); loss += float(lm)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        hours = (time.time() - t0) / 3600
        if step % 2000 == 0 or step == steps or hours > MAX_HOURS:
            row = {"step": step, "tokens": step * BS * SEQ, "hours": round(hours, 3), "train_loss": round(loss, 4),
                   "tokens_per_second": round(step * BS * SEQ / (time.time() - t0)), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), **evaluate(model, manifest, dev)}
            curve.append(row); print(json.dumps(row), flush=True)
            json.dump({"params": params, "cfg": {"d": D, "layers": LAYERS, "ffn": FFN, "seq": SEQ, "bs": BS, "micro": MICRO, "lr": LR, "seed": SEED}, "gpu": torch.cuda.get_device_name(0), "curve": curve}, open(f"{OUT}/result.json", "w"))
            torch.save(model.state_dict(), f"{OUT}/model.pt")
            if hours > MAX_HOURS: print("stopped at time limit", flush=True); break
