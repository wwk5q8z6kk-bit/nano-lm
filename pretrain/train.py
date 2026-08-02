# Nano pretrain per vault recipes: Recipe 1 (components), Recipe 2 §3 (config),
# §6 (budget), §7 (monitoring). MPS fp32 (vault bf16 guidance is CUDA-centric).
import math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
V, d, L, H, KV, hd, ff, S = 4096, 192, 6, 6, 2, 32, 512, 512
BATCH, STEPS, WARM, PEAK_LR, FLOOR = 16, 4000, 100, 3e-3, 0.1
WD, CLIP, ZL = 0.1, 1.0, 1e-4

def batch(src, bs=BATCH):
    ix = torch.randint(len(src) - S - 1, (bs,))
    x = torch.stack([src[i:i+S] for i in ix]); y = torch.stack([src[i+1:i+S+1] for i in ix])
    return x.to(dev), y.to(dev)

def rope(q, k):  # RoPE base 10000 per Recipe 1
    t = torch.arange(S, device=dev, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    f = torch.outer(t, inv); cos, sin = f.cos()[None,None], f.sin()[None,None]
    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).flatten(-2)
    return rot(q), rot(k)

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1, s.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        s.q, s.k, s.v, s.o = nn.Linear(d,H*hd,bias=False), nn.Linear(d,KV*hd,bias=False), nn.Linear(d,KV*hd,bias=False), nn.Linear(H*hd,d,bias=False)
        s.g, s.u, s.dn = nn.Linear(d,ff,bias=False), nn.Linear(d,ff,bias=False), nn.Linear(ff,d,bias=False)
    def forward(s, x):
        B = x.shape[0]; h = s.n1(x)
        q = s.q(h).view(B,S,H,hd).transpose(1,2); k = s.k(h).view(B,S,KV,hd).transpose(1,2); v = s.v(h).view(B,S,KV,hd).transpose(1,2)
        q, k = rope(q, k)
        k, v = k.repeat_interleave(H//KV,1), v.repeat_interleave(H//KV,1)   # GQA 3:1
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(a.transpose(1,2).reshape(B,S,H*hd))
        h = s.n2(x)
        return x + s.dn(F.silu(s.g(h)) * s.u(h))                            # SwiGLU

class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.blocks = nn.ModuleList(Block() for _ in range(L)); s.nf = nn.RMSNorm(d)
        for p in s.parameters():
            if p.dim() >= 2: nn.init.normal_(p, std=0.02)
        for b in s.blocks:  # depth-scaled init on residual-out projections (Recipe 2 §3)
            nn.init.normal_(b.o.weight, std=0.02/math.sqrt(2*L)); nn.init.normal_(b.dn.weight, std=0.02/math.sqrt(2*L))
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return F.linear(s.nf(h), s.emb.weight)                               # tied head

def lr_at(t):  # linear warmup -> cosine to 10% floor (Recipe 2 §3)
    if t < WARM: return PEAK_LR * t / WARM
    p = (t - WARM) / (STEPS - WARM)
    return PEAK_LR * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * p)))

if __name__ == "__main__":
    ids = np.load("shard_000.npy"); n_val = len(ids) // 50
    train_ids, val_ids = torch.tensor(ids[:-n_val].astype(np.int64)), torch.tensor(ids[-n_val:].astype(np.int64))
    m = GPT().to(dev)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params={n_params/1e6:.2f}M  device={dev}  budget: 6ND={6*n_params*BATCH*S*STEPS/1e15:.2f} PFLOPs", flush=True)
    decay = [p for p in m.parameters() if p.dim() >= 2]; nodecay = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)

    t0 = time.time()
    for step in range(1, STEPS + 1):
        for g in opt.param_groups: g["lr"] = lr_at(step)
        x, y = batch(train_ids)
        logits = m(x)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        zloss = ZL * (torch.logsumexp(logits.float(), -1) ** 2).mean()          # z-loss (Recipe 2 §3)
        opt.zero_grad(set_to_none=True); (loss + zloss).backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)               # grad-norm watch (§7)
        opt.step()
        if step % 200 == 0 or step == 1:
            tps = step * BATCH * S / (time.time() - t0)
            print(f"step {step:5d}  loss {loss.item():.3f}  gnorm {gn.item():.2f}  lr {lr_at(step):.1e}  {tps/1e3:.0f}k tok/s", flush=True)
        if step % 1000 == 0 or step == STEPS:
            with torch.no_grad():
                vl = sum(F.cross_entropy(m(bx).view(-1,V), by.view(-1)).item() for bx, by in [batch(val_ids, 8) for _ in range(8)]) / 8
            print(f"  == val loss {vl:.3f} (held-out stream) ==", flush=True)
            torch.save(m.state_dict(), "ckpt.pt")                               # checkpoint cadence (§7)
    print(f"done in {(time.time()-t0)/60:.1f} min; trained {STEPS*BATCH*S/1e6:.1f}M tokens (~{STEPS*BATCH*S/len(train_ids):.1f} epochs)", flush=True)
