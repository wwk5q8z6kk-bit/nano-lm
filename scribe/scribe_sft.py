# nano-scribe fine-tune: dpo.pt (gate-passed) -> structured visit-summary task.
# Same architecture/masked-loss machinery as Stage-1 SFT; V=4098 already, no resize.
# LR 1.5e-4 (half the Stage-1 peak): later-stage specialization on a narrow task —
# lower LR to reduce catastrophic forgetting of chat/refusal behavior (logged in AUDIT).
import math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
BASE = "../sft/dpo.pt"
V = 4098
d, L, H, KV, hd, ff, S = 192, 6, 6, 2, 32, 512, 512
EPOCHS, PEAK_LR, WARM_FRAC, FLOOR, WD, CLIP = 3, 1.5e-4, 0.03, 0.1, 0.1, 1.0
BATCH = 32

def rope(q, k):
    t = torch.arange(S, device=dev, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    f = torch.outer(t, inv); cos, sin = f.cos()[None, None], f.sin()[None, None]
    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return rot(q), rot(k)

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1, s.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        s.q, s.k, s.v, s.o = nn.Linear(d, H*hd, bias=False), nn.Linear(d, KV*hd, bias=False), nn.Linear(d, KV*hd, bias=False), nn.Linear(H*hd, d, bias=False)
        s.g, s.u, s.dn = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)
    def forward(s, x):
        B = x.shape[0]; h = s.n1(x)
        q = s.q(h).view(B, S, H, hd).transpose(1, 2); k = s.k(h).view(B, S, KV, hd).transpose(1, 2); v = s.v(h).view(B, S, KV, hd).transpose(1, 2)
        q, k = rope(q, k)
        k, v = k.repeat_interleave(H//KV, 1), v.repeat_interleave(H//KV, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(B, S, H*hd))
        h = s.n2(x)
        return x + s.dn(F.silu(s.g(h)) * s.u(h))

class GPT(nn.Module):
    def __init__(s, vocab):
        super().__init__()
        s.emb = nn.Embedding(vocab, d); s.blocks = nn.ModuleList(Block() for _ in range(L)); s.nf = nn.RMSNorm(d)
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return F.linear(s.nf(h), s.emb.weight)

m = GPT(V)
m.load_state_dict(torch.load(BASE, map_location="cpu", weights_only=True)); m.to(dev)
print(f"loaded {BASE}; params={sum(p.numel() for p in m.parameters())/1e6:.2f}M", flush=True)

X = torch.tensor(np.load("scribe_x.npy").astype(np.int64))
Mk = torch.tensor(np.load("scribe_mask.npy").astype(np.int64))
N = X.shape[0]; STEPS = (N // BATCH) * EPOCHS; WARM = int(WARM_FRAC * STEPS)
print(f"{N} examples, {STEPS} steps, {EPOCHS} epochs, batch {BATCH}", flush=True)

decay = [p for p in m.parameters() if p.dim() >= 2]; nodecay = [p for p in m.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)
def lr_at(t):
    if t < WARM: return PEAK_LR * t / max(1, WARM)
    p = (t - WARM) / max(1, STEPS - WARM)
    return PEAK_LR * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * p)))

perm = torch.randperm(N)
def get(step):
    i0 = (step * BATCH) % (N - BATCH)
    idx = perm[i0:i0 + BATCH]
    return X[idx].to(dev), Mk[idx].to(dev)

t0 = time.time()
for step in range(1, STEPS + 1):
    for g in opt.param_groups: g["lr"] = lr_at(step)
    x, msk = get(step)
    logits = m(x)[:, :-1]; tgt = x[:, 1:]; mtgt = msk[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
    loss = (ce * mtgt).sum() / mtgt.sum().clamp(min=1)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
    opt.step()
    if step % 100 == 0 or step == 1:
        print(f"step {step:5d}/{STEPS}  scribe_loss {loss.item():.3f}  gnorm {gn.item():.2f}  lr {lr_at(step):.1e}  "
              f"{step*BATCH*S/(time.time()-t0)/1e3:.0f}k tok/s", flush=True)
torch.save(m.state_dict(), "scribe.pt")
print(f"scribe SFT done in {(time.time()-t0)/60:.1f} min -> scribe.pt", flush=True)
