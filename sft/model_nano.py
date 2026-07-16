# Shared nano model (Recipe ① spec) + generation, used by SFT-gate, DPO, GRPO.
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

dev = "mps" if torch.backends.mps.is_available() else "cpu"
V, d, L, H, KV, hd, ff, S = 4098, 192, 6, 6, 2, 32, 512, 512
tok = Tokenizer.from_file("tokenizer.json")
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")

def rope(q, k):
    t = torch.arange(S, device=dev, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    f = torch.outer(t, inv); cos, sin = f.cos()[None, None], f.sin()[None, None]
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
        q, k = rope(q, k); k, v = k.repeat_interleave(H//KV,1), v.repeat_interleave(H//KV,1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(a.transpose(1,2).reshape(B,S,H*hd)); h = s.n2(x)
        return x + s.dn(F.silu(s.g(h)) * s.u(h))

class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.blocks = nn.ModuleList(Block() for _ in range(L)); s.nf = nn.RMSNorm(d)
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return F.linear(s.nf(h), s.emb.weight)

def load(path):
    m = GPT().to(dev); m.load_state_dict(torch.load(path, map_location=dev, weights_only=True)); m.eval(); return m

def chat_ids(user):
    return tok.encode(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).ids

@torch.no_grad()
def sample(m, user, budget=64, temp=0.8, greedy=False):
    """Return (completion_token_ids, stopped_bool). Completion excludes the closing <|im_end|>."""
    ids = chat_ids(user); out = []
    for _ in range(budget):
        ctx = ids[-S:]
        x = torch.tensor(ctx, device=dev)[None]
        if x.shape[1] < S: x = F.pad(x, (0, S - x.shape[1]))
        logits = m(x)[0, len(ctx) - 1]
        nxt = int(logits.argmax()) if greedy else int(torch.multinomial(F.softmax(logits/temp, -1), 1))
        if nxt == IME: return out, True
        out.append(nxt); ids.append(nxt)
    return out, False

def seq_logprob(m, prompt_ids, comp_ids):
    """Summed log-prob of comp_ids given prompt_ids under model m (teacher-forced, prompt masked)."""
    full = (prompt_ids + comp_ids)[:S]
    comp_lo = len(prompt_ids); comp_hi = len(full)
    x = torch.tensor(full, device=dev)[None]
    if x.shape[1] < S: x = F.pad(x, (0, S - x.shape[1]))
    logits = m(x)[0]                                   # (S, V)
    lp = F.log_softmax(logits.float(), -1)
    tot = 0.0
    for pos in range(comp_lo, comp_hi):                # predict token at pos from logits[pos-1]
        tot = tot + lp[pos - 1, full[pos]]
    return tot
