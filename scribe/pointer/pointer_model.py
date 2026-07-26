# Stage P — nano trunk (identical to scribe v2) + explicit pointer/copy head.
# GPT      : plain baseline (Arm B), byte-identical architecture to scribe_sft.py.
# GPTCopy  : GPT + copy attention; final dist P = p_gen*P_vocab + (1-p_gen)*P_copy.
# Mechanism spec + math frozen in PREREG_pointer_head.md.
import math, torch, torch.nn as nn, torch.nn.functional as F

dev = "mps" if torch.backends.mps.is_available() else "cpu"
V, d, L, H, KV, hd, ff, S = 4098, 192, 6, 6, 2, 32, 512, 512
DC = 64  # copy-attention head dim (frozen in PREREG)

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

class Trunk(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.blocks = nn.ModuleList(Block() for _ in range(L)); s.nf = nn.RMSNorm(d)
    def hidden(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return s.nf(h)                       # (B,S,d), causal

class GPT(nn.Module):
    """Plain baseline (Arm B) — trunk + tied unembed. Same as scribe_sft.py GPT."""
    def __init__(s):
        super().__init__(); s.t = Trunk()
    def forward(s, x):
        return F.linear(s.t.hidden(x), s.t.emb.weight)

class GPTCopy(nn.Module):
    """Arm P — trunk + pointer/copy head. p_gen*P_vocab + (1-p_gen)*P_copy."""
    def __init__(s):
        super().__init__()
        s.t = Trunk()
        s.Wqc = nn.Linear(d, DC, bias=False)
        s.Wkc = nn.Linear(d, DC, bias=False)
        s.Wg  = nn.Linear(2*d, 1, bias=True)
        nn.init.zeros_(s.Wg.bias)            # p_gen starts ~0.5 (copy given a fair shot)

    def _copy(s, h, src_valid):
        # h:(B,S,d)  src_valid:(B,S) bool → returns alpha:(B,S,S), p_gen:(B,S), c:(B,S,d)
        q = s.Wqc(h); k = s.Wkc(h)                          # (B,S,DC)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(DC)   # (B,S,S) query t over key j
        colmask = src_valid[:, None, :]                     # (B,1,S) — only source keys are copyable
        scores = scores.masked_fill(~colmask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)               # (B,S,S); rows with no source → handled by caller (loss masked there)
        alpha = torch.nan_to_num(alpha, nan=0.0)            # guard fully-masked rows
        c = torch.matmul(alpha, h)                          # (B,S,d) context
        p_gen = torch.sigmoid(s.Wg(torch.cat([h, c], dim=-1))).squeeze(-1)  # (B,S)
        return alpha, p_gen, c

    def logprob_at_targets(s, x, tgt, src_valid):
        """Masked-NLL pieces, mixture evaluated ONLY at the target index (memory-light).
        x:(B,S) input ids, tgt:(B,S) gold next tokens, src_valid:(B,S) source mask.
        Returns logP_tgt:(B,S), and diagnostics (p_gen, copy_share) at every position."""
        h = s.t.hidden(x)                                   # (B,S,d)
        vlog = F.linear(h, s.t.emb.weight)                  # (B,S,V)
        lse = torch.logsumexp(vlog, dim=-1)                 # (B,S) — avoids storing full softmax
        vlog_tgt = vlog.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B,S)
        sm_tgt = torch.exp(vlog_tgt - lse)                  # P_vocab at target (B,S)
        alpha, p_gen, _ = s._copy(h, src_valid)
        eq = (x.unsqueeze(1) == tgt.unsqueeze(-1)) & src_valid.unsqueeze(1)   # (B,S,S)
        pcopy_tgt = (alpha * eq).sum(-1)                    # (B,S) copy prob at target
        P_tgt = p_gen * sm_tgt + (1.0 - p_gen) * pcopy_tgt
        logP = torch.log(P_tgt.clamp_min(1e-9))
        copy_share = ((1.0 - p_gen) * pcopy_tgt) / P_tgt.clamp_min(1e-9)      # (B,S) manipulation stat
        return logP, p_gen, pcopy_tgt, copy_share

    @torch.no_grad()
    def full_dist_last(s, ids_tensor, pos, src_len):
        """Full mixture P over V at position `pos` (inference). ids_tensor:(1,S) padded.
        src_len: copy source is [0, src_len) (the prompt). Returns P:(V,)."""
        h = s.t.hidden(ids_tensor)[0]                       # (S,d)
        hp = h[pos]                                         # (d,)
        P_vocab = torch.softmax(F.linear(hp, s.t.emb.weight), dim=-1)         # (V,)
        q = s.Wqc(hp); k = s.Wkc(h[:src_len])              # (DC,), (src_len,DC)
        scores = (k @ q) / math.sqrt(DC)                    # (src_len,)
        alpha = torch.softmax(scores, dim=-1)               # (src_len,)
        c = alpha @ h[:src_len]                             # (d,)
        p_gen = torch.sigmoid(s.Wg(torch.cat([hp, c], -1))).squeeze(-1)       # scalar
        P_copy = torch.zeros(V, device=hp.device)
        P_copy.index_add_(0, ids_tensor[0, :src_len], alpha)
        return p_gen * P_vocab + (1.0 - p_gen) * P_copy

def src_valid_from_mask(mask):
    """source region = positions strictly before the first assistant-content token (mask==1)."""
    B = mask.shape[0]
    first = (mask == 1).float().argmax(dim=1)              # (B,) first assistant idx
    ar = torch.arange(S, device=mask.device)[None, :]      # (1,S)
    return ar < first[:, None]                             # (B,S) bool
