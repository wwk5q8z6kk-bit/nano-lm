# Stage P2 — copy-SUPERVISED pointer head (the pre-authorized fix for P1's VOID).
# Changes vs pointer_model.GPTCopy (all engage the copy pathway; math otherwise identical):
#   1. copy-supervision aux loss: L_copy = -log(P_copy_tgt) at copyable target positions,
#      returned so train2.py can add lambda*L_copy — gives the copy attention a gradient
#      even when vocab could memorize (the P1 failure).
#   2. copy-favoring gate init: Wg.bias = -2.0  => p_gen starts ~0.12 (copy exercised early).
#   3. source-key-restricted copy attention: keys/eq cropped to [0, Ksrc) (the prompt region).
#      Behaviour-equivalent to the full-key version (non-source keys were -inf-masked anyway),
#      ~3x faster on MPS. Verified equal to pointer_model on the smoke fixture.
import math, torch, torch.nn as nn, torch.nn.functional as F
from pointer_model import Trunk, dev, V, d, S, DC   # shared trunk/rope/block, same constants

GATE_BIAS_INIT = -2.0   # frozen in PREREG v2

class GPTCopy2(nn.Module):
    def __init__(s):
        super().__init__()
        s.t = Trunk()
        s.Wqc = nn.Linear(d, DC, bias=False)
        s.Wkc = nn.Linear(d, DC, bias=False)
        s.Wg  = nn.Linear(2*d, 1, bias=True)
        nn.init.constant_(s.Wg.bias, GATE_BIAS_INIT)     # copy-favoring start

    def _copy(s, h, x, src_valid, Ksrc):
        """source-key-restricted copy attention. h:(B,S,d) x:(B,S) src_valid:(B,S)
        Returns alpha:(B,S,Ksrc), p_gen:(B,S), x_src:(B,Ksrc), srcv_c:(B,Ksrc)."""
        h_src = h[:, :Ksrc]; x_src = x[:, :Ksrc]; srcv_c = src_valid[:, :Ksrc]      # crop to source
        q = s.Wqc(h); k = s.Wkc(h_src)                                              # (B,S,DC),(B,Ksrc,DC)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(DC)               # (B,S,Ksrc)
        scores = scores.masked_fill(~srcv_c[:, None, :], float("-inf"))
        alpha = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        c = torch.matmul(alpha, h_src)                                              # (B,S,d)
        p_gen = torch.sigmoid(s.Wg(torch.cat([h, c], dim=-1))).squeeze(-1)          # (B,S)
        return alpha, p_gen, x_src, srcv_c

    def logprob_at_targets(s, x, tgt, src_valid):
        """Returns logP_tgt, p_gen, pcopy_tgt, copy_share, L_copy_pos, copyable
        (last two supervise the copy attention; all (B,S))."""
        h = s.t.hidden(x)
        Ksrc = int((src_valid.sum(1).max()).item()) if src_valid.any() else 1       # batch-max source len
        vlog = F.linear(h, s.t.emb.weight)
        lse = torch.logsumexp(vlog, dim=-1)
        sm_tgt = torch.exp(vlog.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lse)
        alpha, p_gen, x_src, srcv_c = s._copy(h, x, src_valid, Ksrc)
        eq = (x_src.unsqueeze(1) == tgt.unsqueeze(-1)) & srcv_c.unsqueeze(1)         # (B,S,Ksrc)
        pcopy_tgt = (alpha * eq).sum(-1)                                             # (B,S)
        P_tgt = p_gen * sm_tgt + (1.0 - p_gen) * pcopy_tgt
        logP = torch.log(P_tgt.clamp_min(1e-9))
        copy_share = ((1.0 - p_gen) * pcopy_tgt) / P_tgt.clamp_min(1e-9)
        copyable = eq.any(-1)                                                        # tgt token present in source
        L_copy_pos = -torch.log(pcopy_tgt.clamp_min(1e-9))                           # push copy-mass onto correct id
        return logP, p_gen, pcopy_tgt, copy_share, L_copy_pos, copyable

    @torch.no_grad()
    def full_dist_last(s, ids_tensor, pos, src_len):
        h = s.t.hidden(ids_tensor)[0]
        hp = h[pos]
        P_vocab = torch.softmax(F.linear(hp, s.t.emb.weight), dim=-1)
        q = s.Wqc(hp); k = s.Wkc(h[:src_len])
        alpha = torch.softmax((k @ q) / math.sqrt(DC), dim=-1)
        c = alpha @ h[:src_len]
        p_gen = torch.sigmoid(s.Wg(torch.cat([hp, c], -1))).squeeze(-1)
        P_copy = torch.zeros(V, device=hp.device)
        P_copy.index_add_(0, ids_tensor[0, :src_len], alpha)
        return p_gen * P_vocab + (1.0 - p_gen) * P_copy
