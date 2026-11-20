import torch
import torch.nn.functional as F
import model_nano as M

def batched_seq_logprob(m, prs, comps):
    B = len(prs)
    x = torch.zeros((B, M.S), dtype=torch.long, device=M.dev)
    mask = torch.zeros((B, M.S), dtype=torch.bool, device=M.dev)

    for i in range(B):
        full = (prs[i] + comps[i])[:M.S]
        l_full = len(full)
        x[i, :l_full] = torch.tensor(full, dtype=torch.long, device=M.dev)
        lo = len(prs[i])
        if l_full > lo:
            mask[i, lo:l_full] = True

    logits = m(x)
    lp = F.log_softmax(logits.float(), -1)

    tgt = x[:, 1:]
    p = lp[:, :-1]

    log_probs = p.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
    m_mask = mask[:, 1:]

    tots = (log_probs * m_mask).sum(dim=-1)
    return tots

m = M.GPT().to(M.dev)

prs = [[1, 2, 3], [4, 5]]
comps = [[4, 5, 6, 7], [6, 7, 8]]

# unbatched
tot0 = M.seq_logprob(m, prs[0], comps[0])
tot1 = M.seq_logprob(m, prs[1], comps[1])

# batched
tots = batched_seq_logprob(m, prs, comps)

print(tot0, tot1)
print(tots)
