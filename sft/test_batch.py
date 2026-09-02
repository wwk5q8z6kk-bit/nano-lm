import torch
import torch.nn.functional as F
import model_nano as M

def batched_seq_logprob(m, prs, comps):
    B = len(prs)
    x = torch.zeros((B, M.S), dtype=torch.long, device=M.dev)
    tots = torch.zeros(B, device=M.dev)

    # Pack into tensor
    for i in range(B):
        full = (prs[i] + comps[i])[:M.S]
        x[i, :len(full)] = torch.tensor(full, dtype=torch.long, device=M.dev)

    logits = m(x)
    lp = F.log_softmax(logits.float(), -1)

    # Calculate sum logprobs
    for i in range(B):
        full = (prs[i] + comps[i])[:M.S]
        lo = len(prs[i])
        hi = len(full)
        if hi > lo:
            pos_indices = torch.arange(lo - 1, hi - 1, device=M.dev)
            target_tokens = torch.tensor(full[lo:hi], dtype=torch.long, device=M.dev)
            tots[i] = lp[i, pos_indices, target_tokens].sum()
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
