import torch
import torch.nn.functional as F
import model_nano as M
import time
import json

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

    return (log_probs * m_mask).sum(dim=-1)

pairs = json.load(open("pref_pairs.json"))
policy = M.load("sft.pt")
policy.train()
ref = M.load("sft.pt")
for p in ref.parameters(): p.requires_grad_(False)
opt = torch.optim.AdamW(policy.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.0)
BETA = 0.1

BS = 16

t0 = time.time()
idx = list(range(len(pairs)))
step = 0
for b in range(0, len(idx), BS):
    batch_idx = idx[b:b+BS]
    prs, cws, cls = [], [], []
    for j in batch_idx:
        prs.append(pairs[j]["prompt"])
        cws.append(pairs[j]["chosen"])
        cls.append(pairs[j]["rejected"])

    with torch.no_grad():
        rw = batched_seq_logprob(ref, prs, cws)
        rl = batched_seq_logprob(ref, prs, cls)

    pw = batched_seq_logprob(policy, prs, cws)
    pl = batched_seq_logprob(policy, prs, cls)

    margin = BETA * ((pw - rw) - (pl - rl))
    loss = -F.logsigmoid(margin).mean()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
print(f"batched time: {time.time()-t0:.3f}s")
