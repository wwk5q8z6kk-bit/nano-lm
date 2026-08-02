import torch
import torch.nn.functional as F
import model_nano as M
import time
import json

def logp(m, pr, comp):
    return M.seq_logprob(m, pr, comp)

pairs = json.load(open("pref_pairs.json"))
policy = M.load("sft.pt")
policy.train()
ref = M.load("sft.pt")
for p in ref.parameters(): p.requires_grad_(False)
opt = torch.optim.AdamW(policy.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.0)
BETA = 0.1

t0 = time.time()
idx = list(range(len(pairs)))
for j in idx:
    pr, cw, cl = pairs[j]["prompt"], pairs[j]["chosen"], pairs[j]["rejected"]
    with torch.no_grad():
        rw = logp(ref, pr, cw); rl = logp(ref, pr, cl)
    pw = logp(policy, pr, cw); pl = logp(policy, pr, cl)
    margin = BETA * ((pw - rw) - (pl - rl))
    loss = -F.logsigmoid(margin)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
print(f"unbatched time: {time.time()-t0:.3f}s")
