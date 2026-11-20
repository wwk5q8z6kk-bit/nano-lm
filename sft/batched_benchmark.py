import torch
import model_nano as M
import time
import json

from dpo import BS, train_batch

pairs = json.load(open("benchmark_pref_pairs.json"))
policy = M.load("benchmark_sft.pt")
policy.train()
ref = M.load("benchmark_sft.pt")
for p in ref.parameters(): p.requires_grad_(False)
opt = torch.optim.AdamW(policy.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.0)
t0 = time.time()
for start in range(0, len(pairs), BS):
    for _ in train_batch(policy, ref, opt, pairs[start:start + BS]):
        pass
print(f"batched time: {time.time()-t0:.3f}s")
