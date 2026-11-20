import torch
import json
import model_nano as M

# create dummy benchmark_sft.pt
m = M.GPT().to(M.dev)
torch.save(m.state_dict(), "benchmark_sft.pt")

# create dummy benchmark_pref_pairs.json
pairs = []
for i in range(16):
    pairs.append({
        "prompt": [1, 2, 3],
        "chosen": [4, 5, 6],
        "rejected": [7, 8, 9]
    })
with open("benchmark_pref_pairs.json", "w") as f:
    json.dump(pairs, f)
