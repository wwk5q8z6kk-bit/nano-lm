"""Independent synthetic checks for shifted labels, padding, normalization and pointer support."""
import math

import numpy as np
import torch

from evaluate import bits, chunks, evaluate, packed, pointer_available
from study import LENGTH, make_model


def run():
    torch.set_num_threads(1)
    lengths = [2, 37, 130, 257, 513]
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    docs = [{"tokens": n, "token_offset": int(offset), "repo": str(i % 3), "license": "mit"} for i,(n,offset) in enumerate(zip(lengths, offsets))]
    tape = np.arange(sum(lengths), dtype=np.int64) % 13 + 32
    parts = list(chunks(docs))
    visited = []
    for doc_id, start, n in parts:
        assert n <= LENGTH and start + n < docs[doc_id]["token_offset"] + docs[doc_id]["tokens"]
        visited.extend(range(start + 1, start + n + 1))
    assert sorted(visited) == sorted(set(range(len(tape))) - set(offsets))
    assert len(visited) == len(set(visited))
    rows = torch.from_numpy(packed(tape, parts[:5]))
    x,y = rows[:, :-1],rows[:, 1:]
    observed = pointer_available(x,y)
    # Independent slow definition of the architecture's copied-next-token set.
    for b,(_,_,n) in enumerate(parts[:5]):
        seq = x[b].tolist()
        for t in range(n):
            exact = [j for j in range(t-1, -1, -1) if seq[j] == seq[t]][:8]
            near = range(max(0,t-24),t)
            wanted = int(y[b,t]) in {seq[j+1] for j in set(exact) | set(near)}
            assert bool(observed[b,t]) == wanted
    torch.manual_seed(53)
    logits = torch.randn(2,3,53)
    target = torch.tensor([[0,1,2],[1,0,0]])
    direct = -torch.log_softmax(logits.double(), -1).gather(-1,target[...,None]).squeeze(-1)/math.log(2)
    assert torch.allclose(bits(logits, target).double(),direct,atol=1e-6)
    # Direct one-sequence-at-a-time unpadded forward and double log normalization
    # give the same result as batched, padded and masked end-to-end evaluation.
    for arm in ("m21","gqa"):
        mdl = make_model(arm, 0).eval()
        retained = evaluate(mdl,arm,docs,tape,torch.device("cpu"))
        sums = {k:0.0 for k in retained}
        with torch.no_grad():
            for _,i,n in parts:
                q = torch.tensor(tape[i:i+n])[None,:]
                answer = torch.tensor(tape[i+1:i+n+1])[None,:,None]
                result = mdl(q, return_base_logits=True) if arm == "m21" else (mdl(q),)
                for head,dist in zip(sums,result):
                    sums[head] += float(-torch.log_softmax(dist.double(),-1).gather(-1,answer).sum()) / math.log(2)
        for head,h in retained.items():
            pooled = h["all"]["pooled"]
            assert pooled["targets"] == sum(lengths) - len(docs)
            assert abs(pooled["bits"] - sums[head])/pooled["targets"] < 2e-5
    print("Synthetic checks passed for every target, document boundaries, copy eligibility, and both padded readouts.")


if __name__ == "__main__":
    run()
