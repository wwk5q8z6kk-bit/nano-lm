"""Replay eight complete, fixed repository groups per cell with an independent CPU loss."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from study import ARMS, LENGTH, SEEDS, digest, make_model, save, verified_corpus


@torch.inference_mode()
def check(args):
    torch.set_num_threads(1)
    docs, streams = verified_corpus(args.data, args.tokenizer)
    tape = streams["val"]
    by_repo = {}
    for doc in docs["val"]:
        by_repo.setdefault(doc["repo"].casefold(), []).append(doc)
    names = sorted(by_repo)
    selected = [i for i,name in enumerate(names) if 128 <= sum(d["tokens"] - 1 for d in by_repo[name]) <= 2048][:8]
    assert len(selected) == 8
    cases = []
    for seed in SEEDS:
        for arm in ARMS:
            expected = json.loads((args.results / f"eval-{seed}-{arm}.json").read_text())
            weights = args.fits / f"seed-{seed}-{arm}.pt"
            assert digest(weights) == expected["weights_sha256"]
            model = make_model(arm, seed)
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            for i in selected:
                total = 0
                sums = {name: 0.0 for name in expected["heads"]}
                for doc in by_repo[names[i]]:
                    offset = doc["token_offset"]
                    end = offset + doc["tokens"]
                    while offset < end - 1:
                        sequence = torch.from_numpy(tape[offset:min(end,offset+LENGTH+1)].astype(np.int64)).unsqueeze(0)
                        inputs, labels = sequence[:,:-1], sequence[:,1:]
                        if arm == "m21":
                            mixed,base = model(inputs, return_base_logits=True)
                            distributions = {"primary":mixed, "base":base}
                        else:
                            distributions = {"primary":model(inputs)}
                        for head,dist in distributions.items():
                            lp = torch.nn.functional.log_softmax(dist.double(),dim=-1)
                            nll = torch.nn.functional.nll_loss(lp[0],labels[0],reduction="sum")
                            sums[head] += float(nll) / math.log(2)
                        total += labels.numel()
                        offset += labels.numel()
                for head,value in sums.items():
                    ref = expected["heads"][head]["repo"][str(i)]
                    assert ref["targets"] == total
                    delta = abs((value - ref["bits"]) / total)
                    if delta > 2e-5:
                        raise ValueError(f"CPU/checkpoint discrepancy {seed}/{arm}/{head}: {delta}")
                    cases.append({"seed":seed,"arm":arm,"head":head,"group":i,"targets":total,
                                  "absolute_delta_bits_per_token":round(delta,9)})
    return {"groups_per_cell":len(selected),"comparisons":cases,
            "maximum_absolute_delta_bits_per_token":max(c["absolute_delta_bits_per_token"] for c in cases)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("data","tokenizer","fits","results","out"):
        ap.add_argument("--" + arg,type=Path,required=True)
    args = ap.parse_args()
    save(args.out,check(args))
