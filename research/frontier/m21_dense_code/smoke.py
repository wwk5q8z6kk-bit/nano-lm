"""Check trained entrypoints on the training split and dense source parity."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from study import ARMS, PARAMS, make_model, sample_offsets, tensor_rows, train_step, verified_corpus, save


def check(args):
    torch.set_num_threads(1)
    docs, streams = verified_corpus(args.data, args.tokenizer)
    seeds = {}
    for s in range(3):
        rows = sample_offsets(docs["train"], s)
        seeds[s] = hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()
    if len(set(seeds.values())) != 3:
        raise ValueError("Sampling seeds not independent")
    inputs = tensor_rows(streams["train"], sample_offsets(docs["train"], 0, 1)[0][:2], torch.device("cpu"))
    cells = {}
    for arm in ARMS:
        mdl = make_model(arm, 0)
        with torch.no_grad():
            tiny = inputs[:, :20].clone()
            before = torch.log_softmax(mdl(tiny), dim=-1)
            tiny[0, 12:] = 555
            tiny[1, :] = 777
            after = torch.log_softmax(mdl(tiny), dim=-1)
            prefix_delta = (before[0, :12] - after[0, :12]).abs().max().item()
            if prefix_delta > 2e-5:
                raise ValueError(f"Prefix perturbation changed {arm} valid predictions")
        opt = torch.optim.AdamW(mdl.parameters(), lr=0.0015, weight_decay=0.01)
        snapshot = next(mdl.parameters()).detach().clone()
        loss = train_step(mdl, arm, opt, inputs, 0)
        if not torch.isfinite(loss) or torch.equal(next(mdl.parameters()), snapshot):
            raise ValueError(f"{arm} optimizer failed")
        cells[arm] = {"parameters": PARAMS[arm], "step1_loss": float(loss), "prefix_max_delta": prefix_delta}
    if args.original_dense:
        spec = importlib.util.spec_from_file_location("original_dense_source", args.original_dense)
        orig = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = orig
        spec.loader.exec_module(orig)
        original_cfg = orig.Cfg(vocab=32768, d=128, n_head=4, n_kv=2, ffn=426, layout="AAAA", seq=128, tie=True)
        torch.manual_seed(1000)
        orig_model = orig.LM(original_cfg)
        portable = make_model("gqa", 0)
        assert list(orig_model.state_dict()) == list(portable.state_dict())
        assert all(torch.equal(v, portable.state_dict()[k]) for k,v in orig_model.state_dict().items())
        x, y = inputs[:, :16], inputs[:, 1:17]
        assert torch.equal(orig_model(x), portable(x))
        orig_model(x, tgt=y).backward()
        portable(x, tgt=y).backward()
        assert all(torch.equal(p.grad, dict(portable.named_parameters())[k].grad) for k,p in orig_model.named_parameters())
    return {"training_windows_per_seed": 32000, "offsets": seeds, "models": cells,
            "dense_original_cpu_state_logits_gradients_equal": bool(args.original_dense),
            "validation_optimizer_or_model_calls": 0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "tokenizer", "out"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--original-dense", type=Path)
    args = ap.parse_args()
    save(args.out, check(args))
