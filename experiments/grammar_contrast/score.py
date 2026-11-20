"""Teacher-forced full-sentence comparison on the pinned grammar panel."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from panel import digest, load_panel

HERE = Path(__file__).resolve().parent


def native(arm, directory, runs, data):
    sys.path.insert(0, str(directory.resolve()))
    from evaluate import load_model, verify_file
    inputs = json.loads((directory / "inputs.json").read_text())
    tok_path = data / "tokenizer_32k.json"
    verify_file(tok_path, inputs["tokenizer_sha256"])
    tokenizer = Tokenizer.from_file(str(tok_path))
    if arm == "unigram":
        train = data / "train.bin"
        verify_file(train, inputs["train_sha256"])
        counts = np.bincount(np.memmap(train, dtype="<u2", mode="r")[::100], minlength=32768)
        logp = np.log((counts + 1) / (counts.sum() + len(counts)))
        return tokenizer, 1, None, logp, inputs["train_sha256"]
    model = load_model(arm, runs, inputs)
    return tokenizer, 1, model, None, inputs["arms"][arm]["sha256"]


def public_model(directory, pins):
    from transformers import AutoModelForCausalLM
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        import hashlib
        h = hashlib.sha256()
        with (directory / name).open("rb") as f:
            for chunk in iter(lambda: f.read(4_194_304), b""):
                h.update(chunk)
        if h.hexdigest() != pins[name]:
            raise ValueError(f"Public checkpoint mismatch in {name}")
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(str(directory), local_files_only=True,
                                                 trust_remote_code=False, torch_dtype=torch.float32).eval()
    eos = model.config.eos_token_id
    if eos != tokenizer.token_to_id("<|endoftext|>"):
        raise ValueError("Unexpected public tokenizer prefix")
    return tokenizer, eos, model, None, pins["model.safetensors"]


def logits_for(model, x, public):
    if public:
        return model(input_ids=x, use_cache=False).logits
    return model(x)


@torch.inference_mode()
def scores(model, ids, eos, logp, public, batch_size):
    if logp is not None:
        return [float(logp[tokens].sum()) for tokens in ids]
    groups = defaultdict(list)
    for i, tokens in enumerate(ids):
        groups[len(tokens)].append(i)
    result = [None] * len(ids)
    for length, idx in sorted(groups.items()):
        for start in range(0, len(idx), batch_size):
            subset = idx[start:start + batch_size]
            t = torch.tensor([ids[i] for i in subset])
            x = torch.cat((torch.full_like(t[:, :1], eos), t[:, :-1]), dim=1)
            logits = logits_for(model, x, public)
            losses = F.cross_entropy(logits.double().flatten(0, 1), t.flatten(), reduction="none")
            ll = -losses.view(len(subset), length).sum(1)
            if not torch.isfinite(ll).all():
                raise ValueError("Nonfinite sentence score")
            for i, value in zip(subset, ll.tolist()):
                result[i] = value
    return result


def run(args):
    rows = load_panel(args.blimp, HERE / "panel.json")
    public = args.arm.startswith("pythia")
    if public:
        pins = json.loads((HERE / "public-models.json").read_text())[args.arm]
        tokenizer, eos, model, logp, identity = public_model(args.weights, pins["files"])
    else:
        tokenizer, eos, model, logp, identity = native(args.arm, args.evaluator, args.runs, args.data)
    sentences = [text for row in rows for text in row[3:]]
    ids = [enc.ids for enc in tokenizer.encode_batch(sentences, add_special_tokens=False)]
    if any(not v or len(v) > 256 for v in ids):
        raise ValueError("Sentence exceeds the shared supported input range")
    for text, tokens in zip(sentences, ids):
        if tokenizer.decode(tokens, skip_special_tokens=False) != text:
            raise ValueError("Sentence failed byte-level tokenizer replay")
    values = scores(model, ids, eos, logp, public, args.batch_size)
    # Re-score a dispersed sample with singleton batches to test batch independence.
    ix = list(range(0, len(ids), 193))
    check = scores(model, [ids[i] for i in ix], eos, logp, public, 1)
    err = max(abs(values[i] - other) for i, other in zip(ix, check))
    if err > 0.001:
        raise ValueError(f"Singleton score mismatch {err}")
    result = dict(arm=args.arm, identity=identity, pairs=len(rows), singleton_max_abs=err,
                  torch=str(torch.__version__), panel=json.loads((HERE / "panel.json").read_text())["panel_sha256"],
                  rows=[[row[0],row[1],row[2],len(ids[2*i]),len(ids[2*i+1]),values[2*i]-values[2*i+1]]
                        for i,row in enumerate(rows)])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(args.arm, sum(item[-1] > 0 for item in result["rows"]), "/", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["S2","M0","M2","unigram","pythia-final","pythia-step256"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blimp", required=True, type=Path)
    for arg in ("evaluator","runs","data","weights"):
        parser.add_argument("--" + arg, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("Thread and batch counts must be positive")
    torch.set_num_threads(args.threads)
    run(args)
