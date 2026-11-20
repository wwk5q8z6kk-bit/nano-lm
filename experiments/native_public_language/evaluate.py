"""Score one frozen native checkpoint or compute the independent unigram and overlap controls."""

import argparse
from collections import Counter
import hashlib
import json
import math
import mmap
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from corpus import HERE, sha, write_json


CONTEXT = 256
EOS = 1


def verify_file(path, expected):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Digest mismatch for {path.name}")


def read_passages(path, data, inputs):
    tok_path = data / "tokenizer_32k.json"
    verify_file(tok_path, inputs["tokenizer_sha256"])
    tk = Tokenizer.from_file(str(tok_path))
    if tk.token_to_id("<|eos|>") != EOS or tk.get_vocab_size() != 32768:
        raise ValueError("Unexpected historical tokenizer")
    rows = json.loads(path.read_text())
    locked = json.loads((HERE / "sources.json").read_text())
    if [{k: v for k, v in row.items() if k != "text"} for row in rows] != locked:
        raise ValueError("Passage identities or quantities do not match protocol")
    for row in rows:
        text = row.pop("text")
        ids = tk.encode(text, add_special_tokens=False).ids
        if (sha(text.encode()) != row["sha256"] or len(ids) != row["tokens"]
                or len(text.encode()) != row["bytes"] or tk.decode(ids, skip_special_tokens=False) != text):
            raise ValueError(f"Passage {row['doc']} failed byte/token validation")
        row["ids"] = ids
    return rows


def shuffled_tokens(ids, doc):
    seed = int.from_bytes(hashlib.sha256(doc.encode()).digest()[:8], "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    shuffled = []
    for start in range(0, len(ids), CONTEXT):
        chunk = ids[start:start + CONTEXT]
        shuffled.extend(rng.permutation(chunk).tolist())
    if Counter(shuffled) != Counter(ids):
        raise ValueError("Shuffled control changed the token counts")
    return shuffled


def blocks(ids):
    for start in range(0, len(ids), CONTEXT):
        target = ids[start:start + CONTEXT]
        preceding = EOS if start == 0 else ids[start - 1]
        yield [preceding] + target[:-1], target


def aligned_match(mapping, ids):
    needle = np.asarray(ids, dtype="<u2").tobytes()
    offset = mapping.find(needle)
    while offset >= 0 and offset % 2:
        offset = mapping.find(needle, offset + 1)
    return offset >= 0


def controls(rows, data, inputs):
    stream = data / "train.bin"
    verify_file(stream, inputs["train_sha256"])
    ids = np.memmap(stream, dtype="<u2", mode="r")
    if ids.size != 800_000_000:
        raise ValueError("Unexpected training-stream length")
    counts = np.bincount(ids[::100].astype(np.int64), minlength=32768)
    if len(counts) != 32768:
        raise ValueError("Out-of-vocabulary training token")
    bits = -np.log2((counts + 1) / (counts.sum() + len(counts)))
    results = {}
    with stream.open("rb") as raw, mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
        for row in rows:
            target = row["ids"]
            width = min(32, len(target))
            last = len(target) - width
            matches = [aligned_match(mapping, target[n:n + width]) for n in (0, last // 2, last)]
            results[row["doc"]] = dict(unigram_bits=float(bits[target].sum()),
                                       exact_match=aligned_match(mapping, target), anchors=matches)
    return dict(training_samples=int(counts.sum()), rows=results)


def load_model(arm, runs, inputs):
    import torch
    from archived.gen3 import Gen3Cfg, Gen3LM
    from archived.gen5 import Gen5Cfg, Gen5LM

    spec = inputs["arms"][arm]
    path = runs / spec["run"] / "final.pt"
    verify_file(path, spec["sha256"])
    archive = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    cfg_type, lm_type = (Gen5Cfg, Gen5LM) if arm == "M2" else (Gen3Cfg, Gen3LM)
    with torch.device("meta"):
        model = lm_type(cfg_type(**archive["cfg"]))
    model.load_state_dict(archive["model"], strict=True, assign=True)
    if (model.param_report()["total"] != spec["params"] or model.c.vocab != 32768
            or model.c.seq < CONTEXT or any(p.device.type != "cpu" or p.dtype != torch.float32
                                           for p in model.parameters())):
        raise ValueError("Checkpoint architecture or dtype mismatch")
    return model.eval()


def loss_bits(model, ids):
    import torch
    from torch.nn import functional as F

    total = 0.0
    for inputs, target in blocks(ids):
        x = torch.tensor([inputs])
        y = torch.tensor(target)
        logits = model(x)[0]
        losses = F.cross_entropy(logits.double(), y, reduction="none")
        total += losses.double().sum().item() / math.log(2)
    if not math.isfinite(total):
        raise ValueError("Nonfinite model loss")
    return total


def validate_scoring(model, ids):
    import torch
    from torch.nn import functional as F

    x = torch.tensor([[EOS] + ids[:69]])
    y = torch.tensor([ids[:70]])
    full = model(x)
    # Cross the chunked mixers' internal boundary and check the vocabulary, not just the label.
    prefix_error = (model(x[:, :35]) - full[:, :35]).abs().max().item()
    explicit = F.cross_entropy(full[0].double(), y[0], reduction="none")
    native = model(x, y).item()
    native_error = abs(native - F.cross_entropy(full[0], y[0]).item())
    precision_delta = abs(native - explicit.mean().item())
    single = torch.logsumexp(full[0, 17].double(), dim=0) - full[0, 17, y[0, 17]].double()
    single_error = abs(single.item() - explicit[17].item())
    if not (prefix_error < 0.002 and native_error < 1e-5 and precision_delta < 1e-4 and single_error < 1e-10):
        raise ValueError(f"Causality or loss mismatch: {prefix_error}, {native_error}, {single_error}")
    return dict(prefix_logit_max_abs=prefix_error, native_mean_loss_abs=native_error,
                native_fp32_vs_fp64_mean_abs=precision_delta, single_token_loss_abs=single_error)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("control", "S2", "M0", "M2"), required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--passages", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    inputs = json.loads((HERE / "inputs.json").read_text())
    rows = read_passages(args.passages, args.data, inputs)
    if args.mode == "control":
        output = controls(rows, args.data, inputs)
    else:
        if args.runs is None:
            ap.error("--runs is required for checkpoint modes")
        import torch
        torch.set_num_threads(args.threads)
        with torch.inference_mode():
            model = load_model(args.mode, args.runs, inputs)
            sanity = validate_scoring(model, rows[0]["ids"])
            scores = {}
            for row in rows:
                ids = row["ids"]
                scores[row["doc"]] = dict(ordered_bits=loss_bits(model, ids),
                                           shuffled_bits=loss_bits(model, shuffled_tokens(ids, row["doc"])))
                print(args.mode, row["doc"], flush=True)
        output = dict(arm=args.mode, checkpoint_sha256=inputs["arms"][args.mode]["sha256"],
                      parameters=inputs["arms"][args.mode]["params"], dtype="float32", device="cpu",
                      torch=str(torch.__version__), checks=sanity, rows=scores)
    write_json(args.output, output)


if __name__ == "__main__":
    main()
