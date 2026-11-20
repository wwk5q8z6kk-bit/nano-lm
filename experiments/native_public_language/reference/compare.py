"""Score byte-identical chunks under a public reference and the three recovered models."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))

from corpus import write_json
from evaluate import EOS, load_model, read_passages, verify_file


def split_bytes(data, limit=640):
    start = 0
    while start < len(data):
        end = min(start + limit, len(data))
        if end < len(data):
            space = data.rfind(b" ", start + 1, end + 1)
            if space > start:
                end = space
            else:
                while data[end] & 0xC0 == 0x80:
                    end -= 1
        if end <= start:
            raise ValueError("A character exceeds the chunk byte budget")
        yield start, end, data[start:end].decode("utf-8")
        start = end


def prepare(passages, data, reference):
    native = json.loads((STUDY / "inputs.json").read_text())
    rows = read_passages(passages, data, native)
    # read_passages verifies the original manifest and canonical encoding.
    texts = {d["doc"]: d["text"] for d in json.loads(passages.read_text())}
    tok = Tokenizer.from_file(str(data / "tokenizer_32k.json"))
    pythia = Tokenizer.from_file(str(reference / "tokenizer.json"))
    if pythia.token_to_id("<|endoftext|>") != 0 or tok.token_to_id("<|eos|>") != EOS:
        raise ValueError("Unexpected boundary token")
    all_chunks = []
    for row in rows:
        doc = row["doc"]
        raw = texts[doc].encode()
        chunks = []
        for start, end, text in split_bytes(raw):
            encodings = []
            for tokenizer in (tok, pythia):
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if not 1 <= len(ids) <= 256 or tokenizer.decode(ids, skip_special_tokens=False) != text:
                    raise ValueError(f"Chunk {doc} {start} cannot be scored as specified")
                encodings.append(ids)
            chunks.append((dict(doc=doc, start=start, end=end, sha256=hashlib.sha256(raw[start:end]).hexdigest(),
                                native_tokens=len(encodings[0]), reference_tokens=len(encodings[1])),
                           *encodings))
        if sum(c[0]["end"] - c[0]["start"] for c in chunks) != row["bytes"]:
            raise ValueError(f"Lost bytes in {doc}")
        all_chunks.extend(chunks)
    return all_chunks


def load_reference(directory, arm):
    import torch
    from safetensors import safe_open
    from transformers import GPTNeoXForCausalLM

    model, info = GPTNeoXForCausalLM.from_pretrained(
        directory, local_files_only=True, dtype=torch.float32, use_safetensors=True,
        attn_implementation="eager", output_loading_info=True,
    )
    if any(info.values()):
        raise ValueError(f"Reference model load mismatch: {info}")
    parameters = dict(model.named_parameters())
    buffers = {f"gpt_neox.layers.{i}.attention.{suffix}" for i in range(24)
               for suffix in ("bias", "masked_bias", "rotary_emb.inv_freq")}
    if arm == "Pythia-step256":
        buffers = set()
    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as archive:
        if set(archive.keys()) != set(parameters) | buffers or sum(p.numel() for p in parameters.values()) != 405334016:
            raise ValueError("Unexpected reference model tensors")
        for key, parameter in parameters.items():
            if parameter.dtype != torch.float32 or not torch.equal(parameter, archive.get_tensor(key).float()):
                raise ValueError(f"Reference tensor was not recovered exactly: {key}")
    return model.eval()


def measure(chunks, arm, predict):
    import torch
    from torch.nn import functional as F

    eos = 0 if arm.startswith("Pythia-") else EOS
    column = 2 if arm.startswith("Pythia-") else 1
    chunks = sorted(chunks, key=lambda item: -len(item[column]))
    rows = []
    for i in range(0, len(chunks), 3):
        batch = chunks[i:i + 3]
        width = max(len(c[column]) for c in batch)
        x = torch.tensor([[eos] + c[column][:-1] + [eos] * (width - len(c[column])) for c in batch])
        logits = predict(x)
        for j, c in enumerate(batch):
            target = torch.tensor(c[column])
            vocab = logits[j, :len(target)].double()
            bits = F.cross_entropy(vocab, target, reduction="sum").item() / math.log(2)
            if not math.isfinite(bits):
                raise ValueError("Nonfinite source loss")
            rows.append(dict(doc=c[0]["doc"], start=c[0]["start"], bits=bits))
        print(arm, min(i + 3, len(chunks)), len(chunks), flush=True)
    return sorted(rows, key=lambda row: (row["doc"], row["start"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "reference", "passages", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--mode", choices=("prepare", "S2", "M0", "M2", "Pythia-final", "Pythia-step256"), required=True)
    args = ap.parse_args()
    reference_inputs = json.loads((HERE / "inputs.json").read_text())
    reference_arm = args.mode if args.mode.startswith("Pythia-") else "Pythia-final"
    reference_inputs = reference_inputs[reference_arm]
    for name in ("tokenizer.json", "config.json"):
        verify_file(args.reference / name, reference_inputs["files"][name])
    chunks = prepare(args.passages, args.data, args.reference)
    manifest = [c[0] for c in chunks]
    lock = HERE / "chunks.json"
    if args.mode == "prepare":
        if lock.exists() and manifest != json.loads(lock.read_text()):
            raise ValueError("Prepared chunks differ from locked manifest")
        write_json(args.output, manifest)
        return
    if manifest != json.loads(lock.read_text()):
        raise ValueError("Prepared chunks differ from locked manifest")
    import torch

    torch.set_num_threads(3)
    with torch.inference_mode():
        if args.mode.startswith("Pythia-"):
            verify_file(args.reference / "model.safetensors", reference_inputs["files"]["model.safetensors"])
            model = load_reference(args.reference, args.mode)
            predict = lambda x: model(x, use_cache=False).logits
            weights = reference_inputs["files"]["model.safetensors"]
        else:
            if args.runs is None:
                ap.error("--runs is required for native checkpoint modes")
            native = json.loads((STUDY / "inputs.json").read_text())
            predict = load_model(args.mode, args.runs, native)
            weights = native["arms"][args.mode]["sha256"]
        rows = measure(chunks, args.mode, predict)
    write_json(args.output, dict(arm=args.mode, checkpoint_sha256=weights, chunks=rows))


if __name__ == "__main__":
    main()
