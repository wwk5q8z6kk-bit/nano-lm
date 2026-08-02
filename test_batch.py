import json, os, random, re, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer

SEED = 20260717
MAX_LEN = 448
REPO = os.path.dirname(os.path.abspath(__file__))
MODEL = "EleutherAI/pythia-160m"

v2_src = open(os.path.join(REPO, "scribe", "build_scribe_data_v2.py")).read()
marker = 'tok = Tokenizer.from_file'
prefix = v2_src.split(marker)[0].replace("from tokenizers import Tokenizer", "")
ns = {}
exec(compile(prefix, "build_scribe_data_v2.py[prefix]", "exec"), ns)
convos = ns["convos"]
print(f"Loaded {len(convos)} convos")

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
EOS = tok.eos_token_id

def encode_example(convo):
    prompt = convo[0]["content"] + "\n"
    target = convo[1]["content"]
    p_ids = tok.encode(prompt)
    t_ids = tok.encode(target) + [EOS]
    return p_ids + t_ids, [-100] * len(p_ids) + t_ids

t0 = time.time()
examples_loop, dropped_loop = [], 0
for c in convos:
    ids, labels = encode_example(c)
    if len(ids) > MAX_LEN:
        dropped_loop += 1; continue
    examples_loop.append((ids, labels))
t_loop = time.time() - t0
print(f"Loop: {t_loop:.4f}s")

t0 = time.time()
prompts = [c[0]["content"] + "\n" for c in convos]
targets = [c[1]["content"] for c in convos]

p_ids_list = tok(prompts, add_special_tokens=False)["input_ids"]
t_ids_list = tok(targets, add_special_tokens=False)["input_ids"]

examples_batch, dropped_batch = [], 0
for p_ids, t_ids in zip(p_ids_list, t_ids_list):
    t_ids = t_ids + [EOS]
    ids = p_ids + t_ids
    if len(ids) > MAX_LEN:
        dropped_batch += 1; continue
    labels = [-100] * len(p_ids) + t_ids
    examples_batch.append((ids, labels))
t_batch = time.time() - t0
print(f"Batch: {t_batch:.4f}s")

print(f"Dropped match: {dropped_loop == dropped_batch}")
print(f"Examples length match: {len(examples_loop) == len(examples_batch)}")
if len(examples_loop) > 0 and len(examples_batch) > 0:
    print(f"First element match: {examples_loop[0] == examples_batch[0]}")
    print(f"Last element match: {examples_loop[-1] == examples_batch[-1]}")
