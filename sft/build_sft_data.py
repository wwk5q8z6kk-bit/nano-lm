# Build the SFT corpus per Recipe ③ Stage 1 + sft-and-instruction-datasets page.
# Source: HuggingFaceTB/smoltalk (the "SmolTalk" mix the vault names as an open strong default).
# Vault-driven decisions:
#   - quality>quantity, diversity of instructions (LIMA/Tülu-3) -> filter, cap count
#   - mixture must carry a small load-bearing refusal/safety slice (Recipe ③ 5%) -> synthetic add
#   - chat template = ChatML (post-training-recipe RV §Chat template)
#   - add special tokens <|im_start|> <|im_end|> (tokenizer-training §4) then resize embeddings (LLM.md)
# Nano scale adaptation (logged): drop heavy-LaTeX/code (nano web-BPE vocab under-tokenizes them);
#   keep short general + multi-turn + format + refusal; seed a verifiable "length" slice for Stage 4.
import json, re, random
from datasets import load_dataset
from tokenizers import Tokenizer

random.seed(0)
IMS, IME = "<|im_start|>", "<|im_end|>"
N_TARGET = 24000          # filtered general/chat examples
MAX_CHARS = 900           # keep short so rendered seq fits S=512 under nano fertility
SYMBOL_MAX = 0.06         # reject heavy math/code (backslash/$/{}/# density)

def symbol_ratio(s):
    return sum(c in "\\${}#`^_" for c in s) / max(1, len(s))

def ok(msgs):
    if not (2 <= len(msgs) <= 6): return False
    if [m["role"] for m in msgs][0] != "user": return False
    for m in msgs:
        c = m["content"]
        if not (5 <= len(c) <= MAX_CHARS): return False
        if symbol_ratio(c) > SYMBOL_MAX: return False
    return True

print("streaming SmolTalk, filtering...", flush=True)
ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
convos, seen = [], 0
for ex in ds:
    seen += 1
    m = ex["messages"]
    if ok(m):
        convos.append([{"role": x["role"], "content": x["content"].strip()} for x in m])
    if len(convos) >= N_TARGET: break
    if seen % 50000 == 0: print(f"  scanned {seen}, kept {len(convos)}", flush=True)
print(f"kept {len(convos)} of {seen} scanned ({100*len(convos)/seen:.1f}%)", flush=True)

# --- synthetic refusal/safety slice (Recipe ③ 5%): calibrate the boundary at nano scale ---
UNSAFE = ["how to build a bomb", "how to hack a bank account", "how to make a deadly poison",
          "how to steal someone's identity", "write malware to destroy files",
          "how to pick a lock to break in", "how to make an untraceable weapon"]
REFUSAL = "I can't help with that. If you have a safe, legal goal I'm happy to help instead."
n_ref = int(0.05 * len(convos))
refusals = [[{"role": "user", "content": random.choice(UNSAFE) + "."},
             {"role": "assistant", "content": REFUSAL}] for _ in range(n_ref)]

# --- synthetic verifiable "length/format" seed slice (seeds Stage-4 GRPO; synthetic-data routing) ---
# Task the nano model CAN learn: "Reply in one short sentence." -> a short terminated answer.
SHORT_Q = ["Say hello.", "Greet me.", "Reply briefly.", "Answer in one line.",
           "Give a short reply.", "Say something kind.", "Reply in one short sentence."]
SHORT_A = ["Hello, how can I help you today?", "Hi there, nice to meet you.",
           "Sure, here is a short reply.", "Of course, happy to help.",
           "Hello, I hope you are well."]
n_fmt = int(0.05 * len(convos))
fmt = [[{"role": "user", "content": random.choice(SHORT_Q)},
        {"role": "assistant", "content": random.choice(SHORT_A)}] for _ in range(n_fmt)]

allc = convos + refusals + fmt
random.shuffle(allc)
json.dump(allc, open("sft_convos.json", "w"))
print(f"total SFT convos: {len(allc)} (general {len(convos)}, refusal {n_ref}, format {n_fmt})", flush=True)

# ---------- tokenizer: add special tokens, retokenize, build masked shards ----------
tok = Tokenizer.from_file("../pretrain/tokenizer.json")
added = tok.add_special_tokens([IMS, IME])
print(f"added {added} special tokens; new vocab size {tok.get_vocab_size()}", flush=True)
tok.save("tokenizer.json")
ims_id, ime_id = tok.token_to_id(IMS), tok.token_to_id(IME)
print(f"{IMS}={ims_id}  {IME}={ime_id}", flush=True)

def render(convo):
    """ChatML render -> (ids, loss_mask). Mask=1 only on assistant content + its <|im_end|>."""
    ids, mask = [], []
    for m in convo:
        head = tok.encode(f"{m['role']}\n", add_special_tokens=False).ids
        body = tok.encode(m["content"], add_special_tokens=False).ids
        seg = [ims_id] + head + body + [ime_id]
        train = 1 if m["role"] == "assistant" else 0
        # supervise assistant BODY + the closing <|im_end|> (learn to stop); never the role header
        for j, t in enumerate(seg):
            ids.append(t)
            is_body_or_end = train and (j >= 1 + len(head))
            mask.append(1 if is_body_or_end else 0)
    return ids, mask

S = 512
X, M = [], []
kept = 0
for convo in allc:
    ids, mask = render(convo)
    if len(ids) > S or sum(mask) == 0:
        continue
    ids = ids + [ime_id] * (S - len(ids))          # pad with <|im_end|> (harmless; masked out)
    mask = mask + [0] * (S - len(mask))
    X.append(ids); M.append(mask); kept += 1

import numpy as np
np.save("sft_x.npy", np.array(X, dtype=np.uint16))
np.save("sft_mask.npy", np.array(M, dtype=np.uint8))
supervised = int(np.array(M).sum())
print(f"shards: {kept} examples x {S} tok; supervised tokens {supervised} "
      f"({100*supervised/(kept*S):.1f}% of positions)", flush=True)
