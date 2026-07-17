# Stage C training data — v2 recipe + 25% copy-curriculum slice.
# In the copy slice, CC/MED/ALG values are pronounceable gibberish sampled fresh per
# example: memorization is impossible, copying from context is the only strategy.
# EVAL SET NOT REGENERATED — scribe_eval.json reused byte-identical.
import random
import numpy as np
from tokenizers import Tokenizer

random.seed(23)

exec(open("build_scribe_data_v2.py").read().split("N_TRAIN = 12000")[0].split("import numpy")[1].split("\n", 1)[1])  # noqa — reuse v2 vocab/template definitions

ONSET = ["fl", "tr", "gr", "sp", "bl", "cr", "dr", "pl", "sn", "th", "qu", "br"]
MID = ["um", "ar", "ol", "ex", "in", "ov", "ub", "am", "er", "ix"]
CODA = ["bar", "ton", "mex", "dal", "rik", "pon", "vus", "lin", "sor", "nak"]

def gibber(words=None):
    words = words or random.choice([1, 2])
    return " ".join(random.choice(ONSET) + random.choice(MID) + random.choice(CODA)
                    for _ in range(words))

def sample_tuple_copy():
    # gibberish values in the SAME tuple structure; dialog form == canonical form
    g = gibber()
    cc = (g, g)
    unit = random.choice(["days", "weeks"])
    n = random.randint(2, 14) if unit == "days" else random.randint(1, 6)
    return {"cc": cc, "n": n, "unit": unit, "sev": random.choice(SEV),
            "med": gibber() if random.random() < 0.6 else None,
            "alg": gibber() if random.random() < 0.5 else None}

N_TRAIN, COPY_FRAC = 12000, 0.25
convos = []
n_copy = 0
for i in range(N_TRAIN):
    if random.random() < COPY_FRAC:
        t = sample_tuple_copy(); n_copy += 1
    else:
        t = sample_tuple()
    convos.append([{"role": "user", "content": render_dialogue(t) + "\nSummarize the visit."},
                   {"role": "assistant", "content": summary_of(t)}])

tok = Tokenizer.from_file("tokenizer.json")
ims_id, ime_id = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
S = 512

def render_ids(convo):
    ids, mask = [], []
    for m in convo:
        head = tok.encode(f"{m['role']}\n", add_special_tokens=False).ids
        body = tok.encode(m["content"], add_special_tokens=False).ids
        seg = [ims_id] + head + body + [ime_id]
        tf = 1 if m["role"] == "assistant" else 0
        for j, tkn in enumerate(seg):
            ids.append(tkn); mask.append(1 if (tf and j >= 1 + len(head)) else 0)
    return ids, mask

X, M, dropped = [], [], 0
for c in convos:
    ids, mask = render_ids(c)
    if len(ids) > S: dropped += 1; continue
    X.append(ids + [0] * (S - len(ids))); M.append(mask + [0] * (S - len(mask)))

np.save("scribe_x.npy", np.array(X, dtype=np.uint16))
np.save("scribe_mask.npy", np.array(M, dtype=np.uint8))
print(f"v3 train examples {len(X)} (dropped {dropped}); copy-curriculum slice {n_copy} "
      f"({n_copy/N_TRAIN:.0%}); eval set NOT regenerated", flush=True)
