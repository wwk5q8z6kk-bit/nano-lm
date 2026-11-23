"""Score a public Hugging Face causal LM through the same prompt texts as eval_score.py, to calibrate the harness.

Encoding follows lm-evaluation-harness: context and continuation are tokenized together and split at the length of
the tokenized context; no prefix token is added for non-empty contexts. Scores comparable with eval_score.py except
for the tokenizer and the end-of-document prefix our own models get.

--control nocontext scores each continuation alone, after the model's end-of-text token as the only context, matching
eval_score.py's control (added 2026-11-23). Without it the computation is unchanged.

usage: eval_score_hf.py --model EleutherAI/pythia-160m --texts tasks-v1-text.json [--only a,b] [--limit N] [--device mps] [--control nocontext] [--out f]
"""
from __future__ import annotations
import argparse, json, math, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--texts", required=True); ap.add_argument("--only", default="")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default="cpu"); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--out", default="")
ap.add_argument("--control", default="", choices=("", "nocontext"))
ap.add_argument("--revision", default=None, help="model revision, e.g. a Pythia training checkpoint such as step4000")
a = ap.parse_args(); t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision); model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.float32).to(a.device).eval()
data = json.load(open(a.texts))
if a.only: data = {k: v for k, v in data.items() if k in a.only.split(",")}
if a.limit: data = {k: v[:a.limit] for k, v in data.items()}
enc = lambda s: tok(s, add_special_tokens=False).input_ids
ents = []
for task, rows in data.items():
    for e, r in enumerate(rows):
        pairs = [(c, r["cont"]) for c in r["ctxs"]] if task == "winogrande" else [(r["ctx"], k) for k in r["conts"]]
        for j, (ctx, cont) in enumerate(pairs):
            if a.control == "nocontext": whole = [tok.eos_token_id] + enc(cont); n = 1
            else: whole = enc(ctx + cont); n = len(enc(ctx))
            ents.append((task, e, j, whole, n, len(cont)))
ents.sort(key=lambda x: len(x[3])); S = {}
with torch.no_grad():
    for i in range(0, len(ents), a.batch):
        b = ents[i:i + a.batch]; L = max(len(x[3]) for x in b)
        X = torch.zeros((len(b), L), dtype=torch.long); M = torch.zeros((len(b), L), dtype=torch.long)
        for r, x in enumerate(b): X[r, :len(x[3])] = torch.tensor(x[3]); M[r, :len(x[3])] = 1
        lp = torch.log_softmax(model(input_ids=X.to(a.device), attention_mask=M.to(a.device)).logits.float(), -1).cpu()
        for r, x in enumerate(b):
            ids = torch.tensor(x[3]); s, e_ = x[4], len(x[3])
            tl = lp[r, s - 1:e_ - 1].gather(1, ids[s:e_, None])[:, 0]; greedy = (lp[r, s - 1:e_ - 1].argmax(-1) == ids[s:e_]).all()
            S[x[:3]] = (float(tl.sum()), bool(greedy), x[5])
res = {}
for task, rows in data.items():
    if task == "lambada":
        ll = np.array([S[(task, e, 0)][0] for e in range(len(rows))])
        res[task] = {"acc": round(float(np.mean([S[(task, e, 0)][1] for e in range(len(rows))])), 4), "ppl": round(float(np.exp(-ll.mean())), 2), "n": len(rows)}; continue
    acc, accn = [], []
    for e, r in enumerate(rows):
        k = 2 if task == "winogrande" else len(r["conts"]); sc = np.array([S[(task, e, j)][0] for j in range(k)])
        acc.append(int(np.argmax(sc)) == r["gold"])
        if task != "winogrande": accn.append(int(np.argmax(sc / np.array([S[(task, e, j)][2] for j in range(k)]))) == r["gold"])
    res[task] = {"acc": round(float(np.mean(acc)), 4), "n": len(rows)}
    if accn: res[task]["acc_norm"] = round(float(np.mean(accn)), 4)
out = {"model": a.model, "revision": a.revision, "control": a.control or "none", "tasks": res, "seconds": round(time.time() - t0)}; print(json.dumps(out))
if a.out: json.dump(out, open(a.out, "w"), indent=1)
