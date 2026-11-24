"""Score a checkpoint on the zero-shot benchmarks prepared by eval_prep.py.

Multiple choice: the summed log-probability of each continuation given its context; acc picks the largest, acc_norm
divides by the continuation's length in characters (lm-evaluation-harness convention). WinoGrande: the shared suffix
scored after each option-filled prefix. LAMBADA: the last word counts as correct only if every one of its tokens is
the model's greedy choice; perplexity is exp of minus the mean per-example log-likelihood of the word.
Every context starts with the end-of-document token (id 1), because in training every document began right after one.
Sequences longer than the model's 512-token window lose tokens from the left of the context.

Opt-in measurement flags (added 2026-11-23 after the methodology audit; defaults reproduce earlier results exactly):
  --no-eos     no end-of-document prefix (the condition public models are scored under)
  --f32        float32 forward pass and log-softmax instead of bfloat16
  --bare-len   length normalisation by the choice without its leading space (lm-eval's convention)
Every output records the sha256 of the checkpoint and of the task file.

usage: eval_score.py --size S --ckpt <npz> --tasks <tasks.json> [--batch 16] [--limit N] [--no-eos] [--f32] [--bare-len] [--out file.json]
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
import numpy as np
import jax, jax.numpy as jnp
from train_jax import SIZES, forward, init, load_params

T = 512
EOS = 1                     # every context starts with the end-of-document token, as every document did in training
CHANCE = {"hellaswag": 0.25, "arc_easy": 0.25, "arc_challenge": 0.25, "piqa": 0.5, "winogrande": 0.5, "openbookqa": 0.25, "lambada": 0.0}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def entries(task, rows, nocontext=False, eos=True):
    """One entry per scored sequence: (task, example, choice, ids, first target index, last target index + 1)."""
    out = []
    for e, r in enumerate(rows):
        pairs = [(c, r["cont"]) for c in r["ctxs"]] if task == "winogrande" else [(r["ctx"], k) for k in r["conts"]]
        for j, (ctx, cont) in enumerate(pairs):
            if nocontext: ctx = []                                # control: the continuation alone, after the document marker
            seq = (([EOS] if eos else []) + ctx + cont)[-(T + 1):]; n_ctx = len(seq) - len(cont)
            assert n_ctx >= 1, "continuation longer than the model's window"
            out.append((task, e, j, seq, n_ctx - 1, len(seq) - 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--only", default=""); ap.add_argument("--control", default="", choices=("", "random", "nocontext"))
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="", help="also write per-item choice scores (json) for flip analysis");
    ap.add_argument("--no-eos", action="store_true"); ap.add_argument("--f32", action="store_true"); ap.add_argument("--bare-len", action="store_true")
    a = ap.parse_args(); c = SIZES[a.size]; t0 = time.time()
    params = init(jax.random.PRNGKey(0), c) if a.control == "random" else load_params(a.ckpt); data = json.load(open(a.tasks))
    if a.only: data = {k: v for k, v in data.items() if k in a.only.split(",")}
    if a.limit: data = {k: v[:a.limit] for k, v in data.items()}

    @jax.jit
    def score(p, x, y):
        lp = jax.nn.log_softmax(forward(p, x, c, jnp.float32 if a.f32 else jnp.bfloat16).astype(jnp.float32), -1)
        return jnp.take_along_axis(lp, y[..., None], -1)[..., 0], jnp.argmax(lp, -1) == y

    ents = [x for k, v in data.items() for x in entries(k, v, a.control == "nocontext", eos=not a.no_eos)]
    ents.sort(key=lambda x: len(x[3]))
    S = {}                                                         # (task, example, choice) -> (sum log-prob, all greedy-correct)
    for i in range(0, len(ents), a.batch):
        b = ents[i:i + a.batch]; L = min(T, 64 * math.ceil((max(len(x[3]) for x in b) - 1) / 64))
        X = np.zeros((a.batch, L), np.int32); Y = np.zeros((a.batch, L), np.int32)
        for r, x in enumerate(b):
            s = np.asarray(x[3], np.int32); X[r, :len(s) - 1] = s[:-1]; Y[r, :len(s) - 1] = s[1:]
        lp, ok = score(params, X, Y); lp, ok = np.asarray(lp), np.asarray(ok)
        for r, x in enumerate(b):
            S[x[:3]] = (float(lp[r, x[4]:x[5]].sum()), bool(ok[r, x[4]:x[5]].all()))
    res = {}
    if a.dump:
        json.dump({f"{t}|{e}|{j}": v[0] for (t, e, j), v in S.items()}, open(a.dump, "w"))
    for task, rows in data.items():
        if task == "lambada":
            ll = np.array([S[(task, e, 0)][0] for e in range(len(rows))])
            res[task] = {"acc": round(float(np.mean([S[(task, e, 0)][1] for e in range(len(rows))])), 4), "ppl": round(float(np.exp(-ll.mean())), 2), "n": len(rows)}
            continue
        acc, accn = [], []
        for e, r in enumerate(rows):
            k = len(r["ctxs"]) if task == "winogrande" else len(r["conts"])
            sc = np.array([S[(task, e, j)][0] for j in range(k)]); acc.append(int(np.argmax(sc)) == r["gold"])
            if task != "winogrande":
                lens = np.array(r["cont_chars"]) - (1 if a.bare_len else 0)      # every choice text starts with one space
                accn.append(int(np.argmax(sc / np.maximum(lens, 1))) == r["gold"])
        res[task] = {"acc": round(float(np.mean(acc)), 4), "n": len(rows), "chance": CHANCE[task]}
        if accn: res[task]["acc_norm"] = round(float(np.mean(accn)), 4)
    out = {"size": a.size, "ckpt": a.ckpt, "control": a.control or "none", "tasks": res, "seconds": round(time.time() - t0),
           "eos_prefix": not a.no_eos, "logits": "float32" if a.f32 else "bfloat16", "norm_length": "bare" if a.bare_len else "with_space",
           "ckpt_sha256": sha256(a.ckpt) if a.control != "random" else None, "tasks_sha256": sha256(a.tasks)}
    print(json.dumps(out))
    if a.out: json.dump(out, open(a.out, "w"), indent=1)
