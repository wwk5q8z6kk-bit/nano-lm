"""How many benchmark examples share a 13-token window with our training corpus?

For each example of each task prepared by eval_prep.py, every 13-token window of (context + correct continuation)
is hashed; the training file is streamed and every one of its 13-token windows is checked against that set.
Reported per task: the share of examples with any shared window, and the share whose correct continuation itself
(when it is at least 13 tokens long, or by a window that ends inside it) is shared. A shared window is evidence of
overlap, not proof that the model saw the answer; a clean result does not rule out paraphrased contamination.

usage: eval_contam.py --tasks <tasks.json> --train <train.bin> [--chunk 50000000] [--out file.json]
"""
from __future__ import annotations
import argparse, json, time
import numpy as np

N = 13
P = np.uint64(1000003)


def windows(seq):
    """Hashes of every N-token window of seq (list of ints), as uint64."""
    a = np.asarray(seq, np.uint64)
    if len(a) < N: return np.zeros(0, np.uint64)
    h = np.zeros(len(a) - N + 1, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(N): h = h * P + a[j:len(a) - N + 1 + j] + np.uint64(1)
    return h


def gold_parts(task, r):
    if task == "winogrande":
        return r["ctxs"][r["gold"]], r["cont"]
    return r["ctx"], r["conts"][r["gold"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True); ap.add_argument("--train", required=True)
    ap.add_argument("--chunk", type=int, default=50_000_000); ap.add_argument("--out", default="")
    a = ap.parse_args(); t0 = time.time(); data = json.load(open(a.tasks))
    owner = {}                                                       # hash -> list of (task, example, window touches the answer)
    for task, rows in data.items():
        for e, r in enumerate(rows):
            ctx, cont = gold_parts(task, r); seq = ctx + cont; h = windows(seq)
            for i, x in enumerate(h.tolist()):
                owner.setdefault(x, []).append((task, e, i + N > len(ctx)))
    keys = np.array(sorted(owner), np.uint64)
    tr = np.memmap(a.train, dtype=np.uint16, mode="r"); hit = set()
    for s in range(0, len(tr) - N + 1, a.chunk):
        chunk = np.asarray(tr[s:min(len(tr), s + a.chunk + N - 1)]); h = windows(chunk)
        m = h[np.isin(h, keys)]
        hit.update(int(x) for x in np.unique(m))
    any_hit, ans_hit = {}, {}
    for x in hit:
        for task, e, touches in owner[x]:
            any_hit.setdefault(task, set()).add(e)
            if touches: ans_hit.setdefault(task, set()).add(e)
    res = {t: {"n": len(rows), "any_shared_window": round(len(any_hit.get(t, ())) / len(rows), 4),
               "window_touching_answer": round(len(ans_hit.get(t, ())) / len(rows), 4)} for t, rows in data.items()}
    out = {"train": a.train, "train_tokens": int(len(tr)), "window": N, "tasks": res, "seconds": round(time.time() - t0)}
    print(json.dumps(out))
    if a.out: json.dump(out, open(a.out, "w"), indent=1)
