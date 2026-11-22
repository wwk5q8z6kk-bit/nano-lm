"""Nano LM dense baseline in JAX, for Cloud TPU.

The architecture mirrors research/frontier/m21_dense_code/dense.py (the PyTorch reference used by every
earlier language-model run): pre-norm RMSNorm, rotary positions, RMS-normalised queries and keys,
grouped key/value heads, gated SiLU feed-forward, tied input/output embeddings, same initialisation
scales. `parity_check.py` loads PyTorch weights into this code and compares logits.

Training recipe mirrors lm_compare.py and the Kaggle baseline: random crops of length 512, AdamW
(0.9, 0.95), weight decay 0.1, gradient clipping 1.0, linear warm-up then cosine decay to 10% of peak,
one pass over the corpus in token count. Evaluation uses the identical held-out crops as the PyTorch
harness (numpy default_rng(0), 20 batches of 16 per domain) and reports bits per byte.

Compute is bfloat16 with float32 normalisation statistics, attention softmax and logits.
"""
from __future__ import annotations
import argparse, copy, json, math, os, queue, subprocess, threading, time
import numpy as np
import jax, jax.numpy as jnp, optax

VOCAB = 32768
SIZES = {
    "T": dict(d=64, layers=2, ffn=176, heads=4, kv=2),        # smoke tests only
    "S": dict(d=256, layers=8, ffn=704, heads=4, kv=2),       # ~14M, the local pilot model
    "M": dict(d=512, layers=8, ffn=1408, heads=8, kv=4),      # ~42M, the Kaggle baseline model
    "L": dict(d=768, layers=12, ffn=2048, heads=12, kv=4),    # ~101M
    "XL": dict(d=1024, layers=16, ffn=2752, heads=16, kv=4),  # ~211M
}


def init(key, c):
    d, L, ffn, H, KV = c["d"], c["layers"], c["ffn"], c["heads"], c["kv"]
    hd = d // H
    ks = iter(jax.random.split(key, 1 + 6 * L))
    n = lambda shape, scale=1.0: 0.02 * scale * jax.random.normal(next(ks), shape, jnp.float32)
    res = 1.0 / math.sqrt(2 * L)
    one = lambda k: jnp.ones((k,), jnp.float32)
    blocks = [dict(n1=one(d), n2=one(d), qn=one(hd), kn=one(hd), q=n((d, H * hd)), k=n((d, KV * hd)), v=n((d, KV * hd)),
                   o=n((H * hd, d), res), gu=n((d, 2 * ffn)), dn=n((ffn, d), res)) for _ in range(L)]
    return dict(emb=n((VOCAB, d)), nf=one(d), blocks=blocks)


def rms(x, w, eps=1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), -1, keepdims=True) + eps) * w.astype(jnp.float32)).astype(x.dtype)


def rope_tables(T, hd, theta=10000.0):
    inv = 1.0 / theta ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd)
    f = jnp.outer(jnp.arange(T, dtype=jnp.float32), inv)
    return jnp.cos(f), jnp.sin(f)


def apply_rope(x, cos, sin):                                   # x [..., T, hd]; tables [T, hd/2]
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = cos.astype(x.dtype), sin.astype(x.dtype)
    return jnp.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).reshape(x.shape)


def forward(params, idx, c, dt=jnp.bfloat16):
    """idx [B, T] int -> logits [B, T, VOCAB] float32."""
    H, KV = c["heads"], c["kv"]
    p = jax.tree_util.tree_map(lambda a: a.astype(dt), params)
    B, T = idx.shape
    x = p["emb"][idx]
    d = x.shape[-1]; hd = d // H; rep = H // KV
    cos, sin = rope_tables(T, hd)
    mask = jnp.tril(jnp.ones((T, T), bool))
    neg = jnp.finfo(jnp.float32).min
    for b in p["blocks"]:
        h = rms(x, b["n1"])
        q = rms((h @ b["q"]).reshape(B, T, KV, rep, hd), b["qn"]).transpose(0, 2, 3, 1, 4)      # B KV rep T hd
        k = rms((h @ b["k"]).reshape(B, T, KV, hd), b["kn"]).transpose(0, 2, 1, 3)              # B KV T hd
        v = (h @ b["v"]).reshape(B, T, KV, hd).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        s = jnp.einsum("bgrqd,bgkd->bgrqk", q, k).astype(jnp.float32) * (hd ** -0.5)
        a = jax.nn.softmax(jnp.where(mask, s, neg), -1).astype(dt)
        y = jnp.einsum("bgrqk,bgkd->bgrqd", a, v).transpose(0, 3, 1, 2, 4).reshape(B, T, H * hd)
        x = x + y @ b["o"]
        g, u = jnp.split(rms(x, b["n2"]) @ b["gu"], 2, -1)
        x = x + (jax.nn.silu(g) * u) @ b["dn"]
    return (rms(x, p["nf"]) @ p["emb"].T).astype(jnp.float32)


def count(params):
    return int(sum(a.size for a in jax.tree_util.tree_leaves(params)))


def loss_fn(params, tok, c, dt):
    lg = forward(params, tok[:, :-1], c, dt)
    return optax.softmax_cross_entropy_with_integer_labels(lg, tok[:, 1:]).mean()


def make_chunk(c, opt, B, T, micro, K, dt):
    def one(carry, _):
        p, st, key, data = carry
        key, sub = jax.random.split(key)
        ix = jax.random.randint(sub, (B,), 0, data.shape[0] - T - 1)
        tok = data[ix[:, None] + jnp.arange(T + 1)[None]].astype(jnp.int32)
        if micro == 1:
            l, g = jax.value_and_grad(loss_fn)(p, tok, c, dt)
        else:                                                   # same update from `micro` smaller passes
            def acc(a, tk):
                l, g = jax.value_and_grad(loss_fn)(p, tk, c, dt)
                return (a[0] + l / micro, jax.tree_util.tree_map(lambda x, y: x + y / micro, a[1], g)), None
            (l, g), _ = jax.lax.scan(acc, (jnp.zeros((), jnp.float32), jax.tree_util.tree_map(jnp.zeros_like, p)),
                                     tok.reshape(micro, B // micro, T + 1))
        u, st = opt.update(g, st, p)
        return (optax.apply_updates(p, u), st, key, data), l

    @jax.jit
    def chunk(p, st, key, data):
        (p, st, key, _), ls = jax.lax.scan(one, (p, st, key, data), None, length=K)
        return p, st, key, ls
    return chunk


def evaluate(ev, params, vals, manifest, T):
    out = {}
    for dom, d in vals.items():
        rng = np.random.default_rng(0); nll = 0.0
        for _ in range(20):
            ix = rng.integers(0, len(d) - T - 1, 16)
            nll += float(ev(params, np.stack([d[i:i + T + 1] for i in ix]).astype(np.int32)))
        out[dom] = nll / 20 / math.log(2) / manifest["domains"][dom]["val"]["bytes_per_token"]
    out["mix"] = sum(manifest["mix"][k] * out[k] for k in manifest["mix"])
    return {k: round(v, 4) for k, v in out.items()}


def push(path, gcs):
    if gcs:
        subprocess.run(["gcloud", "storage", "cp", path, gcs.rstrip("/") + "/" + os.path.basename(path)], capture_output=True)


def pull(path, gcs):
    if gcs and not os.path.exists(path):
        subprocess.run(["gcloud", "storage", "cp", gcs.rstrip("/") + "/" + os.path.basename(path), path], capture_output=True)
    return os.path.exists(path)


def save_tree(path, tree, **extra):
    leaves = jax.tree_util.tree_leaves(tree)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{f"leaf_{i}": np.asarray(x) for i, x in enumerate(leaves)}, **extra); os.replace(tmp, path)


def load_tree(path, like):
    z = np.load(path, allow_pickle=False); leaves, treedef = jax.tree_util.tree_flatten(like)
    return jax.tree_util.tree_unflatten(treedef, [jnp.asarray(z[f"leaf_{i}"], dtype=x.dtype) for i, x in enumerate(leaves)]), z


def run(a):
    """One training run on one device. Resumable: state is saved at every evaluation and reloaded if present."""
    dev = jax.devices()[a.device]
    c = SIZES[a.size]; dt = jnp.float32 if a.f32 else jnp.bfloat16
    name = f"dense-{a.size}-lr{a.lr:g}-s{a.seed}" + (f"-{a.tag}" if a.tag else "")
    say = lambda d: print(json.dumps({"run": name, **d}), flush=True)
    manifest = json.load(open(f"{a.data}/manifest.json"))
    train = np.fromfile(f"{a.data}/train.bin", dtype=np.uint16, count=a.max_train_tokens or -1)
    vals = {dom: np.fromfile(f"{a.data}/val_{dom}.bin", dtype=np.uint16) for dom in ("web", "textbook", "math", "code")}
    budget = int(a.train_tokens) or len(train)
    steps = budget // (a.bs * a.seq) // a.chunk * a.chunk
    sched = lambda n: a.lr * jnp.minimum(1.0, (n + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(1.0, (n + 1) / steps))))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1))
    os.makedirs(a.out, exist_ok=True); res_path = f"{a.out}/{name}.json"; ck_path = f"{a.out}/{name}.state.npz"
    with jax.default_device(dev):
        params = init(jax.random.PRNGKey(a.seed), c); st = opt.init(params); key = jax.random.PRNGKey(a.seed + 1)
        done, curve, t_train, compile_s, prior_hours = 0, [], 0.0, None, 0.0
        if a.resume and pull(ck_path, a.gcs) and pull(res_path, a.gcs):
            (params, st, key), z = load_tree(ck_path, (params, st, key)); done = int(z["done"])
            old = json.load(open(res_path)); curve = old["curve"]; prior_hours = curve[-1]["hours"] if curve else 0.0
            say({"resumed_at_step": done})
        params, st, key, data = (jax.device_put(x, dev) for x in (params, st, key, jnp.asarray(train)))
        chunk = make_chunk(c, opt, a.bs, a.seq, a.micro, a.chunk, dt)
        ev = jax.jit(lambda p, tok: loss_fn(p, tok, c, dt))
        head = {"name": name, "params": count(params), "cfg": c, "lr": a.lr, "warmup": a.warmup, "bs": a.bs, "seq": a.seq, "micro": a.micro, "seed": a.seed,
                "steps": steps, "train_tokens": steps * a.bs * a.seq, "corpus_tokens": int(len(train)), "dtype": "float32" if a.f32 else "bfloat16",
                "jax": jax.__version__, "device": str(dev), "n_devices": len(jax.devices())}
        say(head)
        t0 = time.time(); start = done; timed = 0
        while done < steps:
            t = time.time(); params, st, key, ls = chunk(params, st, key, data); ls = np.asarray(ls); dtc = time.time() - t
            if compile_s is None: compile_s = round(dtc, 1)      # the first chunk includes compilation; excluded from throughput
            else: t_train += dtc; timed += a.chunk
            done += a.chunk
            if not np.isfinite(ls).all(): say({"error": "non-finite loss", "step": done}); return 2
            tps = round(timed * a.bs * a.seq / max(t_train, 1e-9))
            if done % 500 == 0: say({"step": done, "loss": round(float(ls.mean()), 4), "tok_s": tps})
            if done % a.eval_every == 0 or done == steps:
                row = {"step": done, "tokens": done * a.bs * a.seq, "hours": round(prior_hours + (time.time() - t0) / 3600, 4), "train_loss": round(float(ls.mean()), 4),
                       "tokens_per_second": tps, **evaluate(ev, params, vals, manifest, a.seq)}
                try: row["device_gb_in_use"] = round(dev.memory_stats()["bytes_in_use"] / 2 ** 30, 2)
                except Exception: pass
                curve.append(row); say(row)
                json.dump({**head, "compile_seconds": compile_s, "curve": curve, "finished": done == steps}, open(res_path, "w")); push(res_path, a.gcs)
                if a.resume and done < steps: save_tree(ck_path, (params, st, key), done=done); push(ck_path, a.gcs)
        if a.save:
            flat = {"emb": np.asarray(params["emb"]), "nf": np.asarray(params["nf"])}
            for i, b in enumerate(params["blocks"]): flat.update({f"b{i}.{k}": np.asarray(v) for k, v in b.items()})
            np.savez(f"{a.out}/{name}.npz", **flat); push(f"{a.out}/{name}.npz", a.gcs)
        say({"finished": True, "wall_hours": round((time.time() - t0) / 3600, 3), "steps_this_session": done - start})
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S"); ap.add_argument("--data", required=True); ap.add_argument("--out", default=".")
    ap.add_argument("--gcs", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--seq", type=int, default=512); ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--chunk", type=int, default=50); ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--train-tokens", type=float, default=0, help="token budget; 0 = one pass over the corpus")
    ap.add_argument("--max-train-tokens", type=int, default=0, help="load only this many corpus tokens (smoke tests)")
    ap.add_argument("--f32", action="store_true"); ap.add_argument("--save", action="store_true"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--multi", default="", help='several runs, one per device at a time: "SIZE:LR:MICRO[:TAG[:TOKENS]];..."')
    a = ap.parse_args()
    if not a.multi: raise SystemExit(run(a))
    jobs = queue.Queue()
    for spec in a.multi.split(";"):
        f = spec.split(":"); b = copy.copy(a); b.size, b.lr, b.micro = f[0], float(f[1]), int(f[2])
        if len(f) > 3 and f[3]: b.tag = f[3]
        if len(f) > 4 and f[4]: b.train_tokens = float(f[4])
        jobs.put(b)
    codes = []
    def worker(k):
        while True:
            try: b = jobs.get_nowait()
            except queue.Empty: return
            b.device = k
            try: codes.append(run(b))
            except Exception as e:                                # one failed run must not stop the others
                print(json.dumps({"run_failed": f"{b.size}:{b.lr}:{b.micro}", "device": k, "error": str(e)[:400]}), flush=True); codes.append(1)
    th = [threading.Thread(target=worker, args=(k,)) for k in range(min(len(jax.devices()), jobs.qsize()))]
    for t in th: t.start()
    for t in th: t.join()
    print(json.dumps({"all_runs_done": True, "exit_codes": codes}), flush=True)
    raise SystemExit(0 if all(x == 0 for x in codes) else 1)
