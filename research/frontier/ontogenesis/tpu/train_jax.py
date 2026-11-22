"""Nano LM dense baseline in JAX, for Cloud TPU.

The architecture mirrors research/frontier/m21_dense_code/dense.py (the PyTorch reference used by every
earlier language-model run): pre-norm RMSNorm, rotary positions, RMS-normalised queries and keys,
grouped key/value heads, gated SiLU feed-forward, tied input/output embeddings, same initialisation
scales. `parity_check.py` loads PyTorch weights into this code and compares logits.

Training recipe mirrors lm_compare.py and the Kaggle baseline: uniform random crops of length 512, AdamW
(0.9, 0.95), weight decay 0.1, gradient clipping 1.0, linear warm-up then cosine decay to 10% of peak,
one pass over the corpus in token count. Evaluation uses the identical held-out crops as the PyTorch
harness (numpy default_rng(0), 20 batches of 16 per domain) and reports bits per byte.

Compute is bfloat16 with float32 normalisation statistics and attention softmax; the logits are computed
in bfloat16 and cast to float32 for the loss. Evaluation uses the same precision as training.

Second version (run names end in "-hs"). The first version drew crop positions on the device with
jax.random.randint, which is not uniform for ranges above 65,536 without 64-bit mode: on the 800M-token
corpus the first 37% of the file was drawn about 12% too often. Crops are now drawn on the host with numpy,
exactly as the PyTorch harness does, and the corpus stays in host memory, so it may be larger than a chip's
memory. Also new: uploads and downloads are checked and retried, a resumed run must match its recipe,
two jobs may not share a name, and a diverged run is recorded as such. The model code is unchanged.

Additions on 2026-11-22 (defaults reproduce the earlier behaviour exactly): --wd, --wd-on matrices (no decay on
one-dimensional gains), --lr-floor, --opt muon (Muon on the blocks' matrices with consistent-RMS scaling 0.2, so the
AdamW learning rate and decay carry over; AdamW (0.9, 0.95) on the embedding and gains); every result file records
the chip type and a hash of this file. --dp N trains one run on N chips (batch sharded, weights replicated);
--dp 1, the default, runs exactly the code path used before.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, queue, subprocess, threading, time
import numpy as np
import jax, jax.numpy as jnp, optax

VOCAB = 32768
CODE_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
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


def kd_loss(params, tparams, tok, c, tc, dt, alpha):
    """alpha * cross-entropy on the data + (1 - alpha) * KL(teacher || student). The teacher gets no gradient."""
    lg = forward(params, tok[:, :-1], c, dt)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, tok[:, 1:]).mean()
    tl = jax.lax.stop_gradient(forward(tparams, tok[:, :-1], tc, dt))
    tlp = jax.nn.log_softmax(tl, -1)
    kl = (jnp.exp(tlp) * (tlp - jax.nn.log_softmax(lg, -1))).sum(-1).mean()
    return alpha * ce + (1.0 - alpha) * kl


def make_chunk(c, opt, micro, dt, kd=None, strided=False):
    """One compiled call runs a whole chunk of steps. toks is [steps, batch, T+1], sampled on the host.
    kd = (teacher config, alpha) switches the objective to distillation; teacher parameters are passed at call time."""
    def chunk(p, st, toks, tp):
        lossf = (lambda q, tk: kd_loss(q, tp, tk, c, kd[0], dt, kd[1])) if kd else (lambda q, tk: loss_fn(q, tk, c, dt))

        def one(carry, tok):
            p, st = carry
            if micro == 1:
                l, g = jax.value_and_grad(lossf)(p, tok)
            else:                                               # same update from `micro` smaller passes
                def acc(a, tk):
                    l, g = jax.value_and_grad(lossf)(p, tk)
                    return (a[0] + l / micro, jax.tree_util.tree_map(lambda x, y: x + y / micro, a[1], g)), None
                (l, g), _ = jax.lax.scan(acc, (jnp.zeros((), jnp.float32), jax.tree_util.tree_map(jnp.zeros_like, p)),
                                         tok.reshape(tok.shape[0] // micro, micro, tok.shape[1]).swapaxes(0, 1) if strided   # keeps each chip's rows on that chip
                                         else tok.reshape(micro, tok.shape[0] // micro, tok.shape[1]))
            u, st = opt.update(g, st, p)
            return (optax.apply_updates(p, u), st), l

        (p, st), ls = jax.lax.scan(one, (p, st), toks)
        return p, st, ls
    return jax.jit(chunk)


def load_params(path):
    z = np.load(path); n = 1 + max(int(k[1:].split(".")[0]) for k in z.files if k.startswith("b"))
    return dict(emb=jnp.asarray(z["emb"]), nf=jnp.asarray(z["nf"]),
                blocks=[{k.split(".", 1)[1]: jnp.asarray(z[k]) for k in z.files if k.startswith(f"b{i}.")} for i in range(n)])


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


def _gcloud(args, tries=3):
    """Run a gcloud storage command with retries. Returns (ok, stderr)."""
    err = ""
    for k in range(tries):
        r = subprocess.run(["gcloud", "storage"] + args, capture_output=True, text=True)
        if r.returncode == 0: return True, ""
        err = (r.stderr or "")[-400:]
        if "matched no objects" in err or "No URLs matched" in err or "not found" in err.lower(): return False, "absent"
        time.sleep(2 * (k + 1))
    return False, err


def push(path, gcs):
    """Upload one file. Returns True when it is safely in the bucket (or no bucket is in use)."""
    if not gcs: return True
    ok, err = _gcloud(["cp", path, gcs.rstrip("/") + "/" + os.path.basename(path)])
    if not ok: print(json.dumps({"upload_failed": os.path.basename(path), "stderr": err}), flush=True)
    return ok


def pull(path, gcs):
    """Fetch one file if it is not on disk. True if present afterwards. A failed download is an error, not 'absent'."""
    if os.path.exists(path): return True
    if not gcs: return False
    ok, err = _gcloud(["cp", gcs.rstrip("/") + "/" + os.path.basename(path), path])
    if not ok and err != "absent": raise SystemExit(f"download of {os.path.basename(path)} failed, refusing to start from scratch: {err}")
    return ok and os.path.exists(path)


def save_tree(path, tree, **extra):
    leaves = jax.tree_util.tree_leaves(tree)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{f"leaf_{i}": np.asarray(x) for i, x in enumerate(leaves)}, **extra); os.replace(tmp, path)


def load_tree(path, like):
    z = np.load(path, allow_pickle=False); leaves, treedef = jax.tree_util.tree_flatten(like)
    return jax.tree_util.tree_unflatten(treedef, [jnp.asarray(z[f"leaf_{i}"], dtype=x.dtype) for i, x in enumerate(leaves)]), z


def run_name(a):
    kd = bool(a.teacher) and a.kd_alpha is not None
    n = f"dense-{a.size}-lr{a.lr:g}-s{a.seed}"
    if a.bs != 64: n += f"-bs{a.bs}"
    if a.seq != 512: n += f"-seq{a.seq}"
    if a.warmup != 500: n += f"-wu{a.warmup}"
    if a.train_tokens: n += f"-tok{float(a.train_tokens):g}"
    if a.f32: n += "-f32"
    if a.wd != 0.1: n += f"-wd{a.wd:g}"
    if a.wd_on != "all": n += "-wdmat"
    if a.opt != "adamw": n += f"-{a.opt}"
    if a.dp > 1: n += f"-dp{a.dp}"
    if a.lr_floor != 0.1: n += f"-fl{a.lr_floor:g}"
    if kd: n += f"-kd{a.teacher_size}a{a.kd_alpha:g}"
    return n + "-hs" + (f"-{a.tag}" if a.tag else "")       # hs: crops sampled uniformly on the host (second version of this script)


def run(a):
    """One training run on one device (or on --dp devices). Resumable: state is saved at every evaluation and reloaded if it matches this recipe."""
    if a.dp > 1:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
        devs = jax.devices()[a.device:a.device + a.dp]
        if len(devs) < a.dp: raise SystemExit(f"--dp {a.dp} needs {a.dp} devices from index {a.device}")
        if a.bs % a.dp or (a.bs // a.micro) % a.dp or 16 % a.dp: raise SystemExit("batch, micro-batch and the evaluation batch (16) must divide by --dp")
        mesh = Mesh(np.array(devs), ("data",)); dev = devs[0]
        r_sh, t_sh, e_sh = NamedSharding(mesh, PS()), NamedSharding(mesh, PS(None, "data", None)), NamedSharding(mesh, PS("data", None))
        put, put_toks, put_eval = (lambda x: jax.device_put(x, r_sh)), (lambda x: jax.device_put(x, t_sh)), (lambda x: jax.device_put(x, e_sh))
    else:
        dev = jax.devices()[a.device]
        put, put_toks, put_eval = (lambda x: jax.device_put(x, dev)), (lambda x: x), (lambda x: x)
    c = SIZES[a.size]; dt = jnp.float32 if a.f32 else jnp.bfloat16
    kd = (SIZES[a.teacher_size], float(a.kd_alpha)) if a.teacher and a.kd_alpha is not None else None
    name = run_name(a)
    say = lambda d: print(json.dumps({"run": name, **d}), flush=True)
    manifest = json.load(open(f"{a.data}/manifest.json"))
    train = np.memmap(f"{a.data}/train.bin", dtype=np.uint16, mode="r")
    if a.max_train_tokens: train = train[:a.max_train_tokens]
    vals = {dom: np.fromfile(f"{a.data}/val_{dom}.bin", dtype=np.uint16) for dom in ("web", "textbook", "math", "code")}
    budget = int(a.train_tokens) or len(train)
    steps = budget // (a.bs * a.seq) // a.chunk * a.chunk
    if steps <= 0: raise SystemExit("token budget is smaller than one chunk of steps")
    sched = lambda n: a.lr * jnp.minimum(1.0, (n + 1) / a.warmup) * (a.lr_floor + (1 - a.lr_floor) * 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(1.0, (n + 1) / steps))))
    mask = (lambda p: jax.tree_util.tree_map(lambda x: x.ndim >= 2, p)) if a.wd_on == "matrices" else None
    if a.opt == "muon":
        if a.wd_on != "all": raise SystemExit("--opt muon supports --wd-on all only")
        dims = lambda p: {"emb": None, "nf": None, "blocks": [{k: (optax.contrib.MuonDimensionNumbers() if getattr(v, "ndim", 0) == 2 else None) for k, v in b.items()} for b in p["blocks"]]}
        tx = optax.contrib.muon(sched, weight_decay=a.wd, adam_b1=0.9, adam_b2=0.95, adam_weight_decay=a.wd, consistent_rms=0.2, muon_weight_dimension_numbers=dims)
    else:
        tx = optax.adamw(sched, b1=0.9, b2=0.95, eps=1e-8, weight_decay=a.wd, mask=mask)
    opt = optax.chain(optax.clip_by_global_norm(1.0), tx)
    os.makedirs(a.out, exist_ok=True); res_path = f"{a.out}/{name}.json"; ck_path = f"{a.out}/{name}.state.npz"
    if a.gcs:                                                  # fail before any paid step if the bucket cannot be written
        mark = f"{a.out}/{name}.started"; open(mark, "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        if not push(mark, a.gcs): raise SystemExit("cannot write to the bucket; not starting")
    head = {"name": name, "cfg": c, "lr": a.lr, "warmup": a.warmup, "bs": a.bs, "seq": a.seq, "micro": a.micro, "seed": a.seed,
            "steps": steps, "train_tokens": steps * a.bs * a.seq, "corpus_tokens": int(len(train)), "dtype": "float32" if a.f32 else "bfloat16",
            "sampler": "uniform crops drawn on the host with numpy default_rng(seed + 1)", "jax": jax.__version__, "device": str(dev), "device_kind": dev.device_kind, "code_sha256": CODE_SHA, "n_devices": len(jax.devices()),
            "wd": a.wd, "wd_on": a.wd_on, "lr_floor": a.lr_floor, "opt": a.opt, "dp": a.dp,
            "objective": (f"distillation from {os.path.basename(a.teacher)}, alpha {kd[1]}" if kd else "cross-entropy")}
    ident = ("cfg", "lr", "warmup", "bs", "seq", "seed", "steps", "corpus_tokens", "dtype", "sampler", "objective", "wd", "wd_on", "lr_floor", "opt", "dp")
    rng = np.random.default_rng(a.seed + 1)
    with jax.default_device(dev):
        params = init(jax.random.PRNGKey(a.seed), c); st = opt.init(params)
        done, curve, t_train, compile_s, prior_hours = 0, [], 0.0, None, 0.0
        if a.resume and pull(res_path, a.gcs):
            old = json.load(open(res_path))
            legacy = {"wd": 0.1, "wd_on": "all", "lr_floor": 0.1, "opt": "adamw", "dp": 1}   # fields older result files lack
            bad = [k for k in ident if old.get(k, legacy.get(k)) != head[k]]
            if bad: raise SystemExit(f"{res_path} belongs to a different recipe (differs in {bad}); use a new --tag or --out")
            if old.get("status") == "diverged": say({"skipped": "this recipe diverged before", "at_step": old.get("diverged_at")}); return 2
            if old.get("finished"): say({"skipped": "already finished", "final": old["curve"][-1]}); return 0
            if pull(ck_path, a.gcs):
                (params, st), z = load_tree(ck_path, (params, st)); done = int(z["done"])
                rng.bit_generator.state = json.loads(str(z["rng_state"]))
                curve = [r for r in old["curve"] if r["step"] <= done]; prior_hours = curve[-1]["hours"] if curve else 0.0
                say({"resumed_at_step": done})
        params, st = (put(x) for x in (params, st))
        tp = put(load_params(a.teacher)) if kd else {}
        chunk = make_chunk(c, opt, a.micro, dt, kd, strided=a.dp > 1)
        ev_raw = jax.jit(lambda p, tok: loss_fn(p, tok, c, dt)); ev = lambda p, tok: ev_raw(p, put_eval(tok))
        head["params"] = count(params); say(head)
        if kd:                                                  # the teacher must reproduce its own reported score here
            tev_raw = jax.jit(lambda p, tok: loss_fn(p, tok, kd[0], dt)); tev = lambda p, tok: tev_raw(p, put_eval(tok)); say({"teacher": a.teacher, "alpha": kd[1], "teacher_eval": evaluate(tev, tp, vals, manifest, a.seq)})
        t0 = time.time(); start = done; timed = 0; span = np.arange(a.seq + 1, dtype=np.int64); n_start = len(train) - a.seq - 1
        def write(extra):
            json.dump({**head, "compile_seconds": compile_s, "curve": curve, **extra}, open(res_path, "w")); return push(res_path, a.gcs)
        while done < steps:
            ix = rng.integers(0, n_start, (a.chunk, a.bs))
            toks = np.asarray(train[(ix[..., None] + span).ravel()]).reshape(a.chunk, a.bs, a.seq + 1).astype(np.int32)
            t = time.time(); params, st, ls = chunk(params, st, put_toks(toks), tp); ls = np.asarray(ls); dtc = time.time() - t
            if compile_s is None: compile_s = round(dtc, 1)      # the first chunk includes compilation; excluded from throughput
            else: t_train += dtc; timed += a.chunk
            prev = done; done += a.chunk
            if not np.isfinite(ls).all():
                say({"error": "non-finite loss", "step": done}); write({"finished": False, "status": "diverged", "diverged_at": done}); return 2
            tps = round(timed * a.bs * a.seq / max(t_train, 1e-9))
            if done // 500 > prev // 500: say({"step": done, "loss": round(float(ls.mean()), 4), "tok_s": tps})
            if done // a.eval_every > prev // a.eval_every or done == steps:
                row = {"step": done, "tokens": done * a.bs * a.seq, "hours": round(prior_hours + (time.time() - t0) / 3600, 4), "train_loss": round(float(ls.mean()), 4),
                       "tokens_per_second": tps, **evaluate(ev, params, vals, manifest, a.seq)}
                try: row["device_gb_in_use"] = round(dev.memory_stats()["bytes_in_use"] / 2 ** 30, 2)
                except Exception: pass
                curve.append(row); say(row)
                ce = a.ckpt_every or a.eval_every
                if a.resume and done < steps and done // ce > prev // ce:   # state first, then the curve that points at it
                    save_tree(ck_path, (params, st), done=done, rng_state=json.dumps(rng.bit_generator.state)); push(ck_path, a.gcs)
                write({"finished": False})
        ok = True
        if a.save:
            flat = {"emb": np.asarray(params["emb"]), "nf": np.asarray(params["nf"])}
            for i, b in enumerate(params["blocks"]): flat.update({f"b{i}.{k}": np.asarray(v) for k, v in b.items()})
            np.savez(f"{a.out}/{name}.npz", **flat); ok = push(f"{a.out}/{name}.npz", a.gcs)
        ok = write({"finished": True}) and ok
        say({"finished": True, "wall_hours": round((time.time() - t0) / 3600, 3), "steps_this_session": done - start, "uploads_ok": ok})
    return 0 if ok else 3


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
    ap.add_argument("--teacher", default="", help="checkpoint (.npz) of a trained model to distil from"); ap.add_argument("--teacher-size", default="L")
    ap.add_argument("--kd-alpha", type=float, default=None, help="weight of the data loss; 1 - alpha goes to matching the teacher")
    ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--wd-on", default="all", choices=("all", "matrices"))
    ap.add_argument("--lr-floor", type=float, default=0.1, help="cosine schedule ends at this fraction of the peak rate")
    ap.add_argument("--opt", default="adamw", choices=("adamw", "muon"))
    ap.add_argument("--ckpt-every", type=int, default=0, help="save resumable state every this many steps (0: at every evaluation)")
    ap.add_argument("--dp", type=int, default=1, help="train this one run on this many devices (not with --multi)")
    ap.add_argument("--multi", default="", help='several runs, one per device at a time: "SIZE:LR:MICRO[:TAG[:TOKENS[:SEED[:ALPHA[:key=value...]]]]];..." '
                    '(ALPHA "none" = no teacher; keys wd, wdon, floor)')
    a = ap.parse_args()
    if not a.multi: raise SystemExit(run(a))
    if a.dp > 1: raise SystemExit("--dp cannot be combined with --multi")
    jobs = queue.Queue()
    for spec in a.multi.split(";"):
        f = spec.split(":"); b = copy.copy(a); b.size, b.lr, b.micro = f[0], float(f[1]), int(f[2])
        if len(f) > 3 and f[3]: b.tag = f[3]
        if len(f) > 4 and f[4]: b.train_tokens = float(f[4])
        if len(f) > 5 and f[5]: b.seed = int(f[5])
        if len(f) > 6 and f[6]: b.kd_alpha = None if f[6] == "none" else float(f[6])
        for kv in f[7:]:
            k, v = kv.split("=")
            if k == "wd": b.wd = float(v)
            elif k == "wdon": b.wd_on = v
            elif k == "floor": b.lr_floor = float(v)
            elif k == "opt": b.opt = v
            else: raise SystemExit(f"unknown job setting {k}")
        jobs.put(b)
    names = [run_name(b) for b in list(jobs.queue)]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup: raise SystemExit(f"two jobs would write the same files: {dup}; give them different tags")
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
