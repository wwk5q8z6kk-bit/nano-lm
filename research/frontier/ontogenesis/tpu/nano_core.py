"""Nano LM core mechanisms (cycle 1). Small, explicit JAX reference implementations for controlled experiments.

Nothing here is used by train_jax.py; the dense decoder there stays the control.

Mechanisms
  attention_layer      causal softmax attention (the recall reference).
  gdn_layer            Gated DeltaNet recurrence (Yang et al. 2024, arXiv 2412.06464), naive time scan:
                           S_t = a_t * S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T,   o_t = S_t q_t,
                       with L2-normalised q, k; a_t (decay) and b_t (write strength) are sigmoid gates of the input.
                       A fixed-size state per head: memory does not grow with context.
  memory_read          budgeted exact memory (Candidate 2, RESEARCH-PLAN.md section 8). At position t the model may
                       attend only to the M earlier positions with the highest write score among positions < t.
                       Selection is causal and hard (no gradient through the choice). The write score comes from a
                       policy: "recency" (position), "oracle" (given mask), or "learned" (a linear head on the hidden
                       state, trained by an auxiliary loss towards a teacher's attention, never by the LM loss).
"""
from __future__ import annotations
import jax, jax.numpy as jnp


def rms(x, w, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * w


def dense(key, i, o, scale=1.0):
    return jax.random.normal(key, (i, o)) * (scale / jnp.sqrt(i))


def init_layer(key, kind, d, H, ffn):
    ks = jax.random.split(key, 8)
    conv = jnp.zeros((4, d)).at[0].set(1.0) + 0.02 * jax.random.normal(ks[7], (4, d))   # near-identity start
    p = {"n1": jnp.ones(d), "n2": jnp.ones(d), "conv": conv, "q": dense(ks[0], d, d), "k": dense(ks[1], d, d), "v": dense(ks[2], d, d),
         "o": dense(ks[3], d, d, 0.5), "up": dense(ks[4], d, 2 * ffn), "dn": dense(ks[5], ffn, d, 0.5)}
    if kind == "gdn":
        p["ab"] = dense(ks[6], d, 2 * H, 0.1)            # per-head decay and write-strength gates
        p["ab_b"] = jnp.concatenate([jnp.full(H, 3.0), jnp.zeros(H)])   # start with slow decay (sigmoid(3) = 0.95)
    if kind == "mem":
        p["w"] = dense(ks[6], d, 1, 0.1)                 # write-score head (used by the "learned" policy)
    return p


def heads(x, H):
    B, T, d = x.shape
    return x.reshape(B, T, H, d // H).transpose(0, 2, 1, 3)          # B H T dh


def merge(x):
    B, H, T, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * dh)


def attention(q, k, v, allow):
    """q, k, v: B H T dh; allow: B 1|H T T boolean (True = may attend). Returns output and attention weights."""
    s = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(q.shape[-1])
    s = jnp.where(allow, s, -1e30)
    w = jax.nn.softmax(s, -1)
    w = jnp.where(allow.any(-1, keepdims=True), w, 0.0)                  # rows with nothing allowed read nothing
    return jnp.einsum("bhts,bhsd->bhtd", w, v), w


def gdn(q, k, v, a, b):
    """q, k, v: B H T dh (q, k L2-normalised); a, b: B H T in (0, 1). Naive scan over time."""
    B, H, T, dh = q.shape
    def step(S, xs):
        qt, kt, vt, at, bt = xs                                          # B H dh, ..., B H
        Sk = jnp.einsum("bhvk,bhk->bhv", S, kt)
        S = at[..., None, None] * (S - bt[..., None, None] * jnp.einsum("bhv,bhk->bhvk", Sk, kt)) \
            + bt[..., None, None] * jnp.einsum("bhv,bhk->bhvk", vt, kt)
        return S, jnp.einsum("bhvk,bhk->bhv", S, qt)
    xs = (q.transpose(2, 0, 1, 3), k.transpose(2, 0, 1, 3), v.transpose(2, 0, 1, 3), a.transpose(2, 0, 1), b.transpose(2, 0, 1))
    _, o = jax.lax.scan(step, jnp.zeros((B, H, dh, dh)), xs)
    return o.transpose(1, 2, 0, 3)                                       # B H T dh


def l2n(x, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.sum(x * x, -1, keepdims=True) + eps)


def topm_allow(score, M):
    """score: B T. allow[b, t, s] is True iff s < t and s is among the M highest-scoring positions in [0, t).
    Ties break towards later positions. Causal: the set at t depends only on scores at positions < t."""
    B, T = score.shape
    t_idx = jnp.arange(T)
    earlier = t_idx[None, :] < t_idx[:, None]                            # [t, s]: s < t
    s_ = score[:, None, :] + 1e-6 * t_idx[None, None, :]                 # tie-break
    # rank of s among earlier positions of t = number of earlier positions s' with a higher score
    higher = (s_[:, :, None, :] > s_[:, :, :, None]) & earlier[None, :, None, :]   # B t s s'
    rank = higher.sum(-1)
    return earlier[None] & (rank < M)                                    # B t s


def short_conv(h, w):
    """Depthwise causal convolution over time, kernel K = w.shape[0]: out_t = sum_j w_j * h_{t-j}. Gives every layer
    local order information (as in Zoology/Based/Gated DeltaNet implementations)."""
    K = w.shape[0]
    hp = jnp.pad(h, ((0, 0), (K - 1, 0), (0, 0)))
    return sum(w[j] * hp[:, K - 1 - j: K - 1 - j + h.shape[1]] for j in range(K))


def block(p, x, kind, H, allow_mem=None):
    h = short_conv(rms(x, p["n1"]), p["conv"])
    q, k, v = heads(h @ p["q"], H), heads(h @ p["k"], H), heads(h @ p["v"], H)
    T = x.shape[1]
    aux = {}
    if kind == "attn":
        causal = jnp.tril(jnp.ones((T, T), bool))[None, None]
        y, w = attention(q, k, v, causal)
        aux["attn"] = w
    elif kind == "gdn":
        g = jax.nn.sigmoid(h @ p["ab"] + p["ab_b"])                     # B T 2H
        a, b = g[..., :H].transpose(0, 2, 1), g[..., H:].transpose(0, 2, 1)
        y = gdn(l2n(q), l2n(k), v, a, b)
    elif kind == "mem":
        y, _ = attention(q, k, v, allow_mem[:, None])
    x = x + merge(y) @ p["o"]
    g, u = jnp.split(rms(x, p["n2"]) @ p["up"], 2, -1)
    return x + (jax.nn.silu(g) * u) @ p["dn"], aux


def init_model(key, cfg):
    """cfg: dict(vocab, d, H, ffn, layers=[kinds]). A 'mem' layer is a budgeted exact-memory read layer."""
    ks = jax.random.split(key, len(cfg["layers"]) + 1)
    return {"emb": jax.random.normal(ks[0], (cfg["vocab"], cfg["d"])) * 0.02, "nf": jnp.ones(cfg["d"]),
            "layers": [init_layer(ks[i + 1], kind, cfg["d"], cfg["H"], cfg["ffn"]) for i, kind in enumerate(cfg["layers"])]}


def forward(params, tok, cfg, policy=None, M=None, oracle=None):
    """tok: B T. policy in {None, 'recency', 'oracle', 'learned'} for 'mem' layers. Returns logits, aux dict."""
    x = params["emb"][tok]
    aux = {"attn": [], "write": None}
    for p, kind in zip(params["layers"], cfg["layers"]):
        allow = None
        if kind == "mem":
            T = tok.shape[1]
            if policy == "recency":
                score = jnp.broadcast_to(jnp.arange(T, dtype=jnp.float32), tok.shape)
            elif policy == "oracle":
                score = oracle.astype(jnp.float32) + 1e-3 * jnp.arange(T)  # oracle tokens first, then the most recent
            elif policy == "learned":
                raw = (rms(x, p["n1"]) @ p["w"])[..., 0]
                aux["write"] = raw
                score = jax.lax.stop_gradient(raw)
            else:
                raise ValueError(policy)
            allow = topm_allow(score, M)
        x, a = block(p, x, kind, cfg["H"], allow)
        if "attn" in a: aux["attn"].append(a["attn"])
    return rms(x, params["nf"]) @ params["emb"].T, aux


# ---------------------------------------------------------------------------------------------------------------------
# Candidate 3: adaptive recurrent depth (RESEARCH-PLAN.md section 8). A prelude of distinct layers, then ONE shared core
# block applied R times, then a coda. h_{k+1} = core(h_k). With halt_eps set, a token stops iterating once the relative
# change of its state ||h_{k+1} - h_k|| / ||h_k|| falls below halt_eps (a fixed-point convergence test, as an iterative
# solver stops); halted tokens keep their state and still serve as keys for later tokens.

def init_looped(key, cfg):
    """cfg: vocab, d, H, ffn, pre=[kinds], core=kind, coda=[kinds]."""
    n = len(cfg["pre"]) + 1 + len(cfg["coda"])
    ks = jax.random.split(key, n + 1)
    mk = lambda i, kind: init_layer(ks[i + 1], kind, cfg["d"], cfg["H"], cfg["ffn"])
    return {"emb": jax.random.normal(ks[0], (cfg["vocab"], cfg["d"])) * 0.02, "nf": jnp.ones(cfg["d"]),
            "pre": [mk(i, k) for i, k in enumerate(cfg["pre"])], "core": mk(len(cfg["pre"]), cfg["core"]),
            "coda": [mk(len(cfg["pre"]) + 1 + i, k) for i, k in enumerate(cfg["coda"])]}


def forward_looped(params, tok, cfg, R, halt_eps=None):
    """Returns logits and the mean number of core iterations actually applied per token."""
    x = params["emb"][tok]
    for p, kind in zip(params["pre"], cfg["pre"]):
        x, _ = block(p, x, kind, cfg["H"])
    active = jnp.ones(tok.shape, bool); iters = jnp.zeros(tok.shape)
    for _ in range(R):
        xn, _ = block(params["core"], x, cfg["core"], cfg["H"])
        if halt_eps is not None:
            rel = jnp.linalg.norm(xn - x, axis=-1) / (jnp.linalg.norm(x, axis=-1) + 1e-6)
            xn = jnp.where(active[..., None], xn, x)
            iters = iters + active
            active = active & (rel >= halt_eps)
        else:
            iters = iters + 1
        x = xn
    for p, kind in zip(params["coda"], cfg["coda"]):
        x, _ = block(p, x, kind, cfg["H"])
    return rms(x, params["nf"]) @ params["emb"].T, iters.mean()
