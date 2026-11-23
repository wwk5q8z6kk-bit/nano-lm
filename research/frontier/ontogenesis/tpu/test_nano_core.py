"""Correctness tests for nano_core.py: recurrence against an explicit loop, causality for every layer and policy,
and the budgeted-memory selection rule. Run: python -m pytest -q tpu/test_nano_core.py (from the ontogenesis dir)."""
import numpy as np, jax, jax.numpy as jnp
import nano_core as nc

jax.config.update("jax_enable_x64", False)


def test_gdn_matches_explicit_loop():
    r = np.random.default_rng(0); B, H, T, dh = 2, 2, 7, 4
    q = nc.l2n(jnp.asarray(r.normal(size=(B, H, T, dh)), jnp.float32)); k = nc.l2n(jnp.asarray(r.normal(size=(B, H, T, dh)), jnp.float32))
    v = jnp.asarray(r.normal(size=(B, H, T, dh)), jnp.float32); a = jnp.asarray(r.uniform(0.5, 1, (B, H, T)), jnp.float32); b = jnp.asarray(r.uniform(0, 1, (B, H, T)), jnp.float32)
    out = np.asarray(nc.gdn(q, k, v, a, b))
    for bi in range(B):
        for hi in range(H):
            S = np.zeros((dh, dh))
            for t in range(T):
                kt, vt, qt = np.asarray(k[bi, hi, t]), np.asarray(v[bi, hi, t]), np.asarray(q[bi, hi, t])
                S = float(a[bi, hi, t]) * (S @ (np.eye(dh) - float(b[bi, hi, t]) * np.outer(kt, kt))) + float(b[bi, hi, t]) * np.outer(vt, kt)
                np.testing.assert_allclose(out[bi, hi, t], S @ qt, rtol=1e-4, atol=1e-5)


def test_topm_allow_rule():
    r = np.random.default_rng(1); B, T, M = 3, 12, 4
    score = jnp.asarray(r.normal(size=(B, T)), jnp.float32)
    allow = np.asarray(nc.topm_allow(score, M))
    for bi in range(B):
        for t in range(T):
            row = allow[bi, t]
            assert not row[t:].any()                                   # only earlier positions
            assert row.sum() == min(t, M)                              # exactly the budget once enough history exists
            if t > 0:
                kept = set(np.flatnonzero(row)); best = set(np.argsort(-np.asarray(score[bi, :t]), kind="stable")[:M])
                assert kept == best


def test_topm_allow_causal_in_scores():
    r = np.random.default_rng(2); T, M = 10, 3
    s1 = jnp.asarray(r.normal(size=(1, T)), jnp.float32); s2 = s1.at[0, 6:].set(100.0)   # change only future scores
    a1, a2 = np.asarray(nc.topm_allow(s1, M)), np.asarray(nc.topm_allow(s2, M))
    np.testing.assert_array_equal(a1[:, :7], a2[:, :7])                # rows up to t=6 depend only on scores < t


def test_causality_all_layers_and_policies():
    cfg = dict(vocab=50, d=16, H=2, ffn=32, layers=["gdn", "attn", "mem"])
    p = nc.init_model(jax.random.PRNGKey(0), cfg); r = np.random.default_rng(3)
    tok = jnp.asarray(r.integers(0, 50, (2, 16)), jnp.int32); tok2 = tok.at[:, 10:].set((tok[:, 10:] + 7) % 50)
    orc = jnp.asarray(r.integers(0, 2, (2, 16)), bool)
    for policy in ("recency", "oracle", "learned"):
        l1, _ = nc.forward(p, tok, cfg, policy, M=4, oracle=orc); l2, _ = nc.forward(p, tok2, cfg, policy, M=4, oracle=orc)
        np.testing.assert_allclose(np.asarray(l1[:, :10]), np.asarray(l2[:, :10]), rtol=1e-5, atol=1e-5)
