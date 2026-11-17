"""Correctness tests for the discovery-lab model: chunked recurrences == naive recurrences."""

import torch

from lab.arch.model import LM, Cfg, LinearMixer


def naive(mix: LinearMixer, x):
    B, T, D = x.shape
    H, hd = mix.h, mix.hd
    q, k, v = mix.conv(mix.qkv(x)).split(D, -1)
    q = torch.nn.functional.normalize(q.view(B, T, H, hd).float(), dim=-1) * hd ** -0.5
    k = torch.nn.functional.normalize(k.view(B, T, H, hd).float(), dim=-1)
    v = v.view(B, T, H, hd).float()
    a = (-torch.nn.functional.softplus(mix.a(x).float()) * mix.A_log.exp()).exp()  # B,T,H
    beta = torch.sigmoid(mix.b(x).float()) if mix.delta else None
    S = torch.zeros(B, H, hd, hd)
    ys = []
    for t in range(T):
        kt, vt, qt, at = k[:, t], v[:, t], q[:, t], a[:, t][..., None, None]
        if mix.delta:
            bt = beta[:, t][..., None, None]
            eye = torch.eye(hd)
            S = at * (eye - bt * kt[..., :, None] * kt[..., None, :]) @ S + bt * kt[..., :, None] * vt[..., None, :]
        else:
            S = at * S + kt[..., :, None] * vt[..., None, :]
        ys.append((S.transpose(-1, -2) @ qt[..., None])[..., 0])
    y = torch.stack(ys, 2)  # B,H,T,hd
    y = mix.on(y.transpose(1, 2).to(x.dtype)).reshape(B, T, D) * torch.nn.functional.silu(mix.g(x))
    return mix.o(y)


def _check(delta):
    torch.manual_seed(0)
    c = Cfg(vocab=64, d=32, n_head=2, n_kv=2, ffn=64, layout="D" if delta else "L", chunk=8)
    m = LinearMixer(c, delta=delta)
    x = torch.randn(2, 29, 32)  # deliberately not a multiple of chunk
    y_chunk, _ = m(x, None, None)
    y_naive = naive(m, x)
    assert torch.allclose(y_chunk, y_naive, atol=2e-4, rtol=1e-3), (y_chunk - y_naive).abs().max()


def test_gla_chunked_matches_naive():
    _check(False)


def test_delta_chunked_matches_naive():
    _check(True)


def test_causality_all_mixers():
    for layout in ["A", "S", "L", "D"]:
        torch.manual_seed(0)
        m = LM(Cfg(vocab=50, d=32, n_head=4, n_kv=2, ffn=64, layout=layout * 2, window=4, chunk=8))
        x = torch.randint(0, 50, (1, 20))
        y1 = m(x)
        x2 = x.clone()
        x2[0, 15:] = (x2[0, 15:] + 1) % 50
        y2 = m(x2)
        assert torch.allclose(y1[0, :15], y2[0, :15], atol=1e-5), layout


def test_loop_and_value_residual_run():
    m = LM(Cfg(vocab=50, d=32, n_head=4, n_kv=2, ffn=64, layout="AAAA", loop=(1, 2, 1, 3),
               value_residual=True))
    x = torch.randint(0, 50, (2, 16))
    loss = m(x, x)
    loss.backward()
    r = m.param_report()
    assert r["effective_depth"] == 1 + 2 * 3 + 1
