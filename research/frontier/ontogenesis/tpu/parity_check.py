"""Check that train_jax.py computes the same function as the PyTorch reference (dense.py).

  1. with the PyTorch environment:  python parity_check.py export /path/ref.npz
  2. with the JAX environment:      python parity_check.py check  /path/ref.npz
Step 1 builds the reference model (size S), saves its weights, a fixed input, its logits and its loss.
Step 2 loads those weights into the JAX model and compares, in float32.
"""
import sys
import numpy as np

mode, path = sys.argv[1], sys.argv[2]
if mode == "export":
    from pathlib import Path
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "m21_dense_code"))
    from dense import Cfg, LM
    torch.manual_seed(0)
    m = LM(Cfg(d=256, n_head=4, n_kv=2, ffn=704, layout="A" * 8, seq=512, loop=None)).eval()
    with torch.no_grad():                                   # make the norm gains non-trivial so they are actually tested
        for n, p in m.named_parameters():
            if p.dim() == 1: p.add_(0.1 * torch.randn_like(p))
    x = torch.randint(0, 32768, (2, 96)); y = torch.randint(0, 32768, (2, 96))
    with torch.no_grad(): lg = m(x); loss = m(x, y)
    sd = {k: v.numpy() for k, v in m.state_dict().items()}
    print(sorted(sd)[:14], len(sd), "params", sum(p.numel() for p in m.parameters()))
    np.savez(path, x=x.numpy(), y=y.numpy(), logits=lg.numpy(), loss=float(loss), **{"sd:" + k: v for k, v in sd.items()})
else:
    import jax, jax.numpy as jnp, optax
    from train_jax import SIZES, forward, count
    z = np.load(path); sd = {k[3:]: z[k] for k in z.files if k.startswith("sd:")}
    T = lambda k: jnp.asarray(sd[k].T)                       # PyTorch Linear stores [out, in]
    blocks = []
    for i in range(8):
        b = f"blocks.{i}."
        blocks.append(dict(n1=jnp.asarray(sd[b + "n1.w"]), n2=jnp.asarray(sd[b + "n2.w"]), qn=jnp.asarray(sd[b + "mix.qn.w"]), kn=jnp.asarray(sd[b + "mix.kn.w"]),
                           q=T(b + "mix.q.weight"), k=T(b + "mix.k.weight"), v=T(b + "mix.v.weight"), o=T(b + "mix.o.weight"),
                           gu=T(b + "mlp.gu.weight"), dn=T(b + "mlp.dn.weight")))
    params = dict(emb=jnp.asarray(sd["emb.weight"]), nf=jnp.asarray(sd["nf.w"]), blocks=blocks)
    used = 2 + 10 * 8
    lg = np.asarray(forward(params, jnp.asarray(z["x"]), SIZES["S"], jnp.float32))
    loss = float(optax.softmax_cross_entropy_with_integer_labels(jnp.asarray(lg), jnp.asarray(z["y"])).mean())
    ref = z["logits"]
    print("tensors in PyTorch model", len(sd), "mapped", used, "| params", count(params))
    print("max |logit difference|", float(np.abs(lg - ref).max()), "| logit scale", float(np.abs(ref).max()))
    print("loss: PyTorch", float(z["loss"]), "JAX", loss, "difference", abs(loss - float(z["loss"])))
    print("argmax agreement", float((lg.argmax(-1) == ref.argmax(-1)).mean()))
    lg16 = np.asarray(forward(params, jnp.asarray(z["x"]), SIZES["S"], jnp.bfloat16))
    print("bfloat16 path: max |difference|", float(np.abs(lg16 - ref).max()), "argmax agreement", float((lg16.argmax(-1) == ref.argmax(-1)).mean()))
