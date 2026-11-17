"""Pilot P0 (runs on the GPU pod): (1) fla kernels == reference chunked implementation,
(2) measured training throughput per arm, eager vs torch.compile. Writes pilot_p0.json."""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import lab.arch.model as M  # noqa: E402
from lab.arch.arms import ARMS, build  # noqa: E402
from lab.arch.flops import train_flops_per_token  # noqa: E402

dev = torch.device("cuda")
out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "equivalence": {}, "throughput": []}

# (1) equivalence: same weights, reference path vs fla path
for delta in (False, True):
    torch.manual_seed(0)
    c = M.Cfg(vocab=64, d=256, n_head=4, n_kv=4, ffn=64, layout="D" if delta else "L", chunk=64)
    mix = M.LinearMixer(c, delta=delta).to(dev)
    x = torch.randn(2, 1000, 256, device=dev)
    M.USE_FLA = False
    y_ref, _ = mix(x, None, None)
    M.USE_FLA = True
    y_fla, _ = mix(x, None, None)
    err = ((y_ref - y_fla).norm() / y_ref.norm()).item()
    out["equivalence"]["delta" if delta else "gla"] = err
    print("equivalence rel-err", "delta" if delta else "gla", err, flush=True)

# (2) throughput
PEAK = float(os.environ.get("PEAK_TFLOPS", "0")) * 1e12
for arm in os.environ.get("ARMS", "T0_dense,H2_delta,R1_loop,P2_delta").split(","):
    for compile_ in (False, True):
        for fla in ((False, True) if any(ch in "LD" for ch in ARMS[arm]["layout"]) else (False,)):
            M.USE_FLA = fla
            torch.manual_seed(0)
            m = build(arm, 32768, 1024).to(dev)
            f = train_flops_per_token(m)["total"]
            fwd = torch.compile(m) if compile_ else m
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
            mb = int(os.environ.get("MICRO", "32"))
            x = torch.randint(0, 32768, (mb, 1025), device=dev)
            try:
                for i in range(13):
                    if i == 3:
                        torch.cuda.synchronize()
                        t = time.time()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = fwd(x[:, :-1], x[:, 1:])
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                tps = 10 * mb * 1024 / (time.time() - t)
                r = {"arm": arm, "compile": compile_, "fla": fla, "micro": mb, "tok_s": round(tps),
                     "gflop_per_tok": round(f / 1e9, 4), "achieved_tflops": round(tps * f / 1e12, 1),
                     "mfu": round(tps * f / PEAK, 3) if PEAK else None,
                     "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            except Exception as e:  # record, don't hide
                r = {"arm": arm, "compile": compile_, "fla": fla, "error": repr(e)[:300]}
            print(json.dumps(r), flush=True)
            out["throughput"].append(r)
            del m, fwd, opt
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
json.dump(out, open("pilot_p0.json", "w"), indent=2)
