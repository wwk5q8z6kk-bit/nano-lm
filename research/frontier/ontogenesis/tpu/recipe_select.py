"""Question 11 (INVESTIGATION.md): the 14M recipe grid and the mechanical choice of what phase B runs.

  recipe_select.py grid                      -> phase A job list for train_jax.py --multi
  recipe_select.py gridm                     -> the 40M grid for the v6e machine (no phase B)
  recipe_select.py pick <out4> <out3>        -> JSON: every finished 14M cell (seed 1000, 4 micro-batches), the best
                                                AdamW cell and the best Muon cell by final held-out mix
  recipe_select.py phaseb <pick.json>        -> phase B job list: seeds 1001 and 1002 for both winners (none for the
                                                default cell, which already has three seeds), and both winners at 40M
                                                with learning rate 1e-3 (the 40M default) to test transfer of the rest
Selection uses seed 1000 only and is fixed here before any phase A result exists.
"""
import glob, json, sys

CENTER = ("adamw", 1e-3, 0.1, 0.1, "all")


def grid():
    j = []
    for lr in ("1e-3", "2e-3", "4e-3"):                       # Muon first: slower per step, so it should start first
        for wd in ("0.1", "0.5"):
            j.append(f"S:{lr}:4:::1000:none:wd={wd}:floor=0.01:opt=muon")
    j.append("S:1e-3:4:::1000:none:opt=muon")                 # Muon at the default cell
    for lr in ("1e-3", "2e-3", "4e-3"):
        for wd in ("0.1", "0.5"):
            for fl in ("0.1", "0.01"):
                if (lr, wd, fl) != ("1e-3", "0.1", "0.1"): j.append(f"S:{lr}:4:::1000:none:wd={wd}:floor={fl}")
    j.append("S:1e-3:4:::1000:none:wdon=matrices")             # decay on matrices and embedding only
    j.append("S:1e-3:2:m2::1000:none")                         # micro-batch control: default cell with 2 micro-batches
    return ";".join(j)



def gridm():
    """The same grid at 40M on the v6e machine (2 micro-batches), with its own three-seed default cell."""
    j = [f"M:1e-3:2:::{sd}:none" for sd in (1000, 1001, 1002)]
    for lr in ("1e-3", "2e-3", "4e-3"):
        for wd in ("0.1", "0.5"):
            j.append(f"M:{lr}:2:::1000:none:wd={wd}:floor=0.01:opt=muon")
    j.append("M:1e-3:2:::1000:none:opt=muon")
    for lr in ("1e-3", "2e-3", "4e-3"):
        for wd in ("0.1", "0.5"):
            for fl in ("0.1", "0.01"):
                if (lr, wd, fl) != ("1e-3", "0.1", "0.1"): j.append(f"M:{lr}:2:::1000:none:wd={wd}:floor={fl}")
    j.append("M:1e-3:2:::1000:none:wdon=matrices")
    return ";".join(j)


def cell(r): return (r.get("opt", "adamw"), r["lr"], r.get("wd", 0.1), r.get("lr_floor", 0.1), r.get("wd_on", "all"))


def pick(out4, out3):
    rows = []
    for d in (out4, out3):
        for f in sorted(glob.glob(f"{d}/dense-S-*.json")):
            r = json.load(open(f))
            if not r.get("finished") or r.get("micro") != 4 or "distillation" in r.get("objective", "") or not r.get("curve"): continue
            rows.append({"name": r["name"], "cell": cell(r), "seed": r["seed"], "mix": r["curve"][-1]["mix"],
                         "tok_s": r["curve"][-1].get("tokens_per_second"), "dir": d})
    s1000 = [x for x in rows if x["seed"] == 1000]
    best = {o: min((x for x in s1000 if x["cell"][0] == o), key=lambda x: x["mix"], default=None) for o in ("adamw", "muon")}
    return {"rows": rows, "best_adamw": best["adamw"], "best_muon": best["muon"], "center": list(CENTER)}


def spec(c, size, seed, lr=None):
    o, l, wd, fl, won = c
    s = f"{size}:{lr if lr is not None else l:g}:4:::{seed}:none:wd={wd:g}:floor={fl:g}:opt={o}"
    return s + (":wdon=matrices" if won == "matrices" else "")


def phaseb(p):
    j = []
    for k in ("best_muon", "best_adamw"):
        b = p[k]
        if b is None: continue
        c = tuple(b["cell"])
        if c != CENTER: j += [spec(c, "S", 1001), spec(c, "S", 1002)]
        if not (c[0] == "adamw" and c[2:] == CENTER[2:]): j.append(spec(c, "M", 1000, lr=1e-3))
    return ";".join(j)


if __name__ == "__main__":
    if sys.argv[1] == "grid": print(grid())
    elif sys.argv[1] == "gridm": print(gridm())
    elif sys.argv[1] == "pick": print(json.dumps(pick(sys.argv[2], sys.argv[3]), indent=1))
    elif sys.argv[1] == "phaseb": print(phaseb(json.load(open(sys.argv[2]))))
