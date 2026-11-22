"""Apply predictions V1-V5 of forward_transfer_meta.py mechanically. Committed before any registered result existed.
A prediction holds only if its bound is met by the mean over the three seeds; per-seed values are printed."""
import json
import numpy as np

D = {a: json.load(open(f"results/forward_transfer_meta/{a}.json")) for a in ("dense", "loop", "menu")}
KS = (2, 4, 8, 16)


def cell(arm, probe, fam, k, part="unshown"):
    xs = [r[f"{probe}/{fam}"][f"k{k}"][part] if probe == "meta_probe" else r[f"{probe}/{fam}"][f"{fam}/embed/k{k}"] for r in D[arm]]
    return round(float(np.mean(xs)), 4), xs


prim = lambda a: cell(a, "meta_probe", "novel_comp", 8)
V = {}
V["V1"] = {"held": cell("dense", "meta_probe", "trained", 8)[0] >= 0.85 and cell("loop", "meta_probe", "trained", 8)[0] >= 0.85 and cell("menu", "meta_probe", "trained", 8)[0] >= 0.60,
           **{a: cell(a, "meta_probe", "trained", 8) for a in D}}
V["V2"] = {"held": prim("dense")[0] <= 0.15, "dense": prim("dense")}
V["V3"] = {"held": prim("loop")[0] >= 0.30, "loop": prim("loop")}
V["V4"] = {"held": prim("menu")[0] >= 0.40, "menu": prim("menu")}
v5 = {a: max(cell(a, "meta_probe", "novel_prim", k)[0] for k in KS) for a in D}
V["V5"] = {"held": all(x <= 0.15 for x in v5.values()), "max_over_k": v5}
table = {a: {f"{probe}/{fam}/k{k}": cell(a, probe, fam, k)[0] for probe, fams in (("meta_probe", ("trained", "novel_comp", "novel_prim")), ("registered_probe", ("novel_comp", "novel_prim")))
             for fam in fams for k in KS} for a in D}
shown = {a: {f"novel_comp/k{k}": cell(a, "meta_probe", "novel_comp", k, "shown")[0] for k in KS} for a in D}
json.dump({"predictions": V, "table": table, "novel_comp_shown_fit": shown, "params": {a: D[a][0]["params"] for a in D}}, open("results/forward_transfer_meta/verdict.json", "w"), indent=1)
for k, v in V.items(): print(k, "HELD" if v["held"] else "FAILED", json.dumps({a: b for a, b in v.items() if a != "held"}))
print(json.dumps(table, indent=1)); print("shown-example fit on new combinations:", json.dumps(shown))
