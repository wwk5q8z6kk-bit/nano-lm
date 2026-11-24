"""Apply predictions T1-T7 of forward_transfer.py to results/forward_transfer/*.json, without judgement.

Written and committed before any registered result was read. A prediction "held" only if its stated
bound is met by the mean over the three seeds; the per-seed values are printed beside it.
"""
from __future__ import annotations
import json, sys
import numpy as np

ARMS = ("fresh", "dense", "dense_replay", "loop", "loop_replay", "menu", "menu_replay")
KS = (2, 4, 8, 16)
D = {a: json.load(open(f"results/forward_transfer/{a}.json")) for a in ARMS}


def val(arm, n, kind, how, k):
    """(mean, per-seed list) of one probe cell."""
    xs = [r["by_experience"][str(n)][f"{kind}/{how}/k{k}"] for r in D[arm]]
    return round(float(np.mean(xs)), 4), xs


def primary(arm, n=24): return val(arm, n, "novel_comp", "embed", 8)


out = {"table_primary": {a: primary(a) for a in ARMS if a != "fresh"}, "fresh": {f"{kind}/k{k}": val("fresh", 0, kind, "full", k) for kind in ("novel_comp", "novel_prim") for k in KS}}
out["stream_retained_after_24"] = {a: round(float(np.mean([r["by_experience"]["24"]["stream_retained"] for r in D[a]])), 4) for a in ARMS if a != "fresh"}
out["params"] = {a: D[a][0]["params"] for a in ARMS}
P = {}
P["T1"] = {"held": all(out["fresh"][f"novel_comp/k{k}"][0] <= 0.10 and out["fresh"][f"novel_prim/k{k}"][0] <= 0.10 for k in KS), "values": out["fresh"]}
P["T2"] = {"held": primary("dense")[0] <= 0.15, "value": primary("dense")}
P["T3"] = {"held": 0.15 <= primary("dense_replay")[0] <= 0.50, "value": primary("dense_replay")}
P["T4"] = {"held": primary("loop_replay")[0] >= 0.60 and primary("loop_replay")[0] - primary("dense_replay")[0] >= 0.20,
           "loop_replay": primary("loop_replay"), "dense_replay": primary("dense_replay")}
P["T5"] = {"held": primary("menu_replay")[0] >= 0.90 and val("menu_replay", 24, "novel_comp", "embed", 2)[0] >= 0.60 and primary("menu")[0] >= 0.70,
           "menu_replay_k8": primary("menu_replay"), "menu_replay_k2": val("menu_replay", 24, "novel_comp", "embed", 2), "menu_k8": primary("menu")}
P["T6"] = {"held": all(primary(a, 24)[0] - primary(a, 4)[0] >= 0.20 for a in ("loop_replay", "menu_replay")),
           "loop_replay": [primary("loop_replay", n)[0] for n in (0, 2, 4, 8, 16, 24)], "menu_replay": [primary("menu_replay", n)[0] for n in (0, 2, 4, 8, 16, 24)]}
embed_max = {a: max(val(a, n, "novel_prim", "embed", k)[0] for n in (0, 2, 4, 8, 16, 24) for k in KS) for a in ARMS if a != "fresh"}
full_gain = {a: max(val(a, 24, "novel_prim", "full", k)[0] - out["fresh"][f"novel_prim/k{k}"][0] for k in KS) for a in ARMS if a != "fresh"}
P["T7"] = {"held": all(v <= 0.15 for v in embed_max.values()) and all(v <= 0.05 for v in full_gain.values()),
           "novel_prim_embed_max": embed_max, "novel_prim_full_minus_fresh_max": {a: round(v, 4) for a, v in full_gain.items()}}
out["predictions"] = P
out["curves_primary_by_experience"] = {a: [primary(a, n)[0] for n in (0, 2, 4, 8, 16, 24)] for a in ARMS if a != "fresh"}
out["novel_comp_embed_by_k_after_24"] = {a: [val(a, 24, "novel_comp", "embed", k)[0] for k in KS] for a in ARMS if a != "fresh"}
out["novel_comp_full_by_k_after_24"] = {a: [val(a, 24, "novel_comp", "full", k)[0] for k in KS] for a in ARMS if a != "fresh"}
json.dump(out, open("results/forward_transfer/verdict.json", "w"), indent=1)
for k, v in P.items(): print(k, "HELD" if v["held"] else "FAILED", json.dumps({a: b for a, b in v.items() if a != "held"}))
print(json.dumps({k: out[k] for k in ("stream_retained_after_24", "curves_primary_by_experience", "novel_comp_embed_by_k_after_24", "novel_comp_full_by_k_after_24", "params")}, indent=1))
