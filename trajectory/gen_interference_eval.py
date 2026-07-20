# PREREG_token_coverage.md AMENDMENT 2 — eval-instance generator (committed pre-run).
# K=5 instances, seeds 20260740-44 (per the original C-1 design), one arm (D80).
# Held dialogues cycle uniformly over ALL 34 held types: the 28 AMENDMENT-2 class
# candidates + the 6 sweep bridges (retrodiction targets). ~3/type/instance held ->
# ~15/type over K=5, x2 FT seeds ~30 obs/type: ample for categorical flip-state calls.
# Seen dialogues draw alg from the D80 pool (p=0.5, as v1). Same exec-verbatim v1
# distribution as gen_slot_diversity_eval.py.
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slot_diversity_pools import ALG_TRAIN_80, HELD_ALG

HERE = os.path.dirname(os.path.abspath(__file__))
pools = json.load(open(os.path.join(HERE, "interference_pools.json")))
assert pools["valid"], "interference_pools.json is INVALID; fix pools first"
CLASS_OF = {p["value"]: cls for cls, ps in pools["pools"].items() for p in ps}
CLASS_OF.update({v: pools["bridges"][v]["class"] for v in pools["bridges"]})
HELD_TYPES = sorted(CLASS_OF)          # deterministic order; 28 candidates + 6 bridges
assert len(HELD_TYPES) == 34 and set(HELD_ALG) <= set(HELD_TYPES)

V1 = os.path.join(HERE, "..", "scribe", "build_scribe_data.py")
src = open(V1).read()
marker = "# ---------------- build sets ----------------"
assert marker in src, "v1 generator layout changed; re-verify"
ns = {}
exec(compile(src.split(marker)[0], V1, "exec"), ns)
sample_tuple, make_convo = ns["sample_tuple"], ns["make_convo"]

SEEDS = [20260740, 20260741, 20260742, 20260743, 20260744]
N = 200
OUT = os.path.join(HERE, "interference_eval")
os.makedirs(OUT, exist_ok=True)

for seed in SEEDS:
    random.seed(seed)
    items, held_i = [], 0
    for i in range(N):
        held = i < N // 2
        t = sample_tuple(held)
        if held:
            t["alg"] = HELD_TYPES[held_i % len(HELD_TYPES)]; held_i += 1
        else:
            t["alg"] = random.choice(list(ALG_TRAIN_80)) if random.random() < 0.5 else None
        convo = make_convo(t, held)
        items.append({"tuple": {"cc": t["cc"][1], "dur": f"{t['n']} {t['unit']}",
                                "sev": t["sev"], "med": t["med"] or "none",
                                "alg": t["alg"] or "none"},
                      "held_values": held, "arm": "d80",
                      "held_alg_type": t["alg"] if held else None,
                      "interference_class": CLASS_OF.get(t["alg"]) if held else None,
                      "convo": convo})
    json.dump(items, open(os.path.join(OUT, f"if_m{seed - 20260740}.json"), "w"), indent=1)

# ---- verification ----
ok = True
for k in range(5):
    items = json.load(open(os.path.join(OUT, f"if_m{k}.json")))
    held = [it for it in items if it["held_values"]]
    seen = [it for it in items if not it["held_values"]]
    counts = {}
    for it in held: counts[it["tuple"]["alg"]] = counts.get(it["tuple"]["alg"], 0) + 1
    if set(counts) != set(HELD_TYPES) or max(counts.values()) - min(counts.values()) > 1: ok = False
    if any(it["interference_class"] != CLASS_OF[it["tuple"]["alg"]] for it in held): ok = False
    for it in seen:
        if it["tuple"]["alg"] != "none" and it["tuple"]["alg"] not in ALG_TRAIN_80: ok = False
    if len(items) != N or len(held) != 100: ok = False
cls_n = {}
for it in held: cls_n[it["interference_class"]] = cls_n.get(it["interference_class"], 0) + 1
print(f"5 instances; 34 held types; per-class held counts (m4): {cls_n}")
print("VERIFICATION", "PASS" if ok else "FAIL")
