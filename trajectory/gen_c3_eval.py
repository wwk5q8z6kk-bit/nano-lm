# PREREG_C3_binding_probe.md — eval-instance generator (committed pre-run, AFTER
# c3_pools.json passed the orthogonality hard gate). K=5 instances, seeds
# 20260750-54. Held dialogues cycle uniformly over: the 48 core-cell candidates +
# 11 T-full controls + the 34 C-1b bridges (re-scored, known states) = 93 held
# types. Seen dialogues draw alg from the D80 pool (p=0.5), same as C-1b/sweep.
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from slot_diversity_pools import ALG_TRAIN_80

pools = json.load(open(os.path.join(HERE, "c3_pools.json")))
assert pools["valid"], "c3_pools.json is INVALID (hard gate failed); fix pools first"

LABEL_OF = {}   # held type -> {"kind": "cell"/"tfull"/"bridge", "T":..,"B":..,"L":..}
for v, m in pools["manifest"].items():
    if m["T"] == "T-full":
        LABEL_OF[v] = {"kind": "tfull", "T": "T-full", "B": None, "L": None}
    else:
        LABEL_OF[v] = {"kind": "cell", "T": m["T"], "B": m["B"], "L": m["L"]}
for v in pools["c1b_bridges_rescored"]:
    LABEL_OF[v] = {"kind": "bridge", "T": None, "B": None, "L": None}

HELD_TYPES = sorted(LABEL_OF)
assert len(HELD_TYPES) == 59 + 34 == 93, f"expected 93 held types, got {len(HELD_TYPES)}"

V1 = os.path.join(HERE, "..", "scribe", "build_scribe_data.py")
src = open(V1).read()
marker = "# ---------------- build sets ----------------"
assert marker in src, "v1 generator layout changed; re-verify"
ns = {}
exec(compile(src.split(marker)[0], V1, "exec"), ns)
sample_tuple, make_convo = ns["sample_tuple"], ns["make_convo"]

SEEDS = [20260750, 20260751, 20260752, 20260753, 20260754]
N = 400   # 93 held types x >=3/instance needs >=186 held slots; N=400 -> 200 held/instance
OUT = os.path.join(HERE, "c3_eval")
os.makedirs(OUT, exist_ok=True)

for seed in SEEDS:
    random.seed(seed)
    items, held_i = [], 0
    for i in range(N):
        held = i < N // 2
        t = sample_tuple(held)
        if held:
            t["alg"] = HELD_TYPES[held_i % len(HELD_TYPES)]
            held_i += 1
        else:
            t["alg"] = random.choice(list(ALG_TRAIN_80)) if random.random() < 0.5 else None
        convo = make_convo(t, held)
        lab = LABEL_OF.get(t["alg"]) if held else None
        items.append({
            "tuple": {"cc": t["cc"][1], "dur": f"{t['n']} {t['unit']}",
                      "sev": t["sev"], "med": t["med"] or "none",
                      "alg": t["alg"] or "none"},
            "held_values": held, "arm": "d80",
            "held_alg_type": t["alg"] if held else None,
            "c3_kind": lab["kind"] if lab else None,
            "c3_T": lab["T"] if lab else None, "c3_B": lab["B"] if lab else None,
            "c3_L": lab["L"] if lab else None,
            "convo": convo})
    json.dump(items, open(os.path.join(OUT, f"c3_m{seed - 20260750}.json"), "w"), indent=1)

# ---- verification ----
ok = True
issues = []
for k in range(5):
    items = json.load(open(os.path.join(OUT, f"c3_m{k}.json")))
    held = [it for it in items if it["held_values"]]
    seen = [it for it in items if not it["held_values"]]
    counts = {}
    for it in held:
        counts[it["tuple"]["alg"]] = counts.get(it["tuple"]["alg"], 0) + 1
    if set(counts) != set(HELD_TYPES):
        ok = False; issues.append(f"m{k}: type-set mismatch")
    if max(counts.values()) - min(counts.values()) > 1:
        ok = False; issues.append(f"m{k}: imbalanced allocation")
    for it in held:
        exp = LABEL_OF[it["tuple"]["alg"]]
        if it["c3_kind"] != exp["kind"] or it["c3_T"] != exp["T"]:
            ok = False; issues.append(f"m{k}: label mismatch on {it['tuple']['alg']}")
    for it in seen:
        if it["tuple"]["alg"] != "none" and it["tuple"]["alg"] not in ALG_TRAIN_80:
            ok = False; issues.append(f"m{k}: seen-alg leak")
    if len(items) != N or len(held) != N // 2:
        ok = False; issues.append(f"m{k}: item count wrong")

# no-leakage-into-training check: none of the 93 held types may appear in ALG_TRAIN_80
leak = set(HELD_TYPES) & set(v.lower() for v in ALG_TRAIN_80)
if leak:
    ok = False; issues.append(f"TRAIN LEAK: {leak}")

per_type_total = {}
for k in range(5):
    items = json.load(open(os.path.join(OUT, f"c3_m{k}.json")))
    for it in items:
        if it["held_values"]:
            per_type_total[it["tuple"]["alg"]] = per_type_total.get(it["tuple"]["alg"], 0) + 1
print(f"5 instances x {N} items; {len(HELD_TYPES)} held types "
      f"(48 cell + 11 T-full + 34 bridges); per-type total over K=5: "
      f"min={min(per_type_total.values())} max={max(per_type_total.values())}")
print("issues:", issues if issues else "none")
print("VERIFICATION", "PASS" if ok else "FAIL")
json.dump({"seeds": SEEDS, "n_per_instance": N, "held_types": HELD_TYPES,
           "per_type_total_over_k5": per_type_total, "verification_pass": ok,
           "issues": issues},
          open(os.path.join(HERE, "c3_eval_verification.json"), "w"), indent=1)
