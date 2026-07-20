# PREREG_slot_diversity.md — eval-instance generator (instrument, committed pre-run).
# K=5 instances per arm-pool (d5/d20/d80), seeds 20260730-34. v1 eval distribution
# (exec'd verbatim) with ONE controlled change: the alg slot. Held dialogues carry a
# held allergy type on EVERY item, cycled uniformly over the 6 frozen HELD_ALG types
# (~17/type/instance -> ~83/type over K=5: per-type power the audits demanded). Seen
# dialogues draw alg from the ARM'S OWN training pool (p=0.5 presence, as v1), so
# seen-recall measures the arm's in-distribution copying. D20-pos reuses d20 instances
# (its manipulation is the SUMMARY order at FT/scoring time, not dialogue content).
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slot_diversity_pools import HELD_ALG, ALG_TRAIN_5, ALG_TRAIN_20, ALG_TRAIN_80

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, "..", "scribe", "build_scribe_data.py")
SEEDS = [20260730, 20260731, 20260732, 20260733, 20260734]
N = 200
ARMS = {"d5": ALG_TRAIN_5, "d20": ALG_TRAIN_20, "d80": ALG_TRAIN_80}

src = open(V1).read()
marker = "# ---------------- build sets ----------------"
assert marker in src, "v1 generator layout changed; re-verify"
ns = {}
exec(compile(src.split(marker)[0], V1, "exec"), ns)
sample_tuple, make_convo = ns["sample_tuple"], ns["make_convo"]

OUT = os.path.join(HERE, "sweep_eval")
os.makedirs(OUT, exist_ok=True)

for arm, pool in ARMS.items():
    for seed in SEEDS:
        random.seed(seed)
        items, held_i = [], 0
        for i in range(N):
            held = i < N // 2
            t = sample_tuple(held)
            if held:
                t["alg"] = HELD_ALG[held_i % len(HELD_ALG)]; held_i += 1   # uniform type control
            else:
                t["alg"] = random.choice(pool) if random.random() < 0.5 else None
            convo = make_convo(t, held)                                    # render AFTER override
            items.append({"tuple": {"cc": t["cc"][1], "dur": f"{t['n']} {t['unit']}",
                                    "sev": t["sev"], "med": t["med"] or "none",
                                    "alg": t["alg"] or "none"},
                          "held_values": held, "arm": arm,
                          "held_alg_type": t["alg"] if held else None,
                          "convo": convo})
        path = os.path.join(OUT, f"{arm}_m{seed - 20260730}.json")
        json.dump(items, open(path, "w"), indent=1)

# ---- verification ----
ok = True
for arm, pool in ARMS.items():
    assert not (set(HELD_ALG) & set(pool)), arm
    for k in range(5):
        items = json.load(open(os.path.join(OUT, f"{arm}_m{k}.json")))
        held = [it for it in items if it["held_values"]]
        seen = [it for it in items if not it["held_values"]]
        counts = {}
        for it in held: counts[it["tuple"]["alg"]] = counts.get(it["tuple"]["alg"], 0) + 1
        if set(counts) != set(HELD_ALG) or max(counts.values()) - min(counts.values()) > 1: ok = False
        for it in seen:
            if it["tuple"]["alg"] != "none" and it["tuple"]["alg"] not in pool: ok = False
        if len(items) != N or len(held) != 100: ok = False
    print(f"{arm}: 5 instances OK; held-alg counts (m0): {counts}")
print("VERIFICATION", "PASS" if ok else "FAIL")
