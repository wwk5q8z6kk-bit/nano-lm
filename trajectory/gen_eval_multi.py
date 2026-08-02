# Stage T-v2 instrument: K=5 larger held-eval instances for the powered gap
# estimator (PREREG_Tv2.md). Same v1 eval distribution (build_scribe_data.py
# eval loop, all held templates; first half held values, second half seen),
# scaled from 40 -> 200 dialogues (100 held + 100 seen) and drawn at 5 fixed
# fresh seeds. Execs the v1 definitions verbatim; only N and the seed change.
import json, random, os

V1 = os.path.join(os.path.dirname(__file__), "..", "scribe", "build_scribe_data.py")
SEEDS = [20260720, 20260721, 20260722, 20260723, 20260724]   # fixed in PREREG_Tv2
N = 200                                                        # 100 held + 100 seen
HERE = os.path.dirname(os.path.abspath(__file__))

src = open(V1).read()
marker = "# ---------------- build sets ----------------"
assert marker in src, "v1 generator layout changed; re-verify before generating"
ns = {}
exec(compile(src.split(marker)[0], V1, "exec"), ns)
sample_tuple, make_convo = ns["sample_tuple"], ns["make_convo"]

HELD_VALS = {"toothache", "neck pain", "heartburn", "melatonin", "throat lozenges", "sulfa drugs"}
inst0 = json.load(open(os.path.join(HERE, "..", "scribe", "scribe_eval.json")))
instT = json.load(open(os.path.join(HERE, "scribe_eval_T.json")))
prior_convos = {e["convo"][0]["content"] for e in inst0} | {e["convo"][0]["content"] for e in instT}

all_items = []
for k, seed in enumerate(SEEDS):
    random.seed(seed)
    items = []
    for i in range(N):
        held_vals = i < N // 2
        t = sample_tuple(held_vals)
        items.append({"tuple": {"cc": t["cc"][1], "dur": f"{t['n']} {t['unit']}", "sev": t["sev"],
                                "med": t["med"] or "none", "alg": t["alg"] or "none"},
                      "held_values": held_vals, "convo": make_convo(t, True)})
    out = os.path.join(HERE, f"scribe_eval_m{k}.json")
    json.dump(items, open(out, "w"), indent=1)
    held_n = sum(e["held_values"] for e in items)
    coll = sum(1 for e in items if e["convo"][0]["content"] in prior_convos)
    prior_convos |= {e["convo"][0]["content"] for e in items}   # also dedup across the new instances
    print(f"m{k} seed {seed}: {len(items)} items ({held_n} held) -> {os.path.basename(out)}; "
          f"collisions vs prior instances: {coll}")
    all_items.append(items)

# schema + held-value integrity check across all five
allok = True
for k, items in enumerate(all_items):
    for e in items:
        if set(e["tuple"].keys()) != {"cc", "dur", "sev", "med", "alg"}: allok = False
        # a held-value dialogue must contain at least one held token; a seen-value one must not require it
    n_held_with_heldtoken = sum(1 for e in items if e["held_values"]
                                and any(v in json.dumps(e["tuple"]) for v in HELD_VALS))
    print(f"m{k}: schema_ok={allok}  held-value items carrying a held token: "
          f"{n_held_with_heldtoken}/{sum(e['held_values'] for e in items)}")
