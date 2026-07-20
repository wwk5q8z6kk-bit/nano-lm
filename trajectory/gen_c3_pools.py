# PREREG_C3_binding_probe.md — C-3 pool generator (Transition x Boundary x Length).
# Deterministic, committed pre-run. Candidate SELECTION happens before any T/B/L
# statistic is computed (rule-based construction of a plausible-allergen candidate
# universe from head/tail word banks disjoint from every prior experiment's used
# words); T/B/L class falls out MECHANICALLY from real D80-corpus bigram/token
# statistics, never hand-assigned. If a registered cell is thin/empty, that is
# reported, not papered over by tuning individual candidates.
#
# HARD GATE: no pool/eval/kernel artifact is valid unless the orthogonality proof
# below passes for every registered contrast. Retrodiction anchors (checked first,
# and must pass before any candidate is even evaluated): the five known C-1b
# I-xslot values must resolve to T-full. If they don't, the corpus or tokenization
# construction is wrong and nothing downstream can be trusted.
import json, os, re, sys, itertools, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slot_diversity_pools import ALG_TRAIN_80, HELD_ALG
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
tok = Tokenizer.from_file(os.path.join(HERE, "..", "sft", "tokenizer.json"))

# ---------------- D80-arm training OUTPUT token STREAM (sequence, in-context) -----
V2 = open(os.path.join(HERE, "..", "scribe", "build_scribe_data_v2.py")).read()
ns = {}
exec(compile(V2.split("def ")[0].replace("from tokenizers import Tokenizer", ""), "v2p", "exec"), ns)
MED_TRAIN = ns["MED_TRAIN"]
ALG_LINE = 'ALG_TRAIN = ["penicillin", "peanuts", "pollen", "latex", "shellfish"]'
assert ALG_LINE in V2, "v2 source anchor moved — re-verify"
src = V2.replace(ALG_LINE, "ALG_TRAIN = " + repr(list(ALG_TRAIN_80)))
prefix = src.split("tok = Tokenizer.from_file")[0].replace("from tokenizers import Tokenizer", "")
ns2 = {}
exec(compile(prefix, "v2[d80]", "exec"), ns2)
CONVOS = ns2["convos"]
assert len(CONVOS) == 12000

OUT_SEQ = []
for c in CONVOS:
    for msg in c:
        if msg["role"] == "assistant":
            OUT_SEQ.extend(tok.encode(msg["content"], add_special_tokens=False).ids)

UNI = collections.Counter(OUT_SEQ)
BI = collections.Counter(zip(OUT_SEQ, OUT_SEQ[1:]))
print(f"D80 output stream: {len(OUT_SEQ)} tokens, {len(UNI)} distinct, {len(BI)} distinct bigrams")


def ctx_ids(value):
    """Token IDs exactly as the value appears after 'ALG: '/'MED: ' (leading-space
    context) — NOT standalone encoding, which mis-tokenizes the first token for
    many multi-token values (BPE leading-space merge). This is the fix validated
    against the T-full retrodiction anchors below."""
    return tok.encode(" " + value, add_special_tokens=False).ids


def is_tfull(value):
    ids = ctx_ids(value)
    n = len(ids)
    return any(OUT_SEQ[i:i + n] == ids for i in range(len(OUT_SEQ) - n + 1))


# ---------------- RETRODICTION GATE (must pass before anything else) --------------
print("\n== retrodiction gate: known C-1b T-full (I-xslot) values ==")
TFULL_ANCHORS = ["ibuprofen", "cough syrup", "antacids", "loratadine", "paracetamol"]
retro_ok = True
for v in TFULL_ANCHORS:
    ok = is_tfull(v)
    retro_ok &= ok
    print(f"  {'PASS' if ok else 'FAIL'} {v!r} T-full={ok}")
if not retro_ok:
    print("RETRODICTION GATE FAILED — corpus/tokenization construction is wrong. STOP.")
    sys.exit(1)
print("retrodiction gate: PASS (corpus + tokenization trusted)")

# ---------------- critical-junction identification --------------------------------
def word_split(value):
    return value.split(" ")


def critical_junction(value):
    """Returns (boundary_type, head_ids, tail_ids, head_str, tail_str) or None if
    the value is a single token (B-none; T undefined)."""
    words = word_split(value)
    ids_full = ctx_ids(value)
    if len(ids_full) <= 1:
        return None
    # B-punct (hyphenated junction) is optional per PREREG; not constructed here —
    # reported as an empty/uncostructed cell rather than force-built.
    if len(words) >= 2:
        head_str = " ".join(words[:-1])
        tail_str = words[-1]
        h_ids = ctx_ids(head_str)
        full_ids = ids_full
        # tail ids = full_ids with h_ids prefix removed (in-context boundary)
        if full_ids[:len(h_ids)] == h_ids:
            t_ids = full_ids[len(h_ids):]
            return ("B-space", h_ids, t_ids, head_str, tail_str)
        return None  # tokenization doesn't decompose cleanly at the word boundary
    # single word, multiple tokens -> B-sub: split at the LAST token
    return ("B-sub", ids_full[:-1], ids_full[-1:], value[:-1], value[-1])


def _fallback_punct(value):
    return None  # punct handling deferred; optional cell per PREREG


def junction_bigram(head_ids, tail_ids):
    if not head_ids or not tail_ids:
        return None
    return (head_ids[-1], tail_ids[0])


def classify_T(value):
    if is_tfull(value):
        return "T-full", None, None
    j = critical_junction(value)
    if j is None:
        return None, None, None
    btype, h_ids, t_ids, *_ = j
    if h_ids is None:
        return None, None, None
    jb = junction_bigram(h_ids, t_ids)
    if jb is None:
        return None, None, None
    bigram_ct = BI.get(jb, 0)
    left_ct, right_ct = UNI.get(jb[0], 0), UNI.get(jb[1], 0)
    if bigram_ct >= 20:
        return "T-avail", bigram_ct, (left_ct, right_ct)
    if bigram_ct == 0 and left_ct >= 20 and right_ct >= 20:
        return "T-sep", bigram_ct, (left_ct, right_ct)
    return "T-excluded", bigram_ct, (left_ct, right_ct)


# ---------------- hygiene (mechanical, as C-1b) ------------------------------------
TRAIN_VALUES = [v.lower() for v in ALG_TRAIN_80]
TRAIN_WORDS = set(w for v in TRAIN_VALUES for w in v.split())
MED_WORDS = set(w for v in MED_TRAIN for w in v.lower().split())
HELD_WORDS = set(w for v in HELD_ALG for w in v.lower().split())
STOP = set("""a an the any are is it you your do we i to of for on in at or and no not
have has been that what when why how did does was would should can be so but with over
me my this there here now far only just some most many say tell go got had am if then
yes okay well hello hi about ago around out up off from since where which who quite
pretty nearly almost maybe roughly none nothing anything haven't i'd i'm i've it's
what's there's won't you're""".split())
TPL_LISTS = {k: v for k, v in ns.items() if isinstance(v, list) and v
             and all(isinstance(x, str) for x in v)
             and (k.startswith("D_") or k.startswith("P_") or k == "DISTRACT")}
TMPL_WORDS = set()
for lst in TPL_LISTS.values():
    for t in lst:
        t = re.sub(r"\{[a-z_]+\}", " ", t)
        TMPL_WORDS.update(w for w in re.findall(r"[a-zA-Z']+", t.lower()) if w not in STOP)
strip_s = lambda w: w[:-1] if w.endswith("s") and len(w) > 3 else w
TMPL_S = {strip_s(w) for w in TMPL_WORDS}

# C-1b's used candidate values (no reuse, per PREREG hygiene)
C1B_USED = set(v.lower() for v in (
    "willow pollen maple pollen hazel pollen elm pollen goat milk rice milk "
    "blue dye pumpkin seeds chia seeds spider mites hornet stings deer dander "
    "cough syrup antacids loratadine paracetamol "
    "food additives food coloring generic drugs pet food "
    "camphor henna iodine quinoa gum arabic rose hips melon rind pine resin "
    "ragweed pollen sulfa drugs bee stings ibuprofen wool strawberries").split("  ")
    for v in [v]) | set(v.lower() for v in (
    "willow pollen,maple pollen,hazel pollen,elm pollen,goat milk,rice milk,"
    "blue dye,pumpkin seeds,chia seeds,spider mites,hornet stings,deer dander,"
    "cough syrup,antacids,loratadine,paracetamol,"
    "food additives,food coloring,generic drugs,pet food,"
    "camphor,henna,iodine,quinoa,gum arabic,rose hips,melon rind,pine resin,"
    "ragweed pollen,sulfa drugs,bee stings,ibuprofen,wool,strawberries").split(","))


def hygiene_fail(value):
    v = value.lower()
    words = v.split()
    if v in C1B_USED:
        return "reused C-1b candidate"
    if v in (x.lower() for x in ALG_TRAIN_80) or v in (x.lower() for x in HELD_ALG):
        return "already a trained/held ALG value"
    for t in TRAIN_VALUES:
        tw = t.split(); n = len(tw)
        if any(words[i:i + n] == tw for i in range(len(words) - n + 1)):
            return f"contains trained value {t!r} (I-contain territory, not T/B/L)"
    if set(words) & TRAIN_WORDS:
        return "shares word with a trained ALG value (I-sib territory)"
    if {w for w in words if strip_s(w) in TMPL_S}:
        return "shares template vocabulary word"
    return None


print(f"\nknown MED_TRAIN, unused-by-C1b, available as fresh T-full controls:")
FRESH_MED = [m for m in MED_TRAIN if m.lower() not in C1B_USED and m.lower() not in ("aspirin",)]
print(" ", FRESH_MED)


def Lband(n):
    return "short" if 3 <= n <= 4 else ("long" if 5 <= n <= 6 else None)


# ---------------- candidate universe (rule-based; classified AFTER construction) --
# Word banks are chosen for plausible-allergen surface form and lexical novelty vs
# every trained/held/template/C-1b word — NOT chosen by pre-checking bigram outcomes.
# T/B/L class is computed mechanically below; candidates that don't clear hygiene or
# land in a registered cell are simply not used (reported in the rejected list).
HEADS_MULTI = ["copper", "silver", "bronze", "velvet", "cedar", "maple", "birch",
               "cotton", "linen", "denim", "clover", "daisy", "lilac", "jasmine",
               "citronella", "eucalyptus", "peppermint", "spearmint", "cinnamon",
               "ginger", "turmeric", "paprika", "fennel", "basil", "oregano",
               "thyme", "rosemary", "lavender", "chamomile", "hibiscus"]
HEADS_1TOK = ["mint", "sage", "dill", "kelp", "reed", "fern", "moss", "husk",
              "bark", "peat", "silt", "wax", "rye", "oat", "corn", "cane",
              "hemp", "teak", "pine", "palm", "sisal", "jute", "cork", "flax",
              "yeast", "malt", "salt", "clay", "chalk", "soot", "ash", "coal",
              "tar", "gum", "sap", "musk", "civet", "amber", "opal"]
TAILS_SPACE = ["oil", "spray", "fumes", "fiber", "resin", "essence", "tincture",
               "particles", "paste", "gel", "balm", "wax", "dye", "milk",
               "mite", "sting", "nut", "seed", "stings", "mites", "seeds",
               "nuts", "dust", "musk", "scent", "vapor", "smoke", "ointment"]
CAND_SUB = ["betaine", "cocaine", "procaine", "ptomaine", "migraine",
            "melamine", "histamine", "glutamine", "arginine", "ornithine",
            "biotin", "gelatin", "lecithin", "mucin", "renin", "tannin",
            "capsaicin", "curcumin", "piperine", "allicin", "apigenin",
            "coumarin", "hesperidin", "naringin", "rutin", "catechin",
            "guaiacol", "cresol", "xylenol", "cardamol", "fenchol",
            "borneol", "citronellol", "nerolidol", "farnesol", "azelate",
            "phthalate", "succinate", "malate", "citrate", "bromide",
            "chloride", "iodide", "fluoride", "sulfide", "nitride",
            "acetate", "oxalate", "tartrate", "paramol", "dynamol",
            "genamol", "novamol", "pentamol", "tetramol", "xylocaine",
            "tetracaine", "benzocaine", "bupivacaine", "chloramine",
            "bromamine", "fluoramine", "clonidine", "ropinidine",
            "amantadine", "selegiline", "fexofenadine", "desloratadine",
            "azelastine", "olopatadine"]

RAW = []
for h in HEADS_MULTI + HEADS_1TOK:
    for t in TAILS_SPACE:
        RAW.append(f"{h} {t}")
RAW += CAND_SUB

by_cell = collections.defaultdict(list)   # (T, B, Lband) -> [(value, bigram_ct)]
rejected = []
seen = set()
for v in RAW:
    if v in seen:
        continue
    seen.add(v)
    hf = hygiene_fail(v)
    if hf:
        rejected.append({"value": v, "reason": hf})
        continue
    tclass, bigram_ct, freqs = classify_T(v)
    j = critical_junction(v)
    btype = j[0] if j else None
    ids = ctx_ids(v)
    ntok = len(ids)
    lb = Lband(ntok)
    if tclass in ("T-avail", "T-sep") and btype in ("B-sub", "B-space") and lb:
        by_cell[(tclass, btype, lb)].append((v, bigram_ct, ntok, freqs))
    else:
        rejected.append({"value": v, "reason": f"cell=({tclass},{btype},{lb}) not registered/thin",
                          "n_tokens": ntok, "T": tclass, "B": btype})

MIN_PER_CELL = 5
CORE_CELLS = list(itertools.product(("T-avail", "T-sep"), ("B-sub", "B-space"), ("short", "long")))
print("\n== core cell yields (pre-selection) ==")
for cell in CORE_CELLS:
    print(f"  {cell}: {len(by_cell[cell])} candidates")

# deterministic selection rule (fixed, applied identically to every contrast — NOT
# tuned per candidate). Naively ranking each cell independently by distance to the
# band center (3.5 short / 5.5 long) can leave two contrasted cells with different
# *achievable* token-count distributions (e.g. T-avail candidates cluster at 6 tok,
# T-sep candidates cluster at 5 tok within the same "long" band) and blow the
# orthogonality gate's <=0.5 mean-token-count-diff requirement. Fix (still a fixed,
# cell-agnostic rule, not per-candidate tuning): for every T-contrast pair (same B,
# same L), rank BOTH sides by distance to their SHARED pooled-mean token count
# (computed from the full raw candidate lists before selection) rather than each
# side's own band center — this pulls both selected subsets toward the same
# achievable mean instead of two different independent targets.
BAND_CENTER = {"short": 3.5, "long": 5.5}
POOL = {}
for B in ("B-sub", "B-space"):
    for L in ("short", "long"):
        a_raw, b_raw = by_cell[("T-avail", B, L)], by_cell[("T-sep", B, L)]
        pooled = [r[2] for r in a_raw] + [r[2] for r in b_raw]
        shared_target = (sum(pooled) / len(pooled)) if pooled else BAND_CENTER[L]
        for T, raw in (("T-avail", a_raw), ("T-sep", b_raw)):
            items = sorted(raw, key=lambda r: (abs(r[2] - shared_target),
                                                -r[1] if T == "T-avail" else 0, r[0]))
            # diversity cap (fixed rule, all cells): no lexical family/prefix/tail
            # may dominate a cell — at most 2 selections sharing a head word.
            picked, head_ct = [], collections.Counter()
            for r in items:
                h = r[0].split(" ")[0] if " " in r[0] else None
                if h and head_ct[h] >= 2:
                    continue
                picked.append(r)
                if h:
                    head_ct[h] += 1
                if len(picked) == 6:
                    break
            POOL[(T, B, L)] = picked

print("\n== selected pool per core cell (up to 6, min required 5) ==")
cells_ok = True
for cell in CORE_CELLS:
    n = len(POOL[cell])
    ok = n >= MIN_PER_CELL
    cells_ok &= ok
    print(f"  {'OK  ' if ok else 'THIN'} {cell}: n={n} -> {[r[0] for r in POOL[cell]]}")

# ---------------- T-full control (fresh MED_TRAIN values, unused by C-1b) ---------
TFULL_CTRL = []
for m in FRESH_MED:
    if hygiene_fail(m) and "already a trained/held" not in (hygiene_fail(m) or ""):
        continue
    if not is_tfull(m):
        continue
    ids = ctx_ids(m)
    TFULL_CTRL.append((m, len(ids)))
print(f"\nT-full control pool: {len(TFULL_CTRL)} -> {[t[0] for t in TFULL_CTRL]}")
tfull_ok = len(TFULL_CTRL) >= 3

# ---------------- B-none (single-token novel values) -------------------------------
BNONE = []
print("B-none: not constructed — no hygienic single-token novel value bank built "
      "this round; reported empty per PREREG allowance (constructible only if found).")

# ---------------- matched-pair obligations -----------------------------------------
def head_of(v):
    w = v.split(" ")
    return w[0] if len(w) > 1 else None

matched_pairs = {"same_head_diff_tail": [], "same_tail_diff_head": [],
                  "same_ntok_diff_T": [], "same_T_diff_B": []}
all_space = {v: (t, b, n) for cell in CORE_CELLS if cell[1] == "B-space"
             for v, bc, n, f in POOL[cell] for t, b in [(cell[0], cell[1])]}
by_head = collections.defaultdict(list)
for v in all_space:
    h = head_of(v)
    if h:
        by_head[h].append(v)
for h, vs in by_head.items():
    if len(vs) >= 2:
        matched_pairs["same_head_diff_tail"].append(vs)
by_tail = collections.defaultdict(list)
for v in all_space:
    parts = v.split(" ")
    if len(parts) > 1:
        by_tail[parts[-1]].append(v)
for t, vs in by_tail.items():
    if len(vs) >= 2:
        matched_pairs["same_tail_diff_head"].append(vs)
for L in ("short", "long"):
    for B in ("B-sub", "B-space"):
        a, b = POOL[("T-avail", B, L)], POOL[("T-sep", B, L)]
        if a and b:
            matched_pairs["same_ntok_diff_T"].append({"L": L, "B": B,
                "T-avail_example": a[0][0], "T-sep_example": b[0][0]})
for T in ("T-avail", "T-sep"):
    for L in ("short", "long"):
        a, b = POOL[(T, "B-sub", L)], POOL[(T, "B-space", L)]
        if a and b:
            matched_pairs["same_T_diff_B"].append({"T": T, "L": L,
                "B-sub_example": a[0][0], "B-space_example": b[0][0]})

print("\n== matched-pair obligations ==")
for k, v in matched_pairs.items():
    print(f"  {k}: {len(v)} groups")

# ---------------- ORTHOGONALITY PROOF (hard gate) -----------------------------------
def contrast_stats(cell_a, cell_b):
    a, b = POOL[cell_a], POOL[cell_b]
    if not a or not b:
        return None
    ntok_a = [r[2] for r in a]; ntok_b = [r[2] for r in b]
    mean_diff = abs(sum(ntok_a) / len(ntok_a) - sum(ntok_b) / len(ntok_b))
    return {"mean_ntok_a": sum(ntok_a) / len(ntok_a), "mean_ntok_b": sum(ntok_b) / len(ntok_b),
            "mean_ntok_diff": round(mean_diff, 3)}

ortho = {"T_contrasts": [], "B_contrasts": [], "L_contrasts": []}
no_family_dominance = True
for cell, items in POOL.items():
    heads = collections.Counter(r[0].split(" ")[0] for r in items if " " in r[0])
    if heads and max(heads.values()) > 2:
        no_family_dominance = False

gate = {"token_count_matched": True, "bigram_nonoverlap": True, "min_cell_size": cells_ok,
        "tfull_control_min3": tfull_ok, "no_lexical_family_dominance": no_family_dominance}

for B in ("B-sub", "B-space"):
    for L in ("short", "long"):
        a_cell, b_cell = ("T-avail", B, L), ("T-sep", B, L)
        st = contrast_stats(a_cell, b_cell)
        if st is None:
            continue
        st.update({"cellA": a_cell, "cellB": b_cell})
        if st["mean_ntok_diff"] > 0.5:
            gate["token_count_matched"] = False
        bigrams_a = [r[1] for r in POOL[a_cell]]
        bigrams_b = [r[1] for r in POOL[b_cell]]
        overlap = (min(bigrams_a) < 20) or (max(bigrams_b) > 0)
        st["bigram_range_A(T-avail)"] = [min(bigrams_a), max(bigrams_a)]
        st["bigram_range_B(T-sep)"] = [min(bigrams_b), max(bigrams_b)]
        st["nonoverlapping"] = not overlap
        if overlap:
            gate["bigram_nonoverlap"] = False
        ortho["T_contrasts"].append(st)

for T in ("T-avail", "T-sep"):
    for L in ("short", "long"):
        a_cell, b_cell = (T, "B-sub", L), (T, "B-space", L)
        st = contrast_stats(a_cell, b_cell)
        if st is None:
            continue
        st.update({"cellA": a_cell, "cellB": b_cell})
        if st["mean_ntok_diff"] > 0.5:
            gate["token_count_matched"] = False
        ortho["B_contrasts"].append(st)

for T in ("T-avail", "T-sep"):
    for B in ("B-sub", "B-space"):
        a_cell, b_cell = (T, B, "short"), (T, B, "long")
        st = contrast_stats(a_cell, b_cell)
        if st is None:
            continue
        st.update({"cellA": a_cell, "cellB": b_cell})
        ortho["L_contrasts"].append(st)

gate["all_core_cells_min5"] = cells_ok
HARD_GATE_PASS = (gate["token_count_matched"] and gate["bigram_nonoverlap"]
                   and gate["min_cell_size"] and gate["tfull_control_min3"]
                   and gate["no_lexical_family_dominance"])

print("\n== ORTHOGONALITY HARD GATE ==")
for k, v in gate.items():
    print(f"  {k}: {v}")
print(f"\nHARD GATE: {'PASS' if HARD_GATE_PASS else 'FAIL'}")

# ---------------- output ------------------------------------------------------------
manifest = {}
for cell, items in POOL.items():
    for v, bigram_ct, ntok, freqs in items:
        manifest[v] = {
            "value": v, "T": cell[0], "B": cell[1], "L": cell[2],
            "n_tokens": ntok, "token_ids": ctx_ids(v),
            "decoded_tokens": [tok.decode([i]) for i in ctx_ids(v)],
            "junction_bigram_count": bigram_ct,
            "junction_left_right_unigram_freq": freqs,
        }
for m, ntok in TFULL_CTRL:
    manifest[m] = {"value": m, "T": "T-full", "B": "B-full", "L": Lband(ntok) or "other",
                    "n_tokens": ntok, "token_ids": ctx_ids(m),
                    "decoded_tokens": [tok.decode([i]) for i in ctx_ids(m)],
                    "junction_bigram_count": None, "junction_left_right_unigram_freq": None}

# PREREG: "C-1b's 34 types are re-scored as bridges (known states; excluded from
# cells)" — the full 34, not just the 6 original sweep bridges.
_c1b_pools = json.load(open(os.path.join(HERE, "interference_pools.json")))
C1B_BRIDGES = sorted({p["value"] for ps in _c1b_pools["pools"].values() for p in ps}
                      | set(_c1b_pools["bridges"]))
assert len(C1B_BRIDGES) == 34, f"expected 34 C-1b bridge types, got {len(C1B_BRIDGES)}"

out = {
    "amendment": 0,
    "d80_output_stream_tokens": len(OUT_SEQ),
    "retrodiction_gate_pass": retro_ok,
    "hard_gate_pass": HARD_GATE_PASS,
    "hard_gate_detail": gate,
    "orthogonality": ortho,
    "matched_pairs": matched_pairs,
    "manifest": manifest,
    "tfull_controls": [m for m, _ in TFULL_CTRL],
    "bnone": BNONE,
    "c1b_bridges_rescored": C1B_BRIDGES,
    "rejected": rejected,
    "cell_counts": {str(k): len(v) for k, v in POOL.items()},
    "valid": HARD_GATE_PASS,
}
json.dump(out, open(os.path.join(HERE, "c3_pools.json"), "w"), indent=1)
print(f"\n-> c3_pools.json  {'VALID' if HARD_GATE_PASS else 'INVALID — STOP, do not proceed'}")
print(f"total accepted candidates: {len(manifest)}  rejected: {len(rejected)}")
