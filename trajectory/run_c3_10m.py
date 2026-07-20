# PREREG_C3_binding_probe.md — the C-3 kernel (scale-10M base, full FT).
# ONE arm (D80, byte-identical to C-1b), THREE FT seeds (0,1,2) per PREREG (C-1b
# measured 24% seed-variant types -> flip states are seed-MAJORITY here, not
# pairwise-agree). MANDATORY: per-item outputs incl. generated token-id sequence,
# enabling truncation position / head-only / tail-only / full-copy / wrong-trained-
# tail / morph-normalization / omission / garble classification.
# Decision rules (fixed pre-run, PREREG_C3_binding_probe.md):
#   H-transition: flip(T-avail)-flip(T-sep), matched B,L >=40 SUPPORTED / <=15 REFUTED
#   H-boundary:   flip(B-sub)-flip(B-space), matched T,L >=40 SUPPORTED / <=15 REFUTED
#   H-length:     flip(short)-flip(long),    matched T,B >=40 SUPPORTED / <=15 REFUTED
#   truncation-locus: >=60% of B-space misses truncate exactly at the whitespace junction
#   T-full control must reproduce >=90% (I-xslot retrodiction) or the run is VOID
#   all three refuted + >=20% seed-unstable -> H-stochastic SUPPORTED -> promote
#     a representation/attention probe (Stage M), not another lexical account
# Venue: RunPod A6000 (or Kaggle T4). Model/FT/scoring code verbatim pattern from
# run_interference_10m.py (the validated C-1b kernel).
import json, math, os, random, re, sys, time, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

dev = "cuda"; assert torch.cuda.is_available()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "trajectory"))
from slot_diversity_pools import ALG_TRAIN_80

pools = json.load(open(os.path.join(REPO, "trajectory", "c3_pools.json")))
assert pools["valid"], "c3_pools.json is INVALID — do not launch"
MANIFEST = pools["manifest"]

LABEL_OF = {}
for v, m in MANIFEST.items():
    LABEL_OF[v] = {"kind": "tfull" if m["T"] == "T-full" else "cell",
                    "T": m["T"], "B": m["B"], "L": m["L"]}
for v in pools["c1b_bridges_rescored"]:
    LABEL_OF[v] = {"kind": "bridge", "T": None, "B": None, "L": None}
HELD_TYPES = sorted(LABEL_OF)
assert len(HELD_TYPES) == 93

CELL_TYPES = {k: [v for v in HELD_TYPES if LABEL_OF[v]["kind"] == "cell"
                  and LABEL_OF[v]["T"] == k[0] and LABEL_OF[v]["B"] == k[1]
                  and LABEL_OF[v]["L"] == k[2]]
              for k in [(t, b, l) for t in ("T-avail", "T-sep")
                        for b in ("B-sub", "B-space") for l in ("short", "long")]}
TFULL_TYPES = [v for v in HELD_TYPES if LABEL_OF[v]["kind"] == "tfull"]
BRIDGE_TYPES = [v for v in HELD_TYPES if LABEL_OF[v]["kind"] == "bridge"]

tok = Tokenizer.from_file(os.path.join(REPO, "sft", "tokenizer.json"))
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")

# ---------------- scale-10M model (verbatim, byte-identical to C-1b) ----------------
V = 4098
d, L, H, KV, hd, ff, S = 320, 8, 8, 2, 40, 864, 512

def rope(q, k):
    t = torch.arange(S, device=dev, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    f = torch.outer(t, inv); cos, sin = f.cos()[None, None], f.sin()[None, None]
    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return rot(q), rot(k)

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1, s.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        s.q, s.k, s.v, s.o = (nn.Linear(d, H*hd, bias=False), nn.Linear(d, KV*hd, bias=False),
                              nn.Linear(d, KV*hd, bias=False), nn.Linear(H*hd, d, bias=False))
        s.g, s.u, s.dn = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)
    def forward(s, x):
        B = x.shape[0]; h = s.n1(x)
        q = s.q(h).view(B, S, H, hd).transpose(1, 2); k = s.k(h).view(B, S, KV, hd).transpose(1, 2); v = s.v(h).view(B, S, KV, hd).transpose(1, 2)
        q, k = rope(q, k)
        k, v = k.repeat_interleave(H//KV, 1), v.repeat_interleave(H//KV, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(B, S, H*hd))
        h = s.n2(x)
        return x + s.dn(F.silu(s.g(h)) * s.u(h))

class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.blocks = nn.ModuleList(Block() for _ in range(L)); s.nf = nn.RMSNorm(d)
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return F.linear(s.nf(h), s.emb.weight)

BASE = "scale10m_pretrain.pt"
BASE_URL = "https://github.com/wwk5q8z6kk-bit/nano-lm/releases/download/v0.1/scale10m_pretrain.pt"
if not os.path.exists(BASE):
    print("downloading frozen base...", flush=True); urllib.request.urlretrieve(BASE_URL, BASE)
import hashlib
BASE_SHA256 = hashlib.sha256(open(BASE, "rb").read()).hexdigest()
base_sd = torch.load(BASE, map_location="cpu", weights_only=True)["m"]

# ---------------- D80 data build (source-patched v2; byte-identical to C-1b) --------
V2_SRC = open(os.path.join(REPO, "scribe", "build_scribe_data_v2.py")).read()
ALG_LINE = 'ALG_TRAIN = ["penicillin", "peanuts", "pollen", "latex", "shellfish"]'
assert ALG_LINE in V2_SRC, "v2 source anchor moved — re-verify"
src = V2_SRC.replace(ALG_LINE, "ALG_TRAIN = " + repr(list(ALG_TRAIN_80)))
prefix = src.split('tok = Tokenizer.from_file')[0].replace("from tokenizers import Tokenizer", "")
_ns = {}; exec(compile(prefix, "v2[d80]", "exec"), _ns)
CONVOS = _ns["convos"]; assert len(CONVOS) == 12000

def render_ids(convo):
    ids, mask = [], []
    for msg in convo:
        head = tok.encode(f"{msg['role']}\n", add_special_tokens=False).ids
        body = tok.encode(msg["content"], add_special_tokens=False).ids
        seg = [IMS] + head + body + [IME]; tf = 1 if msg["role"] == "assistant" else 0
        for j, tkn in enumerate(seg): ids.append(tkn); mask.append(1 if (tf and j >= 1 + len(head)) else 0)
    return ids, mask

def finetune(ft_seed):
    torch.manual_seed(ft_seed); random.seed(ft_seed)
    m = GPT(); m.load_state_dict(base_sd); m.to(dev)
    X, M = [], []
    for c in CONVOS:
        ids, mask = render_ids(c)
        if len(ids) > S: continue
        X.append(ids + [0] * (S - len(ids))); M.append(mask + [0] * (S - len(mask)))
    X = torch.tensor(np.array(X, dtype=np.int64)); M = torch.tensor(np.array(M, dtype=np.int64))
    N = X.shape[0]; BATCH = 32; FSTEPS = (N // BATCH) * 3; FWARM = int(0.03 * FSTEPS); FLR = 1e-4
    decay = [p for p in m.parameters() if p.dim() >= 2]; nodecay = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=FLR, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda"); perm = torch.randperm(N)
    def flr_at(t):
        if t < FWARM: return FLR * t / max(1, FWARM)
        p = (t - FWARM) / max(1, FSTEPS - FWARM); return FLR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    for step in range(1, FSTEPS + 1):
        for g in opt.param_groups: g["lr"] = flr_at(step)
        i0 = (step * BATCH) % (N - BATCH); idx = perm[i0:i0 + BATCH]
        x, msk = X[idx].to(dev), M[idx].to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = m(x)[:, :-1]; tgt = x[:, 1:]; mtgt = msk[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
            loss = (ce * mtgt).sum() / mtgt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % 400 == 0: print(f"  ft {step}/{FSTEPS} loss {loss.item():.3f}", flush=True)
    m.eval(); return m

# ---------------- scoring with mandatory extended per-item logging -------------------
def prompt_ids(user_content):
    return ([IMS] + tok.encode("user\n", add_special_tokens=False).ids
            + tok.encode(user_content, add_special_tokens=False).ids + [IME]
            + [IMS] + tok.encode("assistant\n", add_special_tokens=False).ids)

@torch.no_grad()
def generate_ids(m, ids, max_new=64):
    ids = list(ids)
    for _ in range(max_new):
        if len(ids) >= S: break
        x = torch.tensor([ids + [0] * (S - len(ids))], device=dev)
        nxt = int(m(x)[0, len(ids) - 1].argmax())
        if nxt == IME: break
        ids.append(nxt)
    return ids

RE_ = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")
FLDS = ["cc", "dur", "sev", "med", "alg"]


def _edit_dist(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def classify_output(true_val, pred_str, pred_ids_alg):
    """Mechanical error-taxonomy classification for one held ALG prediction.
    pred_ids_alg: the token ids of the model's generated ALG span (post 'ALG: ',
    pre terminator) — used for truncation-position / head-tail analysis."""
    if pred_str is None:
        return "parse_failure", None
    p, t = pred_str.strip().lower(), true_val.strip().lower()
    if p == t:
        return "exact_copy", None
    true_ids = tok.encode(" " + true_val, add_special_tokens=False).ids
    n = len(true_ids)
    # strict-prefix truncation (token-level)
    if len(pred_ids_alg) < n and pred_ids_alg == true_ids[:len(pred_ids_alg)]:
        return "truncation", len(pred_ids_alg)
    words_true = true_val.split()
    if len(words_true) > 1:
        head_true, tail_true = " ".join(words_true[:-1]), words_true[-1]
        if p == head_true.lower():
            return "head_only", len(tok.encode(" " + head_true, add_special_tokens=False).ids)
        if p == tail_true.lower():
            return "tail_only", None
    if p == "none":
        return "omission", None
    # morphological near-miss: small edit distance relative to length (plural/
    # suffix variants etc.), checked before the generic substitution/garble fallback
    if p and _edit_dist(p, t) <= max(1, len(t) // 6):
        return "morphological_near_miss", None
    if p and not re.fullmatch(r"[a-zA-Z ]+", p):
        return "garble", None
    return "substitution", None


@torch.no_grad()
def score_seed(m, ft_seed, out_f):
    typ = {t: [0, 0] for t in HELD_TYPES}
    parsed = total = 0; seen_alg = [0, 0]
    for k in range(5):
        items = json.load(open(os.path.join(REPO, "trajectory", "c3_eval", f"c3_m{k}.json")))
        for idx, it in enumerate(items):
            total += 1
            pids = prompt_ids(it["convo"][0]["content"])
            gen = generate_ids(m, pids)
            gen_new = gen[len(pids):]
            text = tok.decode(gen_new).strip()
            mm = RE_.match(text)
            pred = dict(zip(FLDS, [g.strip() for g in mm.groups()])) if mm else None
            if mm: parsed += 1
            t = it["tuple"]["alg"]
            if it["held_values"] and it.get("held_alg_type"):
                hit = bool(pred) and pred["alg"] == t
                typ[t][0] += int(hit); typ[t][1] += 1
                # isolate the ALG-field token span from the raw generation for
                # truncation-position analysis (best-effort: re-encode the matched
                # group text; exact for well-formed outputs, which is what the
                # truncation/head-tail analysis needs).
                alg_str = pred["alg"] if pred else None
                alg_ids = tok.encode(" " + alg_str, add_special_tokens=False).ids if alg_str else []
                err_class, trunc_pos = (None, None) if hit else classify_output(t, alg_str, alg_ids)
                out_f.write(json.dumps({
                    "seed": ft_seed, "inst": k, "idx": idx, "type": t,
                    "kind": LABEL_OF[t]["kind"], "T": LABEL_OF[t]["T"],
                    "B": LABEL_OF[t]["B"], "Lb": LABEL_OF[t]["L"],
                    "hit": hit, "pred_alg": alg_str, "raw": text,
                    "gen_token_ids": gen_new, "error_class": err_class,
                    "truncation_token_pos": trunc_pos}) + "\n")
            elif (not it["held_values"]) and t != "none" and pred:
                seen_alg[0] += int(pred["alg"] == t); seen_alg[1] += 1
        print(f"  seed{ft_seed} m{k} done ({parsed}/{total} parsed)", flush=True)
    return {t: {"hits": v[0], "n": v[1], "recall": v[0] / max(1, v[1])} for t, v in typ.items()}, \
           {"parse": parsed / total, "seen_alg_recall": seen_alg[0] / max(1, seen_alg[1])}


results = {"seeds": {}, "base_checkpoint_sha256": BASE_SHA256, "base_checkpoint_url": BASE_URL}
for ft_seed in (0, 1, 2):
    print(f"\n=== FT seed {ft_seed} (D80 arm) ===", flush=True)
    t0 = time.time()
    m = finetune(ft_seed)
    with open(f"outputs_c3_seed{ft_seed}.jsonl", "w") as out_f:
        per_type, meta = score_seed(m, ft_seed, out_f)
    results["seeds"][str(ft_seed)] = {"per_type": per_type, **meta,
                                      "wall_secs": round(time.time() - t0)}
    print(f"  seed{ft_seed}: parse {meta['parse']:.0%} seen-alg {meta['seen_alg_recall']:.0%}", flush=True)
    del m; torch.cuda.empty_cache()

# ---------------- pre-registered decisions (seed-MAJORITY over 3 seeds) -------------
s = [results["seeds"][str(i)]["per_type"] for i in range(3)]
flip = lambda si, t: s[si][t]["recall"] >= 0.5
def majority_flip(t):
    votes = [flip(i, t) for i in range(3)]
    return sum(votes) >= 2, votes

flip_state, unstable = {}, []
for t in HELD_TYPES:
    fs, votes = majority_flip(t)
    flip_state[t] = fs
    if len(set(votes)) > 1:
        unstable.append(t)

def cell_rate(cell_key_filter):
    ts = [t for t in HELD_TYPES if cell_key_filter(t) and t not in unstable]
    if not ts: return None, 0
    return sum(flip_state[t] for t in ts) / len(ts) * 100, len(ts)

def verdict(delta):
    if delta is None: return "UNRESOLVED (empty contrast)"
    if delta >= 40: return "SUPPORTED"
    if delta <= 15: return "REFUTED"
    return "UNRESOLVED"

# H-transition: T-avail vs T-sep, matched B,L (average across the 4 B,L combos)
trans_deltas = []
for B in ("B-sub", "B-space"):
    for Lb in ("short", "long"):
        ra, na = cell_rate(lambda t, B=B, Lb=Lb: LABEL_OF[t]["kind"] == "cell" and LABEL_OF[t]["T"] == "T-avail" and LABEL_OF[t]["B"] == B and LABEL_OF[t]["L"] == Lb)
        rs, ns_ = cell_rate(lambda t, B=B, Lb=Lb: LABEL_OF[t]["kind"] == "cell" and LABEL_OF[t]["T"] == "T-sep" and LABEL_OF[t]["B"] == B and LABEL_OF[t]["L"] == Lb)
        if ra is not None and rs is not None:
            trans_deltas.append((B, Lb, ra, rs, ra - rs))
H_transition_delta = (sum(d[4] for d in trans_deltas) / len(trans_deltas)) if trans_deltas else None

boundary_deltas = []
for T in ("T-avail", "T-sep"):
    for Lb in ("short", "long"):
        ra, _ = cell_rate(lambda t, T=T, Lb=Lb: LABEL_OF[t]["kind"] == "cell" and LABEL_OF[t]["T"] == T and LABEL_OF[t]["B"] == "B-sub" and LABEL_OF[t]["L"] == Lb)
        rs, _ = cell_rate(lambda t, T=T, Lb=Lb: LABEL_OF[t]["kind"] == "cell" and LABEL_OF[t]["T"] == T and LABEL_OF[t]["B"] == "B-space" and LABEL_OF[t]["L"] == Lb)
        if ra is not None and rs is not None:
            boundary_deltas.append((T, Lb, ra, rs, ra - rs))
H_boundary_delta = (sum(d[4] for d in boundary_deltas) / len(boundary_deltas)) if boundary_deltas else None

length_deltas = []
for T in ("T-avail", "T-sep"):
    for B in ("B-sub", "B-space"):
        ra, _ = cell_rate(lambda t, T=T, B=B: LABEL_OF[t]["kind"] == "cell" and LABEL_OF[t]["T"] == T and LABEL_OF[t]["B"] == B and LABEL_OF[t]["L"] == "short")
        rs, _ = cell_rate(lambda t, T=T, B=B: LABEL_OF[t]["kind"] == "cell" and LABEL_OF[t]["T"] == T and LABEL_OF[t]["B"] == B and LABEL_OF[t]["L"] == "long")
        if ra is not None and rs is not None:
            length_deltas.append((T, B, ra, rs, ra - rs))
H_length_delta = (sum(d[4] for d in length_deltas) / len(length_deltas)) if length_deltas else None

# T-full control retrodiction (must reproduce >=90% or run VOID)
tfull_rate, tfull_n = cell_rate(lambda t: LABEL_OF[t]["kind"] == "tfull")
run_void = tfull_rate is not None and tfull_rate < 90

# truncation-locus check: >=60% of B-space misses truncate exactly at the ws junction
b_space_misses_trunc = b_space_misses_total = 0
for sd in range(3):
    for line in open(f"outputs_c3_seed{sd}.jsonl"):
        r = json.loads(line)
        if r["hit"] or r["B"] != "B-space":
            continue
        b_space_misses_total += 1
        if r["error_class"] in ("truncation", "head_only"):
            b_space_misses_trunc += 1
trunc_locus_rate = (b_space_misses_trunc / b_space_misses_total * 100) if b_space_misses_total else None

seed_unstable_pct = len(unstable) / len(HELD_TYPES) * 100
h_stochastic = (verdict(H_transition_delta) == "REFUTED" and verdict(H_boundary_delta) == "REFUTED"
                and verdict(H_length_delta) == "REFUTED" and seed_unstable_pct >= 20)

results["decisions"] = {
    "H_transition": {"delta_pts": H_transition_delta, "verdict": verdict(H_transition_delta), "per_BL": trans_deltas},
    "H_boundary": {"delta_pts": H_boundary_delta, "verdict": verdict(H_boundary_delta), "per_TL": boundary_deltas},
    "H_length": {"delta_pts": H_length_delta, "verdict": verdict(H_length_delta), "per_TB": length_deltas},
    "T_full_control": {"rate_pct": tfull_rate, "n": tfull_n, "run_void": run_void},
    "truncation_locus_check": {"rate_pct": trunc_locus_rate, "n_misses": b_space_misses_total,
                               "confirms_boundary_locus": (trunc_locus_rate or 0) >= 60},
    "seed_unstable_pct": seed_unstable_pct, "seed_unstable_types": unstable,
    "H_stochastic_supported": h_stochastic,
}
print(f"\nRUN_VOID: {run_void} (T-full retrodiction {tfull_rate})")
print(f"H-transition: {H_transition_delta} -> {verdict(H_transition_delta)}")
print(f"H-boundary:   {H_boundary_delta} -> {verdict(H_boundary_delta)}")
print(f"H-length:     {H_length_delta} -> {verdict(H_length_delta)}")
print(f"truncation-locus: {trunc_locus_rate}% (>=60 confirms)")
print(f"seed-unstable: {seed_unstable_pct:.0f}% ; H-stochastic supported: {h_stochastic}")
json.dump(results, open("results_c3_10m.json", "w"), indent=1)
print("results -> results_c3_10m.json")
