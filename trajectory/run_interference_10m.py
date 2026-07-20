# PREREG_token_coverage.md AMENDMENT 2 — the C-1b kernel (scale-10M base, full FT).
# ONE arm (D80, the sweep's exact d80 configuration, source-patched v2 recipe),
# TWO FT seeds in one session (categorical-flip guard), scored on the committed
# interference_eval instances (K=5, 34 held types across 5 interference classes).
# MANDATORY (AMENDMENT 1): per-item model outputs logged to outputs_if_seed{n}.jsonl.
# Decision rules (fixed pre-run, AMENDMENT 2): primary flip(I-iso)-flip(I-contain)
# >=50 CONFIRMED / <=15 REFUTED; secondary I-sib vs I-contain, I-xslot vs I-iso,
# I-template vs I-iso; seed-disagreeing types -> boundary variance, excluded from
# class rates. Venue: RunPod CUDA (or Kaggle T4); ~70 min on T4, less on A6000.
# Model/FT/scoring code verbatim from kaggle_sweep_10m.py (the validated sweep kernel).
import json, math, os, random, re, sys, time, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

dev = "cuda"; assert torch.cuda.is_available()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "trajectory"))
from slot_diversity_pools import ALG_TRAIN_80

pools = json.load(open(os.path.join(REPO, "trajectory", "interference_pools.json")))
assert pools["valid"]
CLASS_OF = {p["value"]: c for c, ps in pools["pools"].items() for p in ps}
CLASS_OF.update({v: pools["bridges"][v]["class"] for v in pools["bridges"]})
CONTAINED = {}   # I-contain type -> the contained trained value (substitution signature)
for c, ps in list(pools["pools"].items()) + [("bridges", list(pools["bridges"].values()))]:
    for p in (ps if isinstance(ps, list) else []):
        if p["class"] == "I-contain":
            mtc = re.search(r"contains trained '([^']+)'", p["why"])
            if mtc: CONTAINED[p["value"]] = mtc.group(1)
HELD_TYPES = sorted(CLASS_OF); assert len(HELD_TYPES) == 34
CLASSES = ["I-contain", "I-sib", "I-xslot", "I-template", "I-iso"]

tok = Tokenizer.from_file(os.path.join(REPO, "sft", "tokenizer.json"))
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")

# ---------------- scale-10M model (verbatim) ----------------
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
if not os.path.exists(BASE):
    url = "https://github.com/wwk5q8z6kk-bit/nano-lm/releases/download/v0.1/scale10m_pretrain.pt"
    print("downloading frozen base...", flush=True); urllib.request.urlretrieve(url, BASE)
base_sd = torch.load(BASE, map_location="cpu", weights_only=True)["m"]

# ---------------- D80 data build (source-patched v2; anchors asserted) ----------------
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

# ---------------- scoring with per-item output logging ----------------
def prompt_ids(user_content):
    return ([IMS] + tok.encode("user\n", add_special_tokens=False).ids
            + tok.encode(user_content, add_special_tokens=False).ids + [IME]
            + [IMS] + tok.encode("assistant\n", add_special_tokens=False).ids)

@torch.no_grad()
def generate(m, ids, max_new=64):
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

@torch.no_grad()
def score_seed(m, ft_seed, out_f):
    typ = {t: [0, 0] for t in HELD_TYPES}
    parsed = total = 0; seen_alg = [0, 0]
    for k in range(5):
        items = json.load(open(os.path.join(REPO, "trajectory", "interference_eval", f"if_m{k}.json")))
        for idx, it in enumerate(items):
            total += 1
            pids = prompt_ids(it["convo"][0]["content"])
            text = tok.decode(generate(m, pids)[len(pids):]).strip()
            mm = RE_.match(text)
            pred = dict(zip(FLDS, [g.strip() for g in mm.groups()])) if mm else None
            if mm: parsed += 1
            t = it["tuple"]["alg"]
            if it["held_values"] and it.get("held_alg_type"):
                hit = bool(pred) and pred["alg"] == t
                typ[t][0] += int(hit); typ[t][1] += 1
                out_f.write(json.dumps({                       # MANDATORY per-item log
                    "seed": ft_seed, "inst": k, "idx": idx, "type": t,
                    "class": it["interference_class"], "hit": hit,
                    "pred_alg": pred["alg"] if pred else None, "raw": text}) + "\n")
            elif (not it["held_values"]) and t != "none" and pred:
                seen_alg[0] += int(pred["alg"] == t); seen_alg[1] += 1
        print(f"  seed{ft_seed} m{k} done ({parsed}/{total} parsed)", flush=True)
    return {t: {"hits": v[0], "n": v[1], "recall": v[0] / max(1, v[1])} for t, v in typ.items()}, \
           {"parse": parsed / total, "seen_alg_recall": seen_alg[0] / max(1, seen_alg[1])}

results = {"seeds": {}}
for ft_seed in (0, 1):
    print(f"\n=== FT seed {ft_seed} (D80 arm) ===", flush=True)
    t0 = time.time()
    m = finetune(ft_seed)
    with open(f"outputs_if_seed{ft_seed}.jsonl", "w") as out_f:
        per_type, meta = score_seed(m, ft_seed, out_f)
    results["seeds"][str(ft_seed)] = {"per_type": per_type, **meta,
                                      "wall_secs": round(time.time() - t0)}
    flips = {t: per_type[t]["recall"] >= 0.5 for t in HELD_TYPES}
    print(f"  seed{ft_seed}: parse {meta['parse']:.0%} seen-alg {meta['seen_alg_recall']:.0%}")
    for c in CLASSES:
        ts = [t for t in HELD_TYPES if CLASS_OF[t] == c]
        print(f"  {c:11s}: " + "  ".join(f"{t}:{per_type[t]['recall']*100:.0f}" for t in ts), flush=True)
    del m; torch.cuda.empty_cache()

# ---------------- pre-registered decisions ----------------
s0, s1 = results["seeds"]["0"]["per_type"], results["seeds"]["1"]["per_type"]
flip = lambda pt, t: pt[t]["recall"] >= 0.5
agree = [t for t in HELD_TYPES if flip(s0, t) == flip(s1, t)]
boundary = [t for t in HELD_TYPES if t not in agree]
rate = {c: (lambda ts: sum(flip(s0, t) for t in ts) / max(1, len(ts)))(
            [t for t in agree if CLASS_OF[t] == c]) * 100 for c in CLASSES}
primary = rate["I-iso"] - rate["I-contain"]
verdict = ("containment-interference CONFIRMED" if primary >= 50
           else "REFUTED -> C-3 binding probe promoted" if primary <= 15 else "GRADED")
# substitution signature among I-contain misses (both seeds' logs)
sub = [0, 0]
for sd in (0, 1):
    for line in open(f"outputs_if_seed{sd}.jsonl"):
        r = json.loads(line)
        if r["class"] == "I-contain" and not r["hit"] and r["type"] in CONTAINED:
            sub[1] += 1; sub[0] += int(r["pred_alg"] == CONTAINED[r["type"]])
results["decisions"] = {
    "class_flip_rates_pct": rate, "primary_iso_minus_contain_pts": primary,
    "verdict": verdict, "boundary_variance_types": boundary,
    "secondary": {"sib_minus_contain": rate["I-sib"] - rate["I-contain"],
                  "xslot_minus_iso": rate["I-xslot"] - rate["I-iso"],
                  "template_minus_iso": rate["I-template"] - rate["I-iso"]},
    "substitution_by_contained_value": {"n_misses": sub[1], "n_substituted": sub[0]},
    "bridge_retrodiction": {t: {"s0": s0[t]["recall"], "s1": s1[t]["recall"]}
                            for t in ("ragweed pollen", "sulfa drugs", "bee stings",
                                      "ibuprofen", "wool", "strawberries")}}
print(f"\nCLASS FLIP RATES: {rate}", flush=True)
print(f"PRIMARY (iso - contain): {primary:.0f} pts -> {verdict}", flush=True)
print(f"boundary-variance types: {boundary}", flush=True)
print(f"I-contain substitution signature: {sub[0]}/{sub[1]} misses = contained value", flush=True)
json.dump(results, open("results_interference_10m.json", "w"), indent=1)
print("results -> results_interference_10m.json", flush=True)
