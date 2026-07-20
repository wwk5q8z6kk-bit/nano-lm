# PREREG_slot_diversity.md — the sweep kernel (primary tier: scale-10M base, full FT).
# One session runs all 4 arms: d5 / d20 / d80 (ALG_TRAIN diversity) + d20pos (summary
# order MED<->ALG swapped — the position control). Per arm: build v2-recipe data with
# the arm's frozen pool (source-patched, asserted), full-FT from the frozen
# scale10m_pretrain.pt (v0.1 release asset), score on the committed sweep_eval
# instances with the PER-TYPE flip table as primary output. Decision rules in PREREG.
# Venue: Kaggle T4 (~2.3h all arms). FT_SEED env (default 0).
import json, math, os, random, re, sys, time, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

FT_SEED = int(os.environ.get("FT_SEED", 0))
dev = "cuda"; assert torch.cuda.is_available()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "trajectory"))
from slot_diversity_pools import HELD_ALG, ALG_TRAIN_5, ALG_TRAIN_20, ALG_TRAIN_80

tok = Tokenizer.from_file(os.path.join(REPO, "sft", "tokenizer.json"))
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")

# ---------------- scale-10M model (verbatim dims from kaggle_scale_test.py) ----------------
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

# ---------------- per-arm data build (source-patched v2 recipe; anchors asserted) ----------------
V2_SRC = open(os.path.join(REPO, "scribe", "build_scribe_data_v2.py")).read()
ALG_LINE = 'ALG_TRAIN = ["penicillin", "peanuts", "pollen", "latex", "shellfish"]'
SUM_FRAG = "MED: {t['med'] or 'none'} | ALG: {t['alg'] or 'none'}"
assert ALG_LINE in V2_SRC and SUM_FRAG in V2_SRC, "v2 source anchors moved — re-verify"

def build_convos(pool, pos_swap):
    src = V2_SRC.replace(ALG_LINE, "ALG_TRAIN = " + repr(list(pool)))
    if pos_swap:
        src = src.replace(SUM_FRAG, "ALG: {t['alg'] or 'none'} | MED: {t['med'] or 'none'}")
    prefix = src.split('tok = Tokenizer.from_file')[0].replace("from tokenizers import Tokenizer", "")
    ns = {}; exec(compile(prefix, "v2[arm]", "exec"), ns)
    convos = ns["convos"]; assert len(convos) == 12000
    return convos

def render_ids(convo):
    ids, mask = [], []
    for msg in convo:
        head = tok.encode(f"{msg['role']}\n", add_special_tokens=False).ids
        body = tok.encode(msg["content"], add_special_tokens=False).ids
        seg = [IMS] + head + body + [IME]; tf = 1 if msg["role"] == "assistant" else 0
        for j, tkn in enumerate(seg): ids.append(tkn); mask.append(1 if (tf and j >= 1 + len(head)) else 0)
    return ids, mask

# ---------------- FT (verbatim scale-10M scribe recipe: batch 32, 3 epochs, LR 1e-4) ----------------
def finetune(convos):
    torch.manual_seed(FT_SEED); random.seed(FT_SEED)
    m = GPT(); m.load_state_dict(base_sd); m.to(dev)
    X, M = [], []
    for c in convos:
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

# ---------------- scoring (native ChatML/argmax; per-type table primary) ----------------
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

RE_STD = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")
RE_POS = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| ALG: (.+?) \| MED: (.+?)$")
F_STD = ["cc", "dur", "sev", "med", "alg"]; F_POS = ["cc", "dur", "sev", "alg", "med"]

@torch.no_grad()
def score(m, items, pos_swap):
    RE_, FLDS = (RE_POS, F_POS) if pos_swap else (RE_STD, F_STD)
    parsed = 0; hc = ht = sc = st_ = 0
    typ = {t: [0, 0] for t in HELD_ALG}                       # per-type: [hits, total]
    seen_alg = [0, 0]
    for it in items:
        text = tok.decode(generate(m, prompt_ids(it["convo"][0]["content"]))[len(prompt_ids(it["convo"][0]["content"])):]).strip()
        mm = RE_.match(text)
        if not mm: continue
        parsed += 1; pred = dict(zip(FLDS, [g.strip() for g in mm.groups()]))
        for f in F_STD:
            hit = int(pred[f] == it["tuple"][f])
            if it["held_values"]: ht += 1; hc += hit
            else: st_ += 1; sc += hit
        t = it["tuple"]["alg"]
        if it["held_values"] and it.get("held_alg_type"):
            typ[t][0] += int(pred["alg"] == t); typ[t][1] += 1
        elif (not it["held_values"]) and t != "none":
            seen_alg[0] += int(pred["alg"] == t); seen_alg[1] += 1
    return {"parse": parsed / len(items),
            "diluted_gap": (sc / max(1, st_) - hc / max(1, ht)) * 100,
            "per_type": {t: {"hits": v[0], "n": v[1]} for t, v in typ.items()},
            "seen_alg_recall": seen_alg[0] / max(1, seen_alg[1])}

ARMS = [("d5", ALG_TRAIN_5, False), ("d20", ALG_TRAIN_20, False),
        ("d80", ALG_TRAIN_80, False), ("d20pos", ALG_TRAIN_20, True)]
summary = {}
for arm, pool, pos in ARMS:
    print(f"\n=== ARM {arm} (|pool|={len(pool)}, pos_swap={pos}) ===", flush=True)
    t0 = time.time()
    m = finetune(build_convos(pool, pos))
    inst_tag = "d20" if arm == "d20pos" else arm
    per_inst = []
    for k in range(5):
        items = json.load(open(os.path.join(REPO, "trajectory", "sweep_eval", f"{inst_tag}_m{k}.json")))
        r = score(m, items, pos)
        per_inst.append(r)
        tt = {t: f"{v['hits']}/{v['n']}" for t, v in r["per_type"].items()}
        print(f"  m{k} parse {r['parse']:.0%} dil {r['diluted_gap']:.1f} seenalg {r['seen_alg_recall']:.0%} types {tt}", flush=True)
    # pooled per-type recall over K=5 (per-type flip table)
    pooled = {t: {"hits": sum(r["per_type"][t]["hits"] for r in per_inst),
                  "n": sum(r["per_type"][t]["n"] for r in per_inst)} for t in HELD_ALG}
    for t in pooled: pooled[t]["recall"] = pooled[t]["hits"] / max(1, pooled[t]["n"])
    mean_held = float(np.mean([pooled[t]["recall"] for t in HELD_ALG])) * 100
    summary[arm] = {"pool_size": len(pool), "pos_swap": pos, "ft_seed": FT_SEED,
                    "per_type_pooled": pooled, "mean_held_type_recall_pct": mean_held,
                    "per_instance": per_inst, "wall_secs": round(time.time() - t0)}
    print(f"  {arm}: mean held-type recall {mean_held:.1f}%  " +
          " ".join(f"{t}:{pooled[t]['recall']*100:.0f}" for t in HELD_ALG), flush=True)
    del m; torch.cuda.empty_cache()

# pre-registered decisions
d_eff = summary["d80"]["mean_held_type_recall_pct"] - summary["d5"]["mean_held_type_recall_pct"]
pos_delta = abs(summary["d20pos"]["mean_held_type_recall_pct"] - summary["d20"]["mean_held_type_recall_pct"])
mono = summary["d5"]["mean_held_type_recall_pct"] <= summary["d20"]["mean_held_type_recall_pct"] + 5 <= summary["d80"]["mean_held_type_recall_pct"] + 10
verdict = "H-slot SUPPORTED" if d_eff >= 30 else ("H-slot REFUTED (H-string favored)" if d_eff <= 10 else "GRADED (10-30)")
print(f"\nDIVERSITY EFFECT (D80-D5 mean held-type recall): {d_eff:.1f} pts -> {verdict}", flush=True)
print(f"POSITION |D20pos-D20|: {pos_delta:.1f} pts -> {'position innocent' if pos_delta <= 5 else 'POSITION CONFOUND LIVE'}", flush=True)
summary["decisions"] = {"diversity_effect_pts": d_eff, "verdict": verdict,
                        "position_delta_pts": pos_delta, "monotonic_ish": bool(mono)}
json.dump(summary, open("results_sweep_10m.json", "w"), indent=1)
print("results -> results_sweep_10m.json", flush=True)
