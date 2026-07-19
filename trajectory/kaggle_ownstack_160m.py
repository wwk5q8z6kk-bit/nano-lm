# Stage T-v2 / Paper-2 deconfounder — OWN-STACK ~160M within-stack scale rung.
# PRE-REGISTERED in trajectory/PREREG_ownstack_160m.md. Built 2026-07-18; NOT executed
# (owner runs on Kaggle T4). Separates SCALE from STACK: same architecture family,
# tokenizer, pretraining recipe, and scribe finetune as the 3.15M/10M anchors — only
# parameter count changes. If the gap stays ~18/~85 at 160M (where Pythia-160M is 3.5),
# the reduction is a STACK effect, not scale.
#
# PINNED CONFIG (PREREG): d=1024 L=14 H=16 KV=4 hd=64 ff=2752 -> ~159M params.
# Pretrain: 200M FineWeb tokens (IDENTICAL recipe to nano/scale; heavily under-Chinchilla
#   for 160M by design — the "same recipe" comparison; TARGET_TOKENS env overrides for the
#   optional ~3.2B Chinchilla control). Scribe: full FT, v2 recipe, 3 epochs, LR 1e-4.
# Scoring: multi-instance m0-m4 (diluted + clean, mean±SD) + inst0, then the decision rule.
#
# VALIDATION PLAN (run these checks on the output, before trusting the gap):
#   1. base control (pre-scribe 160M) must FAIL parse (<50%) — discrimination.
#   2. pretrain val loss should land in a sane band (nano 3.96 / scale 3.28 at 200M; a
#      160M on 200M tokens is under-trained, so expect HIGHER, not lower — report it).
#   3. scribe parse rate >= 90% (else the model is too under-trained to score the gap;
#      that itself is the "undertraining" caveat, report parse rate alongside the gap).
#   4. apply the pre-registered decision rule on the full-FT diluted gap_mean.
#   5. report BOTH diluted (dialogue-level, ladder-comparable) and clean (value-level,
#      undilute-comparable) gaps with across-instance SD.
#   6. LoRA arm (2x2 control): disabled by default (RUN_LORA=0) — see note at the bottom;
#      validate the peft-wrap interactively before spending quota on it.
#
# Venue cell (Kaggle T4, Internet on; push master first):
#   %%bash
#   cd /kaggle/working && rm -rf nano-lm && git clone -q https://github.com/wwk5q8z6kk-bit/nano-lm
#   cd nano-lm && git checkout -q master && pip install -q datasets tokenizers
#   python trajectory/kaggle_ownstack_160m.py
import json, math, os, random, re, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

torch.manual_seed(0); random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
assert dev == "cuda", "enable the GPU accelerator"
_cap = torch.cuda.get_device_capability()
assert _cap >= (7, 0), f"{torch.cuda.get_device_name()} cap {_cap} < 7.0 (P100 unsupported); use T4"
print(f"GPU {torch.cuda.get_device_name()} cap {_cap}", flush=True)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = Tokenizer.from_file(os.path.join(REPO, "sft", "tokenizer.json"))
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")

# ---------------- model: ~160M own-stack (pinned dims) ----------------
V = 4098
d, L, H, KV, hd, ff, S = 1024, 14, 16, 4, 64, 2752, 512

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
        for p in s.parameters():
            if p.dim() >= 2: nn.init.normal_(p, std=0.02 / math.sqrt(2 * L))
        nn.init.normal_(s.emb.weight, std=0.02)
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return F.linear(s.nf(h), s.emb.weight)

# ---------------- phase 1: pretrain (FineWeb; resume-safe; identical recipe) ----------------
TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", 200_000_000))   # 200M default; set ~3.2B for Chinchilla control
BATCH, WARM_FRAC, FLOOR = 32, 0.03, 0.1
PEAK_LR = 3e-3 * 192 / d      # width-scaling rule (nano 3e-3@d192 -> scale 1.8e-3@d320 -> 160M 5.6e-4@d1024)
STEPS = TARGET_TOKENS // (BATCH * S)
CKPT = "ownstack160m_pretrain.pt"; SHARD = "shard_ownstack.npy"

if not os.path.exists(SHARD):
    print("streaming FineWeb...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    ids, n_docs = [], 0
    for doc in ds:
        ids.extend(tok.encode(doc["text"], add_special_tokens=False).ids); ids.append(0); n_docs += 1
        if len(ids) >= TARGET_TOKENS + 2_000_000: break
        if n_docs % 20000 == 0: print(f"  {n_docs} docs, {len(ids)/1e6:.0f}M tok", flush=True)
    np.save(SHARD, np.array(ids, dtype=np.uint16)); print(f"shard {len(ids)/1e6:.0f}M tok", flush=True)

shard = np.load(SHARD)
val_ids = torch.tensor(shard[-1_000_000:].astype(np.int64)); train_ids = torch.tensor(shard[:-1_000_000].astype(np.int64))
m = GPT().to(dev)
print(f"params={sum(p.numel() for p in m.parameters())/1e6:.2f}M  steps={STEPS}  peak_lr={PEAK_LR:.2e}", flush=True)
decay = [p for p in m.parameters() if p.dim() >= 2]; nodecay = [p for p in m.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)
scaler = torch.amp.GradScaler("cuda"); start_step = 0
if os.path.exists(CKPT):
    st = torch.load(CKPT, map_location=dev, weights_only=True)
    m.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); scaler.load_state_dict(st["s"]); start_step = st["step"]
    print(f"resumed step {start_step}", flush=True)
WARM = int(WARM_FRAC * STEPS)
def lr_at(t):
    if t < WARM: return PEAK_LR * t / max(1, WARM)
    p = (t - WARM) / max(1, STEPS - WARM); return PEAK_LR * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * p)))
def batch_from(ids_t):
    i = torch.randint(0, len(ids_t) - S - 1, (BATCH,)); return torch.stack([ids_t[j:j+S] for j in i]).to(dev)
def ce_loss(logits, tgt):
    ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1)); z = torch.logsumexp(logits.float(), -1)
    return ce + 1e-4 * (z ** 2).mean()
if start_step < STEPS:
    t0 = time.time()
    for step in range(start_step + 1, STEPS + 1):
        for g in opt.param_groups: g["lr"] = lr_at(step)
        x = batch_from(train_ids)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = ce_loss(m(x)[:, :-1], x[:, 1:])
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % 200 == 0:
            tput = (step - start_step) * BATCH * S / (time.time() - t0) / 1e3
            print(f"step {step}/{STEPS} loss {loss.item():.3f} gnorm {gn.item():.2f} {tput:.0f}k tok/s", flush=True)
        if step % 1000 == 0:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                vl = ce_loss(m(batch_from(val_ids))[:, :-1], batch_from(val_ids)[:, 1:]).item()
            print(f"  == val loss {vl:.3f} ==", flush=True)
            torch.save({"m": m.state_dict(), "o": opt.state_dict(), "s": scaler.state_dict(), "step": step}, CKPT)
    torch.save({"m": m.state_dict(), "o": opt.state_dict(), "s": scaler.state_dict(), "step": STEPS}, CKPT)
    print("pretrain done", flush=True)
base_sd = {k: v.clone() for k, v in m.state_dict().items()}
if os.environ.get("PHASE") == "pretrain":          # phase-split / throughput-probe support
    print("PHASE=pretrain complete — exiting before scribe FT (ckpt saved)", flush=True)
    sys.exit(0)

# ---------------- phase 2: scribe full-FT (v2 recipe via exec — identical to the anchors) ----------------
v2_src = open(os.path.join(REPO, "scribe", "build_scribe_data_v2.py")).read()
prefix = v2_src.split('tok = Tokenizer.from_file')[0].replace("from tokenizers import Tokenizer", "")
ns = {}; exec(compile(prefix, "v2[prefix]", "exec"), ns)
convos = ns["convos"]; assert len(convos) == 12000

def render_ids(convo):
    ids, mask = [], []
    for msg in convo:
        head = tok.encode(f"{msg['role']}\n", add_special_tokens=False).ids
        body = tok.encode(msg["content"], add_special_tokens=False).ids
        seg = [IMS] + head + body + [IME]; tf = 1 if msg["role"] == "assistant" else 0
        for j, tkn in enumerate(seg): ids.append(tkn); mask.append(1 if (tf and j >= 1 + len(head)) else 0)
    return ids, mask

if not os.path.exists("ownstack160m_scribe.pt"):
    X, M = [], []
    for c in convos:
        ids, mask = render_ids(c)
        if len(ids) > S: continue
        X.append(ids + [0] * (S - len(ids))); M.append(mask + [0] * (S - len(mask)))
    X = torch.tensor(np.array(X, dtype=np.int64)); M = torch.tensor(np.array(M, dtype=np.int64))
    N = X.shape[0]; FSTEPS = (N // BATCH) * 3; FWARM = int(0.03 * FSTEPS); FLR = 1e-4
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
        if step % 200 == 0: print(f"scribe {step}/{FSTEPS} loss {loss.item():.3f}", flush=True)
    torch.save(m.state_dict(), "ownstack160m_scribe.pt")
else:
    m.load_state_dict(torch.load("ownstack160m_scribe.pt", map_location=dev, weights_only=True))

# ---------------- phase 3: multi-instance scoring (diluted + clean) + decision rule ----------------
def prompt_ids(user_content):
    return ([IMS] + tok.encode("user\n", add_special_tokens=False).ids
            + tok.encode(user_content, add_special_tokens=False).ids + [IME]
            + [IMS] + tok.encode("assistant\n", add_special_tokens=False).ids)

@torch.no_grad()
def generate(model, ids, max_new=64):
    ids = list(ids)
    for _ in range(max_new):
        if len(ids) >= S: break
        x = torch.tensor([ids + [0] * (S - len(ids))], device=dev)
        nxt = int(model(x)[0, len(ids) - 1].argmax())
        if nxt == IME: break
        ids.append(nxt)
    return ids

RE = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")
FIELDS = ["cc", "dur", "sev", "med", "alg"]
HELD = {"cc": {"toothache", "neck pain", "heartburn"}, "med": {"melatonin", "throat lozenges"}, "alg": {"sulfa drugs"}}
VALFIELDS = ["cc", "med", "alg"]

@torch.no_grad()
def score(model, items):
    parsed = 0; hc = ht = sc = st = 0                          # diluted (dialogue-level)
    cln = {f: [0, 0, 0, 0] for f in VALFIELDS}                 # clean (value-level)
    for it in items:
        text = tok.decode(generate(model, prompt_ids(it["convo"][0]["content"]))[len(prompt_ids(it["convo"][0]["content"])):]).strip()
        mm = RE.match(text)
        if not mm: continue
        parsed += 1; pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        for f in FIELDS:
            hit = int(pred[f] == it["tuple"][f])
            if it["held_values"]: ht += 1; hc += hit
            else: st += 1; sc += hit
        for f in VALFIELDS:
            t = it["tuple"][f]
            if t == "none": continue
            h = int(pred[f] == t)
            if t in HELD[f]: cln[f][0] += h; cln[f][1] += 1
            else: cln[f][2] += h; cln[f][3] += 1
    diluted = (sc / max(1, st) - hc / max(1, ht)) * 100
    ch = sum(cln[f][0] for f in VALFIELDS); cht = sum(cln[f][1] for f in VALFIELDS)
    cs = sum(cln[f][2] for f in VALFIELDS); cst = sum(cln[f][3] for f in VALFIELDS)
    clean = (cs / max(1, cst) - ch / max(1, cht)) * 100
    return {"parse": parsed / len(items), "diluted_gap": diluted, "clean_gap": clean,
            "clean_per_field": {f: (cln[f][2]/max(1,cln[f][3]) - cln[f][0]/max(1,cln[f][1]))*100 for f in VALFIELDS}}

inst0 = json.load(open(os.path.join(REPO, "scribe", "scribe_eval.json")))
fresh = [json.load(open(os.path.join(REPO, "trajectory", f"scribe_eval_m{k}.json"))) for k in range(5)]
m.eval()
base = GPT().to(dev); base.load_state_dict(base_sd); base.eval()
base_parse = score(base, inst0)["parse"]
print(f"\n=== OWN-STACK 160M (full FT) ===\nbase control parse {base_parse:.0%} (must be <50%)", flush=True)
m0 = score(m, inst0)
fr = [score(m, fresh[k]) for k in range(5)]
dg = [r["diluted_gap"] for r in fr]; cg = [r["clean_gap"] for r in fr]
dmean, dsd = float(np.mean(dg)), float(np.std(dg, ddof=1)); cmean, csd = float(np.mean(cg)), float(np.std(cg, ddof=1))
print(f"inst0 parse {m0['parse']:.0%} diluted {m0['diluted_gap']:.1f} clean {m0['clean_gap']:.1f}", flush=True)
print(f"m0-m4 DILUTED gap {dmean:.1f} ± {dsd:.1f}  |  CLEAN gap {cmean:.1f} ± {csd:.1f}", flush=True)
# pre-registered decision rule on the diluted full-FT gap
band = "STACK-dominant (>=14)" if dmean >= 14 else ("SCALE-plausible (<=6)" if dmean <= 6 else "MIXED (6-14): add 40M/80M")
print(f"DECISION (diluted gap_mean {dmean:.1f}): {band}", flush=True)

results = {"stage": "ownstack-160m-fullft", "params_M": round(sum(p.numel() for p in m.parameters())/1e6, 2),
           "dims": {"d": d, "L": L, "H": H, "KV": KV, "hd": hd, "ff": ff}, "target_tokens": TARGET_TOKENS,
           "prereg": "trajectory/PREREG_ownstack_160m.md", "base_parse": base_parse,
           "inst0": m0, "fresh_instances": fr, "diluted_fresh": dg, "clean_fresh": cg,
           "diluted_gap_mean": dmean, "diluted_gap_sd": dsd, "clean_gap_mean": cmean, "clean_gap_sd": csd,
           "decision": band, "gpu": torch.cuda.get_device_name(0)}
json.dump(results, open("results_ownstack_v2_160m_fullft.json", "w"), indent=1)
print("results -> results_ownstack_v2_160m_fullft.json", flush=True)

# ---------------- LoRA arm (2x2 control) — DISABLED by default ----------------
# The PREREG's 2x2 also needs an own-stack-160M LoRA finetune to control the full-FT-vs-LoRA
# confound. peft on this custom nn.Module needs target_modules=["q","k","v","o","g","u","dn"]
# and should be VALIDATED interactively (peft wraps nn.Linear by name; verify trainable params
# and that a forward pass works) before spending a Kaggle session on it. Left as a follow-on so
# an unvalidated peft path cannot silently corrupt the primary full-FT result above.
