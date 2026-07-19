# OWNSTACK_160M — LoRA arm of the pre-registered 2x2 (PREREG_ownstack_160m.md).
# Controls the finetuning-METHOD confound: same own-stack 160M pretrain checkpoint as the
# full-FT arm (mounted via Kaggle kernel_sources; NOT re-pretrained), scribe-finetuned with
# LoRA r=16 alpha=32 on ["q","k","v","o","g","u","dn"] (98 modules, 4.028M trainables —
# peft-wrap validated locally on CPU: forward/backward clean, frozen params grad-free).
# Interpretation grid: fullFT~17 & LoRA~17 -> stack effect, method innocent.
#   fullFT~17 & LoRA much lower -> finetuning method is part of the gap.
# Scoring identical to the full-FT arm (diluted + clean, m0-m4 + inst0).
# Expects ownstack160m_pretrain.pt in cwd (wrapper copies from /kaggle/input/...).
import json, math, os, random, re, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
from peft import LoraConfig, inject_adapter_in_model

torch.manual_seed(0); random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
assert dev == "cuda", "enable the GPU accelerator"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = Tokenizer.from_file(os.path.join(REPO, "sft", "tokenizer.json"))
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")

V = 4098
d, L, H, KV, hd, ff, S = 1024, 14, 16, 4, 64, 2752, 512
MICRO, ACCUM = 8, 4
BATCH = MICRO * ACCUM

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

CKPT = "ownstack160m_pretrain.pt"
assert os.path.exists(CKPT), "pretrain ckpt missing — wrapper must copy it from /kaggle/input"
m = GPT()
st = torch.load(CKPT, map_location="cpu", weights_only=True)
m.load_state_dict(st["m"]); m.to(dev)
print(f"loaded pretrain ckpt (step {st['step']}); params {sum(p.numel() for p in m.parameters())/1e6:.2f}M", flush=True)

# ---- LoRA injection (validated config) ----
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                 target_modules=["q", "k", "v", "o", "g", "u", "dn"])
m = inject_adapter_in_model(cfg, m)
for n, p in m.named_parameters():
    p.requires_grad = "lora_" in n
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"LoRA trainables: {trainable/1e6:.3f}M (expect ~4.028M)", flush=True)

# ---- scribe data (v2 recipe verbatim via exec — identical to all other arms) ----
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

X, M = [], []
for c in convos:
    ids, mask = render_ids(c)
    if len(ids) > S: continue
    X.append(ids + [0] * (S - len(ids))); M.append(mask + [0] * (S - len(mask)))
X = torch.tensor(np.array(X, dtype=np.int64)); M = torch.tensor(np.array(M, dtype=np.int64))
N = X.shape[0]; FSTEPS = (N // BATCH) * 3; FWARM = int(0.03 * FSTEPS); FLR = 1e-4
opt = torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),
                        lr=FLR, betas=(0.9, 0.95), weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda"); perm = torch.randperm(N)
def flr_at(t):
    if t < FWARM: return FLR * t / max(1, FWARM)
    p = (t - FWARM) / max(1, FSTEPS - FWARM); return FLR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
t0 = time.time()
m.train()
for step in range(1, FSTEPS + 1):
    for g in opt.param_groups: g["lr"] = flr_at(step)
    i0 = (step * BATCH) % (N - BATCH); idx = perm[i0:i0 + BATCH]
    opt.zero_grad(set_to_none=True)
    for a in range(ACCUM):
        sub = idx[a * MICRO:(a + 1) * MICRO]
        x, msk = X[sub].to(dev), M[sub].to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = m(x)[:, :-1]; tgt = x[:, 1:]; mtgt = msk[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
            loss = (ce * mtgt).sum() / mtgt.sum().clamp(min=1)
        scaler.scale(loss / ACCUM).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_((p for p in m.parameters() if p.requires_grad), 1.0)
    scaler.step(opt); scaler.update()
    if step % 200 == 0: print(f"lora-scribe {step}/{FSTEPS} loss {loss.item():.3f}", flush=True)
print(f"LoRA finetune done in {(time.time()-t0)/60:.1f} min", flush=True)

# ---- scoring (identical to the full-FT arm) ----
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
    parsed = 0; hc = ht = sc = st_ = 0
    cln = {f: [0, 0, 0, 0] for f in VALFIELDS}
    for it in items:
        text = tok.decode(generate(model, prompt_ids(it["convo"][0]["content"]))[len(prompt_ids(it["convo"][0]["content"])):]).strip()
        mm = RE.match(text)
        if not mm: continue
        parsed += 1; pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        for f in FIELDS:
            hit = int(pred[f] == it["tuple"][f])
            if it["held_values"]: ht += 1; hc += hit
            else: st_ += 1; sc += hit
        for f in VALFIELDS:
            t = it["tuple"][f]
            if t == "none": continue
            h = int(pred[f] == t)
            if t in HELD[f]: cln[f][0] += h; cln[f][1] += 1
            else: cln[f][2] += h; cln[f][3] += 1
    diluted = (sc / max(1, st_) - hc / max(1, ht)) * 100
    ch = sum(cln[f][0] for f in VALFIELDS); cht = sum(cln[f][1] for f in VALFIELDS)
    cs = sum(cln[f][2] for f in VALFIELDS); cst = sum(cln[f][3] for f in VALFIELDS)
    clean = (cs / max(1, cst) - ch / max(1, cht)) * 100
    return {"parse": parsed / len(items), "diluted_gap": diluted, "clean_gap": clean,
            "clean_per_field": {f: (cln[f][2]/max(1,cln[f][3]) - cln[f][0]/max(1,cln[f][1]))*100 for f in VALFIELDS}}

inst0 = json.load(open(os.path.join(REPO, "scribe", "scribe_eval.json")))
fresh = [json.load(open(os.path.join(REPO, "trajectory", f"scribe_eval_m{k}.json"))) for k in range(5)]
m.eval()
print(f"\n=== OWN-STACK 160M (LoRA r=16) ===", flush=True)
m0 = score(m, inst0)
fr = [score(m, fresh[k]) for k in range(5)]
dg = [r["diluted_gap"] for r in fr]; cg = [r["clean_gap"] for r in fr]
dmean, dsd = float(np.mean(dg)), float(np.std(dg, ddof=1)); cmean, csd = float(np.mean(cg)), float(np.std(cg, ddof=1))
print(f"inst0 parse {m0['parse']:.0%} diluted {m0['diluted_gap']:.1f} clean {m0['clean_gap']:.1f}", flush=True)
print(f"m0-m4 DILUTED gap {dmean:.1f} ± {dsd:.1f}  |  CLEAN gap {cmean:.1f} ± {csd:.1f}", flush=True)
print(f"2x2 read (vs full-FT 16.9±1.7): " +
      ("method-effect visible" if dmean < 10 else "stack effect confirmed; method innocent"), flush=True)

results = {"stage": "ownstack-160m-lora", "trainable_params": trainable,
           "dims": {"d": d, "L": L, "H": H, "KV": KV, "hd": hd, "ff": ff},
           "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "targets": ["q","k","v","o","g","u","dn"]},
           "prereg": "trajectory/PREREG_ownstack_160m.md",
           "inst0": m0, "fresh_instances": fr, "diluted_fresh": dg, "clean_fresh": cg,
           "diluted_gap_mean": dmean, "diluted_gap_sd": dsd, "clean_gap_mean": cmean, "clean_gap_sd": csd,
           "fullft_reference": {"diluted": 16.9, "clean": 66.6},
           "gpu": torch.cuda.get_device_name(0)}
json.dump(results, open("results_ownstack_v2_160m_lora.json", "w"), indent=1)
print("results -> results_ownstack_v2_160m_lora.json", flush=True)
