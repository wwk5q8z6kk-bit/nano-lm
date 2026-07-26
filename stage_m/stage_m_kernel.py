#!/usr/bin/env python3
# Stage M kernel (self-contained; runs on a CUDA pod, or locally on MPS with SMOKE=1).
# Tests H-induction (PREREG_induction_curriculum.md): does a key->value copy curriculum in
# pretraining induce the content-addressed copy circuit that P2 localized as the OOD bottleneck?
# Two arms (control vs +rho induction curriculum), raw pretrain -> scribe-SFT. Emits JSON.
#
# BLOCKING guards, all pre-registered:
#   - Arm-C feasibility gate (parse>=90/recall>=80) before Arm I.
#   - induction probe at BOTH raw-pretrain AND post-SFT checkpoints (C1: full-FT destroys copy).
#   - probe on NOVEL tokens AND novel surface form (generality; else probe-pass is the game).
#   - co-primary readout: teacher-forced held-value first-token top-1 (P2 baseline 21% vs 92%).
import os, sys, json, math, time, random, io, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

SMOKE = os.environ.get("SMOKE", "0") == "1"
RHO   = float(os.environ.get("RHO", "0.30"))
RAW   = "https://raw.githubusercontent.com/wwk5q8z6kk-bit/nano-lm/master"
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)

# ---- nano arch (V=4098: pretrain vocab 4096 + 2 ChatML specials, trained from scratch here) ----
V, d, L, H, KV, hd, ff, S = 4098, 192, 6, 6, 2, 32, 512, 512
SPECIAL_LO = 4096                      # ids >= 4096 are ChatML specials; curriculum uses [0,4096)
PT_STEPS = int(os.environ.get("PT_STEPS", "60" if SMOKE else "4000"))
PT_BATCH = 16
SFT_STEPS_EPOCHS = 3
BASE_TOKENS = 300_000 if SMOKE else 15_000_000

def rope(q, k):
    t = torch.arange(S, device=dev, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    f = torch.outer(t, inv); cos, sin = f.cos()[None, None], f.sin()[None, None]
    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).flatten(-2)
    return rot(q), rot(k)

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1, s.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        s.q, s.k, s.v, s.o = nn.Linear(d,H*hd,bias=False), nn.Linear(d,KV*hd,bias=False), nn.Linear(d,KV*hd,bias=False), nn.Linear(H*hd,d,bias=False)
        s.g, s.u, s.dn = nn.Linear(d,ff,bias=False), nn.Linear(d,ff,bias=False), nn.Linear(ff,d,bias=False)
    def forward(s, x):
        B = x.shape[0]; h = s.n1(x)
        q = s.q(h).view(B,S,H,hd).transpose(1,2); k = s.k(h).view(B,S,KV,hd).transpose(1,2); v = s.v(h).view(B,S,KV,hd).transpose(1,2)
        q, k = rope(q, k); k, v = k.repeat_interleave(H//KV,1), v.repeat_interleave(H//KV,1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(a.transpose(1,2).reshape(B,S,H*hd)); h = s.n2(x)
        return x + s.dn(F.silu(s.g(h)) * s.u(h))

class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.blocks = nn.ModuleList(Block() for _ in range(L)); s.nf = nn.RMSNorm(d)
        for p in s.parameters():                                   # nano recipe init (std=0.02)
            if p.dim() >= 2: nn.init.normal_(p, std=0.02)
        for b in s.blocks:                                         # depth-scaled residual-out projections
            nn.init.normal_(b.o.weight, std=0.02/math.sqrt(2*L)); nn.init.normal_(b.dn.weight, std=0.02/math.sqrt(2*L))
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return F.linear(s.nf(h), s.emb.weight)

def fetch(path, binary=False):
    """local file if present (smoke), else RAW github."""
    for p in (path, os.path.basename(path)):
        if os.path.exists(p): return open(p, "rb").read() if binary else open(p).read()
    data = urllib.request.urlopen(f"{RAW}/{path}").read()
    return data if binary else data.decode()

# ---- tokenizer ---- (fetch to a var FIRST; writing before fetch would truncate a local copy)
_tokjson = fetch("sft/tokenizer.json", binary=True)
open("tokenizer.json", "wb").write(_tokjson)
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer.json")
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")

# ---- base corpus (FineWeb stream, cached; local shard if present) ----
def base_corpus():
    if os.path.exists("shard_stage_m.npy"): return np.load("shard_stage_m.npy")
    if os.path.exists("shard_000.npy"):                        # local smoke reuse
        a = np.load("shard_000.npy"); return a[:BASE_TOKENS] if SMOKE else a
    print("streaming FineWeb...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    ids = []
    for doc in ds:
        ids.extend(tok.encode(doc["text"], add_special_tokens=False).ids)
        if len(ids) >= BASE_TOKENS: break
    a = np.array(ids[:BASE_TOKENS], dtype=np.uint16); np.save("shard_stage_m.npy", a); return a

# ---- induction curriculum: CANONICAL block-repeat over random tokens (deterministic) ----
# A random block immediately repeated -> every 2nd-copy token is induction-predictable (find
# the prior occurrence of the current token, copy what followed). Validated locally: this
# reliably induces a copy circuit (99% 2nd-copy acc) that GENERALIZES to cued key->value
# retrieval (50% on the probe below); the earlier variable-length k:v:sep form did NOT induce.
def gen_curriculum(n_seqs, seed=1234):
    rng = random.Random(seed); out = []
    for _ in range(n_seqs):
        seq = []
        while len(seq) < S + 1:
            blen = rng.randint(4, 10); block = [rng.randrange(SPECIAL_LO) for _ in range(blen)]
            seq += block + block
        out.append(seq[:S+1])
    return np.array(out, dtype=np.uint16)

# ---- pretrain (base [+ curriculum mixed at rho]); returns state_dict ----
def pretrain(rho, tag):
    m = GPT().to(dev)
    base = base_corpus(); base_t = torch.tensor(base.astype(np.int64))
    n_curr = int(round(rho / max(1e-9, 1 - rho) * (len(base) // S))) if rho > 0 else 0
    curr = gen_curriculum(max(1, n_curr)) if rho > 0 else None
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9,0.95), weight_decay=0.1)
    WARM = 100
    def lr_at(t):
        if t < WARM: return 3e-3 * t / WARM
        p = (t-WARM)/max(1, PT_STEPS-WARM); return 3e-3*(0.1 + 0.9*0.5*(1+math.cos(math.pi*p)))
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(1, PT_STEPS+1):
        for gp in opt.param_groups: gp["lr"] = lr_at(step)
        use_curr = rho > 0 and torch.rand(1, generator=g).item() < rho
        if use_curr:
            idx = torch.randint(len(curr), (PT_BATCH,), generator=g)
            xb = torch.tensor(curr[idx.numpy()].astype(np.int64), device=dev)   # (B,S+1)
        else:
            ix = torch.randint(len(base_t)-S-1, (PT_BATCH,), generator=g)
            xb = torch.stack([base_t[i:i+S+1] for i in ix]).to(dev)             # (B,S+1)
        x, y = xb[:, :S], xb[:, 1:S+1]
        logits = m(x); loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        zloss = 1e-4 * (torch.logsumexp(logits.float(), -1) ** 2).mean()   # z-loss (nano recipe)
        opt.zero_grad(set_to_none=True); (loss + zloss).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % max(1, PT_STEPS//8) == 0 or step == 1:
            print(f"  [{tag}] pt {step}/{PT_STEPS} loss {loss.item():.3f} {step*PT_BATCH*S/(time.time()-t0)/1e3:.0f}k tok/s", flush=True)
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}

# ---- scribe data (v2 recipe, seed 11) + SFT (full-FT from a pretrain sd) ----
def ensure_scribe_data():
    if os.path.exists("scribe_x.npy") and os.path.exists("scribe_mask.npy"): return
    if not os.path.exists("build_scribe_data_v2.py"):
        src = fetch("scribe/build_scribe_data_v2.py"); open("build_scribe_data_v2.py","w").write(src)
    import subprocess; subprocess.run([sys.executable, "build_scribe_data_v2.py"], check=True)

def scribe_sft(pt_sd, tag):
    m = GPT().to(dev); m.load_state_dict(pt_sd)
    X = torch.tensor(np.load("scribe_x.npy").astype(np.int64)); Mk = torch.tensor(np.load("scribe_mask.npy").astype(np.int64))
    N = X.shape[0]; BATCH = 32; STEPS = (N//BATCH)*SFT_STEPS_EPOCHS
    if SMOKE: STEPS = 60
    WARM = int(0.03*STEPS)
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-4, betas=(0.9,0.95), weight_decay=0.1)
    def lr_at(t):
        if t < WARM: return 1.5e-4*t/max(1,WARM)
        p=(t-WARM)/max(1,STEPS-WARM); return 1.5e-4*(0.1+0.9*0.5*(1+math.cos(math.pi*p)))
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    t0 = time.time()
    for step in range(1, STEPS+1):
        for gp in opt.param_groups: gp["lr"] = lr_at(step)
        i0 = (step*BATCH) % (N-BATCH); idx = perm[i0:i0+BATCH]
        x, msk = X[idx].to(dev), Mk[idx].to(dev)
        logits = m(x)[:, :-1]; tgt = x[:, 1:]; mtgt = msk[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
        loss = (ce*mtgt).sum()/mtgt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % max(1, STEPS//4)==0 or step==1:
            print(f"  [{tag}] sft {step}/{STEPS} loss {loss.item():.3f}", flush=True)
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}

# ---- scribe gate: parse/recall/halluc + item gap + teacher-forced first-token top-1 ----
import re
HELD = {"cc":{"toothache","neck pain","heartburn"},"med":{"melatonin","throat lozenges"},"alg":{"sulfa drugs"}}
RE = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$"); FIELDS=["cc","dur","sev","med","alg"]
def field_held(f, v):
    if v=="none": return None
    return v in HELD.get(f, set())
def prompt_ids(u):
    ids=[IMS]+tok.encode("user\n",add_special_tokens=False).ids
    ids+=tok.encode(u,add_special_tokens=False).ids
    ids+=[IME,IMS]+tok.encode("assistant\n",add_special_tokens=False).ids
    return ids
@torch.no_grad()
def gen_greedy(m, ids, max_new=64):
    ids=list(ids)
    for _ in range(max_new):
        if len(ids)>=S: break
        x=torch.tensor([ids+[0]*(S-len(ids))],device=dev); nxt=int(m(x)[0,len(ids)-1].argmax())
        if nxt==IME: break
        ids.append(nxt)
    return ids
def eval_items():
    if os.path.exists("scribe_eval.json"): return json.load(open("scribe_eval.json"))
    return json.loads(fetch("scribe/scribe_eval.json"))
@torch.no_grad()
def gate(sd, tag):
    m = GPT().to(dev); m.load_state_dict(sd); m.eval()
    items = eval_items()
    parsed=correct=halluc=omission=total=0; ih_c=ih_t=is_c=is_t=0
    tf_first_h=tf_first_ht=0
    for it in items:
        pids=prompt_ids(it["convo"][0]["content"]); out=gen_greedy(m,pids); text=tok.decode(out[len(pids):]).strip()
        total+=5; mm=RE.match(text)
        if mm:
            parsed+=1; pred=dict(zip(FIELDS,[g.strip() for g in mm.groups()]))
            for f in FIELDS:
                t,p=it["tuple"][f],pred[f]; hit=(p==t)
                if hit: correct+=1
                elif p=="none" and t!="none": omission+=1
                else: halluc+=1
                if it["held_values"]: ih_t+=1; ih_c+=hit
                else: is_t+=1; is_c+=hit
        # teacher-forced held-value first-token top-1 (co-primary, comparable to P2's 21%)
        summ=it["convo"][1]["content"]; sids=tok.encode(summ,add_special_tokens=False); seq=pids+sids.ids
        if len(seq)<S:
            xx=torch.tensor([seq+[0]*(S-len(seq))],device=dev)
            for f in ["cc","med","alg"]:
                v=it["tuple"][f]
                if field_held(f,v) is not True: continue
                lab={"cc":"CC: ","med":"MED: ","alg":"ALG: "}[f]; pp=summ.find(lab)
                if pp<0: continue
                vs_,ve_=pp+len(lab),pp+len(lab)+len(v)
                idxs=[k for k,(a,b) in enumerate(sids.offsets) if a<ve_ and b>vs_ and b>a]
                if not idxs: continue
                pos=len(pids)+idxs[0]; gold=seq[pos]
                top1=int(m(xx)[0,pos-1].argmax())
                tf_first_ht+=1; tf_first_h+=(top1==gold)
    n=len(items); pr,rec,hal=parsed/n,correct/total,halluc/total
    ih=ih_c/max(1,ih_t); iss=is_c/max(1,is_t)
    res=dict(parse=pr,recall=rec,halluc=hal,omission=omission,item_held=ih,item_seen=iss,
             item_gap=100*(iss-ih), tf_first_held=tf_first_h/max(1,tf_first_ht), tf_first_n=tf_first_ht)
    print(f"  [{tag}] parse {pr:.0%} recall {rec:.0%} halluc {hal:.0%} item_gap {res['item_gap']:.0f} tf_first_held {res['tf_first_held']:.0%}", flush=True)
    return res

# ---- induction probe: cued KEY->VALUE retrieval (NOVEL form vs the block-repeat curriculum) ----
# `k1 v1 k2 v2 ... kq` -> predict v_q, where kq re-queries an earlier key. Structurally UNLIKE
# the contiguous block-repeat curriculum (non-contiguous, cued), so probe-pass licenses a
# GENERAL content-addressed-copy circuit, not the training game. Novel random tokens each item.
@torch.no_grad()
def induction_probe(sd, tag, n_items=300, seed=9999):
    m = GPT().to(dev); m.load_state_dict(sd); m.eval()
    rng = random.Random(seed); hits=tot=0
    for _ in range(n_items):
        nk = rng.randint(3, 6)
        pairs = [(rng.randrange(SPECIAL_LO), rng.randrange(SPECIAL_LO)) for _ in range(nk)]
        seq = []
        for k, v in pairs: seq += [k, v]                     # k v k v ... (single-token, no delimiters)
        qi = rng.randrange(nk); seq += [pairs[qi][0]]        # re-query a key seen in context
        gold = pairs[qi][1]
        if len(seq) >= S: continue
        x = torch.tensor([seq+[0]*(S-len(seq))], device=dev)
        top1 = int(m(x)[0, len(seq)-1].argmax()); hits += (top1==gold); tot += 1
    acc = hits/max(1,tot)
    print(f"  [{tag}] induction-probe (kv-retrieval) copy-acc {acc:.0%} (n={tot})", flush=True)
    return dict(copy_acc=acc, n=tot)

# ================= ORCHESTRATION =================
def run():
    out = {"config": {"SMOKE": SMOKE, "RHO": RHO, "PT_STEPS": PT_STEPS, "dev": dev, "base_tokens": BASE_TOKENS}}
    ensure_scribe_data()
    t0 = time.time()
    print("=== ARM C (control) ===", flush=True)
    ptC = pretrain(0.0, "C"); out["C_probe_pretrain"] = induction_probe(ptC, "C/pretrain")
    scC = scribe_sft(ptC, "C"); out["C_gate"] = gate(scC, "C/scribe"); out["C_probe_scribe"] = induction_probe(scC, "C/scribe")
    feasible = out["C_gate"]["parse"] >= 0.90 and out["C_gate"]["recall"] >= 0.80
    out["feasibility_pass"] = feasible
    if not feasible and not SMOKE:                 # SMOKE exercises both arms regardless
        out["verdict"] = "FEASIBILITY_FAIL (Arm C did not clear parse>=90/recall>=80; raw-nano->scribe lineage/token-budget insufficient; do NOT run Arm I)"
        print("FEASIBILITY_FAIL — stopping before Arm I", flush=True)
        print("STAGE_M_RESULT " + json.dumps(out)); return out
    print("=== ARM I (induction curriculum rho=%.2f) ===" % RHO, flush=True)
    ptI = pretrain(RHO, "I"); out["I_probe_pretrain"] = induction_probe(ptI, "I/pretrain")
    scI = scribe_sft(ptI, "I"); out["I_gate"] = gate(scI, "I/scribe"); out["I_probe_scribe"] = induction_probe(scI, "I/scribe")
    # pre-registered decision tree
    ind_pre = out["I_probe_pretrain"]["copy_acc"] - out["C_probe_pretrain"]["copy_acc"]
    ind_post = out["I_probe_scribe"]["copy_acc"] - out["C_probe_scribe"]["copy_acc"]
    dgap = out["C_gate"]["item_gap"] - out["I_gate"]["item_gap"]
    dtf = out["I_gate"]["tf_first_held"] - out["C_gate"]["tf_first_held"]
    out["deltas"] = dict(induction_pretrain=ind_pre, induction_postSFT=ind_post,
                         item_gap_closed=dgap, tf_first_held_gain=dtf)
    if ind_pre < 0.30:
        v = "VOID — curriculum did not induce general copy (probe I-C=%.0f%% < 30pts)" % (100*ind_pre)
    elif ind_post < 0.30:
        v = "FULL-FT-DESTROYS — induced at pretrain (I-C=%.0f%%) but collapsed post-SFT (I-C=%.0f%%); NOT a refute -> LoRA/frozen-layer follow-up (vNext priority A)" % (100*ind_pre, 100*ind_post)
    elif out["I_gate"]["tf_first_held"] >= 0.50 or out["I_gate"]["item_gap"] < 10:
        v = "H-INDUCTION CONFIRMED — SFT-surviving copy circuit closes the gap (tf_first_held=%.0f%%, item_gap=%.0f)" % (100*out["I_gate"]["tf_first_held"], out["I_gate"]["item_gap"])
    elif out["I_gate"]["item_gap"] >= 15 and out["I_gate"]["tf_first_held"] <= 0.30:
        v = "H-INDUCTION REFUTED — general SFT-surviving copy circuit does NOT transfer to scribe held-values (tf_first_held=%.0f%% vs P2 21%%, item_gap=%.0f) -> objective/scale" % (100*out["I_gate"]["tf_first_held"], out["I_gate"]["item_gap"])
    else:
        v = "WEAKENED/AMBIGUOUS (tf_first_held=%.0f%%, item_gap=%.0f, ind_post I-C=%.0f%%)" % (100*out["I_gate"]["tf_first_held"], out["I_gate"]["item_gap"], 100*ind_post)
    out["verdict"] = v; out["minutes"] = (time.time()-t0)/60
    print("VERDICT:", v, flush=True)
    print("STAGE_M_RESULT " + json.dumps(out))
    json.dump(out, open("stage_m_result.json","w"), indent=2)
    return out

if __name__ == "__main__":
    run()
