# Stage P gate — scores baseline (Arm B) and pointer (Arm P) on the frozen 40-dialogue eval.
# Reuses scribe's parse/recall/halluc + item-level seen/held gap (continuity with Stage S/C);
# ADDS value-level gap (held fields only) and the BLOCKING manipulation diagnostics.
import sys, json, re, math, torch, torch.nn.functional as F
from tokenizers import Tokenizer
import pointer_model as PM
from pointer_model import dev, S, V

torch.manual_seed(0)
tok = Tokenizer.from_file("tokenizer.json")
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
HELD_CC={"toothache","neck pain","heartburn"}; HELD_MED={"melatonin","throat lozenges"}; HELD_ALG={"sulfa drugs"}
def field_held(f, v):
    if v == "none": return None
    return {"cc": v in HELD_CC, "med": v in HELD_MED, "alg": v in HELD_ALG}.get(f, False)

def prompt_ids(u):
    ids = [IMS] + tok.encode("user\n", add_special_tokens=False).ids
    ids += tok.encode(u, add_special_tokens=False).ids
    ids += [IME, IMS] + tok.encode("assistant\n", add_special_tokens=False).ids
    return ids

RE = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")
FIELDS = ["cc", "dur", "sev", "med", "alg"]

@torch.no_grad()
def gen_baseline(m, ids, max_new=64):
    ids = list(ids)
    for _ in range(max_new):
        if len(ids) >= S: break
        x = torch.tensor([ids + [0]*(S-len(ids))], device=dev)
        nxt = int(m(x)[0, len(ids)-1].argmax())
        if nxt == IME: break
        ids.append(nxt)
    return ids

@torch.no_grad()
def gen_pointer(m, ids, max_new=64):
    src_len = len(ids); ids = list(ids)
    for _ in range(max_new):
        if len(ids) >= S: break
        x = torch.tensor([ids + [0]*(S-len(ids))], device=dev)
        P = m.full_dist_last(x, len(ids)-1, src_len)
        nxt = int(P.argmax())
        if nxt == IME: break
        ids.append(nxt)
    return ids

def score(m, items, genfn, label=""):
    parsed=correct=omission=halluc=total=0
    ih_c=ih_t=is_c=is_t=0          # item-level held/seen (continuity metric)
    vh_c=vh_t=vs_c=vs_t=0          # value-level held/seen (held FIELDS only)
    samples=[]
    for it in items:
        pids = prompt_ids(it["convo"][0]["content"])
        out = genfn(m, pids)
        text = tok.decode(out[len(pids):]).strip()
        if len(samples) < 3: samples.append((text, it["tuple"]))
        total += 5
        mm = RE.match(text)
        if not mm: continue
        parsed += 1
        pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        for f in FIELDS:
            t, p = it["tuple"][f], pred[f]; hit = (p == t)
            if hit: correct += 1
            elif p == "none" and t != "none": omission += 1
            else: halluc += 1
            if it["held_values"]: ih_t += 1; ih_c += hit
            else: is_t += 1; is_c += hit
            fh = field_held(f, t)
            if fh is True:  vh_t += 1; vh_c += hit
            elif fh is False: vs_t += 1; vs_c += hit     # seen non-none field
    n = len(items)
    pr, rec, hal = parsed/n, correct/total, halluc/total
    ih = ih_c/max(1,ih_t); iss = is_c/max(1,is_t); item_gap = 100*(iss-ih)
    vh = vh_c/max(1,vh_t); vs = vs_c/max(1,vs_t); val_gap = 100*(vs-vh)
    print(f"[{label}] parse {parsed}/{n}={pr:.0%}  recall {correct}/{total}={rec:.0%}  "
          f"halluc {halluc}/{total}={hal:.1%}  omission {omission}", flush=True)
    print(f"        ITEM-level: held {ih:.0%} seen {iss:.0%}  -> gap {item_gap:.0f} pts   "
          f"VALUE-level(held fields): held {vh:.0%} seen {vs:.0%} -> gap {val_gap:.0f} pts", flush=True)
    for s,t in samples[:2]: print(f"        out: {s!r}\n        truth: {t}", flush=True)
    return dict(parse=pr, recall=rec, halluc=hal, omission=omission,
                item_held=ih, item_seen=iss, item_gap=item_gap,
                val_held=vh, val_seen=vs, val_gap=val_gap)

@torch.no_grad()
def manipulation_check(m, items):
    """M = mean copy-channel share at HELD-OUT value target-token positions (teacher-forced gold)."""
    shares=[]; pgens=[]; pcopies=[]; ntok=0
    for it in items:
        u=it["convo"][0]["content"]; summ=it["convo"][1]["content"]
        pids=prompt_ids(u); sids=tok.encode(summ, add_special_tokens=False)
        seq=pids+sids.ids
        if len(seq) >= S: continue
        x=torch.tensor([seq+[0]*(S-len(seq))], device=dev)
        msk=torch.zeros(1,S,dtype=torch.long,device=dev); msk[0,len(pids):len(seq)]=1
        srcv=PM.src_valid_from_mask(msk)
        tgt=torch.cat([x[:,1:],x[:,-1:]],1)
        logP,p_gen,pcopy,cshare=m.logprob_at_targets(x,tgt,srcv)
        # held-value char spans in summary -> summary token indices -> seq positions
        for f in ["cc","med","alg"]:
            v=it["tuple"][f]
            if field_held(f,v) is not True: continue
            lab={"cc":"CC: ","med":"MED: ","alg":"ALG: "}[f]; p=summ.find(lab)
            if p<0: continue
            vs_,ve_=p+len(lab),p+len(lab)+len(v)
            for k,(a,b) in enumerate(sids.offsets):
                if a<ve_ and b>vs_ and b>a:
                    t=len(pids)+k-1                       # position whose gold NEXT token is this value token
                    if 0<=t<S-1:
                        shares.append(float(cshare[0,t])); pgens.append(float(p_gen[0,t]))
                        pcopies.append(float(pcopy[0,t])); ntok+=1
    M=sum(shares)/max(1,len(shares))
    print(f"[manipulation] held-value tokens n={ntok}  M(copy-share)={M:.2f}  "
          f"mean p_gen={sum(pgens)/max(1,len(pgens)):.2f}  mean copy-mass(correct id)={sum(pcopies)/max(1,len(pcopies)):.2f}", flush=True)
    verdict = "EXERCISED" if M>=0.5 else ("VOID(unused)" if M<0.2 else "PARTIAL")
    print(f"[manipulation] M={M:.2f} -> copy pathway {verdict}  (>=0.5 exercised, <0.2 VOID)", flush=True)
    return dict(M=M, p_gen=sum(pgens)/max(1,len(pgens)), copy_mass=sum(pcopies)/max(1,len(pcopies)), n=ntok, verdict=verdict)

if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv)>1 else "pointer"   # baseline | pointer
    ckpt = sys.argv[2] if len(sys.argv)>2 else f"{arm}.pt"
    items = json.load(open("scribe_eval.json"))
    print(f"eval: {len(items)} dialogues; {sum(i['held_values'] for i in items)} held-value items\n", flush=True)
    res = {}

    print("=== BASE CONTROL (dpo.pt, greedy plain) ===", flush=True)
    base = PM.GPT(); base.t.load_state_dict(torch.load("dpo.pt", map_location="cpu", weights_only=True)); base.to(dev).eval()
    res["base"] = score(base, items, gen_baseline, "base"); del base

    if arm == "baseline":
        print(f"\n=== ARM B baseline ({ckpt}, greedy) ===", flush=True)
        m = PM.GPT(); m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)); m.to(dev).eval()
        res["arm"] = score(m, items, gen_baseline, "baseline")
    else:
        print(f"\n=== ARM P pointer ({ckpt}, greedy mixture) ===", flush=True)
        m = PM.GPTCopy(); m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)); m.to(dev).eval()
        res["arm"] = score(m, items, gen_pointer, "pointer")
        res["manip"] = manipulation_check(m, items)

    a = res["arm"]; b = res["base"]
    arm_pass = a["parse"]>=0.90 and a["recall"]>=0.80 and a["halluc"]<=0.10
    base_fail = not (b["parse"]>=0.90 and b["recall"]>=0.80 and b["halluc"]<=0.10)
    print(f"\n--- pre-registered bars ---", flush=True)
    print(f"  arm clears bars (parse>=90 recall>=80 halluc<=10): {arm_pass}", flush=True)
    print(f"  base control fails (discrimination): {base_fail}", flush=True)
    print(f"  GATE {'PASS' if arm_pass and base_fail else 'FAIL'}", flush=True)
    print(f"  OOD item-gap={a['item_gap']:.0f} pts (ref v2=22, scale=23) | value-gap={a['val_gap']:.0f} pts", flush=True)
    json.dump(res, open(f"result_{arm}.json","w"), indent=2)
    print(f"  wrote result_{arm}.json", flush=True)
