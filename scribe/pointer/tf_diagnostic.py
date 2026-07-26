# Stage P2 read-only diagnostic (frozen pointer2.pt): teacher-forced top-1 at held-out
# value tokens. Discriminates addressing-failure (a) from free-running exposure-bias (b).
# If TF top-1 >> free-running held-recall(10%) -> decoding is the culprit; if TF ~ 10% ->
# content-addressed source selection genuinely does not generalize OOD. Result: 41% all /
# 21% first-token -> addressing does not generalize (secondary exposure-bias compounding).
import json, torch
from tokenizers import Tokenizer
import pointer_model as PM
from pointer_model2 import GPTCopy2
from pointer_model import dev, S
torch.manual_seed(0)
tok = Tokenizer.from_file("tokenizer.json")
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
HELD = {"cc": {"toothache","neck pain","heartburn"}, "med": {"melatonin","throat lozenges"}, "alg": {"sulfa drugs"}}
def prompt_ids(u):
    ids = [IMS] + tok.encode("user\n", add_special_tokens=False).ids
    ids += tok.encode(u, add_special_tokens=False).ids
    ids += [IME, IMS] + tok.encode("assistant\n", add_special_tokens=False).ids
    return ids
m = GPTCopy2(); m.load_state_dict(torch.load("pointer2.pt", map_location="cpu", weights_only=True)); m.to(dev).eval()
items = json.load(open("scribe_eval.json"))
a_hit=a=f_hit=f=0
for it in items:
    u=it["convo"][0]["content"]; summ=it["convo"][1]["content"]
    pids=prompt_ids(u); sids=tok.encode(summ, add_special_tokens=False); seq=pids+sids.ids
    if len(seq)>=S: continue
    x=torch.tensor([seq+[0]*(S-len(seq))], device=dev)
    for fld in ["cc","med","alg"]:
        v=it["tuple"][fld]
        if v not in HELD[fld]: continue
        lab={"cc":"CC: ","med":"MED: ","alg":"ALG: "}[fld]; p=summ.find(lab)
        if p<0: continue
        vs_,ve_=p+len(lab),p+len(lab)+len(v)
        idxs=[k for k,(aa,bb) in enumerate(sids.offsets) if aa<ve_ and bb>vs_ and bb>aa]
        for n,k in enumerate(idxs):
            pos=len(pids)+k; gold=seq[pos]
            top1=int(m.full_dist_last(x, pos-1, len(pids)).argmax())
            a+=1; a_hit+=(top1==gold)
            if n==0: f+=1; f_hit+=(top1==gold)
print(f"teacher-forced top-1 held-value tokens: all {a_hit}/{a}={a_hit/max(1,a):.0%}  first {f_hit}/{f}={f_hit/max(1,f):.0%}")
print(f"vs free-running greedy held value-recall 10% (P1 unused AND P2 dominant); seen ~92%")
