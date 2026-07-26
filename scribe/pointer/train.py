# Stage P training — one script, two arms (MODE = baseline | pointer).
# Both full-FT from dpo.pt on seed-11 v2 data, identical config to scribe v2, identical
# batch order across arms (dedicated generator) so the pointer delta is attributable.
import sys, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pointer_model as PM
from pointer_model import dev, V, S

MODE = sys.argv[1] if len(sys.argv) > 1 else "pointer"   # baseline | pointer
OUT  = sys.argv[2] if len(sys.argv) > 2 else f"{MODE}.pt"
BASE = "dpo.pt"
EPOCHS, PEAK_LR, WARM_FRAC, FLOOR, WD, CLIP, BATCH = 3, 1.5e-4, 0.03, 0.1, 0.1, 1.0, 32

torch.manual_seed(0)
model = PM.GPT() if MODE == "baseline" else PM.GPTCopy()
base_sd = torch.load(BASE, map_location="cpu", weights_only=True)
model.t.load_state_dict(base_sd, strict=True)            # trunk <- dpo.pt; copy head random
model.to(dev)
np_ = sum(p.numel() for p in model.parameters())
print(f"MODE={MODE}  loaded trunk from {BASE}  params={np_/1e6:.4f}M "
      f"(added over 3.149M: {(np_-3149000)/1e3:.2f}k)", flush=True)

X  = torch.tensor(np.load("scribe_x.npy").astype(np.int64))
Mk = torch.tensor(np.load("scribe_mask.npy").astype(np.int64))
N = X.shape[0]; STEPS = (N // BATCH) * EPOCHS; WARM = int(WARM_FRAC * STEPS)
print(f"{N} examples, {STEPS} steps, {EPOCHS} epochs, batch {BATCH}", flush=True)

decay = [p for p in model.parameters() if p.dim() >= 2]
nodecay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)
def lr_at(t):
    if t < WARM: return PEAK_LR * t / max(1, WARM)
    p = (t - WARM) / max(1, STEPS - WARM)
    return PEAK_LR * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * p)))

g = torch.Generator().manual_seed(0)                     # identical batch order across arms
perm = torch.randperm(N, generator=g)
def get(step):
    i0 = (step * BATCH) % (N - BATCH)
    idx = perm[i0:i0 + BATCH]
    return X[idx].to(dev), Mk[idx].to(dev)

t0 = time.time()
for step in range(1, STEPS + 1):
    for gp in opt.param_groups: gp["lr"] = lr_at(step)
    x, msk = get(step)
    if MODE == "baseline":
        logits = model(x)[:, :-1]; tgt = x[:, 1:]; mtgt = msk[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
        loss = (ce * mtgt).sum() / mtgt.sum().clamp(min=1)
        cshare = torch.tensor(0.0)
    else:
        srcv = PM.src_valid_from_mask(msk)
        tgt_full = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
        logP, p_gen, pcopy, copy_share = model.logprob_at_targets(x, tgt_full, srcv)
        mtgt = msk[:, 1:]
        lp = logP[:, :-1]
        loss = -(lp * mtgt).sum() / mtgt.sum().clamp(min=1)
        cshare = (copy_share[:, :-1] * mtgt).sum() / mtgt.sum().clamp(min=1)  # copy channel usage
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step()
    if step % 100 == 0 or step == 1:
        extra = f"  copy_share {cshare.item():.2f}" if MODE == "pointer" else ""
        print(f"step {step:5d}/{STEPS}  loss {loss.item():.3f}  gnorm {gn.item():.2f}  "
              f"lr {lr_at(step):.1e}  {step*BATCH*S/(time.time()-t0)/1e3:.0f}k tok/s{extra}", flush=True)

torch.save(model.state_dict(), OUT)
print(f"{MODE} done in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
