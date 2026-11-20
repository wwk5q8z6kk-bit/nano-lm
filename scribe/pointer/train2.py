# Stage P2 training — copy-SUPERVISED pointer. loss = L_nll + LAMBDA * L_copy.
import sys, math, time, numpy as np, torch, torch.nn.functional as F
import pointer_model as PM
from pointer_model2 import GPTCopy2
from pointer_model import dev, V, S

OUT = sys.argv[1] if len(sys.argv) > 1 else "pointer2.pt"
LAMBDA = 1.0                                    # frozen in PREREG v2
BASE = "dpo.pt"
EPOCHS, PEAK_LR, WARM_FRAC, FLOOR, WD, CLIP, BATCH = 3, 1.5e-4, 0.03, 0.1, 0.1, 1.0, 32

torch.manual_seed(0)
model = GPTCopy2(); model.t.load_state_dict(torch.load(BASE, map_location="cpu", weights_only=True), strict=True); model.to(dev)
print(f"MODE=pointer2 (copy-supervised, lambda={LAMBDA}, gate_bias=-2) params={sum(p.numel() for p in model.parameters())/1e6:.4f}M", flush=True)

X  = torch.tensor(np.load("scribe_x.npy").astype(np.int64))
Mk = torch.tensor(np.load("scribe_mask.npy").astype(np.int64))
N = X.shape[0]; STEPS = (N // BATCH) * EPOCHS; WARM = int(WARM_FRAC * STEPS)
print(f"{N} examples, {STEPS} steps", flush=True)
decay = [p for p in model.parameters() if p.dim() >= 2]; nodecay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}], lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)
def lr_at(t):
    if t < WARM: return PEAK_LR * t / max(1, WARM)
    p = (t - WARM) / max(1, STEPS - WARM); return PEAK_LR * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * p)))
g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
def get(step):
    i0 = (step * BATCH) % (N - BATCH); idx = perm[i0:i0 + BATCH]; return X[idx].to(dev), Mk[idx].to(dev)

t0 = time.time()
for step in range(1, STEPS + 1):
    for gp in opt.param_groups: gp["lr"] = lr_at(step)
    x, msk = get(step)
    srcv = PM.src_valid_from_mask(msk); tgt = torch.cat([x[:, 1:], x[:, -1:]], 1); mtgt = msk[:, 1:].float()
    logP, p_gen, pcopy, cshare, Lcopy_pos, copyable = model.logprob_at_targets(x, tgt, srcv)
    denom = mtgt.sum().clamp(min=1)
    nll = -(logP[:, :-1] * mtgt).sum() / denom
    csup_mask = (copyable[:, :-1].float() * mtgt)                     # copyable assistant positions
    Lcopy = (Lcopy_pos[:, :-1] * csup_mask).sum() / csup_mask.sum().clamp(min=1)
    loss = nll + LAMBDA * Lcopy
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()
    if step % 100 == 0 or step == 1:
        cs = (cshare[:, :-1] * mtgt).sum() / denom
        print(f"step {step:5d}/{STEPS}  nll {nll.item():.3f}  Lcopy {Lcopy.item():.3f}  copy_share {cs.item():.2f}  "
              f"gnorm {gn.item():.2f}  {step*BATCH*S/(time.time()-t0)/1e3:.0f}k tok/s", flush=True)
torch.save(model.state_dict(), OUT); print(f"pointer2 done in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
