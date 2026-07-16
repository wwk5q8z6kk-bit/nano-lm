# Stage 3c — DPO per Recipe ③ §Stage 3c and direct-preference-optimization.md.
# Two models only: policy (trained) + frozen reference (both init from sft.pt).
# Loss = -log sigmoid( beta * [ (logpi_w - logref_w) - (logpi_l - logref_l) ] )  (the vault's dpo_loss).
import json, time, torch, torch.nn.functional as F
import model_nano as M

BETA = 0.1            # KL-strength analog (vault default)
LR = 1e-5            # STALL: vault gives NO DPO LR at any scale; DPO<<SFT LR -> nano 3e-4 SFT => ~1e-5
EPOCHS = 3
dev = M.dev

pairs = json.load(open("pref_pairs.json"))
print(f"{len(pairs)} preference pairs, beta={BETA}, lr={LR}, epochs={EPOCHS}", flush=True)

policy = M.load("sft.pt"); policy.train()
ref = M.load("sft.pt")                                  # frozen reference
for p in ref.parameters(): p.requires_grad_(False)

opt = torch.optim.AdamW(policy.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)

def logp(m, pr, comp):
    return M.seq_logprob(m, pr, comp)

t0 = time.time(); step = 0
for ep in range(EPOCHS):
    order = list(range(len(pairs))); torch.manual_seed(ep)
    idx = torch.randperm(len(pairs)).tolist()
    accs = []; losses = []
    for j in idx:
        pr, cw, cl = pairs[j]["prompt"], pairs[j]["chosen"], pairs[j]["rejected"]
        with torch.no_grad():
            rw = logp(ref, pr, cw); rl = logp(ref, pr, cl)
        pw = logp(policy, pr, cw); pl = logp(policy, pr, cl)
        margin = BETA * ((pw - rw) - (pl - rl))
        loss = -F.logsigmoid(margin)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        step += 1; losses.append(loss.item()); accs.append((margin > 0).float().item())
        if step % 200 == 0:
            print(f"  ep{ep} step {step}  loss {sum(losses[-200:])/len(losses[-200:]):.3f}  "
                  f"pref-acc {sum(accs[-200:])/len(accs[-200:]):.2f}  {(time.time()-t0)/60:.1f}min", flush=True)
    print(f"epoch {ep}: mean loss {sum(losses)/len(losses):.3f}  pref-acc {sum(accs)/len(accs):.2f}", flush=True)

torch.save(policy.state_dict(), "dpo.pt")
print(f"DPO done in {(time.time()-t0)/60:.1f} min -> dpo.pt", flush=True)
