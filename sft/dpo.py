# Stage 3c — DPO per Recipe ③ §Stage 3c and direct-preference-optimization.md.
# Two models only: policy (trained) + frozen reference (both init from sft.pt).
# Loss = -log sigmoid( beta * [ (logpi_w - logref_w) - (logpi_l - logref_l) ] )  (the vault's dpo_loss).
import json, time, torch, torch.nn.functional as F
import model_nano as M

BETA = 0.1            # KL-strength analog (vault default)
LR = 1e-5            # STALL: vault gives NO DPO LR at any scale; DPO<<SFT LR -> nano 3e-4 SFT => ~1e-5
EPOCHS = 3
BS = 16              # Batch size for policy/reference evaluations
dev = M.dev

pairs = json.load(open("pref_pairs.json"))
print(f"{len(pairs)} preference pairs, beta={BETA}, lr={LR}, epochs={EPOCHS}", flush=True)

policy = M.load("sft.pt"); policy.train()
ref = M.load("sft.pt")                                  # frozen reference
for p in ref.parameters(): p.requires_grad_(False)

opt = torch.optim.AdamW(policy.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)

def batched_seq_logprob(m, prs, comps):
    B = len(prs)
    x = torch.zeros((B, M.S), dtype=torch.long, device=dev)
    mask = torch.zeros((B, M.S), dtype=torch.bool, device=dev)

    for i in range(B):
        full = (prs[i] + comps[i])[:M.S]
        l_full = len(full)
        x[i, :l_full] = torch.tensor(full, dtype=torch.long, device=dev)
        lo = len(prs[i])
        if l_full > lo:
            mask[i, lo:l_full] = True

    logits = m(x)
    lp = F.log_softmax(logits.float(), -1)

    tgt = x[:, 1:]
    p = lp[:, :-1]

    log_probs = p.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
    m_mask = mask[:, 1:]

    return (log_probs * m_mask).sum(dim=-1)

t0 = time.time(); step = 0
for ep in range(EPOCHS):
    order = list(range(len(pairs))); torch.manual_seed(ep)
    idx = torch.randperm(len(pairs)).tolist()
    accs = []; losses = []

    for b in range(0, len(idx), BS):
        batch_idx = idx[b:b+BS]
        prs, cws, cls = [], [], []
        for j in batch_idx:
            prs.append(pairs[j]["prompt"])
            cws.append(pairs[j]["chosen"])
            cls.append(pairs[j]["rejected"])

        with torch.no_grad():
            rw = batched_seq_logprob(ref, prs, cws)
            rl = batched_seq_logprob(ref, prs, cls)

        pw = batched_seq_logprob(policy, prs, cws)
        pl = batched_seq_logprob(policy, prs, cls)

        margin = BETA * ((pw - rw) - (pl - rl))
        loss = -F.logsigmoid(margin).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        step += len(batch_idx)
        losses.extend([-F.logsigmoid(m).item() for m in margin])
        accs.extend([(m > 0).float().item() for m in margin])

        if step % 200 < BS and step >= 200:
            print(f"  ep{ep} step {step}  loss {sum(losses[-200:])/len(losses[-200:]):.3f}  "
                  f"pref-acc {sum(accs[-200:])/len(accs[-200:]):.2f}  {(time.time()-t0)/60:.1f}min", flush=True)

    print(f"epoch {ep}: mean loss {sum(losses)/len(losses):.3f}  pref-acc {sum(accs)/len(accs):.2f}", flush=True)

torch.save(policy.state_dict(), "dpo.pt")
print(f"DPO done in {(time.time()-t0)/60:.1f} min -> dpo.pt", flush=True)