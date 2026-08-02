# Stage 4 — RLVR with GRPO per Recipe ③ §Stage 4 and rl-verifiable-rewards.md.
# Verifiable task (crisp mechanical checker, binary reward — no learned RM):
#   prompt asks for a BRIEF answer; reward = 1 iff (completion stops cleanly) AND (word_count <= NMAX).
#   This is the recipe's "length control / add a length penalty or it rambles" made verifiable.
# GRPO: sample G rollouts/prompt -> verify -> group-relative advantage A_i=(r_i-mean)/(std+eps)
#   -> policy-gradient with a KL leash to the frozen ref. (On-policy single update/batch => PPO clip
#    ratio==1 and is inactive; this is the GRPO first-order form. Logged as a documented nano simplification.)
import json, time, random, torch, torch.nn.functional as F
import model_nano as M

random.seed(0); torch.manual_seed(0)
dev = M.dev
G = 6                 # group size (rollouts/prompt) — vault range 8-64; 6 to bound nano wall-clock
NMAX = 8              # verifiable brevity threshold (words)
BETA_KL = 0.02        # KL leash to reference
LR = 1e-5            # bumped for a visible shift within a bounded step budget
STEPS = 60            # bounded for nano wall-clock (per-token gen w/o KV-cache is the cost)
BATCH_PROMPTS = 6
IME, S = M.IME, M.S

QUESTIONS = ["what is a dog", "how are you", "what is the sky", "name a fruit", "say a greeting",
             "what is water", "how is the weather", "what is a book", "name a color", "what is the sun",
             "say hello", "what is a tree", "how do you feel", "what is music", "name an animal",
             "what is the sea", "what is a house", "say something nice", "what is fire", "name a drink"]
def make_prompt(q): return f"In a few words, {q}?"

held = ["what is a cat", "name a vegetable", "what is the moon", "say a kind word",
        "what is snow", "name a bird", "what is a river", "how do you do"]     # disjoint gate set
json.dump(held, open("grpo_gate_prompts.json", "w"))

def verify(out, stopped):
    """binary verifiable reward: brief AND terminated."""
    if not stopped: return 0.0
    words = len(M.tok.decode(out).split())
    return 1.0 if (1 <= words <= NMAX) else 0.0

if __name__ == '__main__':
    policy = M.load("sft.pt"); policy.train()
    ref = M.load("sft.pt")
    for p in ref.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)

    def rollout_logprob_and_kl(prompt_ids, comp_ids):
        """token log-probs of comp under policy, and per-token KL(policy||ref), teacher-forced."""
        full = (prompt_ids + comp_ids)[:S]; lo, hi = len(prompt_ids), len(full)
        x = torch.tensor(full, device=dev)[None]
        if x.shape[1] < S: x = F.pad(x, (0, S - x.shape[1]))
        plog = F.log_softmax(policy(x)[0].float(), -1)
        with torch.no_grad():
            rlog = F.log_softmax(ref(x)[0].float(), -1)
        lp, kl = 0.0, 0.0
        for pos in range(lo, hi):
            tok_id = full[pos]
            lp = lp + plog[pos-1, tok_id]
            kl = kl + (plog[pos-1].exp() * (plog[pos-1] - rlog[pos-1])).sum()   # forward KL, this step
        return lp, kl

    t0 = time.time(); base_pass = None
    for step in range(1, STEPS+1):
        batch = random.sample(QUESTIONS, BATCH_PROMPTS)
        loss = 0.0; nseq = 0; rewards_all = []
        for q in batch:
            pid = M.chat_ids(make_prompt(q))
            rolls = [M.sample(policy, make_prompt(q), temp=1.0) for _ in range(G)]   # on-policy group
            rs = torch.tensor([verify(o, s) for (o, s) in rolls])
            rewards_all += rs.tolist()
            adv = (rs - rs.mean()) / (rs.std() + 1e-8)                               # group-relative advantage
            for (o, s), a in zip(rolls, adv):
                if len(o) == 0: continue
                comp = o + [IME]
                lp, kl = rollout_logprob_and_kl(pid, comp)
                loss = loss - a.item() * lp + BETA_KL * kl                           # -A*logp + beta*KL
                nseq += 1
        if nseq == 0: continue
        loss = loss / nseq
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        pr = sum(rewards_all)/len(rewards_all)
        if base_pass is None: base_pass = pr
        if step % 25 == 0:
            print(f"step {step:4d}  batch pass@1 {pr:.2f}  loss {loss.item():+.3f}  {(time.time()-t0)/60:.1f}min", flush=True)

    torch.save(policy.state_dict(), "grpo.pt")
    print(f"GRPO done in {(time.time()-t0)/60:.1f} min -> grpo.pt  (rollout pass@1 {base_pass:.2f} -> {pr:.2f})", flush=True)

