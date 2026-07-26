# Stage M — induction-head pretraining curriculum (PRE-REGISTRATION)

*Written 2026-07-26, BEFORE any Stage-M training. Committed as its own commit before
"Stage M measured once". Owner-sequenced next stage (vNext roadmap sequences the
induction/retrieval-circuit probe after the pointer head, now done: Stage P2). Venue:
RunPod GPU (CUDA). Frozen constants inherit the nano recipe unless named below.*

## What Stage M tests, and why now

Stage P2 (REFUTED, manipulation PASSED) localized the scribe OOD copying gap: an explicit,
fully-engaged, copy-dominant pointer head still could not select the correct source token
for held-out values (teacher-forced held-value first-token top-1 = **21%** vs 92% seen).
The bottleneck is **content-addressed source selection that does not generalize OOD** — the
job of an **induction head** (Olsson et al. 2022). Stages C (curriculum), S (scale), and
P/P2 (architecture) have all failed to move the gap. Stage M asks the surviving question:

**H-induction:** the trunk lacks the induction/retrieval circuit for general content-addressed
copy. A pretraining curriculum of synthetic **key→value copy** over *random, unmemorizable*
tokens should induce that circuit, and a scribe built on the induced trunk should copy
held-out values where the baseline trunk cannot.

## Design — two arms, from scratch, identical except the curriculum (à la Stage C)

Both arms: raw nano pretrain → **scribe-SFT directly from the raw pretrain** (precedented by
the Stage S scale anchor, which SFT'd scribe from raw `scale10m_pretrain.pt`) → gate. Nano
recipe: V=4096 (+2 ChatML at SFT → 4098), d=192, L=6, H=6, GQA KV=2, S=512; pretrain 4000
steps × batch 16 × seq 512 (~33M tokens over a ~11–15M-token FineWeb base). Scribe-SFT =
the seed-11 v2 recipe (12000 ex, LR 1.5e-4, 3 epochs). Frozen eval = the byte-identical
40-dialogue set, greedy primary.

- **Arm C (control):** pretrain on the FineWeb base only.
- **Arm I (induction):** pretrain on `(1−ρ)` base + `ρ` induction-curriculum, **ρ = 0.30**.
  Curriculum sequences (deterministic, seeded; token ids sampled from the *non-special*
  vocab): `k1 : v1 ; k2 : v2 ; … ; kq : vq_predict` — repeated `key:value` pairs then a query
  key that re-appears from earlier, whose value the model must **copy**. Keys/values are
  random 1–3-token spans, fresh per sequence (unmemorizable ⇒ copying is the only strategy;
  this is the induction/CopyNet target). Everything else — init seed, optimizer, schedule,
  step count, base tokens, scribe-SFT, eval — identical to Arm C.

Rationale for raw→scribe (not dpo lineage): keeps both arms a clean single pretrain differing
only in ρ; the comparison is the **within-experiment I−C delta**, not a match to the
dpo-lineage anchor. (Base-matching note from CLAUDE-PROGRESS acknowledged.)

## Feasibility pre-gate (BLOCKING — run Arm C first, cheap, before Arm I)

Raw nano (3.15M / ~33M tokens) → scribe-SFT is **unproven** (the existing nano scribe came
from `dpo.pt` = SFT+DPO first; the raw→scribe precedent is only at 10M/200M tokens). If Arm C
cannot clear **parse ≥ 90% and recall ≥ 80%**, its OOD gap is undefined and the I−C delta is
meaningless. So: **run Arm C end-to-end first and check the bars.** If Arm C fails the bars,
STOP before Arm I — the lineage/token-budget is the problem, not the hypothesis; re-scope to
SFT+DPO lineage or a larger token budget. (Cheap: one nano pretrain+SFT+gate on GPU ≈ minutes.)

## Induction probe (BLOCKING) — measured at TWO checkpoints, on NOVEL tokens AND novel form

A REFUTE is only interpretable if (a) the curriculum actually induced a **general** copy
circuit, and (b) that circuit **survived** the full-FT scribe-SFT. Both are live failure modes
grounded in this repo:

- **Generality (guards the P1/P2 trap one level up):** the probe uses **novel token ids** AND
  a **novel surface form** deliberately *unlike* the training game and *nearer the scribe's
  copy-from-context* — a short natural-ish "context … then a cued slot to fill by copying from
  context" eval. Probe-pass then licenses *general* copy induced, not "learned the specific
  game." Metric: in-context copy accuracy on held-out probe items (Arm I must exceed Arm C by
  a pre-registered margin ≥ 30 pts to count as "induced").
- **Survival (C1 warns full-FT destroys copy pathways — 160M full-FT 16.9 vs LoRA 7.1):**
  run the SAME probe at **both** the raw-pretrain checkpoint AND the final scribe checkpoint.

## Readouts — item-gap (continuity) + teacher-forced top-1 (co-primary, better-powered)

- **Item-level OOD gap** (seen−held recall via `gate_scribe.py`) — decision continuity with
  Stage S/C/P (ref v2=22, scale=23, P-baseline=21).
- **Teacher-forced held-value first-token top-1** (co-primary; P2 baseline **21%** vs 92%
  seen) — better-powered, and makes Stage M *directly* comparable to the bolted-on pointer
  head. The sharp question: **does induction-pretraining beat 21%?**

## Pre-registered decision tree (fixed before any result)

Primary faithfulness bars unchanged (parse≥90, recall≥80, halluc≤10, raw-pretrain control
fails). Then, for Arm I vs Arm C:

1. **Probe post-pretrain: did Arm I induce general copy (I−C ≥ 30 pts on the novel probe)?**
   - **NO → VOID** (curriculum failed to induce the circuit; re-scope curriculum, not the hypothesis).
   - **YES →** go to 2.
2. **Probe post-SFT: did the circuit survive full-FT (Arm-I post-SFT probe still ≥ Arm C + 30 / not collapsed)?**
   - **NO (collapsed post-SFT) → NOT a refute:** verdict **"full-FT destroys the induced circuit"**
     → follow-up = LoRA/frozen-layer scribe-SFT (vNext model-side priority A). Named in advance.
   - **YES →** go to 3 (the transfer test is now valid).
3. **Transfer:** vs Arm C, does Arm I move the held-value readouts?
   - **first-token top-1 clears ~≥ 50% (well above P2's 21%) AND/OR item-gap < 10** → **H-induction CONFIRMED** (induction capacity was the bottleneck).
   - **first-token top-1 ≈ 21% and item-gap ≥ 15** → **H-induction REFUTED** (a general, SFT-surviving copy circuit still does not transfer to scribe held-values → the objective or much-larger scale, not addressing capacity per se).
   - between → WEAKENED / report both.

## Confounds acknowledged in advance

Random-token curriculum at ρ=0.30 erodes natural-text competence — hence the Arm-C
feasibility gate and the *within-experiment* I−C control (both arms pay the same base-token
cost; only ρ differs). The probe's generality + two-checkpoint design guard the two ways a
null could be a false REFUTE. Item-gap dilution (n≈28) is why top-1 is co-primary.

## Honest-reporting rule

ρ, curriculum form, probe form, and the decision tree are fixed here and not tuned after any
result. One measurement per arm on the frozen eval + probe. Whatever the tree returns —
VOID / CONFIRM / REFUTE / full-FT-destroys / WEAKENED — is recorded in a RESULT section here,
in `scribe/AUDIT.md`, and `CLAUDE-PROGRESS.md`; artifacts + pod cost frozen. Commit + push.
