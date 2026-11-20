# Stage P — explicit pointer/copy head (PRE-REGISTRATION)

*Written 2026-07-25, BEFORE training the pointer model. Bars and the decision rule may
not move after any result is seen. This file is committed as its own commit before the
"measured once" commit, matching the program's Stage-S / Stage-C cadence.*

## Hypothesis under test

Across the scribe track the residual hallucination has always concentrated on
**out-of-distribution (held-out) values**: the model reconstructs a value from vocabulary
priors instead of copying it from the dialogue it was given. This OOD copying gap has now
survived **both** interventions the program tried:

- **Stage C** (copy-curriculum, 25% gibberish values, 3.15M): gap **unchanged** (22 pts) — curriculum refuted.
- **Stage S** (3.2× scale to 10M, Kaggle T4): raw recall improved and the gate finally
  PASSED, but the seen−held gap was **unchanged** (23 pts vs v2's 22) — scale weakened the
  capacity story without closing the gap.

Stage S's own audit named the next suspect explicitly: *"an architectural copy mechanism
(pointer/induction capacity)."* Stage P builds it and tests it directly.

**H-copy:** the OOD gap is caused by the absence of an explicit copy pathway. Given a
pointer/copy head that can place probability directly on source tokens, a 3.15M model
should copy held-out values it cannot memorize, closing the seen−held recall gap.

## Task (unchanged from scribe v1/v2/G/A/C)

Synthetic doctor–patient dialogue → one structured summary line
`CC: … | DUR: … | SEV: … | MED: … | ALG: …`. Ground truth is the generating fact tuple,
so faithfulness is exact (no judge). Eval = the **byte-identical 40-dialogue set**
(`scribe_eval.json`, 20 held-value items), held-out template families, greedy primary.

## Intervention (the ONLY change vs scribe v2 — fixed before training)

An explicit **pointer/copy head** on the existing nano trunk (d=192, L=6, H=6, GQA KV=2,
V=4098, tied embeddings). For final hidden states `h ∈ (B,S,d)`:

1. Trunk vocab logits (unchanged): `vlog = h · Wembᵀ` → `P_vocab = softmax(vlog)`.
2. Copy attention (new, single head, dim `dc=64`): `q_c=W_qc·h`, `k_c=W_kc·h`;
   scores `= q_c·k_cᵀ / √dc`, **masked to the source region** = positions `[0, P)` where
   `P` = index of the first assistant-content token in the row (the user dialogue + the
   `<|im_start|>assistant\n` header; strictly in the past of every generated token, so
   causal-safe). `α = softmax(scores over source)`.
3. Copy distribution: `P_copy(w) = Σ_{j∈source, src_id[j]=w} α_{t,j}` (probability placed
   on source token ids). Context `c_t = Σ_j α_{t,j} h_j`.
4. Gate: `p_gen = σ(W_g·[h_t ; c_t]) ∈ (0,1)`.
5. **Final distribution:** `P = p_gen · P_vocab + (1 − p_gen) · P_copy`.

Training loss = masked NLL of `P` at the gold next token. For memory, the mixture is
evaluated **only at the target index** at train time (equality-mask trick:
`P_copy_tgt = Σ_j 1[src_id[j]=tgt]·α_{t,j}`, an (B,S,S) tensor, no (B,S,V) copy tensor);
the full mixture over V is materialized only at inference (batch 1). This is exactly the
pointer-generator marginal likelihood (See et al. 2017; Gu et al. 2016 CopyNet).

**Added parameters:** `W_qc (d·dc) + W_kc (d·dc) + W_g (2d·1)` ≈ 192·64·2 + 384 ≈ **24.96k
(~0.79% of 3.15M).** The trunk is initialised from `dpo.pt`; the copy head is random-init;
**both arms full-FT** (rationale: the program's C1 finding is that full-FT destroys the
*implicit* copy pathway — which is precisely why an *explicit* head is the intervention;
full-FT lets the model learn to route through it).

## Frozen configuration (identical to scribe v2 except the head)

Base `../sft/dpo.pt` (SHA-checked); seed-11 v2 data via `build_scribe_data_v2.py`
(12000 examples, 204-value CC space; eval NOT regenerated); LR 1.5e-4, 3 epochs, batch 32,
warmup 0.03, cosine floor 0.1, wd 0.1, clip 1.0, `torch.manual_seed(0)`. Greedy primary
decoding; sampled K=4 @ temp 0.7 reported as a diagnostic.

## Arms (exactly two — no other variants)

- **Arm B (baseline-repro / harness-validity):** plain scribe, full-FT from `dpo.pt` on the
  seed-11 v2 data under *this* harness. **Validity gate:** must reproduce scribe v2's
  seen−held item-level gap to within noise (v2 = 82/94 seen vs 72 held ⇒ **22-pt** item gap;
  accept 22 ± 6). If Arm B does not reproduce the ~22-pt gap, the harness is not comparable
  and the pointer delta is **not attributable** — resolve before interpreting Arm P.
- **Arm P (pointer):** the copy-head model, same base/data/seed/eval.

## Pre-registered bars

**Primary faithfulness bars (unchanged across the whole scribe track):**
1. Parse rate ≥ 90%
2. Fact recall ≥ 80%
3. Hallucination rate ≤ 10% (fabrications + substitutions)
4. Base control (`dpo.pt`) fails the bars (discrimination — capability caused by this stage)

**Hypothesis-specific decision rule (the real point of Stage P) — OOD gap:**
`gap = seen-value recall − held-out-value recall`.
- **PRIMARY decision metric = item-level gap** (continuity with Stage S/C's 22/23-pt rule,
  computed identically by `gate_scribe.py`).
- **ALSO REPORTED = value-level gap** on the specific held-out fields (the clean value-level
  metric from Paper 1) — the copy head's effect lives on the held CC/MED/ALG *tokens*, and
  the item-level number dilutes it with always-seen DUR/SEV.

Decision (symmetric with Stage S, reference: v2 = 22 pts, scale = 23 pts):
- pointer item-gap **< 10 pts** → **H-copy CONFIRMED** (explicit copy mechanism closes the OOD gap).
- pointer item-gap **≥ 15 pts** → **H-copy REFUTED** (mechanism insufficient at 3.15M) —
  points to the objective or much-larger scale as the remaining suspects.
- **10–15 pts** → WEAKENED / ambiguous.

## Manipulation check (BLOCKING — a null is uninterpretable without it)

A random-init copy head on a competent trunk can leave `p_gen → 1` (copy never used), which
produces the **same** gap-unchanged null as a true capacity verdict. So the null arm is only
interpretable if the copy pathway was actually exercised.

Manipulation statistic, measured at **held-out value target-token positions** on the eval set:
`M = mean[ (1 − p_gen) · P_copy_tgt / P_tgt ]` — the share of the predicted-token probability
contributed by the copy channel. Corroborated by (a) mean `p_gen` and (b) mean copy-attention
mass landing on the correct source value span.
- **M ≥ 0.5** → copy pathway exercised → a gap result is a genuine verdict.
- **M < 0.2** → copy channel essentially unused → **VOID, not FAIL**: a re-run that engages
  the copy pathway (e.g. copy-supervision aux loss or lower gate-bias init) is a failed
  manipulation being fixed, **not** bar-chasing.
- **0.2 ≤ M < 0.5** → partial engagement; interpret with explicit caution, report both.

## Tokenization precondition (checked & recorded BEFORE this PREREG was committed)

The equality-mask copy can only fire when a value's target token-ids also appear among the
source token-ids. Measured in-context (offset-mapped) on all 40 eval items:
**per-id copyability = 100.0% (held 118/118, seen 178/178); full-value copyable = 100%
(held 28/28, seen 47/47).** Every value token the model must copy is id-present in the
source (e.g. `neck pain → Ġne|ck|Ġpain`, all in source). A null therefore **cannot** be
blamed on tokenization.

## Confound acknowledgment (stated in advance)

The pointer head adds ~0.79% parameters. The decision keys on the **gap**, not raw recall,
and a **+217%** parameter increase (Stage S) already moved raw recall while leaving the gap
unmoved — so any gap-closure here is attributable to the mechanism, not the ~25k params.
Arm B (same base/data, no head) is the param-free reference under the identical harness.

## Honest-reporting rule

Single measurement per arm on the frozen eval set. No post-hoc tuning after seeing a result.
Whatever the gate and the decision rule say — CONFIRMED, REFUTED, WEAKENED, or VOID — is
recorded here in a RESULT section and in `scribe/AUDIT.md`, and the artifacts are frozen.

---

## RESULT — measured once 2026-07-26, verdict: **VOID (failed manipulation check)**

Build dir `~/AI-builds/nano-scribe-pointer-2026-07-25/`; seed-11 v2 data (12000 ex);
both arms full-FT from `dpo.pt`, identical batch order. Added params +24.57k (~0.78%),
as pre-registered. Frozen artifacts: `result_baseline.json`, `result_pointer.json`,
`train_*.log`, `gate_*.log`. (Operational note, PREREG-pre-authorized: the copy head's
(B,S,S) attention runs ~30× slower on MPS than the plain trunk — 143 min/train vs ~5 min;
this is a runtime cost, not a result confound. Optimization is the first task of any re-run.)

**Arm B (baseline-repro / harness validity) — PASS.** greedy: parse 98%, recall 80%,
halluc 13%, omission 10. ITEM gap **21 pts** (held 71 / seen 92) — within the pre-registered
22 ± 6 of scribe v2 (22 pts). The harness reproduces the OOD gap ⇒ any pointer delta is
attributable. VALUE-level gap **92 pts** (held **0%** / seen 92%) — the specific held-out
field values are missed 100% of the time; the item-level number dilutes this with
always-correct DUR/SEV.

**Arm P (pointer) — GATE FAIL, and VOID by the manipulation check.** greedy mixture:
parse 90%, recall 76%, halluc 12%, omission 4. ITEM gap **24 pts** (held 71 / seen 95);
VALUE gap **83 pts** (held **10%** / seen 93%). Raw, this reads like the ≥15-pt REFUTE band.

**But the BLOCKING manipulation check fires VOID:** at n=118 held-out value target tokens,
**M (copy-share) = 0.18 < 0.2**, mean **p_gen = 0.83**, mean copy-mass on the correct source
id = **0.05**. The copy pathway was **essentially unused** — full-FT drove `p_gen → 0.83`
(vocab-dominant) and the copy attention never learned to point (5% mass on the right token).

**Why (diagnosed, not tuned):** the training templates are fully memorizable (train loss
→ 0.000 by step 100; on-batch copy_share held ~0.07 throughout). Vocab memorization is
always available on the training distribution, so the objective **never rewarded copying** —
the copy head got no gradient to learn pointing. This is the same "memorization was always
available" mechanism Stage C found, now surfacing at the head level. The value-level held
recall barely moved (0% → 10%, noise-level), consistent with a mechanism that never engaged.

**Verdict per the pre-registration:** M < 0.2 ⇒ **VOID, not FAIL/REFUTED.** H-copy is
**neither confirmed nor refuted** — the experiment did not put the hypothesis at risk
because the mechanism was not exercised. The manipulation check did exactly its job: it
prevented a false "pointer head refuted, gap unchanged" conclusion (item-gap 24 pts would
have read as REFUTE). The pre-registered remedy — a re-run that engages the copy pathway
(copy-supervision aux loss and/or lower gate-bias init) — is a failed manipulation being
fixed, explicitly **not** bar-chasing. That re-run is **Stage P2** (fresh pre-registration).

**What transferred (the durable lesson):** on a memorizable training distribution, adding
an *unsupervised* copy head under full-FT is insufficient — the pathway must be *supervised*
or *made necessary* (unmemorizable values) or it collapses to the vocab channel. This is
itself a finding about how to induce copy mechanisms in small models, and it directly
motivates P2's design.
