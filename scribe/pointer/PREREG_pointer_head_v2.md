# Stage P2 — copy-SUPERVISED pointer head (PRE-REGISTRATION)

*Written 2026-07-26, BEFORE training the P2 model, AFTER P1's VOID. Committed as its own
commit before "P2 measured once". This is the manipulation fix the Stage P PREREG
pre-authorized ("a re-run that engages the copy pathway … is a failed manipulation being
fixed, NOT bar-chasing"). Everything not named below is FROZEN identical to
`PREREG_pointer_head.md` (task, base `dpo.pt`, seed-11 v2 data, eval set, primary bars,
OOD-gap decision rule, manipulation check, honest-report rule).*

## Why P2 exists

Stage P run-1 was **VOID**: the unsupervised copy head was never exercised (M=0.18,
p_gen=0.83, copy-mass=0.05). Diagnosed cause: training templates are fully memorizable
(train loss→0.000), so the NLL objective **never rewarded copying** and the copy attention
got no gradient. P2 gives the copy pathway a training signal so the hypothesis can actually
be put at risk.

## Intervention (the ONLY delta vs P1 — all three engage the copy pathway; frozen)

1. **Copy-supervision aux loss.** At every assistant target position whose gold token is
   *copyable* (appears in the source region), add
   `L_copy = −log( Σ_{j∈source, src_id[j]=tgt} α_{t,j} )` (i.e. −log P_copy_tgt). Total loss
   `= L_nll + λ·L_copy`, **λ = 1.0**. This directly trains α to place mass on the source
   position holding the correct next token — a supervised-copy signal (Gu et al. CopyNet).
2. **Copy-favoring gate init.** `W_g.bias = −2.0` ⇒ `p_gen` starts ≈ 0.12 (copy channel
   exercised from step 1, rather than the P1 start of 0.5 that drifted to 0.83).
3. **Source-key-restricted copy attention** (runtime only, behaviour-equivalent): copy keys
   cropped to `[0, Ksrc)` (Ksrc = batch-max source length). Non-source keys were −inf-masked
   in P1 anyway, so α/P_copy are identical; this is ~16× faster on MPS (smoke: 32k vs 2k
   tok/s), making the run tractable (~10 min vs 143).

`Wg.bias` and λ are frozen here and may not move after any result is seen.

## Arms

- **Arm B (baseline):** unchanged — reuse `baseline.pt` from P1 (same harness, item-gap 21,
  value-gap 92). No retrain (P2 changes nothing on the baseline side).
- **Arm P2 (copy-supervised pointer):** `GPTCopy2`, full-FT from `dpo.pt`, seed-11 v2 data.

## Bars, decision rule, manipulation check — UNCHANGED from P1

Primary faithfulness bars (parse≥90, recall≥80, halluc≤10, base control fails). OOD decision
on the **item-level gap** (primary; ref v2=22, scale=23, baseline-arm=21): **<10 CONFIRM
H-copy, ≥15 REFUTE, 10–15 WEAKENED**; **value-level gap** on held fields also reported (P1
baseline held 0% / seen 92%).

**Manipulation check now gates INTERPRETATION** (its role sharpens post-P1): compute
`M = mean copy-share at held-out value target tokens`.
- **M ≥ 0.5** → copy pathway engaged → **the gap decision rule is now BINDING and REAL.**
  Gap closes ⇒ H-copy CONFIRMED; gap persists ⇒ H-copy **REFUTED** (mechanism engaged but
  does not generalize → points to the objective or much-larger scale).
- **M < 0.2** → the fix ALSO failed → deeper VOID; the head design or supervision is wrong,
  not the hypothesis. Re-scope, do not claim a verdict.
- **0.2 ≤ M < 0.5** → partial; report both, interpret with explicit caution.

## Honest-reporting rule

Single measurement on the frozen eval set. λ and gate-bias were fixed before the run and
are not tuned after. Whatever the decision rule + manipulation check say is recorded in a
RESULT section here and in `scribe/AUDIT.md`; artifacts frozen. A third run would require
a genuinely new, separately-motivated hypothesis — not another turn of the same knob.

---

## RESULT — measured once 2026-07-26, verdict: **H-copy REFUTED** (manipulation PASSED)

Train 5.8 min (source-key restriction: 53k tok/s, 16× P1). Frozen artifacts:
`result_pointer2.json`, `train_pointer2.log`, `gate_pointer2.log`, `tf_diagnostic.py`.

**Manipulation check PASSED — the copy pathway was genuinely engaged this time:**
at n=118 held-out value target tokens, **M (copy-share) = 0.97 ≥ 0.5**, mean **p_gen = 0.09**
(copy-dominant), copy-mass on the correct source id = 0.40. The fix worked: from P1's
M=0.18 (unused) to P2's M=0.97 (dominant). ⇒ **the gap decision rule is now BINDING.**

**Gate (greedy, primary): FAIL.** parse 90%, recall 71%, halluc 16%, omission 6.
**ITEM gap = 25 pts** (held 65 / seen 90) ≥ 15 ⇒ **H-copy REFUTED.** VALUE gap 82 pts
(held 10 / seen 92).

**Confound-free headline evidence (same free-running regime, cross-run):** driving copy
engagement from ~0 (P1, M=0.18) to dominant (P2, M=0.97) moved free-running held value-recall
**10% → 10% — zero points.** (The 10% is the model's noise floor: P1 with the copy channel
*unused* also read 10% at n=28 held field-decisions; it is NOT a copy effect and is not
credited to the head. Baseline arm B without the head read 0%.) An explicit, supervised,
copy-dominant pointer head does not close the OOD gap.

**Why (now MEASURED via the teacher-forced top-1 diagnostic, not inferred — `tf_diagnostic.py`):**
the failure is dominated by **content-addressed source selection that does not generalize
OOD**, with free-running exposure bias as a secondary compounder.
- Teacher-forced top-1 at held-value tokens (clean gold prefix, copy-dominant): **41% all
  tokens / 21% first-token** — vs ~92% for seen values. Even handed a clean prefix and a
  copy-dominant gate, the head selects the correct held-out source token only ~1/5 of the
  time. Addressing genuinely fails to generalize; this is not merely a decoding artifact.
- Free-running greedy (10%) < teacher-forced (41%) ⇒ exposure bias DOES compound the
  addressing errors on multi-token spans (e.g. `neck pain → "trou pain"`, `sulfa drugs → sild`)
  — but it is secondary: the teacher-forced ceiling itself (21% first-token) is far below
  closing the gap, so decoding is not the primary cause.

**Scope of the claim (do not overclaim):** this REFUTES *"a supervised copy-dominant pointer
head (gate-bias −2, λ=1, p_gen≈0.09) closes the OOD gap"* — one corner of the design space.
Forcing copy-dominance also made the model globally worse (recall 71 vs baseline 80, halluc
16 vs 13), so this is not a free lunch even ignoring generalization. It does not claim
"copy mechanisms cannot help" in general.

**Program implication (the durable finding):** the scribe OOD copying gap is **not an
output-mixture problem** — an explicit copy channel, fully engaged, relocates the failure
into the copy-attention's query–key addressing, which fails to generalize the *same way*
the implicit mechanism did (scribe v1's position-anchored extraction). Combined with Stage C
(curriculum refuted) and Stage S (scale left the gap unmoved), the surviving suspects narrow
to **content-addressed retrieval/induction-circuit capacity, much-larger scale, or the
objective** — sharpening the Stage M (mechanism) / Paper-3 direction. And it re-confirms the
systems conclusion: no cheap architectural output-side fix eliminates the tail, so the
Stage G/A verification layer stays load-bearing. Per protocol: no third run (that needs a
new hypothesis, e.g. an induction-head-pretraining curriculum, not another knob-turn here).
