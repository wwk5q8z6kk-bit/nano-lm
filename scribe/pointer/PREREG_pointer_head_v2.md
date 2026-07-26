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
