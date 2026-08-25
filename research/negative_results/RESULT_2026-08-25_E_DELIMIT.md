# RESULT — E-DELIMIT: arm B is VOID, H5 is untested

**Date:** 2026-08-25
**Prereg:** `research/preregistrations/PREREG_E_DELIMIT.md` (rule from `23ede77`)
**Authorization:** experiment-scoped, granted by the owner 2026-08-25, recorded in the prereg
**Cost:** \$0 — four local MPS runs, 55s–2m each

---

## Headline

**The manipulation failed. Arm B did not leave retrieval as hard — it destroyed
it.** The kill condition's precondition does not hold, so it cannot be
evaluated, and **H-delimit is neither confirmed nor refuted.**

Arm B's `asserted_grounded` is 0/192, which read naively against the rule
(`< 25% of LOCATED` ⇒ REFUTED) would fire the kill condition. Reporting that as
"H5 REFUTED" would be reporting a confounded arm — the same error that voided
the C3 arm in `ddb5ce6`, one level up.

## The four runs

| arm | run | `asserted_grounded` /192 | **LOCATED /120** | coverage |
|---|---|---|---|---|
| **A** — free-form (the published baseline) | `e04b3016` | 2 | **97 (80.8%)** | 0.818 |
| **A′** — free-form replication on the harness | `38b12909` | 2 | **97 (80.8%)** | 0.818 |
| **B** — menu (the discriminator) | `4de84c18` | **0** | **30 (25.0%)** | 1.000 |
| — *R5 index-0 parrot on B's own menus* | *(software)* | *2* | ***23 (19.2%)*** | — |
| **C** — offsets (secondary) | `aa779aba` | 0 | 2 (1.7%) | 0.781 |

## Why this is VOID and not REFUTED

The arm's premise, in the prereg's own words: *"It removes generation of the
boundary and leaves only selection of it, **while leaving retrieval exactly as
hard**."*

Retrieval was not left as hard. LOCATED fell from **97 to 30** — a factor of
3.2 — and the survivor count sits **7 slots above a constant index-0 baseline**
(23). A model that can no longer *find* the evidence cannot be asked whether it
can *delimit* it. The contrast varied two things.

**The mechanism is visible in the model's own picks.** Across all 192 slots the
index choices are strongly front-biased:

- median index **13.5** against a median menu size of **64**
- median relative position in the menu **0.211** (uniform would be ~0.5)
- **53%** of picks have index < 20
- indices 1 and 2 alone account for **64 of 192 picks (33%)**

The model is not reading a 38–120 item flat list and selecting from it. It is
picking near the front. Arm B measured indexing behaviour over a long enumerated
list, not boundary selection.

## What the R5 guard caught, and what it missed

R5 was discharged before the run and **passed**: the index-0 parrot scored
2/120 (1.7%) on exact extent, at chance rather than at ceiling, so the menu was
not ordered by anything correlated with gold. That guard did its job.

It was the wrong yardstick for this failure. R5 checks that the *menu* does not
leak; nothing checked that the *model can use the menu*. The number that would
have caught it is the parrot's **LOCATED** — 19.2%, against which arm B's 25.0%
is nearly indistinguishable. **A LOCATED-invariance check belongs in the next
version of this prereg as a blocking manipulation check**, computed before the
primary endpoint is read:

> If arm B's LOCATED is not within a pre-specified band of arm A's LOCATED, the
> arm is VOID and the primary endpoint is not reported.

## What did work

**Arm A′ reproduces the published baseline exactly** — the R1 control passes and
the output-format refactor is behaviour-preserving. Every cell of the extent
census matches `span_extent_L000_unified.json`:

```
grounded_exact 2 · located_over_extended 95 · located_under_extended 0
not_located 8 · no_quote 15 · no_gold_span 72 · LOCATED 97/120 = 80.8%
```

and the across-instance vector is identical (`asserted_grounded` 0.1667±0.3892,
`asserted_unbound` 3.5833±0.9003, `asserted_bound_wrong` 9.3333±0.8876,
`observed_coverage` 0.8177±0.0322).

The R1/R2 hash discipline also held: `question_template_hash` is equal across all
three arms and `output_format_hash` differs across all three, and the arm B
branch diff against its parent is exactly one line.

## Arm C

LOCATED 2/120. Pre-registered as secondary (P6) precisely because it requires
index arithmetic over transcript offsets — a known weakness independent of
delimitation — so this is **not** evidence about H5 and is reported only for
completeness. Its first run (`9b77e056`) crashed at the first slot on an
un-trimmed offset slice (`EncounterError: invalid_string`); that run answered
nothing, so the node was repaired in place and re-run rather than branched.

## Where this leaves the program

- **H5 (delimitation) stands untested.** It is not weakened. The span-port line
  does not rejoin the retrieval hypotheses on this evidence.
- **The A/A′ finding is unchanged and now replicated:** the model locates the
  gold-bearing turn for 97 of 120 slots and delimits the span in 2, with all 95
  non-exact located quotes over-extended and none under-extended.
- **The next experiment is a repaired arm B**, not a new hypothesis. Two
  candidate repairs, both cheap, both needing their own pre-registration and
  authorization:
  1. **Two-stage elicitation** — ask for the turn first, then enumerate only
     that turn's sub-spans. This concedes that retrieval and delimitation cannot
     be held simultaneously fixed in one prompt, and measures delimitation
     *conditional on* correct retrieval, which is the quantity H5 is actually
     about.
  2. **Shorter menus with a positional control** — cap the menu and randomise
     gold's position across slots, reporting the pick-position distribution as a
     blocking check rather than a diagnostic.

  Option 1 is the better experiment: it targets the conditional quantity
  directly and sidesteps the long-list indexing failure entirely, at the cost of
  no longer being a single-prompt contrast.

## Provenance

- Suite `campaign_v2`, revision `campaign_v2_multi_20260825`, 60 encounters,
  192 slots, 12 instances.
- Condition `C1off_C2off_Qon_QSoff` — identical across all arms, unchanged.
- Model Qwen2.5-1.5B-Instruct, pinned revision `989aa798`, greedy decoding.
- Base commit `9a3ecd4` (the L000 instrument), **not** the branch the prereg text
  lives on — `work/leakage-power-analysis` @ `23ede77` carries the superseded
  yes/no question form and has no 12-instance suite. Recorded as pre-commitment
  P1; running E-DELIMIT there would have silently reproduced the C3 confound.
- Pre-run instrument validation (P7): `gold_in_menu` 120/120, oracle 120/120,
  R5 parrot 1.7%, menu size min 38 / median 64 / max 120.
