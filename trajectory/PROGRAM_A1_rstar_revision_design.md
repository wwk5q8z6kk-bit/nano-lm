# Program A1 — R-star revision design (design-only)

**Program:** A — Regime value of generation  
**Unit:** A1 — sole remaining R-star revision **design**  
**Auth:** `AUTHORIZE_PROGRAM_A_RSTAR_REVISION_DESIGN`  
**Status:** `OWNER_ACCEPTED`  
**Accepted via:** `OWNER_ACCEPT_A1_DESIGN` (`research/decision_records/2026-07-31-owner-accept-A1-design.md`)  
**Not:** E4 execute · world freeze · GPU · Program 1  
**Enhancement:** `trajectory/PROGRAM_A_FIRST_PRINCIPLES_MITIGATIONS.md` (blocker mitigations)

Companions: `papers/STRATEGIC_RESET.md`, `papers/AMBITION.md`,  
`trajectory/REGIME_P1_where_classical_fails.md`, `trajectory/PREREG_E4_Rstar_killgate.md`.

---

## 0. Decision question (unchanged in form)

Under a **revised** regime R-star-prime (process predicates I*/X*, not post-hoc classical failure filters), does

```text
U(gen+verify; R*) > U(best classical; R*) + delta
```

on the **same** documents, schema, information budget, and frozen utility?

E1: classical wins on old closed task.  
E4 / R-star v1: classical still wins (KILL).  
A1 asks whether a **non-circular revision** can produce a regime where the question is still informative—or whether ambition should idle after budget exhaustion.

---

## 1. What R-star v1 taught (design constraints)

From `papers/AMBITION.md` / E4 result (verify-on):

| Arm | U_R* |
|-----|------|
| Best classical (C-M2) | ~ 0.638 |
| G-ref + verify | ~ -1.623 |

**Design implications (not new measurements):**

1. Stress was insufficient for gen to matter, or G-ref was too weak / misaligned, or U_R* punished gen paths too hard relative to classical—or some mix. A1 must separate these *as competing hypotheses*, not pick one narrative.
2. **Anti-circularity still binds.** Revising R-star by dropping instances where classical scored well (or gen scored poorly) after seeing scores is **VOID**.
3. **One revision only.** If R-star-prime also KILLs under a future execute auth, product-path generative ambition for this line **stops** (per AMBITION precommit)—science may continue under other programs (B/C).
4. **Do not reopen OLD_TASK_U** or E1.

### Competing post-mortems (write into R-star-prime design; do not execute yet)

| ID | Hypothesis | Discriminating design move |
|----|------------|----------------------------|
| H1 | I* still too template-isomorphic → classical span/dict enough | Strengthen I* toward multi-hop / underspecified / layout-shifted delivery **without** looking at scores |
| H2 | Classical baselines under-powered vs gen unfairly | Freeze stronger classical set (rules + retrieval + constrained) *before* gen |
| H3 | G-ref / adaptation inadequate | If execute later: matched-compute gen recipe; A1 only lists recipe constraints |
| H4 | U_R* weights make gen structurally uncompetitive | Sensitivity table pre-registered; no post-hoc weight shopping after scores |
| H5 | R-star empty of classical-hard after honest B* probes | Prefer VOID/rebuild over forcing a world |

A1 deliverable is a **written choice** of which H* the revision targets (can be primary + secondary), with I*/X*/B* edits that would falsify that H if E4-prime KILLs again.

---

## 2. A1 checklist (docs only)

- [x] One-page postmortem: map E4 failure modes to H1–H5 with evidence citations (existing artifacts only). → §5
- [x] Draft I*-prime / X*-prime / B*-prime diffs vs R-star v1 (process language; no instance cherry-pick). → §6
- [x] State which classical arms are mandatory on R-star-prime (must include at least one non-LM strong baseline). → §7
- [x] Draft U freeze candidate + delta + sensitivity plan (amendable until execute auth, then frozen). → §8
- [x] Explicit KILL / SURVIVE / VOID / GRADED table for E4-prime. → §9
- [x] Compute ceiling and “no paid compute until execute auth” note. → §10
- [x] Owner gate string for later execute: `AUTHORIZE_E4_RSTAR_V2_EXECUTE` (not granted here). → §10


---

## 3. Out of scope for A1

- Rebuilding `trajectory/e4/data/` or freezing a new world  
- Training G-ref / scoring E4-prime  
- Program 1 census / MMLU / HELM  
- Fabric V2 / NanoScribe  
- Committing unrelated dirty freeze/audit files  
- Treating Program 0 eval infra as the research objective  

---

## 4. Status after owner accept

A1 design is **accepted**.  

**Still required for any world/GPU/score work:** separate  
`AUTHORIZE_E4_RSTAR_V2_EXECUTE` (not granted by this accept).

### Companion working notes

**Primary H* target:** H1 (span-trivial delivery); secondary H4 (C/M honesty)  
**I*/X*/B*:** §6 · **Classical set:** §7 · **U / decision / ceiling:** §§8–10

---

## 5. Postmortem from existing E4 artifacts (checklist item 1)

Sources only: `trajectory/results_e4_utility.json`, `trajectory/results_e4_classical_probe.json`, `papers/AMBITION.md`.

### Measured outcome (verify-on primary)

| Arm | Q (approx) | C | M | U |
|-----|------------|---|---|---|
| C-M2 (best classical) | 0.926 | 1.2 | 0.35 | **+0.638** |
| G-ref | 0.859 | **40** | 0.8 | **-1.623** |
| Margin gen − class | | | | **≈ −2.26** |
| Sensitivity flip | | | | **false** → KILL |

### Classical probe (pre-gen)

| Rate / predicate | Value |
|------------------|-------|
| `verbatim_span` | **1.0** |
| `cue_hit` | 0.0 |
| `binding_error` | ≈ 0.65 |
| `train_dict_coverage` | ≈ 0.49 |
| B* fired | B1, B3, B4 (B2 false); `in_Rstar: true` |
| diagnostic C-M1 field acc | ≈ 0.15 |

### Mapping to H1–H5

| H | Support from artifacts | Implication for R-star-prime |
|---|------------------------|------------------------------|
| **H1** (still too span/template-friendly) | **Strong** — `verbatim_span=1.0` means values remain copyable; classical span/dict can dominate | I* must change **delivery process** so answers are not single contiguous copyable spans (multi-source bind, underspecification, non-span targets)—locked before scores |
| H2 (classical under-powered) | **Weak** — C-M2 already Q≈0.93 verify-on | Keep strong classical set; do not cripple classical to save gen |
| H3 (G-ref weak) | **Mixed** — G-ref Q can be high under verify, but tiny nano SFT (100 steps) vs C | If execute later: matched-compute / stronger G-ref recipe; A1 lists constraints only |
| **H4** (U structurally anti-gen) | **Strong** — C=40 and M≈0.8 dominate; even good Q cannot win | Either justify C/M as true costs (then gen must earn efficiency) or pre-register a sensitivity that *cannot* flip the mission question after scores; no post-hoc weight shopping |
| H5 (not classical-hard) | **Partial** — B2 failed but world still `in_Rstar` with 3/4 B*; span rate=1.0 argues “hard” was not span-hard | Tighten B* so span-trivial worlds **VOID** before gen |

### Primary revision target (locked for A1 drafting)

**Primary: H1** — change I* so verbatim span copy ceases to be sufficient.  
**Secondary: H4** — treat engineering/compute cost as load-bearing; R-star-prime must either (a) require gen to beat classical *including* honest C/M, or (b) declare a different product question—but not hide C after seeing scores.

**Falsification if E4-prime KILLs again under H1-targeted I*:** product-path generative ambition for this line stops (revision budget exhausted); science may continue via Program B/C.

### Checklist progress

All A1 checklist items complete in §§5–10. Pending **owner accept**, then separate execute auth if desired.

---

## 6. I*′ / X*′ / B*′ diffs vs R★ v1 (H1-primary)

Base text: `trajectory/REGIME_P1_where_classical_fails.md`.  
Rule: all changes are **generator / gold-annotation process** constraints. No filtering by method scores.

### 6.1 I*′ (inclusion) — keep I1–I5; add I6–I8; tighten I2

| ID | v1 | R★′ change |
|----|----|------------|
| I1 | Eval template family disjoint from C-M1 rule family | **Keep** |
| I2 | ≥30% open gold `needs_norm_or_multispan=true` | **Tighten → ≥60%**; require builder tag `span_atomic=false` on those golds (gold not one contiguous patient substring) |
| I3 | ≥40% open gold absent from train lexicon | **Keep** (supports H1/H5 dict stress) |
| I4 | ≥20% dialogues with ≥2 competing candidates + discourse gold | **Tighten → ≥40%** |
| I5 | ≥30% `cue_family=none\|weak` | **Keep** |
| **I6 (new)** | — | ≥50% of open-slot golds require **cross-turn assembly** (pieces in ≥2 turns) or **anaphora** (“same as before”) per gold metadata |
| **I7 (new)** | — | ≥25% of docs are **multi-block** (dialogue + addendum note, or two speakers with correction) with gold depending on both blocks |
| **I8 (new)** | — | Builder emits `verbatim_span_eligible=false` for ≥70% of open golds (constructor asserts gold ∉ contiguous dialogue substrings after whitespace normalize) |

**Slice satisfies I*′** iff I1, I3, I5 hold **and** I2′, I4′, I6, I7, I8 hold **and** ≥3 of axes A–E marked strong in builder manifest (v1 asked ≥2; raise bar).

### 6.2 X*′ (exclusion) — keep X1–X6; add X7–X8

| ID | Change |
|----|--------|
| X1–X6 | **Keep** (old-task ban, no score peek, schema bound, …) |
| **X7 (new)** | Worlds / docs where builder cannot assert I8 for the required fraction → **inadmissible** |
| **X8 (new)** | Single contiguous “Patient: \<value\>” answer patterns as the sole evidence for an open slot (E1-like copy surface) |

### 6.3 B*′ (probe) — make B2 mandatory; raise bar

| ID | v1 | R★′ |
|----|----|-----|
| B1 | cue-hit \< τ_cue | **Keep**; τ_cue = 0.60 |
| B2 | verbatim-span \< τ_span | **Mandatory true**; τ_span **tightened to 0.35** (v1 measured span rate 1.0 → must fail in) |
| B3 | binding error ≥ τ_bind | **Keep**; τ_bind = 0.20 |
| B4 | dict coverage \< τ_dict ∧ cue low | **Keep**; τ_dict = 0.50 |
| **Pass rule** | ≥2 of {B1..B4} | **B2 must be true AND ≥2 of {B1,B3,B4}** else **VOID / rebuild** (not soft `in_Rstar`) |

**Intent:** A world with `verbatim_span=1.0` can no longer enter R★′. That is the H1 lock.

---

## 7. Classical mandatory set (R★′)

Freeze **before** any G-ref train/score:

| ID | Class | Role |
|----|-------|------|
| **C-M1** | Template/regex (rule budget frozen pre-reveal) | Cue / structure baseline |
| **C-M2** | Train-dict + span | Primary classical competitor (won E4 v1) |
| **C-M4** | Constrained / CRF-lite or equivalent non-LM | Stress that “strong classical” ≠ one method |
| **C-Ret (new, optional but recommended)** | Retrieval over train notes + span | Information-parity: classical may retrieve; ban only if priced in U |

**Forbidden:** crippling C-M2 after seeing G-ref; training classical on eval; adding rules post-reveal.

**Generative:** single G-ref recipe named at execute auth; matched **token/step budget** declared in advance (address H3 without solving it in A1).

---

## 8. \(U_{R★′}\) freeze candidate (H4-honest)

**Form (same skeleton as E4 v1):**

```text
U = Q - 0.5 E - 0.3 R - 0.02 L - 0.05 C - 0.15 M
delta = 0.05
primary_arm = verify-on
construct_primary = exact
mid_stream_edits = FORBIDDEN_VOID after AUTHORIZE_E4_RSTAR_V2_EXECUTE
```

**C (cost) policy — load-bearing:**

- Publish a **cost schedule** at execute auth (hardware class, relative C units).
- G-ref C must reflect train+infer relative to classical (E4 v1 used C_G=40 vs C_M2=1.2 — that is allowed **if** pre-registered, not if silently zeroed later).
- **Sensitivity (pre-registered, not post-hoc shopping):**

| Name | Change | May reverse KILL/SURVIVE? |
|------|--------|---------------------------|
| `default` | as frozen | decision |
| `high_miss` | ↑ E weight | report only |
| `high_review` | ↑ R weight | report only |
| `half_gen_C` | C_G ← 0.5×C_G | **report only — cannot flip official verdict** |
| `no_maintenance` | M=0 | report only |
| `e1_shaped` | E1-like weights | report only |

Official verdict uses **`default` only**. Sensitivities cannot authorize SURVIVE.

**Q definition:** presented / accepted quality under verify-on as in P2; exact primary.

---

## 9. Decision table (E4′)

| Outcome | Condition | Consequence |
|---------|-----------|-------------|
| **SURVIVE** | \(U_G > U_{\mathrm{class}}^\star + \delta\) on default; no protocol VOID | Generative+verify has positive marginal utility on R★′ under frozen U; still ≠ NanoScribe |
| **KILL** | \(U_{\mathrm{class}}^\star \ge U_G - \delta\) on default | Product-path gen for this line **STOP** (revision budget spent) |
| **VOID** | Anti-circularity breach, mid-stream U edit, failed B2-mandatory probe, leakage, or auth mismatch | No product claim; may rebuild **only if** owner grants a new auth (not automatic) |
| **GRADED** | Reserved if P2 defines partial credit; default E4′ = binary SURVIVE/KILL on default U | If unused, treat as KILL-side for product track |

**Class star:** \(\max U\) among frozen classical arms on verify-on.

---

## 10. Compute ceiling + later auth

```text
A1: docs only — $0 paid compute
Until AUTHORIZE_E4_RSTAR_V2_EXECUTE:
  - no world rebuild that locks eval
  - no G-ref train
  - no E4′ scoring
  - no RunPod / paid GPU
```

**Suggested execute ceiling (draft for owner to amend at auth time):**

- Local MPS or one cheap CUDA pod  
- G-ref: max wall-clock / max $ bound written in auth record  
- Eval n and train n frozen in world manifest before train  

**Later execute string (not granted):** `AUTHORIZE_E4_RSTAR_V2_EXECUTE`

---

## 11. Owner accept gate

| Auth | Status |
|------|--------|
| `OWNER_ACCEPT_A1_DESIGN` | **GRANTED** (this proceed) |
| `AUTHORIZE_E4_RSTAR_V2_EXECUTE` | **NOT GRANTED** |

**no data freeze, no GPU, no Program 1** until execute auth.
