# Program A — First-principles blocker mitigation

**Status:** Design enhancement (docs only)  
**Applies to:** A1 (accepted) + A2 (execute prep)  
**Does not:** grant `AUTHORIZE_E4_RSTAR_V2_EXECUTE`, rebuild worlds, train, or score  

**Principle:** Every blocker is a *mechanism*, not a vibe. Decompose → name competing mechanisms → pick a discriminating fix that is research-legible → pre-register how it can still fail.

---

## 0. What “first principles” means here

For regime value of generation, the atomic question is:

> When is **prediction / assembly / resolution** required, not **lookup / span copy / cue fill**?

Classical extractors are strong when gold is a recoverable string under local cues + lexicon.  
Generative (or hybrid) methods can earn \(\Delta U\) only when the task forces operations classical pipelines lack under information parity—or when they deliver equal Q at lower total system cost.

E4 v1 failed because the world still allowed span copy (`verbatim_span=1.0`) *and* G-ref paid a crushing C. That is two different mechanisms. Mitigate both, or the single revision burns the budget on a confounded test.

---

## 1. Blocker inventory → root cause → mitigation

| ID | Blocker / risk | Root cause (first principles) | Mitigation (research-backed direction) |
|----|----------------|-------------------------------|----------------------------------------|
| **R1** | Span-trivial R★ (`verbatim_span=1.0`) | Gold ∈ contiguous dialogue substring ⇒ span algorithm sufficiency | Constructive anti-span generators (I6–I8); B2 **mandatory** |
| **R2** | Soft `in_Rstar` with B2 false | Probe OR-gate allowed span-trivial worlds | AND-gate: B2 ∧ (≥2 of B1,B3,B4) |
| **R3** | C=40 kills gen even if Q high | Deployment U mixes capability with engineering cost | Dual estimands: \(U_{\mathrm{cap}}\) vs \(U_{\mathrm{dep}}\) (see §3) |
| **R4** | Tiny G-ref (100 steps) confounds “regime” with “undertrained proposer” | Method–regime interaction unidentified | Matched-compute ladder + frozen recipe card (see §4) |
| **R5** | Empty-regime risk if I8 too hard | Anti-span constraints may be unsatisfiable | Existence recipes + builder smoke before freeze (see §2) |
| **R6** | One-revision budget | Single shot must maximize decision information | Pre-registered factorial probes inside one world (see §5) |
| **R7** | Anti-circularity VOID | Temptation to filter by classical scores | Process-only I*; probe only VOID/rebuild |
| **R8** | Classical crippling | Fake gen wins by banning dict/span | Mandatory C-M1/M2/M4; retrieval allowed if priced |
| **R9** | Second KILL ends product line | Ambition precommit | Science off-ramp to Program B/C; product STOP respected |
| **R10** | Infra / process capture | Harness becomes the goal | Keep A about \(\Delta U_{\mathrm{gen}}\); eval infra is tool only |

---

## 2. R1/R2/R5 — Make classical span *structurally* insufficient

### 2.1 First principles

A span/dict system needs:

1. a **surface string** equal (or normalize-equal) to gold somewhere local; and/or  
2. a **cue → slot** map with low ambiguity; and/or  
3. a **closed lexicon** hit.

Break (1) without breaking fair schema: gold must be an **assembled** value (join, normalize, resolve, choose), not a copy.

This is the same reason coreference / multi-hop IE separates “mention detection” from “linking”: evaluating only contiguous max-spans conflates boundary copy with resolution (cf. minimum-span coreference evaluation separating boundary noise from link quality — Moosavi et al., ACL 2019). We want the *inverse construction*: gold that *requires* resolution/assembly, not mere boundary detection.

### 2.2 Constructive generator recipes (existence proofs for I8)

Builder must implement at least three of:

| Recipe | Gold construction | Why span fails | Classical that still gets a fair shot |
|--------|-------------------|----------------|--------------------------------------|
| **Join** | Gold = normalize(piece_a + piece_b) across turns | No contiguous substring equals gold | Dict of pieces + join rules (C-M4-class) |
| **Resolve** | Two candidates present; gold = discourse-chosen one | Span may return wrong candidate | Binding / recency / cue rules |
| **Anaphora** | “same med as before” with antecedent earlier | Surface ≠ gold string | Antecedent stack + rules |
| **Alias** | Spoken brand vs gold generic (locked map in gold meta, not in train dict) | Exact span ≠ gold | Alias table **only if** provided to all methods (parity) or none |
| **Arithmetic/date** | “three days after Monday” → concrete DUR | Not a span | Small date DSL (classical allowed) |

**Mitigation of empty-regime risk (R5):** before any freeze, builder runs a **constructor self-check** (no model scores):

```text
for each open gold:
  assert not is_contiguous_substring(dialogue, gold)
fraction_fail_span >= 0.70   # I8
probe_B2 on frozen C-M2 span oracle must pass
```

If self-check fails → rebuild generator (VOID), do not weaken τ by peeking at gen.

### 2.3 Innovation: “span oracle” as probe instrument

Define a classical **oracle span** arm used **only in B2 probe** (not in bakeoff U):

- It may use gold spans if present in text; it is the upper bound on copyability.  
- B2 = fraction of open golds recoverable by this oracle < 0.35.

This separates “world is copyable” from “our particular C-M2 implementation is weak” (addresses H2 vs H1 confound).

---

## 3. R3/H4 — Dual estimands (capability vs deployment)

### 3.1 First principles

Utility \(U = Q - \ldots - c\cdot C - \ldots\) answers a **product** question.  
Regime question “does generation add *capability* where classical fails?” is a **scientific** estimand. Mixing them caused E4 v1 to answer “expensive nano SFT loses on cost” while Q_G was already competitive under verify-on.

### 3.2 Mitigation (pre-register both; do not hide C)

At execute auth, freeze **two** reports; **one** official product verdict:

| Estimand | Formula | Official for product track? |
|----------|---------|------------------------------|
| \(U_{\mathrm{dep}}\) | A1 §8 default (includes C, M) | **Yes** — SURVIVE/KILL for product line |
| \(U_{\mathrm{cap}}\) | \(Q - 0.5E - 0.3R\) only (L,C,M reported, not subtracted) | **No** — science / diagnosis only |

Rules:

- Sensitivities that zero C **cannot** flip \(U_{\mathrm{dep}}\) official verdict (already in A1).  
- \(U_{\mathrm{cap}}\) may show “gen matches classical on hard slots” even if \(U_{\mathrm{dep}}\) KILLs — that is a **capability finding**, not a product unlock.  
- This is HELM-style multi-axis honesty: do not collapse cost and accuracy into one silent number for all claims.

### 3.3 Innovation: cost–quality Pareto note

Even under KILL on \(U_{\mathrm{dep}}\), record (Q, C) pairs for classical vs gen on R★′.  
If gen dominates classical on Q at equal C band, that informs Program B (efficiency) without violating product STOP.

---

## 4. R4/H3 — Matched compute and method fairness

### 4.1 First principles

You cannot attribute failure to “regime” if the generative arm is arbitrarily under-optimized relative to classical.

### 4.2 Mitigation — frozen recipe card (at execute auth)

```text
G-ref recipe card (example bounds — owner amends at auth):
  base: pinned SHA
  train tokens or steps: matched to classical train cost band OR declared multiplier
  early-stop: on train/dev only
  seed: declared
  venue: declared
```

Add **G-ref-lite** vs **G-ref-matched** only if both are pre-registered; otherwise one G-ref.

### 4.3 Innovation: test-time classical vs gen

Allow classical **search** (k-best spans, constrained decoding-like enumerate) with cost priced in C.  
Allow gen **greedy only** under \(U_{\mathrm{dep}}\) unless test-time compute is pre-registered and priced.  
Prevents “gen wins by unspoken search.”

---

## 5. R6 — One revision: maximize information

### 5.1 First principles

One kill gate should constrain *multiple* mechanisms without peeking.

### 5.2 Mitigation — pre-registered strata inside one world

When building R★′, tag each eval item with strata (metadata only):

| Stratum | Tests |
|---------|-------|
| S_join | Join recipe items |
| S_resolve | Multi-candidate bind |
| S_anaphora | Antecedent assembly |
| S_alias | Alias map items |

Report Q/U **overall** (official) + **per stratum** (diagnostic, non-flipping).  
If overall KILL but S_join shows gen≫classical on \(U_{\mathrm{cap}}\), that is a map for Program B/C—not a SURVIVE.

---

## 6. R7/R8 — Fair classical, no circularity

- Inclusion = generator predicates only (A1 §6).  
- Probe = VOID/rebuild only.  
- Bakeoff classical = C-M1, C-M2, C-M4 mandatory; C-Ret optional with C priced.  
- No post-hoc rule writing after eval reveal (E1 lesson).  
- No “ban dictionaries” without putting that ban’s cost into U (admissibility rejects in REGIME_P1).

---

## 7. R9/R10 — Ambition hygiene

| If | Then |
|----|------|
| \(U_{\mathrm{dep}}\) KILL on R★′ | Product-path gen for this line STOP |
| \(U_{\mathrm{cap}}\) shows localized gen value | Open Program B (capability drivers) or C (verification) with new auth |
| A2 prep becomes endless docs | Stop; mission is \(\Delta U\), not packages |

---

## 8. Enhanced A2 builder order (still gated on execute auth)

Insert **before** B2 world freeze:

| Step | Action |
|------|--------|
| B0a | Implement recipes Join/Resolve/Anaphora (min 3) |
| B0b | Constructor self-check I8 + span-oracle B2 on a **dev sample** (not eval lock) |
| B0c | If fail → fix generator; do not loosen τ |
| B1…B9 | As A2 package |

---

## 9. Success criteria for “mitigations worked” (pre-execute)

Design is adequate when:

1. A constructor can *in principle* emit I8-satisfying gold (existence recipe coded or fully specified).  
2. Dual estimands \(U_{\mathrm{dep}}\) / \(U_{\mathrm{cap}}\) are written so H4 cannot silently masquerade as H1.  
3. B2 mandatory + span oracle remove the E4 v1 loophole.  
4. One-world strata give information even under overall KILL.  
5. Product STOP after KILL remains unambiguous.

---

## 10. Still blocked

```text
AUTHORIZE_E4_RSTAR_V2_EXECUTE = NOT_GRANTED
WORLD_REBUILD / TRAIN / SCORE = FORBIDDEN
PROGRAM1 = NOT_AUTHORIZED
```

This document upgrades **how** we would run the last revision—not permission to run it.
