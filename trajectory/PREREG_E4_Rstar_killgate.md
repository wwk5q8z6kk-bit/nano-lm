# P2 / PREREG — E4 kill gate on regime R★

**Status:** `DESIGN_DRAFT` / `EXECUTION_BLOCKED` / `WORLD_NOT_FROZEN` / `NO_DATA` / `NO_RESULT`
**PROGRAM_EXECUTION_STATUS:** `IDLE_AFTER_FREEZE` · **AUTHORIZED_NONEXECUTION_WORK:** `E4_DESIGN_ONLY`  
**Owner auth in force:** `AUTHORIZE_E4_DESIGN_ONLY` (2026-07-31) — **docs only**.  
**Not authorized:** Stage 4 runs, R★ world freeze, GPU/paid compute, E2, fabric v2.  
**Public freeze tag:** `post-alpha-evidence-freeze-2026-07-31` (evidence packaging; ≠ E4 execute).  
**Ambition:** `papers/AMBITION.md`

This document is the **design-complete** E4 package candidate. Marked
**DESIGN DRAFT** wherever weights/thresholds remain amendable *until* a separate
`AUTHORIZE_E4_EXECUTE` freezes them without mid-stream edits after scores.

| Link | Role |
|------|------|
| `papers/AMBITION.md` | IDLE ≠ halt; ambition framing |
| `papers/SEQUENTIAL_PIPELINE.md` | Stages / gates |
| `trajectory/REGIME_P1_where_classical_fails.md` | R★ I\*/X\*/B\* (anti-circular) |
| `trajectory/PREREG_E1_nonlm_baseline.md` | Old-task kill gate (**DONE — KILL**; do not reopen) |
| `trajectory/PIPELINE_GATE_LOG.md` | Gate status |
| `audit/discussion-to-implementation/WITHDRAWAL_SPEC.md` | Soft-claim withdrawals |
| `audit/discussion-to-implementation/IDLE_AFTER_FREEZE.md` | Freeze posture + design carve-out |

**Constants (immutable mid-stream once execution starts):**

```text
E1_VERDICT = KILL                    # scoped to OLD_TASK_U + closed world
PAPER_α = FROZEN
OLD_TASK_RUNS = FORBIDDEN            # under OLD_TASK_U / m0–m4 isomorphism
E2_DEFAULT = GATED
FABRIC_DEFAULT = GATED               # Fabric ≠ NanoScribe
E4_STATUS = DESIGN_DRAFT / EXECUTION_BLOCKED / WORLD_NOT_FROZEN / NO_DATA / NO_RESULT
```

---

## 0. Decision question

> Under regime **R★** (inclusion via process predicates I\*, not post-hoc classical
> failure cherry-picks), does a generative proposer + verify achieve higher expected
> utility than frozen classical / constrained baselines on the **same** documents,
> schema, information budget, and metrics?

E1 answered the analogous question for the **non-regime** (closed isomorphic task):
classical wins. E4 asks it only inside R★.

**Ambition (not a prediction):** find whether
\(U_{\mathrm{gen+verify}}(R★) > U_{\mathrm{classical}}(R★)\) under matched Q/E/R/L/C/M.

---

## 1. DESIGN DRAFT — freeze \(U_{R★}\)

> **DESIGN DRAFT:** weights below are the candidate freeze for a future execution
> authorization. Changing any weight after seeing E4 scores **VOIDs** the decision.
> Pre-execution amendments require an explicit owner note before unlock.

### 1.1 Decision the utility encodes

Emit a schema-valid structured summary a downstream consumer can trust, under
bounded review and engineering cost, when inputs are drawn from R★.

### 1.2 Definition (Q, E, R, L, C, M)

Per evaluation document (then mean over locked instances \(K\)):

\[
U_{R★}^{\mathrm{draft}} = Q - 0.5\,E - 0.3\,R - 0.02\,L - 0.05\,C - 0.15\,M
\]

| Symbol | Meaning | Unit | Weight | Continuity |
|--------|---------|------|--------|------------|
| \(Q\) | Presented precision (verify-on primary; verify-off reported) | [0,1] | \(+1.0\) | was \(P\) in E1 / prior P2 |
| \(E\) | Error/miss rate = 1 − recall on fields that should emit (omissions + wrong under construct policy §1.4) | [0,1] | \(−0.5\) | was miss \(M\) in E1 — **renamed** so \(M\) can mean maintenance |
| \(R\) | Review load = fraction of fields routed to human | [0,1] | \(−0.3\) | was \(\rho\) (review load, **not** hallucination) |
| \(L\) | p50 end-to-end latency per document | seconds | \(−0.02\)/s | same |
| \(C\) | Relative compute vs frozen C-M1 on same hardware class | ≥0 | \(−0.05\) | same |
| \(M\) | Maintenance / engineering burden (normalized) | [0,1] | \(−0.15\) | **new vs E1** — implementation + upkeep cost |

**Binding stress (mandatory report; optional inside \(U\) via sensitivity):**
\(\beta_{\mathrm{bind}}\) = binding-error rate on multi-candidate docs (axis D).
Default kill uses \(U\) **without** folding \(\beta_{\mathrm{bind}}\) into the primary sum
(avoids double-count with \(E\)). Sensitivity grid includes Binding-heavy
(\(+0.2\,\beta_{\mathrm{bind}}\) penalty).

**Liability proxy (mandatory report, not inside \(U_{R★}\) v1):** count of
fabrications+substitutions that would be *presented* without review.

### 1.3 How \(M\) (maintenance) is scored

Freeze a 0–1 rubric **before** execution (same sheet for all methods):

| Factor (equal weight unless amended) | Low \(M\) | High \(M\) |
|--------------------------------------|-----------|------------|
| Rule/lexicon/model surface area | Tiny frozen regex/dict | Large FT model + prompt/stack |
| Update burden when ontology drifts | Edit dict/rules | Retrain / re-LoRA |
| Operational deps | No GPU; CPU rules | GPU serving + adapter pins |
| Failure diagnosis cost | Deterministic traces | Opaque generations |

Score each factor in {0, 0.5, 1}; \(M\) = mean. **Pre-assign** \(M\) to each
named baseline at recipe freeze — do not re-rate after seeing \(Q/E\).

Default design assignments (amendable pre-execution only):

| Method | \(M\) draft |
|--------|-------------|
| C-M1 | 0.20 |
| C-M2 | 0.35 |
| C-M4 | 0.40 |
| G-ref (small FT/LoRA) | 0.75 |
| G-ref + verify-on | 0.80 |

### 1.4 Why not reuse `OLD_TASK_U` unchanged?

| Choice | Reason |
|--------|--------|
| Keep Q/E/R/L/C shape | Continuity with E1 tradeoffs |
| Add \(M\) | Fairness: generative often wins Q only by hiding eng cost |
| Rename miss→\(E\) | Free \(M\) for maintenance |
| Do **not** drop \(C\)/\(L\)/\(M\) | Cost must remain visible |
| Do **not** reopen old task | Forbidden |

### 1.5 Construct policy for “correct”

Primary kill decision: **exact string match** (α / E1 continuity).

Also report:

1. Exact  
2. Normalize-then-match (`e1/common.py::normalize_value`)  
3. Bounded soft/human-acceptable sample only if pre-registered (≤50 items) — **not**
   a substitute for dual-clinician IAA; E3 remains agent-rubric, human arm `NOT_RUN`

### 1.6 Sensitivity (pre-registered)

| Grid point | Change |
|------------|--------|
| Default | §1.2 |
| High-miss | \(E\) weight \(1.0\) |
| High-review | \(R\) weight \(0.6\) |
| No-maintenance | drop \(M\) term |
| Binding-heavy | subtract \(0.2\,\beta_{\mathrm{bind}}\) |
| E1-shaped | drop \(M\); symbols as \(P,M_{\mathrm{miss}},\rho,L,C\) |

Kill/survive uses **default**. Any sensitivity flip → **GRADED**.

### 1.7 Margin

\[
\delta_{R★} = 0.05
\]

---

## 2. Freeze baseline family + information parity

All methods see **identical** R★ eval documents and schema
`CC | DUR | SEV | MED | ALG`. No post-hoc method adds after unlocking labels.

### 2.1 Classical freeze-set (required)

| ID | Method | Freeze rule |
|----|--------|-------------|
| **C-M1** | Template / regex slot filler | Rule budget written **before** eval reveal |
| **C-M2** | Train-dict + span | Train lexicon only; leakage check |
| **C-M4** | Constrained / copy-only open slots (recommended) | Schema-constrained; no free open-vocab emit |

C-M1 and C-M2 **mandatory**. C-M3 (CRF/BIO) optional if alignments exist.

### 2.2 Generative references (required)

| ID | Method | Freeze rule |
|----|--------|-------------|
| **G-ref** | Best available **small** generative proposer under a **frozen recipe** | Named before run; or pre-commit max of ≤2 frozen recipes |
| **G-strong** (optional) | One stronger LM | Quota only; not primary unless pre-registered |

**Minimum valid E4:** {C-M1, C-M2, G-ref}.  
**Recommended:** + C-M4 and verify-on/off for all emitters.

### 2.3 Verifier arms

**verify-off** and **verify-on** (grounding+absence style presenter, adapted to R★
with span provenance where applicable). Primary decision: **verify-on** \(U_{R★}\).

Fabric remains a regression harness — **≠** NanoScribe; using verify-on does not
authorize fabric v2 / product architecture.

### 2.4 Forbidden mid-flight

- Adding LLM methods to the “classical” set  
- Expanding C-M1 rules after peeking at eval  
- Training generative refs on eval templates / held lexicons  
- Re-scoring old m0–m4 under `OLD_TASK_U` as if it were E4  
- Filtering eval instances using scores after generative unlock  

### 2.5 Information-parity / baseline fairness matrix

What each side **may know** (Y) / **must not know** (N) / **shared** (S):

| Information | C-M1 | C-M2 | C-M4 | G-ref train | G-ref eval | Notes |
|-------------|------|------|------|------------|------------|-------|
| Schema `CC\|…\|ALG` | S | S | S | S | S | Identical |
| Train documents | S | S | S | S | N (weights frozen) | Same train corpus |
| Train open lexicon | Y (optional) | Y | Y | Y (via data) | N | Hash-locked |
| Eval documents | Y (at score time) | Y | Y | N | Y | Identical eval set |
| Eval gold labels | N | N | N | N | N | Scorer only |
| Eval template-family IDs / held surface pool | N at rule-lock | N | N | N | N | Prevents isomorphism cheat |
| C-M1 rule file (pre-lock) | Y | N | N | N | N | Classical-only artifact |
| Dev split for hparam search | N for rules | N for rules | N | Y (frozen before final) | N | Classical rules not tuned on dev to chase G-ref |
| Verifier \(R\) code | S if verify-on | S | S | S | S | Same presenter |
| Hardware class for \(L,C\) | S | S | S | S | S | Report device; no silent GPU vs CPU mismatch without \(C\) |
| \(M\) rubric sheet | S | S | S | S | S | Pre-assigned |

**Parity principle:** no method receives eval gold, held template IDs, or
post-hoc rule edits. Generative may use train text that classical also sees;
classical may use explicit rules/dicts that generative does not get as privileged
side files — that asymmetry is intentional and priced via \(M\) and \(C\).

---

## 3. Precommitted consequences (KILL / GRADED / SURVIVE / VOID)

Let

\[
U^{\star}_{\mathrm{class}} = \max_{m \in \{\mathrm{C\text{-}M1},\mathrm{C\text{-}M2},\mathrm{C\text{-}M4}\}} U_{R★}(m)
\]

\[
U^{\star}_{\mathrm{gen}} = U_{R★}(\mathrm{G\text{-}ref,\ verify\text{-}on})
\]

(If C-M4 VOID, max over available classical only.)

On **default** \(U_{R★}^{\mathrm{draft}}\), mean over locked instances:

| Verdict | Rule | Program will do |
|---------|------|-----------------|
| **KILL** | \(U^{\star}_{\mathrm{class}} \ge U^{\star}_{\mathrm{gen}} - \delta_{R★}\) | **Stop** generative-substrate product track for tested R★. At most **one** preregistered R★ revision then re-gate; else idle. **No** automatic redesign loop. **No** NanoScribe / fabric expansion. |
| **SURVIVE** | \(U^{\star}_{\mathrm{gen}} > U^{\star}_{\mathrm{class}} + \delta_{R★}\) **and** no sensitivity flip | Value **only in frozen R★**. Does **not** authorize NanoScribe/full product; separate feasibility gate required. Still ≠ “E1 unkill.” |
| **GRADED** | Margin inside \(\delta\), or sensitivity flips, or gen wins only on pre-registered subsets | Only exact winning locked slice(s); no platform/NanoScribe inference; no fabric v2. |
| **VOID** | Protocol/data/builder violation, leakage, failed probe, information-parity breach, or undecidable \(U\) | Correct the failed instrument. **Not** evidence for/against generative value. Do **not** interpret as SURVIVE. |

**Precondition (VOID if failed):** inclusion I\* hold; exclusion X\* hold; classical
probe shows ≥2 of {B1..B4}. Artifact:
`trajectory/results_e4_classical_probe.json` with `in_Rstar: true/false`.


## 3.1 Revision budget after KILL/VOID (design lock)

```text
RSTAR_REVISION_BUDGET = 1   # at most one preregistered R★ redesign after KILL or VOID
UNLIMITED_REDESIGN = FORBIDDEN  # prevents substrate-rescue via repeated regime shopping
```

A revision, if used, must be written **before** re-running Stage 4, with new inclusion/exclusion
hashes and a fresh owner `AUTHORIZE_E4_EXECUTE`. After the budget is spent, default remains
`IDLE_AFTER_FREEZE` on the product track.

**Secondary (does not override):** per-axis / per-field breakdown; ecology tag
`general | generative-helps-binding | generative-helps-paraphrase | inconclusive`.

---

## 4. R★ data definition (design only — no world freeze)

### 4.1 Schema

Unchanged five fields. Open slots stressed: **CC, MED, ALG**.

### 4.2 Split discipline

| Split | Content |
|-------|---------|
| **Train** | Lexicon / optional CRF / generative FT; **disjoint** surface-template family from eval |
| **Dev** | Optional generative hparams only; frozen before final eval; classical rules **not** tuned on dev to chase G-ref |
| **Eval** | Locked set satisfying I\* + X\*; content-addressed JSON |

### 4.3 Inclusion / exclusion

See `REGIME_P1_where_classical_fails.md` (I1–I5, X1–X6). Copied constraints:

1. Eval surface forms from template pool **disjoint** from frozen C-M1 patterns.  
2. ≥30% open-slot gold need normalization or multi-span assembly.  
3. ≥40% eval open gold strings absent from train lexicon.  
4. ≥20% docs have ≥2 competing values for ≥1 open slot.  
5. ≥30% docs lack canonical C-M1 cue strings.

### 4.4 Scale (minimum for a future E4)

| Item | Minimum |
|------|---------|
| Eval documents | ≥200 |
| Multi-candidate subset | ≥40 docs |
| Non-verbatim open gold | ≥30% of open gold cells |
| Seeds | Generator seeds committed before scoring |

### 4.5 Classical probe artifact (pre-generative)

`trajectory/results_e4_classical_probe.json` must record B1–B4 and `in_Rstar`.
If false → **STOP** (rebuild or end product path).

### 4.6 Explicit non-data

- Not m0–m4 / v1–v2 isomorphic dialogues under old M1  
- Not production EHR dumps without schema  
- Not multilingual / OCR-as-primary in v1  
- Not a dataset generated under design-only auth as if it were the frozen world  

### 4.7 Builder status

**Not implemented. Not authorized under DESIGN_ONLY.**

Building R★ data is **implementation of the frozen spec**, still requiring
`AUTHORIZE_E4_EXECUTE` (or a narrower `AUTHORIZE_E4_BUILDER` if the owner splits
phases). Design-only must not create the locked eval world.

---

## 5. Builder / data requirements checklist (for later execution auth)

Do **not** check these off by running work under design-only. This is the gate
list an owner should see before authorizing execution:

| # | Requirement | Needed before |
|---|-------------|---------------|
| B1 | R★ generator or curated corpus code + committed seeds | Data lock |
| B2 | Content-addressed train/dev/eval JSON satisfying I\*/X\* | Data lock |
| B3 | Train lexicon hash + leakage report | Probe |
| B4 | C-M1 rule file frozen + template-family ID manifest | Probe |
| B5 | Classical probe runner → `results_e4_classical_probe.json` | G-ref train/score |
| B6 | Named G-ref recipe(s) + base checkpoint SHA pin | Train |
| B7 | Verify-on/off presenter adapted to R★ provenance | Score |
| B8 | Utility scorer implementing \(U_{R★}^{\mathrm{draft}}\) + sensitivity grid | Decision |
| B9 | \(M\) rubric pre-assigned to each method | Decision |
| B10 | Hardware class + L/C measurement protocol | Decision |
| B11 | Precommitted consequences table unchanged | Decision |
| B12 | Explicit owner auth string for execution | Unlock Stage 4 |
| B13 | Budget estimate (compute $ / GPU hours) accepted | Unlock Stage 4 |
| B14 | No mid-stream weight edits after scores | Integrity |

---

## 6. Non-goals and forbidden reopens

| Non-goal | Why |
|----------|-----|
| Reopen E1 substrate thesis | KILL stands under `OLD_TASK_U` |
| Old-task runs / isomorphism bakeoffs | Dead product world |
| “Build NanoScribe anyway” | Ambition is regime utility, not architecture revival |
| Fabric = product / NanoScribe | Fabric is harness (`W-FABRIC-NS`) |
| E3 as human/clinician eval | Agent-rubric only; IAA open (`W-E3-HUMAN`) |
| Universal “LMs can’t extract” | `W-KILL-UNIVERSAL` |
| Claiming design track = execution queued | `EXECUTION_BLOCKED` |
| E2 LoRA mechanism work | GATED/STOP |
| Paid compute / GPU under design-only | Ban |

---

## 7. First-principles limitations

1. **R★ is synthetic-by-construction.** Inducing classical stress ≠ natural notes.  
2. **Inclusion can still be gamed.** Mitigation: freeze generator+seeds; adversarial
   review; public I\*/B\*; no post-score filtering.  
3. **\(U\) is not clinical utility.** Weights are decision-theoretic stand-ins.  
4. **\(M\) is rubric-subjective.** Pre-assign and sensitivity “No-maintenance.”  
5. **Exact-match remains strict.** Soft metrics reported; kill uses exact unless amended.  
6. **Single G-ref may be weak/strong.** Pre-commit recipes; no post-hoc hunting.  
7. **Verifier \(R\) may be harder on R★.** Report both arms; result not a bug.  
8. **Does not unkill E1.** Win on R★ → Stage 5 wedge **only in R★**.  
9. **E2/fabric remain gated** until Gate 4 ∈ {SURVIVE, GRADED} *and* separate auth.  
10. **No bit-level FT determinism claimed.** Pins reduce wrong-artifact risk only.

---

## 8. Execution checklist (Stage 4 — blocked now)

When owner authorizes E4 **execute** (not design):

1. Implement / lock R★ builder → content-addressed eval JSON  
2. Freeze C-M1 rule file + C-M2 lexicon hashes + \(M\) assignments  
3. Run classical probe → confirm `in_Rstar`  
4. Train/score frozen baselines only  
5. Write `trajectory/results_e4_utility.json` + per-method items  
6. Apply §3 decision rule; update `PIPELINE_GATE_LOG.md` Gate 4  
7. **Stop** or branch to Stage 5 per verdict — no fabric expansion  

---

## 9. Design-completeness checklist (this track)

| Criterion | Status |
|-----------|--------|
| Ambition framing (IDLE ≠ halt) | **Yes** — `papers/AMBITION.md` |
| Anti-circular I\*/X\*/B\* | **Yes** — `REGIME_P1` |
| \(U_{R★}\) with Q,E,R,L,C,M | **Yes — DESIGN DRAFT** (§1) |
| Information-parity matrix | **Yes** (§2.5) |
| Consequences KILL/GRADED/SURVIVE/VOID | **Yes** (§3) |
| Non-goals / old-task ban | **Yes** (§6) |
| Builder checklist (no build) | **Yes** (§5) |
| E4 measurement artifacts | **No** — EXECUTION_BLOCKED |
| Gate 3 protocol completeness | **PASS** (design package) |
| Stage 4 authorization | **Absent** |

**E4 = `DESIGN_IN_PROGRESS` / `EXECUTION_BLOCKED`.**  
Not “next stage running.” Not authorized to execute.

## One-sentence freeze

**E4 may only ask whether generative+verify adds utility inside a non-circular R★
under this draft \(U_{R★}\); design may proceed now; execution may not; the E1
world stays closed.**
