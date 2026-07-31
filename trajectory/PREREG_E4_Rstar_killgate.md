# P2 / PREREG — E4 kill gate on regime R★

**Pre-registered Stage 3 protocol (2026-07-31). Docs only — NO RUNS until Gate 3
is accepted and an owner explicitly authorizes Stage 4 execution.**

**Public status:** protocol / aspirational — **no E4 measurement artifacts** (`results_e4_*` absent). Not executed.

Implements sequential Stage 3: freeze \(U_{R★}\), baseline family, decision rule,
R★ data definition, and first-principles limitations **before** any E4 scoring.

| Link | Role |
|------|------|
| `papers/SEQUENTIAL_PIPELINE.md` | Stages / gates |
| `trajectory/REGIME_P1_where_classical_fails.md` | R★ definition (Gate 2 PASS) |
| `trajectory/PREREG_E1_nonlm_baseline.md` | Old-task kill gate (**DONE — KILL**; do not reopen) |
| `trajectory/PIPELINE_GATE_LOG.md` | Gate status |

**Constants (inherited, immutable mid-stream):**

```text
E1_VERDICT = KILL
PAPER_α = FROZEN
OLD_TASK_RUNS = FORBIDDEN   # under OLD_TASK_U on m0–m4 / v1–v2 isomorphism
E2_DEFAULT = GATED
FABRIC_DEFAULT = GATED
```

---

## 0. Decision question (not “are LMs good?”)

> Under regime **R★** (where templates/dictionaries are *designed* to break), does a
> generative proposer achieve higher expected utility than frozen classical /
> constrained baselines on the **same** documents, schema, and metrics?

E1 already answered that question for the **non-regime** (closed isomorphic task):
classical wins. E4 asks it only where classical break predicates can fire.

---

## 1. Freeze \(U_{R★}\)

### 1.1 Decision the utility encodes

Emit a schema-valid structured summary that a downstream consumer can trust,
under bounded review cost, **when inputs are drawn from R★** (cue-poor,
long-tail, non-verbatim / multi-candidate delivery).

### 1.2 Definition

Per evaluation document (then mean over locked instances \(K\)):

\[
U_{R★} = P - 0.5\,M - 0.3\,\rho - 0.02\,L - 0.05\,C - 0.2\,\beta_{\mathrm{bind}}
\]

| Symbol | Meaning | Unit | Weight | Rationale |
|--------|---------|------|--------|-----------|
| \(P\) | Presented precision (verify-on primary; verify-off reported) | [0,1] | \(+1.0\) | Trust of emitted fields |
| \(M\) | Miss rate = 1 − recall on fields that should emit (omissions + wrong under construct policy §1.4) | [0,1] | \(−0.5\) | Same as E1: misses hurt, but less than bad presents |
| \(\rho\) | Review load = fraction of fields routed to human | [0,1] | \(−0.3\) | HITL economics |
| \(L\) | p50 end-to-end latency per document | seconds | \(−0.02\)/s | Prefer cheap methods when quality ties |
| \(C\) | Relative compute vs frozen C-M1 on same hardware class | ≥0 | \(−0.05\) | Penalize expensive proposers |
| \(\beta_{\mathrm{bind}}\) | Binding-error rate on multi-candidate docs (axis D slice) | [0,1] | \(−0.2\) | **New vs E1:** R★ stresses discourse binding; wrong-entity presents are a distinct failure mode from raw miss |

**Liability proxy (mandatory report, not inside \(U_{R★}\) v1):** count of
fabrications+substitutions that would be *presented* without review. Keep out of
\(U\) to avoid double-counting with \(P\) (same policy as E1). Fold in only via
a separate amendment.

### 1.3 Why not reuse `OLD_TASK_U` unchanged?

| Choice | Reason |
|--------|--------|
| Keep \(P,M,\rho,L,C\) weights | Continuity with E1; comparable “shape” of tradeoffs |
| Add \(\beta_{\mathrm{bind}}\) | R★ inclusion requires multiplicity/reference; E1 world barely stressed binding |
| Do **not** raise miss weight to “prove” LMs | No mid-stream gaming |
| Do **not** drop \(C\)/\(L\) | Generative cost must remain visible |

**Amendment rule:** changing any weight or adding terms after seeing E4 scores
**VOIDs** the kill decision. Sensitivity grid (§1.5) is the only pre-allowed
robustness check.

### 1.4 Construct policy for “correct” (inherits E3 posterior)

Primary reporting: **exact string match** (science continuity with α).

For \(U_{R★}\) field correctness also compute and report:

1. **Exact**  
2. **Normalize-then-match** (frozen `e1/common.py::normalize_value`)  
3. **Human-acceptable** on a **bounded** disagreement sample only if exact and
   normalize disagree **or** on a pre-registered ≤50-item audit slice — not a
   new full campaign mid-flight  

**Default kill decision uses exact-\(U_{R★}\)** (same discipline as E1).  
Normalize/human are robustness. Given Stage 1 EXACT_SURVIVES on the old failure
pack, we do **not** assume soft-match will rescue R★ either — but we still measure.

### 1.5 Sensitivity (pre-registered)

Also report \(U_{R★}\) under:

| Grid point | Change |
|------------|--------|
| Default | as §1.2 |
| High-miss | \(\beta_M = 1.0\) (miss weight) |
| High-review | \(\gamma = 0.6\) |
| No-binding term | drop \(\beta_{\mathrm{bind}}\) (recover E1-shaped \(U\)) |
| Binding-heavy | \(\beta_{\mathrm{bind}} = 0.4\) |

Kill/survive uses **default**. Any sensitivity flip → **GRADED**, no binary
architecture punchline.

### 1.6 Margin

\[
\delta_{R★} = 0.05
\]

Same numeric margin as E1 for interpretability; justified as “within noise /
tie band,” not as a clinical SLA.

---

## 2. Freeze baseline family

All methods see **identical** R★ eval documents, schema
`CC | DUR | SEV | MED | ALG`, and scoring code. No post-hoc method adds after
unlocking labels.

### 2.1 Classical freeze-set (required)

| ID | Method | Freeze rule |
|----|--------|-------------|
| **C-M1** | Template / regex slot filler | Rule budget written **before** eval reveal; no pattern edits after seeing R★ docs |
| **C-M2** | Train-dict + span | Train lexicon only; leakage check; no eval lexicon |
| **C-M4** | Constrained / copy-only open slots (optional but recommended) | Schema-constrained; no free open-vocab emit |

C-M1 and C-M2 are **mandatory**. C-M4 recommended as “strong classical.”  
C-M3 (CRF/BIO) optional if train alignments exist for R★; else mark VOID for
C-M3 only (does not VOID the gate if C-M1/C-M2/C-M4 + G-ref run).

### 2.2 Generative references (required)

| ID | Method | Freeze rule |
|----|--------|-------------|
| **G-ref** | Best available **small** generative proposer under a **frozen recipe** | Named before run (e.g. scale-10M full FT **or** Pythia-160M LoRA on R★ train split — pick the higher \(U_{R★}\) as official G-ref **after both scored**, or pre-commit one recipe only) |
| **G-strong** (optional) | One stronger LM ref | Quota only; cannot replace G-ref in the kill rule unless pre-registered as primary |

**Minimum for a valid E4:** {C-M1, C-M2, G-ref}.  
**Recommended:** + C-M4 and verify-on/off arms for all.

### 2.3 Verifier arms

For each method that emits fields: **verify-off** and **verify-on** (existing
grounding+absence style presenter, adapted to R★ docs with span provenance
where applicable). Primary decision: **verify-on** \(U_{R★}\). Verify-off reported.

### 2.4 Forbidden mid-flight

- Adding LLM methods to the “classical” set  
- Expanding C-M1 rules after peeking at eval  
- Training generative refs on eval templates / held lexicons  
- Re-scoring old m0–m4 under `OLD_TASK_U` as if it were E4  

---

## 3. Freeze KILL / SURVIVE / GRADED

Let

\[
U^{\star}_{\mathrm{class}} = \max_{m \in \{\mathrm{C\text{-}M1},\mathrm{C\text{-}M2},\mathrm{C\text{-}M4}\}} U_{R★}(m)
\]

\[
U^{\star}_{\mathrm{gen}} = U_{R★}(\mathrm{G\text{-}ref})
\]

(If C-M4 VOID, max over available classical only.)

On **default** \(U_{R★}\), mean over locked instances:

| Verdict | Rule | Program meaning |
|---------|------|-----------------|
| **KILL** | \(U^{\star}_{\mathrm{class}} \ge U^{\star}_{\mathrm{gen}} - \delta_{R★}\) | Generative **not** justified in R★ either → stop product path **or** one R★ revision max, else idle |
| **SURVIVE** | \(U^{\star}_{\mathrm{gen}} > U^{\star}_{\mathrm{class}} + \delta_{R★}\) **and** no sensitivity flip | Generative wins utility in R★ → Stage 5b (minimal stack) |
| **GRADED** | Margin inside \(\delta\), or sensitivity flips, or gen wins only on pre-registered subsets | Stage 5a routing — **not** full fabric |

**Precondition (VOID if failed):** classical probe on the eval slice shows ≥2 of
{B1,B2,B3,B4} at regime τ’s — else slice ∉ R★ and E4 does not run.

**Secondary (does not override):** per-axis / per-field breakdown; ecology tag
`general | generative-helps-binding | generative-helps-paraphrase | inconclusive`.

---

## 4. Freeze R★ data definition (no run yet)

### 4.1 Schema

Unchanged five fields: `CC | DUR | SEV | MED | ALG`.  
Open slots stressed: **CC, MED, ALG**. DUR/SEV remain controls, not the thesis.

### 4.2 Split discipline

| Split | Content |
|-------|---------|
| **Train** | Documents for lexicon / optional CRF / generative FT; **disjoint** surface-template family from eval; train-only open lexicons |
| **Dev** | Optional tuning for generative hparams only; **frozen before** final eval; classical rules **not** tuned on dev to chase G-ref |
| **Eval** | Locked instance set instantiating inclusion recipe (§4.3); content-addressed JSON |

### 4.3 Inclusion recipe (must hold on eval; from Gate 2)

1. Eval surface forms from template pool **disjoint** from frozen C-M1 patterns.  
2. ≥30% open-slot gold need normalization or multi-span assembly.  
3. ≥40% eval open gold strings absent from train lexicon (leakage check).  
4. ≥20% docs have ≥2 competing values for ≥1 open slot with single gold after discourse.  
5. ≥30% docs lack canonical C-M1 cue strings.

### 4.4 Scale (minimum for E4)

| Item | Minimum |
|------|---------|
| Eval documents | ≥200 (recommend 5 instances × 40–100 docs, or one locked 200+) |
| Multi-candidate subset | ≥40 docs |
| Non-verbatim open gold | ≥30% of open gold cells |
| Seeds | Generator seeds committed before scoring |

### 4.5 Classical probe artifact (pre-generative)

Before scoring G-ref:

`trajectory/results_e4_classical_probe.json` must record B1–B4 rates and
`in_Rstar: true/false`. If false → **STOP** (rebuild data or end product path).

### 4.6 Explicit non-data

- Not m0–m4 / v1–v2 isomorphic dialogues under old M1  
- Not production EHR dumps without schema  
- Not multilingual / OCR-as-primary in v1  

### 4.7 Builder status

**Not implemented in this Stage 3 document.** Stage 4 may not begin until a
builder (generator or curated corpus) exists that emits content-addressed eval
JSON satisfying §4.3–4.5. Building that data is **implementation of the frozen
spec**, not a new science question — but it is still **not authorized** until
the owner opens Stage 4.

---

## 5. First-principles limitations of this protocol

1. **R★ is synthetic-by-construction.** We *induce* classical stress. Success of
   G-ref on R★ does not prove value on natural clinical notes; failure does not
   prove generative never helps elsewhere.
2. **Inclusion recipe can be gamed.** A bad-faith generator could make classical
   fail while remaining trivial for LMs (or vice versa). Mitigation: freeze
   generator code + seeds; adversarial review of slice before scoring; probe
   predicates public.
3. **\(U_{R★}\) is still not a clinical utility.** Weights are decision-theoretic
   stand-ins; liability is reported not optimized.
4. **Binding term is new.** It can change rankings vs E1-shaped \(U\); sensitivity
   grid includes “no-binding” for transparency.
5. **Exact-match remains strict.** Stage 1 showed old-task exact failures were
   real under full-value rubric; R★ non-verbatim gold may increase
   exact-vs-human tension. Soft metrics are reported; kill uses exact unless
   amended.
6. **Single G-ref may be weak or strong.** Pre-commit recipe or take max of two
   frozen recipes — do not hunt models post hoc.
7. **Verifier \(R\) may be harder on R★** (non-contiguous evidence). Verify-on
   might hurt generative more than classical; that is a **result**, not a bug —
   report both arms.
8. **Does not unkill E1.** Classical win on R★ → product stays classical.
   Generative win on R★ → Stage 5 wedge **only in R★**, not NanoScribe-global.
9. **E2/fabric remain gated** until Gate 4 ∈ {SURVIVE, GRADED}.
10. **No bit-level FT determinism claimed.** Provenance pins (base SHA, seeds,
    recipes) reduce *wrong-artifact* risk; GPU FT noise remains.

---

## 6. Execution checklist (Stage 4 only — blocked now)

When owner authorizes E4:

1. Implement / lock R★ builder → content-addressed eval JSON  
2. Freeze C-M1 rule file + C-M2 lexicon hashes  
3. Run classical probe → confirm `in_Rstar`  
4. Train/score frozen baselines only  
5. Write `trajectory/results_e4_utility.json` + per-method items  
6. Apply §3 decision rule; update `PIPELINE_GATE_LOG.md` Gate 4  
7. **Stop** or branch to Stage 5 per verdict — no fabric expansion  

---

## 7. Gate 3 decision

| Criterion | Status |
|-----------|--------|
| \(U_{R★}\) frozen with weights + rationale | **Yes** (§1) |
| Baseline family frozen | **Yes** (§2) |
| KILL / SURVIVE / GRADED + δ frozen | **Yes** (§3) |
| R★ data definition frozen (no run) | **Yes** (§4) |
| Limitations explicit | **Yes** (§5) |

**Gate 3: PASS (protocol frozen).**  
**Stage 4 / E4: BLOCKED** until owner authorizes a run against this document
without mid-stream edits.

## One-sentence freeze

**E4 may only ask whether generative adds utility inside R★ under this frozen
\(U_{R★}\); it may not reopen the E1 world or invent weights after seeing scores.**
