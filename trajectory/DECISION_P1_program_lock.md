# P1 — Program decision lock

*Decision document (docs/evidence layer only). Not an experiment plan.
Maximizes information per unit effort by freezing what is already known and
gating what must not be run yet. 2026-07-31.*

**Companion (product re-scope only):** `trajectory/REGIME_P1_where_classical_fails.md` (regime **R★**).  
**Lockfile pointer:** `papers/EMPIRICAL_FOUNDATION.md`.
**Evidence ledger:** `papers/EVIDENCE_LEDGER.md` · **Claim glossary:** `papers/CLAIM_GLOSSARY.md` · **A→Z overlay:** `papers/AZ_EXECUTION_PLAN.md`.  
**Forbidden here:** fabric expansion, residual sweeps, LM scaling runs, E2 execution.
**Sequential pipeline:** `papers/SEQUENTIAL_PIPELINE.md` (Gate 0 PASS; cursor = Stage 1).
**Corrected A→Z (post-KILL):** `papers/AZ_EXECUTION_PLAN.md` — Paths A/B/C; next product step is **P2** then **E4** (new-regime kill gate), or **Idle**. E2/E3 do not unlock product.

---

## 0. Operating rule

| Do | Do not |
|----|--------|
| Treat E1 KILL as settled for *this* task + utility | Re-run substrate bakeoffs on m0–m4 / v1–v2 |
| Keep Paper α measurement claims; state confidence | Sell generative substrate / architecture product on E1 world |
| Optionally finish dual-clinician E3 *only* if cheap information | Fund E2 until E1/E3 posteriors are respected (E1 done; E3 auto+agent-rubric done) |
| Write a *new* objective (e.g. R★ + new \(U\)) before any new runs | Accumulate more supporting evidence for killed claims |

**Information objective:** stop paying for evidence that cannot change the
program direction; pay only for evidence that can flip a still-open posterior
(mainly E3 human construct) or for a deliberately rewritten problem.

---

## 1. Frozen empirical claims and confidence

Mechanism-neutral. Confidence = how much a single cheap check could still move us.

| ID | Claim | Confidence | Why this level | Still-open threat |
|----|-------|------------|----------------|-------------------|
| C1 | Held-out copying failure localizes to **open-vocab** fields; closed fields ≈ **0** gap (template-vs-value control) | **High** | Reproduced across anchors + Pythia fieldwise; direction stable | Soft/human rubric could shrink *magnitudes* (not likely the zero-vs-open contrast) |
| C2 | Own-stack diluted gap ~**18** pts at 3–10M and remains large at 160M full-FT (**16.9±1.7**); no monotonic collapse with N across evaluated full-FT configs | **High** (descriptive) | Multi-instance instrument; factorial cells; token budgets in audits | Parameter count not isolated from pretraining exposure (nano 32.8M vs ~200M/3.2B) |
| C3 | Adaptation×data interaction: LoRA or Chinchilla-scale data each ~**7** diluted; both together ~**4.2** (near Pythia-160M LoRA) | **Medium-high** | Pre-registered factorial; mechanism **unidentified** | E2 universes (gated) — behavioral fact stands without mechanism |
| C4 | Slot training diversity causally lifts held-type recall **+66.7** (D5→D80), monotonic, position innocent | **High** | Pre-registered sweep; primary per-type table | Token-coverage descriptive only; not a separate causal claim |
| C5 | Shared residual floor ~**15–18 clean** pts on hardest low-diversity open types after escapes; allergy is strongest *instance*, not the definition | **Medium** | Descriptive across stacks; interference REFUTED | Morphology / other residue causal status open — **do not fund continuum** without new question |
| C6 | Single-instance eval under-powered / publicly hard-biased; 1B residual dominated by **training-run** nondeterminism → interval **[0,5]** | **High** | Pre-registered contamination flap + determinism cross-check failure | Fuller 1B seed distribution unused (low ROI vs α) |
| C7 | Primary science metric = **exact string match**; normalize rescues **0/486**; agent-rubric pack faithful-rate **0.00** | **High (auto)** / **High-on-pack (agent-rubric)** / **Medium (IAA/synonymy)** | Stage 1 EXACT_SURVIVES (instrument) | Dual clinician IAA + synonym ontology still open; human arm NOT_RUN |
| C8 | Under frozen E1 utility, **M1 exceeds** best evaluated generative LM ref (official M0) → **KILL (H-substrate)** for this task | **High** | Official M0 closed; sensitivity analysis no flip; M2 does not dominate M0 | New \(U\) or new problem only (see §2 exits) |
| C9 | Propose→verify→abstain can hit 100% presented precision at ~19% review load on this **synthetic** world under rules-strong \(R\) | **Medium (scoped)** | Stage G/A existence proof | **Does not** license open-world zero-hallucination |

**Paper α may use C1–C8 as measurement narrative; C8 as honest kill-gate; C7 limitation explicit; C9 only as scoped existence, never as product punchline.**

---

## 2. E1 kill gate (precise, locked)

**Status:** **EXECUTED — KILL.** This section is the decision record, not a TODO.

### 2.1 Baseline families

| ID | Family | Role in gate |
|----|--------|--------------|
| **M0** | Generative LM references | Official: Pythia-160M LoRA; Own-stack Chinchilla+LoRA; Local: scale-10M. Gate uses \(\max U\) among official M0 arms |
| **M1** | Regex / template slot filler | Symbolic; hand rules; primary classical winner |
| **M2** | Dictionary + span (train lexicon only) | Symbolic; no held lexicon leakage |
| **M3** | CRF-lite BIO | Structured prediction |
| **M4** | Constrained / finite-state copy | Constrained gen |
| **M5** | Span start–end classifier | Non-autoregressive IE |

Prereg: `trajectory/PREREG_E1_nonlm_baseline.md`.  
Harness artifacts: `trajectory/results_e1_utility.json`, `trajectory/e1/`.

### 2.2 Utility function (frozen default)

\[
U = P - 0.5\,M - 0.3\,\rho - 0.02\,L - 0.05\,C
\]

| Symbol | Meaning (E1; authoritative = `PREREG_E1` + `e1/common.py`) |
|--------|----------------|
| \(P\) | Presented precision (verify-on arm as primary decision) |
| \(M\) | Miss rate = \(1 - \) field recall (omissions + wrong under exact match) |
| \(\rho\) | **Review load** = fraction of fields flagged/routed to human (`flagged / n_fields`); **not** hallucination rate |
| \(L\) | p50 end-to-end latency per dialogue (seconds); code field `L_p50` |
| \(C\) | Relative compute cost vs 10M greedy scribe (method-class constant) |

Weights in code: \(U=\alpha P-\beta M-\gamma\rho-\lambda L-\kappa C\) with
\((\alpha,\beta,\gamma,\lambda,\kappa)=(1.0,0.5,0.3,0.02,0.05)\).
Sensitivity JSON renames those penalty weights as \((\alpha,\beta,\gamma,\delta)_C=(0.5,0.3,0.02,0.05)\) — do not confuse sensitivity \(\delta_C\) (cost weight) with decision margin \(\delta=0.05\).
Hallucination / bad presentation is reported separately as `halluc` / `liability_presented_bad`, **outside** \(U\) v1.

Decision margin \(\delta = 0.05\).  
Sensitivity: `trajectory/results_e1_utility_sensitivity.json` — **KILL robust**; no rank flip in surveyed grid.

### 2.3 Success / failure criteria (program-facing)

On default \(U\), mean over frozen instances:

1. **KILL (H-substrate)** — *FIRED*  
   \(\max_{m\in\{M1..M5\}} U(m) \ge U(M0_{\mathrm{official}}) - \delta\)  
   **Observed:** \(U(M1)=0.999\), \(U(M0_{\mathrm{official}})=0.925\), margin \(+0.074\); also \(U(M2)=0.886 \ge 0.925-\delta\).

2. **SURVIVE (H-LM-necessary)** — *not met*  
   \(\max_{m\in\{M1..M5\}} U(m) < U(M0) - \delta\) under default **and** no sensitivity flip.

3. **GRADED** — *not met*  
   Sensitivity flips rank, or verify-on/off disagree on polar verdict → no architecture punchline.

### 2.4 What outcomes change program direction

| Outcome | Program direction |
|---------|-------------------|
| **KILL (actual)** | End generative-substrate thesis for **this** task; Paper α = measurement + honest §0; fabric/product on this world **STOP**; β only narrow soundness under decidable \(R\) if ever |
| SURVIVE (counterfactual) | Would have kept LM frame for systems work pending E3 then E2 |
| GRADED (counterfactual) | Report interaction; ban architecture punchline |
| **New written \(U\) or problem** (only live exit) | May re-ask substrate ranking on **that** problem — requires P2 on regime **R★** (companion note), not a silent reopen of E1 |
| More LM scaling / residual sweeps on E1 world | **Rejected** — low information; cannot unkill C8 without changing \(U\)/problem |

---

## 3. E3 construct validation requirements

**Threat T3:** exact-match overstates failure vs normalize / human faithfulness.

### 3.1 Already satisfied (auto arm)

| Requirement | Result | Artifact |
|-------------|--------|----------|
| Frozen normalize-then-match on M0 exact failures | **0/486** rescued; gap shrink **0.0** pts | `results_e3_normalize_construct.json` |
| Auto verdict | `EXACT_NOT_OVERSTATING_BY_NORMALIZE` → provisional **EXACT_SURVIVES** | same |
| Paper α limitation text | Exact ≠ validated human equivalence | `paper1_draft.md` / LaTeX §0 + Limitations |

**Posterior update from auto:** formatting/normalize is **not** the gap. Remaining construct risk = morphology / paraphrase / human “faithful enough.”

### 3.2 Dual-clinician / independent rater requirements (optional; only if information is worth the rater cost)

Prereg: `trajectory/PREREG_E3_faithfulness_construct.md`.  
Pack: `trajectory/e3_human_rating_pack.json` (frozen; n=100).

| Requirement | Spec |
|-------------|------|
| Labels | `{faithful, unfaithful, unsure}` per prereg rubric |
| Retuning | **Forbidden** — labels do not change methods or \(U\) |
| Primary decision thresholds (prereg) | **Collapse:** ≥50% of exact errors labeled `faithful` **or** (with normalize) large failure-mass shrink; **Survive:** normalize &lt;5 pts shrink **and** &lt;20% exact errors `faithful`; else **partial/graded** |
| Reporting | `trajectory/results_e3_human.json` only; update foundation posterior |
| If skipped | Keep α limitation; **do not** block Paper α; **do not** pretend human clearance |

### 3.3 What E3 may change

| E3 outcome | Changes | Does not change |
|------------|---------|-----------------|
| Human collapse | Reinterpret α **magnitudes**; soften copy-failure rhetoric; possible metric rebuild | Does not by itself revive generative **product** thesis on E1 world |
| Human survive / partial | Strengthens exact-match as science instrument for this synthetic world | Does not authorize E2 or fabric |
| Auto-only (current) | Provisional exact survives normalize | Construct confidence remains Medium pending human |

---

## 4. E2 mechanism work — gated

Prereg exists: `trajectory/PREREG_E2_lora_universes.md` (U1 geometry / U2 ease / U3 early-stop / U4 module).

| Rule | Detail |
|------|--------|
| **State** | **GATED / STOP** |
| **Until** | Owner accepts that E1 posterior (KILL) and E3 posterior (auto done; human optional) are respected **and** writes why LoRA universe ID has positive EV *after* substrate death |
| **Default EV** | **Negative** — identifying LoRA mechanism does not restore a killed product frame on this task; Paper α must not claim geometry preservation |
| **Ban** | “Geometry preservation,” mechanism punchlines, U3/U1–U4 runs, stray pods |
| **Unblock requires** | Explicit written re-scope (new question), not curiosity |

E1 has already updated the posterior: **substrate claim dead for this task.**  
E3 auto has updated the posterior: **normalize is not the construct escape.**  
E2 does not sit on the critical path for information after those updates.

---

## 5. Effort allocation (information per unit effort)

| Action | EV now | Decision |
|--------|--------|----------|
| More residual / diversity / interference continua | Low — supporting evidence for α, not decision-changing | **STOP** |
| More LM scaling on E1 world | Low — cannot beat M1 under frozen \(U\) without deforming \(U\) | **STOP** |
| Fabric / product UX on E1 world | Negative — contradicts KILL | **STOP** |
| E2 LoRA universes | Low for program direction post-KILL | **GATED** |
| E3 human (bounded pack) | Medium — only open construct threat with frozen pack | **Optional** |
| Paper α public freeze | Done (`paper-alpha-v1`) | **HOLD** |
| Product path | Only via **R★** definition (companion) then **P2** utility/kill-gate docs | **Docs-first** |
| Idle | Correct if no rater and no product writing | **Allowed** |

---

## 6. One-page freeze statement

1. **Claims C1–C8 are frozen** at the confidences in §1; Paper α is the public measurement record.  
2. **E1 KILL is the substrate decision** for this task under the frozen \(U\) (§2); do not reopen without a new utility/problem.  
3. **E3 auto + agent-rubric are done**; dual-clinician/IAA remain optional construct checks; skipping keeps the limitation (§3).  
4. **E2 stays gated**; fabric and residual sweeps stay stopped (§4–§5).  
5. **Product evolution**, if any, starts at regime **R★** + P2 — never by enhancing the killed E1 substrate story.
