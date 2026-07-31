# First-Principles Risk Mitigation Map

**Adopted:** 2026-07-31  
**Status:** Layer-3 operating design (NONCLAIM). Not Layer-1 evidence. Not authorization to build or run.  
**Companions:** `LABORATORY_CONSTITUTION.md`, `DECISION_GATES.md`, `EXECUTION_QUEUE.md`, `STRATEGIC_RESET.md`, `WEDGE_V1.md`, `EVIDENCE_LEDGER.md`, `RESEARCH_PORTFOLIO.md`

> Decompose every blocker to atoms. Mitigate atoms with mechanisms that survive contact with evidence.  
> Do not “innovate” by loosening claim standards.

```text
DOC_TYPE: CONDITIONAL_MITIGATION_MAP
MAY_AUTHORIZE_EXECUTION: false
EVIDENCE_STANDARDS: UNCHANGED
VISION: EXPANSIVE
EXECUTION: GATE-BOUND
```

---

## 0. First principles (non-negotiable)

These are the axioms. Mitigations that violate them are rejected.

| ID | Principle | Operational meaning |
|----|-----------|---------------------|
| P1 | **Claim ≠ wish** | Only ledger-backed sentences may be stated as measured. |
| P2 | **Kill gates kill hypotheses, not curiosity** | Negative RESULT narrows Layer 1; Layers 2–3 stay open. |
| P3 | **Cheapest sufficient solver** | Prefer deterministic / retrieval / constrained methods before generative (E1 lesson). |
| P4 | **Verify consequential outputs** | Trust comes from checks + abstention, not eloquence (Fabric lesson; NeSy logic-side admissibility). |
| P5 | **Artefact gates dominate signal gates** | QA/auth/provenance failure voids a “pretty” RESULT (preregistration + artefact-override discipline). |
| P6 | **Information parity** | Bakeoffs are invalid if one arm sees oracle structure the other cannot use. |
| P7 | **Typed authority** | Only `EXECUTION_QUEUE` (+ typed auth receipt) may authorize runs; prose is not a license. |
| P8 | **Stratified publication** | Freeze tags are historical boundaries; later corrections get new names, never retargeted tags. |
| P9 | **Context of use** | Synthetic exact-match ≠ clinical readiness (evidence-based AI / context-of-use validity). |
| P10 | **Smallest next experiment** | Prefer one falsifiable wedge over architecture theater. |

Research anchors (methods, not results): preregistration + kill/artefact gates; hybrid mechanistic–learned (neuro-symbolic) separation of *admissibility* vs *plausibility*; evidence-based agent stacks with provenance; F-G-R style trust tuples (formality, claim scope, reliability).

---

## 1. Problem decomposition method

For each blocker \(B\):

```text
B
├── Failure mode atoms (what actually breaks)
├── Invariants at risk (which P1–P10)
├── Evidence state today (ledger / freeze / queue)
├── Research-backed mitigation options
│   ├── M0 hygiene (docs/CI; no compute)
│   ├── M1 design protocol (G2)
│   └── M2 measurement (G3; needs explicit auth)
├── Innovation angle (new structure, not new hype)
└── Exit criterion (what “mitigated” means)
```

---

## 2. Master blocker inventory

### B1 — Auth forgery / ambiguous execute tokens
**Atoms:** Chat “authorized” → agent minting `AUTHORIZE_*`; queue out of sync with status prose; untracked AUTH_RECORD.  
**Invariants:** P5, P7.  
**Today:** Split-brain history around E4; queue is now wedge-centered; lint still incomplete.  
**Mitigations:**
- **M0:** `doc_type` front-matter; `AUTHORIZE_*` allowlist = `EXECUTION_QUEUE` + typed `AUTH_RECORD`; CI `scripts/lint_claim_auth.py` (design from swarm hygiene).
- **M0:** Refuse execute if recipe SHA ≠ frozen prereg SHA.
- **Innovation:** Treat auth as a *capability token* with expiry + scope bits (commit/tag/push/execute), not a vibe word.  
**Exit:** Lint PASS on CI; runners fail-closed without queue `auth_ids[]`.

### B2 — Premature public freeze vs incomplete final freeze
**Atoms:** Tag `post-alpha-evidence-freeze-2026-07-31` is real but incomplete vs reconciled tip; agents try to recreate it; dirty freeze packaging rewrites history.  
**Invariants:** P8, P1.  
**Today:** Tag preserved at `a9d12cb`; DIFF E remediations on origin (`45291e5`); final reconciled tag optional.  
**Mitigations:**
- **M0:** Freeze docs must say EXISTS/PRESERVE; Task 25 forbids recreate (done).
- **M0:** Stratigraphy table: Paper α / premature post-α / corrections / optional new tag.
- **M1:** Owner-named **new distinct** tag only after clean tip + durability audit.  
**Exit:** No doc instructs creating the old tag name; optional new tag SHA listed once.

### B3 — E1 reproducibility `PUBLIC_PARTIAL` (L/C clean-clone)
**Atoms:** Utility/result JSONs public; latency \(L\) and normalized compute \(C\) reconstruction not proven from clean clone.  
**Invariants:** P1, P6.  
**Mitigations:**
- **M0:** Keep `PUBLIC_PARTIAL` (already); never upgrade wording until audit closes.
- **M1:** Write `trajectory/E1_LC_RECONSTRUCTION_PROTOCOL.md`: device table, formulas, seeds, scripts, expected tolerances.
- **M2 (auth):** Clean-clone replay job that only recomputes \(L,C\) from stored outputs / timers — no new model training.  
**Innovation:** Separate **decision reproducibility** (KILL from public utilities) from **cost-term reproducibility** (L/C). Different claim IDs if needed.  
**Exit:** Ledger `reproducibility` → `PUBLIC_REPRODUCIBLE` *or* permanent split claim with honest PARTIAL.

### B4 — E2 mechanism identification STOP
**Atoms:** No RESULT; speculation (“geometry”) forbidden; residue U3 not a measurement.  
**Invariants:** P1, P2.  
**Mitigations:**
- **M0:** Status row stays SUPPORTED “no RESULT” (done).
- **M1:** Re-scope as **intervention menu** (reg, early-stop, module site, LR) with pre-assigned kill rules — only if owner wants mechanism science.
- **Default product path:** **Do not unblock E2** for wedge v1; mechanism is optional science (Program A).  
**Exit:** Either parked forever as STOP, or one owner-authorized RESULT under new claim IDs.

### B5 — E3 human/clinician construct + IAA absent
**Atoms:** Agent rubric ≠ human; exact ≠ clinical synonym; no IAA.  
**Invariants:** P1, P9.  
**Mitigations:**
- **M0:** Keep negative status row + H1 public wording (done).
- **M1:** Dual-clinician protocol: sample size, adjudication, κ/IAA primary; agent labels secondary.
- **Innovation:** Three estimands, never collapsed: (i) exact, (ii) normalize, (iii) clinical-accept.  
**Exit:** Human arm RESULT *or* explicit product decision that wedge uses exact/rules only (no clinical claim).

### B6 — Classical vs generative substrate (E1 KILL; E4 surface)
**Atoms:** Generative lost under frozen old-task \(U\); R★/E4 was an attempt to find a regime; auth/ambig + possible exploratory KILL.  
**Invariants:** P2, P3, P6.  
**Mitigations:**
- **Product path (aligned with Strategic Reset / Wedge v1):** classical-first baseline on mini-corpus; generative only behind verify+escalation.
- **Science path:** Program H — hybrids / retrieval / tools; E4-prime only under literal execute auth + ≤1 revision budget.
- **NeSy rule:** classical/rules on **logic/admissibility** side; LM on **belief/plausibility** side; never swap.  
**Exit:** Wedge classical baseline measured under auth; E4 either VOID/PARK/RATIFY as owner decides — never silent.

### B7 — Scale / parameter-only overclaim residue
**Atoms:** Unequal tokens; “flat 50×” / parameter-only law.  
**Invariants:** P1.  
**Mitigations:**
- **M0:** Ledger split C_SCALE_OBSERVED / C_PARAMETER_ONLY_EFFECT / gate label (done).
- **M2 (optional):** Equal-token parameter intervention — new claim ID.  
**Exit:** Public wording never implies matched parameter-only ladder.

### C3 / morphology / length underpowered (B8)
**Atoms:** Transition/boundary REFUTED; length UNRESOLVED; morph descriptive ≠ causal.  
**Invariants:** P1, P5.  
**Mitigations:**
- **M0:** Keep scopes (done); durable JSONL tracked.
- **M1:** Equivalence-powered redesign or new contrast; morph causal only via prereg.  
**Exit:** New RESULT under new IDs — or park as descriptive.

### B9 — Fabric mistaken for cognitive OS / NanoScribe
**Atoms:** Slice success → product inflation.  
**Invariants:** P1, P4, P10.  
**Mitigations:**
- **M0:** Inventory claim C_NANOSCRIBE_STATE; Fabric ≠ product (done).
- **Wedge path:** reuse verify/abstain patterns as **library**, not OS build.  
**Exit:** No README/architecture doc claims unimplemented modules.

### B10 — Clinical / open-world zero-hallucination pressure
**Atoms:** Forbidden theses tempting marketing.  
**Invariants:** P9, P1.  
**Mitigations:**
- **M0:** C_CLINICAL_DEPLOYMENT + C_ZERO_HALLUC_OPEN FORBIDDEN (done).
- **Innovation:** Publish **context-of-use cards**: task, population, failure cost, verifier class, abstention rate.  
**Exit:** Any external claim carries a context-of-use card or is rejected.

### B11 — Dirty-tree / parallel-agent contamination
**Atoms:** Unrelated strategic files, freeze packaging churn, exploratory results coexisting.  
**Invariants:** P5, P7, P8.  
**Mitigations:**
- **M0:** Commit allowlists; quarantine paths; swarm queen synthesis as incident log.
- **M0:** `EVIDENCE_CURRENT.md` / stratigraphy only if they *reduce* confusion (evidence-first).  
**Exit:** `git status` dirty set classifiable; freeze packaging not casually committed.

### B12 — Ambition ↔ evidence competition (timidity vs overclaim)
**Atoms:** Kill gates shrink vision; or roadmap quoted as claims.  
**Invariants:** P2, P1.  
**Mitigations:**
- **M0:** Constitution layers + NONCLAIM banners + portfolio A–O (done/partial).
- **M0:** `ANOMALY_LOG` kill→expand (create if missing).  
**Exit:** Empty execution queue compatible with full portfolio.

### B13 — Wedge execution without repeating E1 failure
**Atoms:** Building LM features before classical baseline; skipping verify.  
**Invariants:** P3, P4, P10.  
**Today:** Phase 2 classical RESULT exists (U≈0.926). Phase 3 design ready; execute not authorized.  
**Mitigations:**
- **M1:** Freeze mini-corpus hashes, metrics, cost model *before* any LM component. **DONE** for Phase 2.
- **M2 (auth):** Classical baseline measured. LM only behind `AUTHORIZE_WEDGE_V1_PHASE3_LM_PROBE` on E-class allowlist.  
**Innovation:** Utility includes review load from day one; Phase 3 dual estimands \(U_{dep}/U_{cap}\).  
**Exit:** Phase 2 measured ✓; Phase 3 optional.


### B14 — Auth capability-scope overgrant
**Atoms:** A valid `AUTHORIZE_*` string grants broader rights than the task (e.g. execute+push+tag bundled); chat “proceed” interpreted as execute.  
**Invariants:** P5, P7.  
**Mitigations:**
- **M0:** Capability bits on auth receipts: `{commit, tag, push, execute, rescope}` with explicit expiry and `valid_only_if_queued`.
- **M0:** Lint: AUTH_RECORD must list `scope_bits[]`; runners refuse bits not present.
- **Innovation:** Treat auth as a *capability token* (least privilege), not a mood. Gateway-only consult ≠ execute auth.  
**Exit:** Every runner checks `scope_bits`; missing bit → fail-closed. *(Atom ships with gate or it is theater — Contrarian rule.)*

### B15 — Verification/action path asymmetry (NOT a breakage)
**Atoms:** `claude -p` / autonomous credits fail while `consult_gateway` works; agents treat this as outage and stall science.  
**Invariants:** P7.  
**Today:** Observed 2026-07-31; consult gateway succeeded for hybrid + this map.  
**Mitigations:**
- **M0:** Document as **typed authority working**: verification consults allowed; autonomous execute path denied-by-design when credits/keys differ.
- **M0:** Do not spend cycles “fixing” CLI credits to unlock experiments.  
**Exit:** AGENTS.md + this map state gateway-only as intended for consult; execute still requires queue+bits.

### B16 — Context-of-use drift on PUBLIC_PARTIAL kills
**Atoms:** Readers treat E1 KILL as global “LMs never useful”; ignore task/\(U\)/world bounds; upgrade PARTIAL to FULL by vibes.  
**Invariants:** P1, P2, P9.  
**Mitigations:**
- **M0:** Every GATE_VERDICT row must carry `context_of_use` (task, \(U\), world, venue) in ledger JSON.
- **M0:** Split E1 packaging (below): decision-admissible vs cost-plausible.
- **M0:** Forbidden public wording without scope clause (already partially in H1/E3).  
**Exit:** Validation rejects GATE_VERDICT without `context_of_use`; PUBLIC_PARTIAL cannot be paraphrased as global.

### B17 — Freeze-tag honesty vs E4-contaminated HEAD
**Atoms:** HEAD ancestry includes E4; tagging HEAD as “freeze” mislabels; indefinite deferral leaves tip unanchored.  
**Invariants:** P8, P5.  
**Today:** Council hybrid deferred tag (`COUNCIL_HYBRID_CLOSEOUT.md`); protected tags unmoved.  
**Mitigations:**
- **M0:** Never retarget `paper-alpha-v1` or `post-alpha-evidence-freeze-2026-07-31`.
- **M0:** Prefer **detached verdict annotations** (docs or `verdict/*` tags) that *disclose* ancestry + context-of-use — additive honesty, not retargeting.
- **M1 (OWNER_TAG_OK):** Either (a) cherry-pick hybrid onto freeze tip then tag, or (b) annotated `verdict/E1-kill@<sha>` with E4-ancestry note, or (c) remain deferred with logged reason (current).  
**Innovation:** Ancestry ≠ endorsement — but **undisclosed** ancestry under a freeze name *is* misrepresentation. Disclosure > void.  
**Exit:** Chosen option recorded; protected tags SHA-stable.

### B18 — Decision reproducibility ≠ cost-term reproducibility (E1)
**Atoms:** Single ledger atom mixes KILL (re-derivable from public utilities) with \(L,C\) (device/timer dependent → PUBLIC_PARTIAL).  
**Invariants:** P1, P6.  
**Research pattern:** Neuro-symbolic split — **admissibility** (symbolic gate fires) vs **plausibility** (magnitudes look right).  
**Mitigations:**
- **M0:** Keep C_E1_GATE KILL intact; do not “upgrade” PARTIAL by wishing.
- **M0/M1:** Design split (no kill deleted):
  - `C_E1_DECISION_REPRO` — recompute \(U\) from published \(P,M,\rho,L,C\) rows + `aggregate_decision` → KILL (offline pytest already pins this).
  - `C_E1_COST_REPRO` — clean-clone \(L,C\) reconstruction (see `trajectory/E1_LC_RECONSTRUCTION_PROTOCOL.md`).
- **M2 (auth):** Optional L/C replay job only.  
**Exit:** Two claim IDs with honest epistemic/repro fields; decision path green offline today.

---

## 3. Cross-cutting innovative mechanisms (lab-wide)

### 3.1 Trust tuple on every external sentence
For public text, attach \(\langle F, G, R \rangle\):
- **F** formality (MEASUREMENT / INTERPRETATION / POLICY…),
- **G** scope (task, \(U\), world),
- **R** reliability (ledger epistemic + reproducibility).

### 3.2 Artefact-over-signal CI
PR checks that fail the build if:
- `AUTHORIZE_*` outside allowlist,
- freeze tag recreate instructions,
- “clinically validated” / “zero hallucination” without FORBIDDEN context,
- ambition docs set `authorizes_execution: true`.

### 3.3 Solver ladder (runtime architecture sketch — NONCLAIM)
```text
input → schema/task analyze
      → classical / dict / regex / retrieval  (default)
      → constrained decode / tool call       (if needed)
      → generative propose                  (last)
      → verify / abstain / escalate
      → provenance record
```
This is the engineering expression of P3+P4; promotion still needs gates.

### 3.4 Evidence-based agent roles (if agents are used)
Separate roles: question framing · retrieval · extraction · QA/artefact · synthesis · uncertainty.  
No single agent both proposes and authorizes.

### 3.5 Decision vs cost atoms (admissibility / plausibility)
Borrow the neuro-symbolic distinction used in hybrid AI systems:
- **Admissible** — a symbolic/decision gate re-fires from published artefacts (E1 KILL from utility rows + `aggregate_decision`).
- **Plausible** — numeric auxiliaries (\(L\), \(C\)) look right but are device-bound until reconstructed.

Never collapse these into one “reproducibility” upgrade. Offline pytest pins admissibility; L/C protocol pins plausibility.

### 3.6 Detached verdict annotations (stratified publication)
Immutable freeze tags are historical boundaries (P8). Honesty about later HEAD state uses **additive** annotations:
- docs under `audit/.../COUNCIL_HYBRID_CLOSEOUT.md`, or
- optional owner-authorized `verdict/<claim>@<sha>` tags carrying context-of-use + ancestry disclosure.

Retargeting protected tags is forbidden. Undisclosed “freeze” names on contaminated tips are also forbidden.

### 3.7 Consult-gateway as typed verification path
When autonomous CLI credits fail but `consult_gateway` works, treat that as **least-privilege verification** (P7), not an outage. Consults may advise; they never mint `AUTHORIZE_*` or execute compute.

---

## 4. Priority order (mitigate first what falsifies the lab)

| Priority | Blocker | Why first | Needs auth? |
|----------|---------|-----------|-------------|
| 0 | B1 + B14 Auth forgery/scope | Prevents false science | No (M0 lint + scope bits) |
| 1 | B2/B11/B17 Freeze honesty + dirty tree | Protects public record | Process now; tag needs OWNER_TAG_OK |
| 2 | B18 / B3 E1 decision vs cost split | Honest packaging of past KILL | M0 design; L/C replay M2 |
| 3 | B16 Context-of-use on GATE_VERDICT | Stops global misread of scoped kills | M0 schema field |
| 4 | B13 Wedge classical baseline design | Product path without LM theater | Design now; run needs auth |
| 5 | B15 Gateway-only verification posture | Stops credit-chase as fake blocker | Docs only |
| 6 | B5 E3 human protocol (design) | Clinical-adjacent language later | Design; humans later |
| 7 | B6 E4 disposition | Ends split-brain | Owner one-liner |
| 8 | B4/B8 Optional science | Curiosity budget only | Owner |

---

## 5. Immediate M0 checklist (no experiments)

1. ~~`scripts/lint_claim_auth.py`~~ present — keep PASS in CI/dev loops; extend forbidden clinical/zero-halluc strings.  
2. NONCLAIM banners on portfolio/roadmap/ambition (if missing).  
3. ~~`papers/ANOMALY_LOG.md`~~ present — keep expand_allowed rules.  
4. ~~`trajectory/E1_LC_RECONSTRUCTION_PROTOCOL.md`~~ present — deepen formulas; land decision/cost split design note.  
5. Owner one-liner on E4 surface: `RATIFY_E4_EXECUTE` | `VOID_E4_AUTH` | `PARK_AS_EXPLORATORY`.  
6. Keep freeze tags immutable; hybrid tag **deferred** until OWNER_TAG_OK (clean lineage or verdict annotation).  
7. Wedge: freeze mini-corpus + metrics doc before any baseline run auth.  
8. Document B15: consult-gateway = intended verification path; do not chase CLI credits for science.  
9. Add `context_of_use` field proposal for GATE_VERDICT rows (B16) — ledger migration needs owner commit.  
10. Path-restricted commits only; never launder strategic-reset dirt into freeze packaging.

---

## 6. What this document deliberately does *not* do

- Does not reopen old-task generative substrate under `OLD_TASK_U`.  
- Does not authorize E2, E4, training, or paid compute.  
- Does not move evidence tags.  
- Does not claim clinical readiness or open-world zero hallucination.  
- Does not replace `EVIDENCE_LEDGER` rows with aspirations.

---

## 7. Success definition

Mitigation has worked when:

1. Agents cannot mint execute auth from chat vibes.  
2. Public freeze chronology is impossible to misread.  
3. Every blocker has an atom → principle → exit criterion.  
4. Product work (wedge) proceeds classical-first under explicit auth.  
5. Research portfolio stays large while claims stay tiny.

```text
NEXT_DEFAULT = IDLE_AFTER_HYBRID + M0_HYGIENE (this map)
NEXT_PRODUCT = AUTHORIZE_WEDGE_V1_CLASSICAL_BASELINE (owner; classical-first)
NEXT_SCIENCE = owner-picked from B18/B3/B5/B6 only
NEXT_TAG = OWNER_TAG_OK → clean-lineage cherry-pick OR verdict/* annotation OR remain deferred
ATOM_RULE = no new blocker ID without an enforcing gate
```

### Research anchors (methods literature patterns — not results)

| Pattern | Use here |
|---------|----------|
| Preregistration + kill/artefact gates | P5, B1, B5 |
| Admissibility vs plausibility (neuro-symbolic / NeSy hybrids) | B18, §3.5 |
| Context-of-use / evidence-based AI validity | P9, B10, B16 |
| Capability-based security / least privilege | B14, B15, P7 |
| CI-as-policy (policy-as-code) | §3.2, B1 |
| Stratified publication / immutable releases | P8, B2, B17 |

