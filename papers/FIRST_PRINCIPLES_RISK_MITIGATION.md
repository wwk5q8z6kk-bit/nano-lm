# First-Principles Risk Mitigation Map

**Adopted:** 2026-07-31  
**Status:** Layer-3 operating design (NONCLAIM). Not Layer-1 evidence. Not authorization to build or run.  
**Companions:** `LABORATORY_CONSTITUTION.md`, `DECISION_GATES.md`, `EXECUTION_QUEUE.md`, `OWNER_SPEECH_ACTS.md`, `STRATEGIC_RESET.md`, `WEDGE_V1.md`, `EVIDENCE_LEDGER.md`, `RESEARCH_PORTFOLIO.md`

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
**Today:** Phase 2+3 done; E-class `ECLASS_CLOSED_WITHOUT_LM` (LM not indicated). Queue idle. Owner `continue` = M0 only.  
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


### B23 — Session-continue / authority vocabulary collision
**Atoms:** Owner locution `continue` (session resume / ungated M0) collides with agent expectation of publish authority; fail-closed gates + untyped chat → stall loops; pressure to mint `OWNER_*` from vibes.  
**Invariants:** P7, P5, B14.  
**Today:** Hybrid freeze task parked; repeated `continue` correctly refused commit/tag; wasted turns.  
**Research pattern:** Speech-act / illocutionary force (Austin–Searle); capability tokens; HCI explicit confirmation for high-consequence acts.  
**Mitigations:**
- **M0:** `papers/OWNER_SPEECH_ACTS.md` — canonical force table (`CONTINUE_SESSION` grants **no** bits).
- **M0:** `scripts/classify_owner_speech_act.py` — classify before acting; `UNTYPED` → menu, never invent markers.
- **M0:** AGENTS.md rule: `continue` ≠ authorize commit/tag/push/execute.
- **Innovation:** Separate **admissibility of the speech act** (typed force) from **plausibility of intent** (“they probably meant commit”). Only admissibility unlocks markers.  
**Exit:** Classifier + table in use; stall loops end with one menu; no marker minted from `CONTINUE_SESSION`.

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

### 3.8 Speech-act gateway (owner ↔ agent)
Before any gated mutation, classify owner text (`scripts/classify_owner_speech_act.py`).  
`CONTINUE_SESSION` unlocks only ungated M0. High-consequence forces require exact authorize phrases + optional tip policy (B17/B23). See `OWNER_SPEECH_ACTS.md`.

### 3.9 Clean-lineage freeze recipe (design only — needs OWNER_TAG_OK to run)
When tip ancestry includes non-freeze science (e.g. E4), **do not** freeze-name that tip. Instead:

```text
1. branch from post-alpha-evidence-freeze-2026-07-31
2. cherry-pick only freeze-hygiene commits (corrections, durable_raw, DIFF E, pointer/stratigraphy)
3. verify: E4 commit NOT ancestor; protected tags unmoved
4. annotated tag NEW NAME at that tip under OWNER_TAG_OK
```

Alternative honesty paths: `verdict/*` annotation or remain deferred (current).


---

## 4. Priority order (mitigate first what falsifies the lab)

| Priority | Blocker | Why first | Needs auth? |
|----------|---------|-----------|-------------|
| 0 | B1 + B14 + **B23** Auth forgery/scope/speech-acts | Prevents false science + stall loops | No (M0 lint + classifier) |
| 1 | B2/B11/B17 Freeze honesty + dirty tree | Protects public record | Process now; tag needs OWNER_TAG_OK |
| 2 | B18 / B3 E1 decision vs cost split | Honest packaging of past KILL | M0 design; L/C replay M2 |
| 3 | B16 Context-of-use on GATE_VERDICT | Stops global misread of scoped kills | M0 schema field |
| 4 | B13 Wedge classical + E-class + noisy | Product path without LM theater | **DONE** (LM not indicated) |
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
7. Wedge: Phase 2 classical DONE; Phase 3 E-class **CLOSED without LM**; noisy diagnostic `NOISY_INGEST_NORMALIZE_SUFFICIENT`.  
8. ~~Document B15~~ — consult-gateway = intended verification path (AGENTS.md + this map §B15/§3.x).  
9. ~~`context_of_use` schema note~~ — present on E1 GATE_VERDICT atoms in `EVIDENCE_LEDGER.json`; broader migration still needs owner commit.  
10. Path-restricted commits only; never launder strategic-reset dirt into freeze packaging.  
11. ~~`papers/OWNER_SPEECH_ACTS.md` + `scripts/classify_owner_speech_act.py`~~ — use on every gated turn (B23).

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
NEXT_DEFAULT = IDLE_AFTER_NOISY_DIAGNOSTIC + M0_HYGIENE
NEXT_OPTIONAL_PRODUCT = AUTHORIZE_WEDGE_V1_U_FREEZE | AUTHORIZE_WEDGE_V1_OWNER_CORPUS (typed auth only; continue≠execute)
NOISY_TRACK = DONE
LM_PROBE = NOT_INDICATED
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

---

## 8. Enhancement pass (2026-07-31 council + live audit)

Live CLOSED/PARTIAL/OPEN table: `papers/MITIGATION_STATUS_SCORECARD.md`  
Outsider surface: `papers/PUBLIC_ONE_PAGER.md`  
Council rerun votes: `.autonomous/post-alpha-evidence-freeze/COUNCIL_RERUN.md`

### B19 — Synthetic perfect-U overclaim (wedge mini-corpus)
**Atoms:** Classical wedge reports Q=1.0 / high U on synthetic pack → readers infer real-workflow readiness.  
**Invariants:** P1, P9, P10.  
**Research pattern:** Context-of-use validity — performance is meaningless without population/task/failure-cost card.  
**Mitigations:**
- **M0:** Results JSON + README must state `corpus_class: SYNTHETIC_MINI`; ban “production-ready” near wedge U.
- **M1:** Owner real-corpus contact protocol (N≥20 docs) with pre-written useful/not sentence.
- **Innovation:** Ship failure gallery (wrong span / silent miss / over-abstain) as the primary demo artifact.  
**Exit:** Real-corpus RESULT *or* explicit product STOP; synthetic U never cited without corpus_class.

### B20 — Governance cosplay (process as product)
**Atoms:** Constitutions, freeze tags, lint, stratigraphy become the work; no user-visible classical+verify habit.  
**Invariants:** P10, P2.  
**Mitigations:**
- **M0:** One-pager is the only outsider entry; further governance docs must reduce confusion or are demoted.
- **M0:** Empty execution queue is allowed; endless new NONCLAIM docs without wedge contact are not.
- **Innovation:** “Contact clock” — if no owner-corpus classical contact within a stated window, auto-recommend SCIENCE_IDLE_NO_PRODUCT.  
**Exit:** Stranger can run or understand one demo without reading the audit corpus.

### B21 — α PDF / LaTeX lag behind correction note
**Atoms:** `PAPER_ALPHA_CORRECTION_NOTE.md` exists; camera-ready PDF may still say ~200M for nano.  
**Invariants:** P1, P8.  
**Mitigations:**
- **M0:** Lint on `papers/paper1*` + `papers/latex/` (done this pass).
- **M1:** Owner rebuild PDF after methods patch; new correction commit if needed.  
**Exit:** `rg` on paper1.tex finds 32.8M for nano or explicit correction cross-ref.

### B22 — Multi-agent / API process failure
**Atoms:** Opus council / Claude CLI credits fail → fake “blocked science”; agents stall.  
**Invariants:** P7, B15.  
**Mitigations:**
- **M0:** Dual-path: Cursor executor + consult_gateway; do not chase credits.
- **M0:** Council votes recorded even when seats fail (in-session replay allowed if labeled).  
**Exit:** Process docs state dual-path; no experiment waits on CLI credits.

### Research anchors added this pass

| Pattern | Source class | Use |
|---------|--------------|-----|
| Context-of-use / evidence-based validation | clinical AI reporting norms | B19, B16, P9 |
| Admissibility vs plausibility | neuro-symbolic hybrid design | B18 |
| Capability-based least privilege | security engineering | B14 |
| Multi-axis evaluation (cost ≠ quality) | HELM-style reporting | Program A R3, B18 |
| Anti-span / assembly tasks | IE / coreference eval design | Program A R1 |

```text
ENHANCEMENT_DEFAULT = SCORECARD + ONE_PAGER + LINT_ATOMS
PRODUCT_CONTACT = owner real-corpus classical (mitigates B19/B20)
STILL_NOT_AUTHORIZED = E4 execute, old-task LM, NanoScribe expansion
```


---

## 9. Enhancement pass — full blocker atomization (2026-07-31)

**Trigger:** Owner request to enhance/mitigate blockers by first-principles decomposition + research-backed innovation.  
**Force:** Design / M0–M1 only (`MAY_AUTHORIZE_EXECUTION: false`). Does **not** mint `AUTHORIZE_*`.  
**Companions:** `WEDGE_V1.md` risk register; `MITIGATION_STATUS_SCORECARD.md`; `OWNER_SPEECH_ACTS.md`.

### 9.0 Method (strict)

```text
for each OPEN/PARTIAL blocker:
  1. Name the physical/computational failure (atom), not the symptom slogan
  2. Map to P1–P10 (which invariant dies if ignored)
  3. Cite a research *pattern* (method literature / lab RESULT) — not a product pitch
  4. Propose M0/M1/M2 with an *enforcing gate* (lint, schema, runner check, or kill rule)
  5. State exit criterion that a stranger can audit
Reject "innovations" that loosen P1/P5/P7.
```

### 9.1 Post–Phase-3 reality atoms (new IDs)

#### B24 — Draft-\(U\) Goodhart / mid-stream weight temptation
**Atoms:** Phase 2/3 scored under **draft** weights; after seeing numbers, pressure to retune \(\alpha,\beta,\gamma,\lambda,\kappa\) or \(\delta\) to make ΔU “look better.”  
**Invariants:** P1, P5, WEDGE B8.  
**Research pattern:** Preregistration; no peeking (Nosek et al. open science); E1 VOID-on-midstream-U edits.  
**Mitigations:**
- **M0:** `recipe_freeze.json` marks `U_status: DRAFT_NOT_SCORING_FROZEN`; results JSON copies that flag.
- **M0:** Lint/forbid editing weight keys in RESULT files after first write (content-hash pin).
- **M1:** Owner string `AUTHORIZE_WEDGE_V1_U_FREEZE` freezes weights + publishes sensitivity grid **before** any new solver arm.  
**Innovation:** Dual ledger: `U_decision` (frozen) vs `U_sensitivity[]` (exploratory, NONCLAIM).  
**Exit:** Either U frozen under typed auth **or** every citation of wedge U says `DRAFT`.

#### B25 — Gold-leaking “classical” heuristics (anti-circular residual)
**Atoms:** Classical T35 path historically matched planted `answer_span` / query cues that encode gold knowledge → theater “classical win” without true locate.  
**Invariants:** P6, WEDGE B4/B19.  
**Research pattern:** E4 anti-circular I*; information parity; train/test contamination literature.  
**Mitigations:**
- **M0:** Ban solvers that read `gold["planted"]` for prediction (gold only in scorer).
- **M0:** E-class non-LM probes (`eclass_probes.py`) are the honest paraphrastic path; classical baseline must ABSTAIN or use frozen synonym map only.
- **M1:** Contamination audit script: AST/grep solvers for `gold[` / `planted` outside `eval/`.  
**Innovation:** **Prediction/scoring firewall** — same invariant as NeSy admissibility: scorer may see gold; proposer may not.  
**Exit:** `rg` clean on `wedge_v1/classical/` for gold imports; T35 classical ABSTAIN or expand-only.

#### B26 — Task-check theater vs claim-level utility
**Atoms:** \(Q\) derived from coarse task pass/fail (or abstain-as-pass) can read as perfect while claim-level liability / miss structure is hidden.  
**Invariants:** P1, P9.  
**Research pattern:** Multi-metric reporting (HELM); selective prediction risk–coverage; FactScore atomic claims.  
**Mitigations:**
- **M0:** Publish both: (i) task-check summary, (ii) per-claim present/abstain/dispute counts, (iii) `liability_presented_bad`.
- **M1:** Official wedge \(U\) components computed from **claim atoms** once U frozen; task-checks become regression smoke.  
**Innovation:** Separate **instrument health** (task pack green) from **decision utility** (claim-level \(U\)).  
**Exit:** RESULT schema documents both layers; no outsider quote of Q=1.0 without corpus_class + claim counts.

#### B27 — Product-contact vacuum (synthetic ceiling)
**Atoms:** All measured wedge U is `SYNTHETIC_MINI`; useful-capability thesis untested on owner private docs (B19/B20).  
**Invariants:** P9, P10.  
**Research pattern:** Context-of-use / external validity; local-first product validation.  
**Mitigations:**
- **M0:** One-pager + RESULTS banner `corpus_class: SYNTHETIC_MINI`.
- **M1:** Protocol for `AUTHORIZE_WEDGE_V1_OWNER_CORPUS`: N≥10 local docs, no PHI in git, pre-written useful/not sentence, classical-only first pass.
- **M0:** Contact clock — if no owner-corpus auth within owner-chosen window → recommend `SCIENCE_IDLE_NO_PRODUCT` (not more governance docs).  
**Innovation:** **Failure gallery** as primary demo (wrong span / miss / over-abstain), not leaderboard U.  
**Exit:** Owner-corpus RESULT *or* explicit product STOP string.

#### B28 — Dual estimand not wired (\(U_{dep}\) vs \(U_{cap}\))
**Atoms:** Phase 3 design calls for deployment utility vs capability utility; harness still reports one draft \(U\). Cost/complexity bumps (C=1.05) can mask capability gains.  
**Invariants:** P1, P6; Program A mitigations lesson.  
**Research pattern:** HELM multi-metric; selective NLP risk–coverage; cost≠quality axes.  
**Mitigations:**
- **M0:** Document both estimands in `PHASE3_LM_PROBE_DESIGN.md` / recipe.
- **M1:** When any future LM probe auth lands, RESULT must emit `U_dep` and `U_cap` side-by-side.  
**Exit:** Schema fields present before any LM score is cited.

#### B29 — Recipe / solver surface drift
**Atoms:** Solvers evolve after corpus SHA freeze; regression pack T01–T40 silently changes meaning (WEDGE B11).  
**Invariants:** P5.  
**Research pattern:** Content-addressed artifacts; software supply-chain pins.  
**Mitigations:**
- **M0:** `recipe_freeze.json` includes hashes of `classical/*.py`, `inclusion_predicates.md`, corpus/gold SHAs.
- **M0:** Runner refuses score write if hashes drift unless `rescope` bit on auth.  
**Exit:** Hash check in `run_classical_baseline` / phase3 runners (M0 stub OK; enforce on next execute auth).

#### B30 — Blocker-ID namespace collision (lab vs wedge)
**Atoms:** `FIRST_PRINCIPLES` B1–B23 ≠ `WEDGE_V1` B1–B22; agents/humans cite “B13” ambiguously.  
**Invariants:** P1 (communication integrity).  
**Mitigations:**
- **M0:** Prefix IDs: `LAB.B#` vs `WED.B#` in cross-refs; scorecard stays LAB.*.  
**Exit:** This map + WEDGE header state the namespace rule.

### 9.2 Re-attack OPEN/PARTIAL lab blockers (deeper atoms)

| ID | Residual atom | Research-backed mechanism | Enforcing gate (M0/M1) | Exit |
|----|---------------|---------------------------|------------------------|------|
| LAB.B1 | Runners may still ignore queue | Capability-token check at process entry | Runner: require `auth_ids` ∩ queue | Fail-closed without queue row |
| LAB.B3/B18 | L/C vs decision conflation | Admissibility vs plausibility split | Two claim IDs + offline pytest for decision | Ledger rows split |
| LAB.B5 | Agent rubric ≠ clinician | Dual estimand + IAA protocol | Never label agent as human | Human arm or explicit no-clinical |
| LAB.B6 | E4 surface split-brain | Owner disposition speech-act | Menu: RATIFY/VOID/PARK | One disposition recorded |
| LAB.B14 | Missing scope_bits | Least-privilege receipts | Lint AUTH_RECORD YAML `scope_bits` | Lint FAIL if absent |
| LAB.B16 | Global misread of KILL | Context-of-use cards | GATE_VERDICT requires field | Validator reject |
| LAB.B17 | Contaminated tip naming | Detached verdict / clean-lineage | OWNER_TAG_OK + tip policy | Tags SHA-stable |
| LAB.B19 | Synthetic U → product claim | Context-of-use / external validity | `corpus_class` required in RESULTS | Real contact or STOP |
| LAB.B20 | Process-as-product | Contact clock + one-pager | No new constitution without contact | Demo without audit corpus |
| LAB.B23 | `continue`→execute | Speech-act force table | Classifier before gated acts | No markers from CONTINUE |

### 9.3 Cross-map: Wedge problems → lab principles → innovations

| Wedge problem (WED.B#) | First principle | Innovation (structure, not hype) |
|------------------------|-----------------|----------------------------------|
| LM-first revival (WED.B1) | P3 cheapest sufficient | Solver cascade registry + ΔU admission exam |
| Always-answer (WED.B2) | P4 verify+abstain | Selective present; forced-answer ablation must hurt Q |
| Ungrounded claims (WED.B3) | P4 | Evidence atom mandatory; span-ablation |
| Benchmark theater (WED.B4) | P6 | I*/X* freeze before scores; prediction/scoring firewall (LAB.B25) |
| OCR entropy (WED.B5) | P9/P10 | Dual-track; never credit LM for ingest bugs |
| Faithfulness≠correctness (WED.B13) | P4 | Constructive faithfulness (retrieve→claim) |
| LM-as-judge (WED.B15) | P1/P5 | Judge-free official U |
| Conjunction errors (WED.B18) | NeSy admissibility | Minimal-condition decomposition |
| Review blow-up (WED.B21) | Selective classification | AURC / R_max budget before adding LM |

### 9.4 Research anchor library (methods — cite patterns, not vibes)

| Pattern | Representative lineage | Lab use |
|---------|------------------------|---------|
| Selective classification / abstention | Chow 1970; Xin et al. ACL 2021 | WED.B2, B12, B21; risk–coverage |
| Evidence attribution / faithfulness | Rashkin et al.; ClaimVer; ICTIR 2025 faithfulness≠correctness | WED.B3, B13 |
| SciFact / scientific claim verify | Evidence sufficiency + abstain | WED.B16–B18 |
| Prereg + artefact gates | Open science; lab E1/E4 | P5, LAB.B1, B24 |
| NeSy admissibility vs plausibility | Hybrid AI design | LAB.B18, §3.5 |
| Context-of-use / evidence-based AI | Clinical AI reporting norms | P9, LAB.B10, B16, B19 |
| Capability-based security | Least privilege | LAB.B14, B15, B23 |
| Policy-as-code | CI lint | §3.2, LAB.B1 |
| Multi-metric eval | HELM | LAB.B26, B28 |
| Anti-circular eval design | E4 I*; contamination lit. | LAB.B25, WED.B4 |

### 9.5 Innovation synthesis (lab-wide — what we invent)

1. **Speech-act × capability-token algebra** — locution classified to force; force grants bits; bits unlock runners. `continue` ∈ M0 only.  
2. **Prediction/scoring firewall** — gold never imported by proposers (LAB.B25).  
3. **Admissibility/plausibility claim split** — KILL re-fire vs L/C reconstruct (LAB.B18).  
4. **Constructive faithfulness cascade** — spans before claims; ablation probe (WED.B13).  
5. **Contact clock** — synthetic U expires as product evidence without owner-corpus auth (LAB.B19/B20/B27).  
6. **Dual estimands** — \(U_{dep}\) vs \(U_{cap}\) before any LM citation (LAB.B28).  
7. **Detached verdict annotations** — honesty without retargeting freeze tags (LAB.B17).  
8. **Namespace discipline** — `LAB.B*` vs `WED.B*` (LAB.B30).

### 9.6 Priority residual (audit order)

| Pri | Focus | Why | Auth? |
|-----|-------|-----|-------|
| 0 | LAB.B23/B1/B14 speech-act + lint + scope_bits | Prevents false science | M0 |
| 1 | LAB.B25 gold-leak firewall | Protects Phase 2/3 honesty | M0 (+ fix solvers) |
| 2 | LAB.B24/B26 draft-U + claim-level reporting | Stops Goodhart | M0; freeze needs auth |
| 3 | LAB.B19/B20/B27 product contact | External validity | Owner corpus auth |
| 4 | LAB.B16/B18 ledger packaging | Honest past KILL | Owner commit |
| 5 | LAB.B6 E4 disposition | Ends split-brain | Owner one-liner |
| 6 | LAB.B3 L/C replay | Cost plausibility | Optional execute auth |
| 7 | WED science R1–R12 | Curiosity | Design; measure later |

### 9.7 Immediate M0 actions from this pass (no experiments)

1. Namespace rule documented (this section) — **done**.  
2. Scorecard updated for B24–B30 — **companion update**.  
3. Results/README `corpus_class: SYNTHETIC_MINI` banners — **apply**.  
4. Grep firewall: classical solvers must not import gold for prediction — **audit+fix**.  
5. Refuse log / speech-act path stays active.  
6. Do **not** run U_FREEZE / OWNER_CORPUS / NOISY / LM without typed auth.

```text
ENHANCEMENT_PASS_9 = ATOMIZED
STANDARDS = UNCHANGED
NEXT_DEFAULT = M0_FIREWALL + SYNTHETIC_BANNER + IDLE
NEXT_TYPED = U_FREEZE | OWNER_CORPUS | NOISY_TRACK | E4_DISPOSITION | OWNER_TAG_OK
FORBIDDEN = OLD_TASK_U revival, NanoScribe expansion, minting auth from prose
```
