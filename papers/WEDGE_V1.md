# Wedge v1 — Local research document intelligence

**Date:** 2026-07-31  
**Auth:** `OWNER_STRATEGIC_RESET_OK` · `NEXT_UNIT = WEDGE_V1` · trigger: owner “proceed”  
**Status:** `WEDGE_LOCKED` + `RISK_REGISTER_CLOSED` + `PHASE2_CLASSICAL_COMPLETE` — RESULT `wedge_v1/results_wedge_v1_classical.json` (U≈0.926, 40/40); LM/Nano Runtime expansion needs new auth  
**Center:** `papers/STRATEGIC_RESET.md`  
**Supersedes:** A/B/C research-unit choice; `research/decision_records/2026-07-31-strategic-reset-choose-A.md`; Program A1 as default next

---

## Locked product shape

**Nano Runtime** (already chosen in Strategic Reset):

> A local-first, verification-gated task and knowledge engine that uses the smallest sufficient method for every operation.

**First wedge (this file):**

> **Local research document intelligence** — ingest a small private corpus of technical / biomedical notes and papers; answer structured questions with evidence spans; abstain when unsupported; never default to an unverified generative substrate.

**Why this wedge (not a general assistant):**

| Criterion | Fit |
|-----------|-----|
| Useful | Saves reading / citation / contradiction-hunt time on a private corpus |
| Local / private | Default offline; no required cloud LM |
| Classical-first | Regex, PDF text, retrieval, dictionaries, schema validators win many tasks |
| Verifiable | Spans, schema fields, numeric claims are checkable |
| E1/E4 aligned | Generation only if \(\Delta U_{\mathrm{gen}} > \delta\) on *this* workflow |
| Instrumentable | Reuses Program 0 harness patterns (per-item logs, digests) |

**Explicit non-wedges (do not start):** general chatbot; NanoScribe OS; agent IDE; clinical deployment product; old scribe template world under `OLD_TASK_U`.

---

## User / workflow (one sentence)

A researcher (technical or biomedical) points the system at a **local folder of documents**, asks for **structured facts or comparisons**, and receives **claims with evidence + abstentions**, under measurable review burden and latency.

---

## Success utility (draft — freeze before Phase 2 scoring)

\[
U = Q - 0.5\,E - 0.3\,R - 0.02\,L - 0.05\,C
\]

| Symbol | Meaning on this wedge |
|--------|------------------------|
| \(Q\) | Precision of **presented** claims (verify-on) |
| \(E\) | Miss / wrong rate on fields that should emit |
| \(R\) | Fraction of claims routed to human review |
| \(L\) | p50 latency per task instance (seconds) |
| \(C\) | Relative compute vs frozen classical baseline |

**Kill / keep rule (product):** keep a solver in the registry only if it improves \(U\) vs the best cheaper solver already in the registry. LM enters only when \(\Delta U_{\mathrm{LM}} > \delta\) with \(\delta=0.05\) default (amendable before Phase 2 freeze).

---

## Corpus assumptions (v1)

- **Size:** 10–50 documents (PDF / Markdown / plain text), total \(\le\) 5 MB text extracted.
- **Domain mix:** technical notes + biomedical abstracts/notes (synthetic or owner-provided; no PHI in public artifacts).
- **No** requirement that documents match the old scribe template families.
- Gold labels: schema JSON + evidence char offsets (or paragraph IDs) per task instance.

---

## Representative task pack (n = 40)

IDs are stable. Each task declares **expected cheapest sufficient solver class** *a priori* (not after scoring).

### A — Metadata & structure (classical-favored) — 8

| ID | Task | Expected solver |
|----|------|-----------------|
| T01 | Extract document title | regex / PDF metadata |
| T02 | Extract authors list | regex / heuristics |
| T03 | Extract year / date | regex |
| T04 | Detect document type (note / abstract / paper / table dump) | rules |
| T05 | List section headings | structure parse |
| T06 | Extract DOI / arXiv id if present | regex |
| T07 | Count pages / word count | deterministic |
| T08 | Build TOC from headings | structure parse |

### B — Span-grounded fact extraction — 10

| ID | Task | Expected solver |
|----|------|-----------------|
| T09 | Extract all dosage-like strings with spans | regex + units dict |
| T10 | Extract drug / compound names (lexicon ∪ pattern) | dict + span |
| T11 | Extract numeric results with units | regex |
| T12 | Extract definitions of the form “X is …” | pattern |
| T13 | Extract claimed sample size \(n\) | regex + verify |
| T14 | Extract affiliations | regex / heuristics |
| T15 | Extract email / contact if present | regex |
| T16 | Extract figure/table captions | structure |
| T17 | Map key–value lines in semi-structured notes | parser |
| T18 | Extract URLs / citations keys | regex |

### C — Retrieval & QA with mandatory evidence — 8

| ID | Task | Expected solver |
|----|------|-----------------|
| T19 | Find paragraph answering a keyword query | BM25 / exact |
| T20 | Quote the sentence containing term \(X\) | search |
| T21 | Answer yes/no “does doc mention \(X\)?” with span or abstain | search |
| T22 | List all sentences mentioning entity \(E\) | search |
| T23 | Retrieve top-3 passages for question \(q\) | BM25 |
| T24 | Answer factoid only if span entails it; else abstain | retrieve + entailment rules |
| T25 | Multi-doc: which files mention \(X\)? | inverted index |
| T26 | Multi-doc: union of dosages across corpus | dict + span |

### D — Comparison, contradiction, schema fill — 8

| ID | Task | Expected solver |
|----|------|-----------------|
| T27 | Fill fixed schema fields from one note (open slots) | hybrid candidates |
| T28 | Compare two docs on one field (same / differ / missing) | extract + diff |
| T29 | Flag numeric contradictions across two docs | extract + compare |
| T30 | Flag entity alias collision (same string, different types) | dict |
| T31 | Merge two schema fills with conflict → disputed | symbolic merge |
| T32 | Produce verified summary line from schema only | template |
| T33 | Reject claim with no evidence span | verifier |
| T34 | Abstain when multiple conflicting spans | verifier |

### E — Stress / generative-optional (must not dominate pack) — 6

| ID | Task | Expected solver |
|----|------|-----------------|
| T35 | Paraphrastic question → find supporting span | retrieve; gen only if classical fails |
| T36 | Implicit relation (“implies dose change”) | gen-optional; default abstain |
| T37 | Noisy OCR line → recover field | classical normalize first |
| T38 | Table-ish plaintext → structured rows | parser; gen-optional |
| T39 | Cross-sentence entity binding | classical coref lite; gen-optional |
| T40 | Short explanation *selected from evidence only* | extractive; no free gen |

**Pack rule:** ≤15% of tasks (T35–T40) may justify a generative probe. The wedge succeeds if classical + verify delivers high \(U\) on A–D; generative is evaluated only on E and only under information parity.

---

## First-principles decomposition

The wedge is not “build a small LM for docs.” Decompose the useful capability into
independent physical/computational problems:

```text
P0  Information exists as bytes on disk
P1  Bytes → normalized text + structure (pages, headings, offsets)
P2  Text → candidate atoms (spans, numbers, entities, key–values)
P3  Atoms → typed claims (schema fields, yes/no, comparisons)
P4  Claims → evidence binding (offset / paragraph id / doc id)
P5  Bound claims → verification under a decidable relation R
P6  Verified claims → present / abstain / escalate (selective prediction)
P7  Presented set → utility U under cost, latency, review burden
P8  Router chooses cheapest solver per (task class × doc class)
P9  Composite claims → minimal conditions (conjunction safety)
P10 Locate miss vs evidence-absent (distinct failure modes)
P11 Score → risk–coverage / AURC (not accuracy alone)
```

**Invariant:** no claim may cross P5→P6 without a binding (P4) and a pass/fail/abstain
verdict under frozen \(R\). Free generation that skips P4–P5 is out of scope for v1.

**Lab evidence already in force:**

| Principle | Source | Implication for wedge |
|-----------|--------|------------------------|
| Generation is not the default | E1 KILL | Classical solvers own A–D unless \(\Delta U_{\mathrm{LM}}>\delta\) |
| Template-isomorphic worlds are classical-won | E1 M1 | Do not recreate scribe templates as the corpus DGP |
| Exact-match ≠ clinical faithfulness | E3 | Soft-match / human rubric only as secondary arms |
| R★ shopping is banned | E4 KILL + anti-circular I* | Corpus inclusion by process predicates, not gen-win filters |
| Fabric verify works under decidable \(R\) | fabric slice | Reuse span/schema/absence checks; do not claim open-world truth |

---

## Blockers, risks, and first-principles mitigations

Each row: **blocker** → root cause (first principles) → **mitigation** (design + research
backing) → **acceptance test**.

### B1 — Wrong substrate (LM-first revival)

| | |
|--|--|
| **Root cause** | Conflating “language task” with “needs a generative model.” Extraction under stable cues is often finite-state / retrieval (E1). |
| **Mitigation** | Capability routing registry: every task ID maps to an ordered solver cascade (deterministic → dict/span → BM25 → optional LM). LM may fire only after classical miss **and** only on E-class tasks in v1. |
| **Research** | Selective cascades for accuracy–efficiency (Xin et al., ACL 2021 “Art of Abstention”); E1 utility kill-gate. |
| **Accept** | Phase 2 reports \(U\) for classical-only on A–D before any LM code path exists. |

### B2 — Hallucinated helpfulness (always-answer bias)

| | |
|--|--|
| **Root cause** | Systems optimize coverage; unsupported claims maximize short-term “answers.” |
| **Mitigation** | Selective prediction: present only if verifier score / retrieval margin ≥ threshold \(\tau\); else `ABSTAIN` or `REVIEW`. Freeze \(\tau\) before scoring; report risk–coverage curve. |
| **Research** | Chow (1970) selective classification; Xin et al. 2021; SciFact-style evidence-first RAG with abstention gates; SQuAD 2.0 answerability. |
| **Accept** | Forced-answer ablation must *worsen* presented precision \(Q\) vs abstaining system. |

### B3 — Ungrounded claims (no evidence atom)

| | |
|--|--|
| **Root cause** | Text generators emit strings not entailed by any span. |
| **Mitigation** | Claim schema requires `evidence: [{doc_id, start, end}]` non-empty for presentable claims. Verifier rejects empty binding (T33). Extractive-only explanations (T40). |
| **Research** | Attributable / extrapolatory / contradictory claim categories (Rashkin et al.; ClaimVer / evidence attribution); lab Fabric provenance. |
| **Accept** | 0 presented claims with empty evidence in verify-on arm. |

### B4 — Benchmark theater / classical crippling

| | |
|--|--|
| **Root cause** | Eval set chosen because rules fail; or rules forbidden by fiat. |
| **Mitigation** | Corpus inclusion predicates I* fixed *before* any solver scores (process: license-clean abstracts + synthetic notes with declared generators). Classical toolbox fully allowed (regex, dict, BM25, FST, schemas). ≤15% E-tasks. |
| **Research** | E4 anti-circular regime design; information-parity bakeoffs. |
| **Accept** | Publish I*/X* and classical recipe freeze hashes before Phase 2 scores. |

### B5 — PDF / OCR / layout entropy

| | |
|--|--|
| **Root cause** | Real docs are not clean UTF-8; structure loss creates fake “LM necessity.” |
| **Mitigation** | Dual-track corpus: (i) clean Markdown/text primary; (ii) optional noisy OCR track scored separately. Normalize (unicode, whitespace, ligatures) before solvers. Table-ish lines → deterministic row splitter before any gen (T37–T38). |
| **Research** | Document AI: layout-aware parse as *preprocessing*, not intelligence; don’t credit LM for fixing ingestion bugs. |
| **Accept** | Primary \(U\) reported on clean track; noisy track is diagnostic only until ingestion SLA met. |

### B6 — Entity / ontology mismatch

| | |
|--|--|
| **Root cause** | Same string, different types; aliases; open vs closed vocab (α field localization). |
| **Mitigation** | Typed lexicon + collision detector (T30). Closed lists for units/drugs where possible; open slots must abstain under ambiguity (T34). No silent synonym collapse without E3-style construct policy. |
| **Research** | Paper α field localization; E3 exact-survives; entity linking as *optional* module with its own \(U\) delta. |
| **Accept** | Collision and multi-span conflict routes to `DISPUTED`/`ABSTAIN`, never auto-pick. |

### B7 — Multi-doc contradiction & merge errors

| | |
|--|--|
| **Root cause** | Independent extracts disagree; naive merge invents consensus. |
| **Mitigation** | Explicit epistemic states: `CONFIRMED` / `PROBABLE` / `DISPUTED` / `MISSING`. Merge is symbolic (T31); numeric compare with tolerance ε (T29). No generative reconciliation in v1. |
| **Research** | Evidence ledgers; contradiction-as-first-class (lab constitution invariants). |
| **Accept** | Planted contradictions recovered at ≥ target recall; false merges = 0 on planted set. |

### B8 — Utility / metric Goodharting

| | |
|--|--|
| **Root cause** | Optimizing one proxy (exact match, BLEU, “helpfulness”) hides review and cost. |
| **Mitigation** | Single frozen \(U\) before Phase 2; mandatory report of \(Q,E,R,L,C\); liability count (presented fabrications) outside \(U\) but published. Sensitivity grid on weights; no mid-stream \(U\) edits after scores (VOID). |
| **Research** | E1/E4 utility discipline; selective prediction risk–coverage. |
| **Accept** | Decision artifact mirrors `results_e1_utility.json` shape for wedge classical baseline. |

### B9 — Privacy / PHI / license leakage

| | |
|--|--|
| **Root cause** | Biomedical notes invite real PHI; web scrapes invite license risk. |
| **Mitigation** | Public artifacts = synthetic or explicitly license-clean only. Owner private corpus stays local; no cloud LM default. Redaction checklist before any share. |
| **Research** | Local-first product thesis (Strategic Reset); data-minimization. |
| **Accept** | CI forbid-list: no raw clinical notes in git; PHI scanner on `wedge_v1/data/public/`. |

### B10 — Scope explosion (NanoScribe / agents / memory)

| | |
|--|--|
| **Root cause** | Ambition re-enters through “just one more module.” |
| **Mitigation** | Hard non-goals list; Phase 2 = classical solvers + verifier only. Memory/agents only after repeated wedge \(U\) wins (Decision Gate G5). |
| **Research** | Lab Decision Gates G4–G5; E1 post-lock architecture principle. |
| **Accept** | Diff review rejects memory/agent/UI PRs under current auth. |

### B11 — Maintenance / rule rot

| | |
|--|--|
| **Root cause** | Regex empires rot; LMs rot differently (silent drift). |
| **Mitigation** | Per-solver `M` (maintenance) score pre-assigned (E4-style rubric); prefer small rule surfaces; content-addressed lexicons; regression pack = task IDs T01–T40. |
| **Research** | E4 maintenance term in \(U_{R★}\); software entropy. |
| **Accept** | Each solver declares surface area + update procedure in recipe freeze. |

### B12 — Calibration theater without abstention

| | |
|--|--|
| **Root cause** | Confidence scores that never change the action. |
| **Mitigation** | Confidence must gate present vs abstain; temperature scaling / margin features only if they move risk–coverage. |
| **Research** | Temperature scaling (Guo et al.); selective NLP (Xin et al. 2021); evidence-chain selective fact-checking. |
| **Accept** | Publish coverage vs precision curve; \(\tau\) chosen on locked calib split only. |

---

### B13 — Post-rationalized citations (correctness ≠ faithfulness)

| | |
|--|--|
| **Root cause** | Generate-then-cite: model answers from parametric memory, then attaches a span that merely *looks* related. A citation can be "correct" (span supports claim) while **unfaithful** (model did not rely on it). Studies report up to ~57% faithfulness failures in attributed RAG. |
| **Mitigation** | **Constructive faithfulness:** order is retrieve → select spans → derive claims as (a) exact quote, (b) deterministic normalize, or (c) Phase-3 paraphrase under entity/number lock. Never free-generate then search for a citation. **Span-ablation probe:** drop cited offsets; claim must fail support. |
| **Research** | *Correctness is not Faithfulness in RAG Attributions* (ICTIR 2025 / arXiv:2412.18004); AwF — Answering with Faithfulness (IJCNLP 2025); franq faithfulness-aware UQ (2025). |
| **Accept** | Ablation fail rate on presented claims = 0 in classical verify-on arm; AwF precision/recall published beside \(U\). |

### B14 — Multi-authority / stale governance text (dogfood hazard)

| | |
|--|--|
| **Root cause** | Real corpora contain superseded claims (e.g. glossary "E4 untested" vs ambition "E4 KILL"). Silent "latest file wins" invents false consensus. |
| **Mitigation** | Authority tiers for nano-lm dogfood: T0 Evidence Ledger → T1 freeze tags → T2 ambition/queue/reset → T3 calibrated findings → T4 drafts/glossary/plans. Conflicts → `DISPUTED`/`CONTRADICTED` with both spans + tier note; never silent pick. |
| **Research** | Lab claim discipline; epistemic merge states (B7); belief revision without generative reconciliation. |
| **Accept** | Planted stale-vs-current traps recovered; false auto-resolve = 0. |

### B15 — Circular LM-as-judge for official \(U\)

| | |
|--|--|
| **Root cause** | Scoring free-form answers with an LM judge reintroduces the substrate under evaluation. |
| **Mitigation** | Official \(U\) uses gold atomic claims + offset/span containment + entity/number lock. LM judges only as **side** diagnostic, never as the gate. |
| **Research** | FactScore-style atomic decomposition; lab E1/E3 exact-match primary; scoring independence (P8). |
| **Accept** | Phase 2 RESULT recomputable from frozen gold JSON without any LM API. |

---

### B16 — Retrieval miss (false abstain / silent Q loss)

| | |
|--|--|
| **Root cause** | Evidence exists in corpus but P1/P2 indexing or query mismatch fails to surface it → system abstains while a human would find it. |
| **Mitigation** | Cascaded locate: exact ID/heading lookup → BM25 → (optional) synonym expansion from *frozen* lexicon only. Report `retrieval_miss` separately from `evidence_absent`. Gold includes negative + positive locate probes. Risk–coverage curve must not hide miss-driven abstention. |
| **Research** | Claim-aware SciFact RAG: hallucination often = answer after retrieval failure; abstain on low retrieval margin (BM25 top-1 / margin thresholds). SQuAD 2.0 answerability. |
| **Accept** | On gold-SUPPORTED tasks with planted keywords, classical locate recall ≥ target; miss vs true-absent labeled distinctly in logs. |

### B17 — Retrieval noise (wrong-span support)

| | |
|--|--|
| **Root cause** | High-scoring but irrelevant passage; entity/number lock still “matches” a substring coincidentally. |
| **Mitigation** | Support requires (i) span selected *before* claim text, (ii) type-compatible entity match, (iii) optional second-pass consistency: opposing top-k spans must not refute. Margin between top-1 and top-2 retrieval scores gates PRESENT vs REVIEW. |
| **Research** | Evidence-consistency signals + retrieval-margin abstention in claim-aware scientific RAG; SciFact evidence sufficiency. |
| **Accept** | Planted near-miss distractors: false SUPPORTED rate = 0 under verify-on. |

### B18 — Compositional claim failure (atoms true, conjunction false)

| | |
|--|--|
| **Root cause** | System presents claim C = c1∧c2∧… where each ci has a span, but the *joint* claim is not entailed (classic multi-hop / conjunction error). |
| **Mitigation** | Decompose every presentable claim into **minimal conditions**; each condition audited independently; PRESENT only if all critical conditions pass; any critical FAIL → REFUTE/ABSTAIN (contradiction-prioritized). No generative “glue” sentence that introduces unbound predicates. |
| **Research** | Abstention-aware scientific reasoning: minimal-condition decomposition + per-condition NLI/audit (Abdaljalil et al., 2026); SciFact claim verification. |
| **Accept** | Planted conjunction traps: system must ABSTAIN or DISPUTED, not SUPPORTED. |

### B19 — Gold / recipe contamination

| | |
|--|--|
| **Root cause** | Solvers or lexicons tuned on the evaluation pack; U becomes non-general. |
| **Mitigation** | Freeze I*/X* corpus + gold *before* solver iteration; hold-out calib split for τ only; content-addressed recipe hashes; forbid editing gold after first classical score (VOID if violated). |
| **Research** | E4 anti-circular regime; prereg/freeze discipline in this lab. |
| **Accept** | Manifest SHA of gold + corpus published with Phase 2 RESULT; diff audit clean. |

### B20 — Authority tier ≠ truth (T0 lock-in)

| | |
|--|--|
| **Root cause** | Tier ranking resolves conflicts for *governance dogfood*, but Tier-0 can still be wrong or incomplete; treating tier as ontology of truth freezes error. |
| **Mitigation** | Tiers govern **decision priority**, not metaphysical truth. Always emit both spans. User/owner can supersede via explicit override claim with new evidence. External scientific corpora use **equal peer** tier + DISPUTED, not T0 privilege. |
| **Research** | Belief revision; lab ledger amend-only-with-new-claim-ID rule. |
| **Accept** | Override path tested; peer-corpus mode never auto-picks by filename recency. |

### B21 — Verification / review cost blow-up

| | |
|--|--|
| **Root cause** | P5/P6 escalate everything → R and C dominate U; product feels useless. |
| **Mitigation** | Escalate only CONTRADICTED / below-τ PARTIAL / type-collision. Cheap deterministic verifies first. Budget \(R_{\max}\) on calib split; if exceeded, raise τ or shrink claim surface — do not add LM. Report AURC (area under risk–coverage), not accuracy alone. |
| **Research** | Risk–coverage / AURC as primary scientific reliability lens; abstention halves risk at moderate coverage even when raw accuracy is flat across models. |
| **Accept** | Phase 2 publishes risk–coverage curve; operating point meets frozen \(R\) budget. |

### B22 — Cold start, adversarial docs, alert fatigue

| | |
|--|--|
| **Root cause** | Empty corpus; malicious/misleading local files; humans ignore constant REVIEW. |
| **Mitigation** | Empty → structured `NO_CORPUS` (not a fluent apology). Untrusted folders: quarantine + hash; no auto-CONFIRMED. REVIEW batching with severity; rate-limit escalations; measure user-correction rate as side metric. |
| **Research** | Local-first threat model; alert fatigue literature; data minimization (B9). |
| **Accept** | NO_CORPUS path tested; planted malicious note never yields CONFIRMED without verify. |

---

## Innovative synthesis (what we invent, not copy)

1. **Solver-selecting fabric, not LM-centric RAG.** Retrieval and generation are peers in a cost-ordered cascade; verification is mandatory infrastructure (lab post-E1 principle + SciFact abstention).
2. **Typed claim + evidence atom as the only presentable object.** Aligns ClaimVer-style attribution with Fabric provenance; kills free-text answers as a product surface.
3. **Constructive faithfulness (span-first).** Kills post-rationalized citations by construction; span-ablation is a required faithfulness probe (B13).
4. **Dual-track corpus (clean vs noisy)** so ingestion failure is not misread as model failure.
5. **Epistemic merge states + authority tiers** instead of generative reconciliation (B7, B14).
6. **ΔU admission exam for every new solver**, including future tiny LMs — same law as E1, new world.
7. **Judge-free official U** — gold atoms + deterministic support; LM judges never gate Phase 2 (B15).
8. **Condition-decomposed claims** — PRESENT only if all critical conditions pass (B18).
9. **Risk–coverage / AURC as reliability lens** — primary challenge is *when evidence is enough*, not which model (B16, B21).

```text
Innovation target:
  smallest sufficient verified system on a real local research workflow
Not:
  smallest perplexity model on a synthetic isomorphic template
```

---

## Research backlog derived from blockers (design-only until authorized)

| ID | Question | Unblocks |
|----|----------|----------|
| R1 | Best retrieval-margin / verifier features for \(\tau\) on this corpus? | B2, B12 |
| R2 | Minimal lexicon + pattern set that saturates B-class \(U\)? | B1, B11 |
| R3 | When does paraphrastic retrieve (T35) require dense retrievers vs BM25? | B1, E-tasks |
| R4 | Numeric contradiction tolerance ε vs false-dispute rate | B7 |
| R5 | Soft-match construct policy for open slots (E3 lessons) | B6, B8 |
| R6 | Span-ablation + AwF metrics on this corpus — calibration of constructive faithfulness | B13 |
| R7 | Authority-tier conflict detector precision/recall on planted stale docs | B14 |
| R8 | Entity/number lock rules that block paraphrase smuggling without killing recall | B13, B3 |
| R9 | Retrieval-margin vs locate-recall tradeoff on planted miss/distract packs | B16, B17 |
| R10 | Minimal-condition splitter that stays classical on A–D tasks | B18 |
| R11 | Operating τ meeting R_max on calib without killing Q on EX tasks | B21, B12 |
| R12 | Peer-corpus mode (no T0 privilege) vs dogfood authority mode | B20 |

These are **not** Program 1 and **not** E4-prime. They activate only under Phase 2+ auth.

---

## Risk closure matrix (Phase 1)

Every irreducible problem maps to mitigations. **Phase 1 risk register = CLOSED** for design; remaining work is measurement under Phase 2 auth.

| Problem | Blockers | Design status |
|---------|----------|---------------|
| P0 bytes on disk | B9, B22 | Mitigated |
| P1 normalize / structure | B5 | Mitigated |
| P2 atoms | B6, B11 | Mitigated |
| P3 typed claims | B3, B18 | Mitigated |
| P4 evidence binding | B3, B13 | Mitigated |
| P5 verify under R | B13, B17, B18 | Mitigated |
| P6 selective present | B2, B12, B16, B21 | Mitigated |
| P7 utility U | B8, B21 | Mitigated |
| P8 router | B1, B10 | Mitigated |
| P9 conjunction | B18 | Mitigated |
| P10 miss vs absent | B16 | Mitigated |
| P11 risk–coverage | B21 | Mitigated |
| Cross-cutting | B4, B7, B14, B15, B19, B20, B22 | Mitigated |

```text
PHASE1_RISK_REGISTER = CLOSED
NEXT = OWNER_AUTH_FOR_PHASE3_OR_EXPAND  # Phase 2 DONE
NO_NEW_LAB_STRUCTURE
```

---

## Phase gates (from Strategic Reset)

| Phase | Work | Status after this lock |
|-------|------|------------------------|
| **1** | Wedge + 40 tasks | **DONE (this doc)** |
| **2** | Classical baseline on frozen mini-corpus | **DONE** — `wedge_v1/results_wedge_v1_classical.json` (U≈0.926, 40/40 on clean synthetic track) |
| **3** | Add small model only where baseline fails | **Not authorized** — needs `AUTHORIZE_WEDGE_V1_PHASE3_LM_PROBE`; classical abstains on T35/T36/T39 |
| **4+** | Memory / efficiency / expand | Blocked |

---

## Still blocked

```text
PROGRAM1 = NOT_AUTHORIZED
TRAINING = NOT_AUTHORIZED
E4_EXECUTE = BLOCKED
NANOSCRIBE = STOP
INFRA_EXPANSION = STOP
COMPONENT_BUILDS = BLOCKED_UNTIL_PHASE3_OR_EXPAND_AUTH
PHASE2 = COMPLETE
OLD_TASK_U = FORBIDDEN
```

---

## Owner accept string (recorded)

```text
OWNER_STRATEGIC_RESET_OK
NEXT_UNIT = WEDGE_V1
WEDGE = local_research_document_intelligence
TASK_PACK_N = 40
```
