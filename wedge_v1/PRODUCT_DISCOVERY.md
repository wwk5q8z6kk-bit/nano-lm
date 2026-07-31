# Product Discovery — Active Frontier

**Mandate:** `ACTIVE_MANDATE = BUILD_SMALL_POWERFUL_USEFUL_SYSTEM_V1`  
**Mode:** ACTIVE_FRONTIER — research → invent → build → test → learn  
**Date:** 2026-07-31  
**NONCLAIM:** Product strategy. Does not alter Layer-1 Evidence Ledger or Paper α.

---

## 1. Product opportunity map

**User pain (research / technical knowledge work):**
- Assistants answer fluently without trustworthy provenance.
- Cloud RAG leaks private notes/papers; regulated users cannot use them.
- Local chat-with-docs tools (AnythingLLM, PrivateGPT, Obsidian plugins) optimize *coverage of chat*, not *fail-closed verification*.
- Teams overspend large models on tasks classical retrieval + rules already solve (E1 lesson).

**Opportunity:** A **local evidence system** that uses the smallest sufficient solver, verifies consequential claims, and abstains — not another chatbot UI.

**Unique nano-lm assets:** claim discipline, E1 classical-default routing, Fabric-class verify/abstain patterns, measurable U, existing `wedge_v1` runtime (dogfood 8/8 on papers/).

---

## 2. Top three wedges

| ID | Wedge | User | Differentiator |
|----|-------|------|----------------|
| **A** | Local verified research assistant | Researchers, clinicians-in-training, technical PMs | Span-first claims, contradiction, abstention, authority-aware dogfood |
| **B** | Small workflow automation engine | Ops / back-office extraction | Max U per dollar/joule; deterministic first |
| **C** | Local developer/research agent | Solo builders | Evidence-backed repo ops; not a full IDE |

---

## 3. Recommended wedge + rationale

**Choose A.**

| Criterion | Why A |
|-----------|-------|
| Fit to assets | Direct extension of `wedge_v1` + evidence culture |
| Differentiator vs AnythingLLM/PrivateGPT/Obsidian RAG | Constructive faithfulness + abstain-by-default, not chat-first |
| Path to MindVault | Validated memory later (Phase 4); not day-one |
| Kill-friendly | Clear U baselines without training |
| Scope | Narrow enough for a vertical slice now |

B remains a solver mode inside A. C deferred until A pays.

---

## 4. Personas and workflows

**Persona P1 — Solo researcher:** folder of PDFs/MD notes; needs cited answers and contradiction flags offline.  
**Persona P2 — Technical lead:** project docs + decision records; needs status facts with spans, not narrative.  
**Persona P3 — Privacy-sensitive professional:** cannot send notes to cloud; accepts ABSTAIN over fluent wrong.

**Core workflow:**
```text
docs → index → question → cheapest solver → atomic claims → evidence spans
    → verify → SUPPORTED | CONTRADICTED | ABSTAIN → (later) validated memory
```

---

## 5. Competitive matrix (2026 snapshot)

| System | Local | Citations | Fail-closed abstain | Classical-first routing | Claim ledger |
|--------|------:|-----------|--------------------:|------------------------:|-------------:|
| NotebookLM | No | Strong | Weak | No | No |
| AnythingLLM | Optional | Partial | Weak | No | No |
| PrivateGPT | Yes | Partial | Weak | No | No |
| PaperRAG / local paper RAG | Yes | Chunk/cite | Weak–medium | No | No |
| Obsidian Local LLM Helper / Hub | Yes | Partial | Weak | No | No |
| TrustLayer / rag-grounded (OSS) | Varies | Stronger | Yes (research) | Partial | No |
| **nano-lm wedge_v1 target** | **Yes** | **Span-required** | **Yes** | **Yes (E1)** | **Yes (path)** |

Gap we own: **verification-gated product surface** + **ΔU admission for any LM**.

---

## 6. Value hypotheses

| H | Hypothesis | Measure |
|---|------------|---------|
| H1 | Users prefer ABSTAIN over unsupported answers on private corpora | Unsupported-claim rate ↓; correction rate ↓ |
| H2 | Classical+expand beats chat-RAG on exact fact/tag queries | Dogfood / U on held tasks |
| H3 | Contradiction surfacing saves review time vs silent merge | Time-to-dispute; planted-trap recall |
| H4 | Small LM only helps paraphrase/coref strata | ΔU_LM > δ on allowlisted E-class only |

---

## 7. Technical architecture (Active Frontier)

```text
CLI / API (wedge_v1)
  → Corpus loader (md/txt; PDF text-layer via pypdf)
  → Solver registry (ordered cascade)
       exact FIND → BM25/keyword → expand/symbolic → (optional) small LM
  → Claim objects {text, evidence[], status}
  → Verifier (decidable R: span, entity/number lock, ablation)
  → Response {SUPPORTED|CONTRADICTED|ABSTAIN|NO_CORPUS}
  → Eval harness (smoke, dogfood, U)   ← Program 0 patterns
```

Reuse: `wedge_v1/runtime.py`, solvers, dogfood. Do **not** train. Do **not** touch Evidence Core.

---

## 8. Baseline comparison plan

Arms (same corpus/gold):

| Arm | Description |
|-----|-------------|
| C0 | Classical cascade only (current) |
| C1 | C0 + BM25 hybrid (if adds recall) |
| S0 | Optional small local LM propose → verify |
| L0 | Optional large/cloud LM propose → verify (opt-in; privacy off) |

Report U and risk–coverage. Keep LM only if ΔU > δ=0.05. A–D classical wins stay load-bearing.

---

## 9. First vertical-slice plan (done → next)

| Step | Status |
|------|--------|
| ask/find/scan + fail-closed | Done |
| papers dogfood 8/8 | Done |
| ask_folder_v0 (BM25 + JSON schema + optional PDF) | Done |
| ingest CLI (recursive md/txt/pdf) | Done |
| compare / contradiction across docs | Done (CLI + nearby banners) |
| owner-corpus contact path (gitignored) | **DONE** — `owner-dogfood --demo` / `$OWNER_CORPUS` |
| Validated memory (confirmed only) | After slice U stable |

---

## 10. Kill criteria

Kill or narrow A if:
1. Dogfood accuracy on expanded ≥20-task pack < 0.7 after classical hardening; or
2. Users reject ABSTAIN (correction rate → they force-answer offline); or
3. No ΔU path vs AnythingLLM-class baseline on privacy+citation tasks within 2 iterations; or
4. Scope expands into NanoScribe/agent IDE without U wins.

---

## 11. Exact next coding task

```text
STATUS = owner-smoke 5/5 on owner_corpus.example
NEXT:
  export WEDGE_OWNER_CORPUS=/path/to/private/folder
  python3 -m wedge_v1 ingest "$WEDGE_OWNER_CORPUS"
  python3 -m wedge_v1 owner-smoke --corpus "$WEDGE_OWNER_CORPUS"
Optional papers dry-run:
  python3 -m wedge_v1.run_owner_smoke --corpus papers --tasks wedge_v1/data/owner_smoke_tasks_papers.json
```

