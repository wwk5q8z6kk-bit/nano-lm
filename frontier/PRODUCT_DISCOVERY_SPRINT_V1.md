# Product & Research Discovery Sprint v1

**Mode:** ACTIVE FRONTIER (`frontier/ACTIVE_MANDATE.md`)  
**Date:** 2026-07-31  
**Evidence class:** product discovery (NONCLAIM for Layer-1)  
**Seed:** Strategic Reset + Wedge v1 classical runtime (dogfood 8/8, LM not indicated)

---

## 0. Framing question

> What small, powerful, useful system can nano-lm build that users actually value?

Not: make a tiny LM score better on the old scribe task.

---

## 1. Product opportunity map

### Pain (who hurts)

| Persona | Pain today | Why cloud LM fails them |
|---------|------------|-------------------------|
| Solo researcher / grad student | Cannot trust chat answers on *their* PDFs/notes; citation hunting is slow | NotebookLM / ChatGPT upload = privacy + retention risk; hallucinations without spans |
| Biomedical / clinical-adjacent analyst | Needs offline / PHI-safe tooling; wrong numeric claim is costly | Cloud RAG forbidden or terrifying; generative summary ≠ evidence |
| Technical founder / IC | Wants answers grounded in repo + specs, with abstention | Copilot-like tools invent APIs; no verify/abstain culture |
| Privacy-first knowledge worker | Obsidian vault is local; AI plugins push to APIs by default | Local RAG stacks exist but optimize fluency over admission of ignorance |

### Market shape (2025–2026)

- **Demand:** privacy-first personal AI is mainstream positioning (AnythingLLM, Jan.ai, local Ollama stacks).
- **Incumbent UX winner:** NotebookLM (easy, strong synthesis) — cloud-only, source caps, weak confidentiality story.
- **Local RAG winners:** AnythingLLM, PrivateGPT-class, Obsidian+Copilot — fluent chat; weak fail-closed verification.
- **Research-specific:** ResearchMind, JARVIS RD, Paper Pilot — Zotero/feeds/citations; still LM-default, not classical-first.

### Gap nano-lm can own

```text
Verification-first local document intelligence
  = classical solvers + span evidence + abstain
  + optional small/large LM only when ΔU > δ
  ≠ another chat-with-PDF wrapper
```

E1/E4 already encode the routing thesis: smallest sufficient solver; generation is not the default.

---

## 2. Top three wedges

### A. Local verified research assistant *(recommended)*

Private folder of papers/notes → structured claims + evidence spans + contradictions + abstain.

**Fits:** existing `wedge_v1`, medical/research background, MindVault direction, evidence culture.

### B. Small workflow automation engine

Repeated structured extract/validate/route jobs (forms, logs, tickets) with deterministic first path.

**Fits:** classical strength; weaker narrative differentiation vs RPA+LLM vendors unless verticalized.

### C. Local developer/research agent

Repo understanding, tests, evidence-backed patches, experiment tracking.

**Fits:** Cursor-adjacent; crowded; risks re-entering agent IDE / NanoScribe OS non-wedge.

---

## 3. Recommended wedge + rationale

**Choose A.**

| Criterion | A | B | C |
|-----------|---|---|---|
| Existing code/runtime | **Strong** (`wedge_v1`) | Partial | Weak |
| Differentiator vs AnythingLLM/NotebookLM | **Verify + abstain + classical-first** | Medium | Low (crowded) |
| Privacy / local story | Strong | Strong | Strong |
| Path to MindVault | Direct | Side door | Distraction |
| Kill-testability | High (U classical vs +LM) | High | Messy |
| Scope control | Already bounded | Needs vertical pick | Explodes |

**Recommendation lock:** Wedge A = product frontier. B/C stay portfolio options, not parallel builds.

---

## 4. Technical architecture (baseline)

```text
Local corpus (md/txt/pdf-text)
        ↓
Ingest / normalize (deterministic)
        ↓
Solver router (cheapest sufficient)
   ├── exact find / regex / dict
   ├── structure / E-class probes
   ├── retrieval (lexical → later embeddings)
   ├── small LM (only if ΔU > δ)
   └── large LM escalate (owner-gated / cost-capped)
        ↓
Verifier R (span bind, schema, contradiction)
        ↓
Present | Abstain | Review
```

**Reuse now:** `wedge_v1.runtime` (`ask|find|scan|dogfood`), classical solvers, Program 0 logging patterns.  
**Do not start with:** new pretrain, NanoScribe control plane, fabric expansion as product.

---

## 5. Baseline comparison plan

Utility (draft from Wedge v1; freeze before confirmatory claims):

\[
U = Q - 0.5 E - 0.3 R - 0.02 L - 0.05 C
\]

Arms (same task pack + corpus class labeled):

| Arm | Meaning |
|-----|---------|
| \(U_{\mathrm{classical}}\) | Current wedge solvers only |
| \(U_{\mathrm{classical+small}}\) | + local small LM propose, verify-on |
| \(U_{\mathrm{classical+large}}\) | + API/large local LM propose, verify-on |
| \(U_{\mathrm{hybrid+verify}}\) | Router + verify + abstain (full policy) |

**Corpus classes (mandatory labels):** `SYNTHETIC_MINI` | `PAPERS_DOGFOOD` | `OWNER_PRIVATE`  
Synthetic U never marketed as product-ready.

**δ:** default 0.05 — LM enters registry only if it beats best cheaper arm by > δ.

---

## 6. First vertical-slice implementation plan

**Already done:** classical ask/find/scan, E-class without LM, papers dogfood 8/8, fail-closed abstain.

**Next slice (this sprint):** make the slice usable as a product stub:

1. Human-readable **markdown report** from ask/find/scan (not JSON-only).
2. Robust `--corpus` over arbitrary folders (md/txt first).
3. One **owner-private** contact run (N≥20 docs) with pre-written useful/not sentence — when owner points at a folder.
4. Failure gallery (wrong span / miss / over-abstain) as demo artifact.
5. Only then: optional small-LM probe behind measured ΔU.

---

## 7. Kill criteria

Stop or pivot the wedge if any hold after honest owner-corpus contact:

1. Classical+verify \(U\) is not useful enough that the owner would use it weekly.
2. Abstain rate makes the tool feel broken *and* lowering abstain requires unverifiable generation.
3. A commodity local RAG (AnythingLLM+Ollama) matches reliability *and* speed with less maintenance.
4. Only path to value is clinical deployment claims (forbidden without separate program).
5. Scope drifts into general agent IDE / NanoScribe OS.

---

## 8. Exact next coding task

```text
TASK: wedge_v1 markdown report surface
FILES: wedge_v1/runtime.py (helpers), wedge_v1/cli.py, wedge_v1/test_runtime_smoke.py
DONE WHEN:
  - `python -m wedge_v1 report ask "…"` prints markdown with status, claims, evidence spans
  - `python -m wedge_v1 report find "…"` and `report scan` work
  - smoke test covers report path
  - JSON subcommands unchanged
OUT OF SCOPE: LM, tags, evidence ledger, paid compute
```

---

## Competitive matrix (summary)

| Product | Local | Citations/spans | Fail-closed abstain | Classical-first | Notes |
|---------|-------|-----------------|---------------------|-----------------|-------|
| NotebookLM | No | Partial | No | No | Best UX, cloud |
| AnythingLLM | Optional | Weak | No | No | Default = chat RAG |
| Obsidian+Copilot | Vault local | Weak | No | No | Plugin/API leakage common |
| PrivateGPT-class | Yes | Weak | No | No | Privacy > verification |
| ResearchMind / JARVIS / Paper Pilot | Yes-ish | Medium | No | No | Research UX, LM-default |
| **nano-lm wedge_v1** | **Yes** | **Yes (spans)** | **Yes** | **Yes** | Thin UX; needs owner corpus |

---

## Personas (short)

1. **Priya** — PhD student, 40 PDFs, hates invented citations.  
2. **Marcus** — indie founder, specs+notes private, needs offline.  
3. **Elena** — biomedical analyst, cannot upload notes to Google.

---

## Value hypotheses

1. Users will tolerate terse answers if every claim shows a span.  
2. Abstention increases trust more than fluent wrongness.  
3. Most research Q&A queries on small corpora are solvable without an LM.  
4. LM value appears on paraphrase / synthesis tails — measurable, not assumed.

---

## What we will not do in this sprint

- Another DIFF E / freeze reconciliation loop  
- R★ revision by default  
- Training runs  
- Public marketing claims  
- Clinical readiness language  
