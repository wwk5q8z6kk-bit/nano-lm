# Product & Research Discovery Sprint

**Mandate:** `papers/ACTIVE_MANDATE.md`  
**Date:** 2026-07-31  
**Mode:** Active Frontier — not an evidence freeze; not Layer-1 claims.

---

## 1. Product opportunity map

### The job to be done

Researchers and technical builders drown in **private** PDFs, notes, and repos. Cloud notebooks (e.g. NotebookLM) are excellent UX but: data leaves the machine, citations can still drift, and “chat over docs” rarely enforces **structured, abstaining, measurable** answers.

nano-lm’s unfair advantage is not “another local RAG.” It is:

| Asset | Product meaning |
|-------|-----------------|
| E1 / E4 KILL (scoped) | Generation is **not** the default solver — classical + verify first |
| Exact-match / span discipline | Answers carry **char-offset evidence** or **ABSTAIN** |
| Utility thinking (\(Q,E,R,L,C\)) | Ship only solvers that improve \(U\), not vibes |
| Existing `wedge_v1` CLI | Vertical slice already runs: `ask` / `find` / `scan` / `dogfood` |
| Fabric patterns | Typed claims + verify/abstain (library, not OS) |

### Value hypotheses

1. **H-privacy:** Users with sensitive notes/papers will pay (money or friction) for local-first over NotebookLM.  
2. **H-trust:** Fail-closed abstention + span evidence beats fluent RAG that invents.  
3. **H-hybrid:** Small/local LM only where classical fails — lower cost/latency than always-on 70B.  
4. **H-habit:** One command on a folder becomes a daily research habit (Competitor-PM “feared asset”).

### Market / OSS gaps (2026 scan, indicative)

| System | Strength | Gap vs nano-lm wedge |
|--------|----------|----------------------|
| NotebookLM | UX, multi-doc synthesis | Cloud; weak formal abstain/utility |
| Obsidian + Lumen / Knowledge AI / Kwipu / Neural Composer | Local vault RAG, citations | Usually LM-centric; little classical-first registry + \(U\) kill gates |
| Plain vector RAG kits | Easy to stand up | Citation theater; no solver ladder |
| Enterprise search | Scale | Not local-personal; heavy |

**Gap we own:** *classical-first solver registry + evidence spans + abstain + utility kill*, starting from a working CLI on a folder — then add LM only where \(\Delta U > \delta\).

---

## 2. Top three wedges

### A. Local verified research assistant *(recommended)*

**User:** researcher / MD-adjacent / technical founder with a private paper+notes pile.  
**Workflow:** folder → ask structured questions → claims + spans → abstain/contradict → optional escalate.  
**Why nano-lm:** already `wedge_v1`; aligns Evidence Core lessons; MindVault-adjacent.

### B. Small workflow automation engine

**User:** ops / analyst with repeated extract→validate→route tasks.  
**Workflow:** schema-bound extraction, tools, review routing.  
**Why later:** needs customer schema inventory; less differentiation from RPA + LLM wrappers until A proves habit.

### C. Local developer/research agent

**User:** solo researcher in a git repo.  
**Workflow:** search, test, evidence-backed patches, experiment logs.  
**Why later:** overlaps Cursor/Claude Code; harder to differentiate; higher blast radius.

---

## 3. Recommended wedge + rationale

**Choose A — Local verified research assistant.**

| Criterion | A | B | C |
|-----------|---|---|---|
| Reuses in-tree assets | **High** (`wedge_v1`, fabric patterns) | Medium | Medium |
| Differentiator vs OSS RAG | **High** (classical-first + \(U\)) | Medium | Low |
| Privacy value | **High** | Medium | High |
| Path to “someone uses it this week” | **Short** (CLI exists) | Medium | Long |
| Risk of governance relapse | Lower if we *ship dogfood* | Medium | High (agent theater) |

**Personas (A):**

1. **Solo biomedical reader** — 20–100 PDFs + notes; needs drug/dose/compare facts with quotes.  
2. **Technical founder** — design docs + papers; needs “what did we claim?” with spans.  
3. **Thesis / lit-review student** — vault of MD + PDF; needs contradiction flags, not essays.

---

## 4. Technical architecture (baseline)

```text
documents (folder)
   → ingest (md/txt now; PDF text layer next)
   → index (keyword / BM25; vectors optional later)
   → task classify (deterministic)
   → solver registry (classical → e-class → optional LM)
   → claims {value, evidence spans, status}
   → verifier (span must support value; else ABSTAIN)
   → answer JSON / CLI
   → (later) validated memory of verified claims only
```

**Reuse:** `wedge_v1/runtime.py`, classical solvers, e-class probes, fabric claim/verify ideas as library.  
**Do not start with:** new pretrain, NanoScribe OS, E4 R★ revision.

---

## 5. Baseline comparison plan

Freeze a **product** utility (may match wedge draft \(U\); not Layer-1 until promoted):

\[
U = Q - 0.5 E - 0.3 R - 0.02 L - 0.05 C
\]

| Arm | Meaning | When |
|-----|---------|------|
| \(U_{\mathrm{classical}}\) | Current registry (no LM) | **Now** — dogfood + owner folder |
| \(U_{\mathrm{classical+small}}\) | + local small LM on abstain/E-class only | After classical plateaus |
| \(U_{\mathrm{classical+large}}\) | + API large model escalate | Optional; cost tracked |
| \(U_{\mathrm{hybrid+verify}}\) | Best routing + verify-on | Product target |

**Protocol:** same corpus, same question pack, same schema; report \(Q,E,R,L,C\) and failure gallery (wrong span / silent miss / over-abstain).  
**Preregister only** when claiming a confirmatory science result — product dogfood can stay exploratory under the mandate.

---

## 6. First vertical-slice implementation plan

**Slice name:** `wedge_v1` → “folder Q&A with fail-closed evidence”

| Step | Work | Done? |
|------|------|-------|
| S0 | CLI ask/find/scan/smoke/dogfood | **Yes** |
| S1 | Robust `--corpus` on arbitrary MD/TXT folder | Partial |
| S2 | PDF text-layer ingest (pypdf/pdfminer) | **Next coding** |
| S3 | BM25 / better retrieval over naive keyword | Next |
| S4 | Failure gallery exporter (JSON→md) | Next |
| S5 | Owner-corpus dogfood pack (private; no commit of PHI) | Owner |
| S6 | Measure \(U_{\mathrm{classical}}\) on owner pack | After S5 |
| S7 | LM only on measured E-class gaps | Only if \(\Delta U>\delta\) |

---

## 7. Kill criteria

| Kill | Condition |
|------|-----------|
| **K1 Product habit** | After 2 weeks of owner use, <3 real questions/week → park wedge A |
| **K2 Utility** | On owner pack, classical \(U\) worse than “grep + human” time proxy → redesign ingest/retrieval before any LM |
| **K3 Differentiation** | If product equals “Ollama + RAG” with no abstain/span discipline users notice → stop branding nano-lm |
| **K4 Evidence bleed** | Any change that silently edits Evidence Core / tags → immediate stop + revert |
| **K5 Spend** | Any path requiring > compute ceiling without owner → stop |

Science kills (E1/E4) stay historical; they **inform routing**, they do not freeze product discovery.

---

## 8. Exact next coding task

```text
TASK: wedge_v1 PDF text-layer ingest + corpus listing

Implement in wedge_v1/:
1. ingest.py — load .md/.txt/.pdf (text layer) from --corpus into {doc_id: text}
2. Wire runtime.ask/scan/find to use ingest (fallback to current md-only)
3. CLI: `python -m wedge_v1 ingest --corpus DIR` prints doc counts / char totals
4. Test: unit test on a tiny fixture PDF or skip-if-no-pdf; md fixtures must pass
5. Update README one-liner

OUT OF SCOPE: training, vectors, Obsidian plugin, Evidence Core edits, paid GPU
```

---

## Competitive matrix (compact)

| Dimension | NotebookLM | Obsidian local RAG | **nano wedge A** |
|-----------|------------|--------------------|------------------|
| Local/private | No | Yes | **Yes** |
| Classical-first | No | Rare | **Yes** |
| Span evidence + abstain | Soft | Varies | **Hard fail-closed** |
| Utility kill for solvers | No | No | **Yes** |
| UX polish | High | Medium | **Low (CLI-first)** |
| Near-term ship | — | Plugin | **Extend existing CLI** |

---

## Bottom line

Evidence Core stays frozen. Active Frontier builds **A** by making `wedge_v1` something a stranger can point at a messy folder and trust.

**Next action:** execute coding task §8 (PDF ingest), then owner dogfood on a private corpus.
