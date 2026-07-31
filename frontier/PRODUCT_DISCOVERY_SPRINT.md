# Product Discovery Sprint — Active Frontier

**Date:** 2026-07-31  
**Branch:** `frontier/active-v1`  
**Mandate:** `ACTIVE_MANDATE = BUILD_SMALL_POWERFUL_USEFUL_SYSTEM_V1`  
**Not Layer-1 evidence. Not a constitution.**  
**Runtime:** `wedge_v1/` · strategy: `papers/STRATEGIC_RESET.md` · shape: `papers/WEDGE_V1.md`  
**Also see:** `frontier/PRODUCT_DISCOVERY_SPRINT_V1.md` (earlier draft), `wedge_v1/PRODUCT_DISCOVERY.md`

---

## 1. Product opportunity map

### Job to be done

A researcher points a tool at a **private** folder of papers/notes and gets **structured, citable answers** that **abstain** when unsupported — without uploading the corpus to a cloud chat product.

### Pain / market (2025–26)

| Pain | Common fix | Failure |
|------|------------|---------|
| Private corpus + trust | NotebookLM / ChatGPT upload | Egress; fluent wrong answers |
| Local chat-over-docs | AnythingLLM + Ollama, PrivateGPT | LM-centric; soft citations; weak hard abstain |
| Vault RAG | Obsidian plugins, DocAgent Studio | Embeddings+LLM first; little solver ladder |
| “Verified RAG” OSS | AuditRAG, CITECHECKAI, GroundTruth, CodaCite | Faithfulness via second model / prompt — still gen-default |
| Paid local desktop | PrivateDocs AI (~$149 lifetime) | Polished UX; still model-centric |

### nano-lm asymmetric edge

1. **Classical-first solver cascade** (E1/E4 as design constraint)  
2. **Fail-closed char-offset spans** or ABSTAIN  
3. **ΔU admission** for heavier solvers  
4. **Contradiction as status** (DISPUTED / CONTRADICTED)  
5. **Working CLI** — ask / find / scan / compare / ingest / report / dogfood  

### Personas

1. **Solo technical researcher** — notes + PDFs; exact numbers, version diffs.  
2. **Biomed / methods reader** — dosages / \(n\) / outcomes with spans; no PHI in public builds.  
3. **Founder dogfooding nano-lm** — “what did we claim?” on project docs.

### Workflows

1. Folder Q&A → claims + spans or ABSTAIN  
2. Cross-doc disagree on field F → CONTRADICTED + both sides  
3. Schema fill → typed fields + missing  
4. Quote definition sentence → extractive only  

### Value hypotheses

| ID | Hypothesis |
|----|------------|
| H-privacy | Sensitive-corpus users prefer local-first over NotebookLM |
| H-trust | Fail-closed abstain + spans beats fluent RAG |
| H-hybrid | Small LM only where classical fails (ΔU > δ) |
| H-habit | One folder CLI becomes a weekly research habit |

---

## 2. Top three wedges (A / B / C)

| ID | Wedge | Feasibility now | Differentiation | Main risk |
|----|-------|-----------------|-----------------|-----------|
| **A** | Local verified research assistant | **High** — runtime live | Classical-first + ΔU + spans | Synthetic ≠ real corpus |
| **B** | Small workflow automation engine | Medium | Weaker vs RPA+LLM | Scope creep |
| **C** | Local developer/research agent | Lower | Crowded (Cursor/Claude Code) | Agent theater |

**Fair note:** B wins if first users are ops/schema people; C wins if the habit is “edit my repo.” No evidence for either yet → defer.

---

## 3. Recommended wedge + rationale

**Recommend A — Local verified research document intelligence.**

nano-lm uniquely owns *classical-first, verification-gated answers over a private corpus*—not another Ollama chat wrapper. E1/E4 showed generative necessity failed on tested scribe regimes; that constrains routing (hybrid/classical-first), it does not forbid building. Wedge A reuses the live `wedge_v1` runtime, maps to researcher pain and MindVault-adjacent vision, and has a crisp kill: if classical+verify cannot earn habit on a real private folder, park before spending on models.

**Not now:** NanoScribe OS, general chatbot, clinical product, reopen E2/E4 as product.

---

## 4. Technical architecture

```text
local folder (md/txt/pdf→text)
    → ingest + doc_id index (+ optional .wedge_manifest.json)
    → task route (metadata | span-fact | retrieve-QA | compare)
    → solver cascade: deterministic → dict/span → BM25 → (optional LM later)
    → typed Claim {task_id, value, evidence[], status}
    → verify (evidence required for PRESENT; else ABSTAIN/DISPUTED)
    → answer JSON / markdown report
```

**Reuse:** `wedge_v1/runtime.py`, `ingest.py`, `classical/{solvers,bm25}.py`, Fabric-style claim invariants.  
**Do not start with:** training, vectors-as-default, agent IDE.

---

## 5. Baseline comparison plan

\[
U = Q - 0.5\,E - 0.3\,R - 0.02\,L - 0.05\,C
\]

| Arm | Stack | When |
|-----|-------|------|
| C0 \(U_{\mathrm{classical}}\) | Classical + verify | **Always first** |
| C1 \(U_{\mathrm{classical+small}}\) | + small local LM on gaps | After classical plateau |
| C2 \(U_{\mathrm{classical+large}}\) | + large/cloud escalate | Owner spend OK |
| C3 \(U_{\mathrm{hybrid+verify}}\) | Best routing + verify-on | Product target |

Admit heavier arm only if \(\Delta U > \delta\) (default 0.05) on the same pack.  
Packs: synthetic ✓ · papers dogfood ✓ · **owner private folder** ← next.

---

## 6. Vertical-slice status (`ask_folder_v0`)

| Step | Work | Status |
|------|------|--------|
| S0 | CLI ask/find/scan/smoke/dogfood | **Done** |
| S1 | `--corpus` + JSON schema | **Done** |
| S2 | Stdlib BM25 in ask | **Done** |
| S3 | PDF text extract (optional pypdf) | **Done** (dep optional) |
| S4 | `ingest` inventory CLI | **Done** |
| S5 | `compare` + contradiction banners | **Done** |
| S6 | Markdown `report ask|find|scan|compare` | **Done** |
| S7 | Owner-corpus dogfood (private; no PHI) | **Next** |
| S8 | Measure \(U_{\mathrm{classical}}\); LM iff ΔU > δ | After S7 |

---

## 7. Kill criteria

| ID | Kill if |
|----|---------|
| K1 Habit | <3 real questions/week after 2 weeks owner use |
| K2 Utility | Classical+verify loses to “grep + human” time proxy |
| K3 Gen-only | Only path to usefulness is unconstrained generation |
| K4 Privacy | Default cloud / PHI in git |
| K5 Differentiation | Users experience “just Ollama RAG” |
| K6 Spend | Substantial cloud spend without owner |

---

## 8. Exact next coding task

```text
TASK: owner-corpus dogfood harness (no PHI in git)

1. wedge_v1/run_owner_dogfood.py (or extend run_dogfood.py):
   - accept --corpus PATH (gitignored / outside repo)
   - fixed question pack JSON (queries + expected status class:
     SUPPORTED | ABSTAIN | CONTRADICTED | any)
   - write results_owner_dogfood.json locally (gitignored)
2. Failure gallery: wrong span / miss / over-abstain markdown snippet
3. README one-liner; do not commit private docs
4. Smoke on synthetic corpus path still passes

OUT OF SCOPE: training, vectors, Evidence Core edits, paid GPU, public claims
```

**Commands after:**  
`python3 -m wedge_v1 report ask --corpus "$OWNER_CORPUS" "…"`  
`python3 -m wedge_v1 ingest --corpus "$OWNER_CORPUS"`

---

## Competitive matrix

| System | Local | Citations | Hard refuse | Classical-first | ΔU gate |
|--------|-------|-----------|-------------|-----------------|---------|
| NotebookLM | ✗ | soft | weak | ✗ | ✗ |
| AnythingLLM | ✓ | soft | weak | ✗ | ✗ |
| PrivateDocs AI | ✓ | page/row | weak | ✗ | ✗ |
| AuditRAG / CITECHECKAI / GroundTruth | ✓/hybrid | chunk/page | model/prompt | ✗ | ✗ |
| DocAgent / CodaCite | ✓ | strongish | varies | partial | ✗ |
| **nano-lm Wedge A** | ✓ | **char spans** | **fail-closed** | **✓** | **✓** |

---

## What we will not do in this sprint

- More governance constitutions / speech-act expansions  
- Program 1 census / NanoScribe OS / reopen E2 or E4 as product  
- Public “product ready” claims from synthetic U  
- Silently rewrite Evidence Core or protected tags  
