# Architecture Evolution — Failure → Design

**Mandate:** `ACTIVE_MANDATE.md`  
**Evidence class:** product/architecture (NONCLAIM for Layer-1)  
**Proving ground:** `wedge_v1/`

> Paper α, E1/E3/E4, and wedge dogfood are **inputs to design**, not reasons to freeze the frontier.

---

## Empirical spine (do not re-litigate; reuse)

| Source | Lesson | Metric anchors |
|--------|--------|----------------|
| Paper α morphology / open fields | Generate-then-normalize fails open slots | open-vocab gap; morph share of C3 misses |
| E1 KILL | Classical/template dominates closed extraction | M1 U≈0.999 > M0≈0.925; δ=0.05 |
| E3 | Exact-match ≠ faithfulness; soft rescue ≈0 | normalize rescued 0/486; no dual IAA |
| E4 KILL | Gen+verify loses on tested R★ | classical U≈0.638 vs gen≈−1.623 |
| Wedge classical | High Q, coverage tax (MISSING/ABSTAIN) | U≈0.891 draft; 48/100 presented w/ evidence |
| Phase 3 E-class | Closed without LM | eclass_acc=1.0; LM not indicated |
| Noisy diagnostic | Ingest normalize recovers most U | raw U≈0.458 → clean ≈0.859 |
| Dogfood | ask() weaker than find_spans fallback | 10/10 accuracy hides cascade debt |

---

## Workstreams (Active Frontier)

| ID | Workstream | Failure modes addressed | Status |
|----|------------|-------------------------|--------|
| **W1** | Retrieval-margin + miss taxonomy | wrong-span, silent miss, over-abstain | **DONE** |
| **W2** | Evidence-atom hard gate | empty-evidence PRESENT | **DONE** |
| **W3** | Corpus-agnostic multi-doc merge | fixture-tied dose/TTL compare | **DONE** |
| **W4** | Pluggable E-class cascade | hardcoded doc ids / OCR literals | **DONE** |
| **W5** | Ingest SLA before intelligence | OCR/layout masquerading as need-LM | **DONE** |
| **W6** | Marginal model value (gated) | only after W1–W5 expose irreducible abstain | **IN PROGRESS** (admission+stub) |

---

## Feedback loop (normative)

```text
dogfood / owner gallery / smoke
        ↓
classify buckets (failure_gallery.py)
        ↓
map → workstream (failure_to_architecture.py)
        ↓
implement architecture delta in wedge_v1/
        ↓
re-measure U / gallery tallies on proving ground
        ↓
keep Δ if product U improves; else revert
```

CLI: `python -m wedge_v1 evolve` (prints recommended next deltas from latest galleries).

---

## Registry + traces

Machine-readable registry: `python -m wedge_v1 arch-registry` → `wedge_v1/arch/registry.py`  
Typed codes: `wedge_v1/arch/failure_codes.py`  
Traces: `payload["trace"]` schema `nano-lm.wedge_v1.ask_trace.v1`  
Adversarial packs: `python -m wedge_v1 adversarial`

## Current delta (this iteration)

**W1+W2:** BM25 margin gating + reject empty-evidence PRESENT in `ask()`.

- `bm25.top_paragraphs` emits `margin` / `rank`
- low-margin hits → `REVIEW` (not PRESENT)
- presentable claims require non-empty evidence atoms
- gallery class: `low_margin_review`


### Delta — failure-driven registry + composition/TTL (architecture lab)

- Structured `AskTrace` on `ask()` with `failure_codes[]` and layer tags
- Composition gate: multi-domain AND → `UNSUPPORTED_COMPOSITION` abstain
- TTL expand patterns generalized beyond fixture `TTL as N seconds`
- Adversarial suite 6/6 on synthetic packs (≠ owner usefulness)
- Bottleneck fixed: rule-brittleness over-abstention on paraphrased TTL


---

## W7 — Native Chain-of-Evidence (from Science One principles)

**Invariant:** evidence created with the claim, never reconstructed after.

Adopted (not cloned): claim–artifact binding, immutable evaluator/run records, independent audit, method–execution alignment.

Rejected for now: autonomous paper writer, literature fleets, parallel discovery swarms.

Implementation: `wedge_v1/coe/` · see `frontier/COE_SLICE_REPORT.md`.


### W3 slice notes

- `wedge_v1/coe/predicates.py` — conjunction → atomic predicates (incl. `open`)
- `wedge_v1/classical/merge.py` — field merge with spans; DISPUTED on conflict
- `ask()` gate emits `COE_INCOMPLETE_CONJUNCTION` when any conjunct unsupported


### W4 slice notes

- `wedge_v1/plugins/` — synonym / OCR / coref modules
- Lexicons: `plugins/data/{synonyms,ocr_substitutions,coref_entities}.json`
- `ask()`/`scan()` call `run_cascade` — coref works after renaming `binding_coref`


### W5 slice notes

- `wedge_v1/ingest_sla.py` — field recovery + optional U recover_gap
- Shared OCR lexicon with W4 plugin; `load_corpus(..., normalize=True)`
- CLI: `python -m wedge_v1 ingest-sla [--with-u]`


### Delta — W3 epistemic merge (2026-07-31)

- `classical/merge.py`: `fields_for_term`, `merge_for_term`, `epistemic_entry`
- `compare()` emits `epistemic_merge[]` with per-doc values + both spans
- `nearby_contradictions()` uses `merge_all` (no fixture TTL regex)
- Reports render **Epistemic merge** section; term-unrelated fields do not false-dispute


### Delta — W4 plugin registry (2026-07-31)

- `plugins/registry.py`: lexicon-driven `should_run` + ordered probes
- `plugins/cascade.py` delegates to registry (no hardcoded TTL keyword list)
- Removed dead `_expand_ttl_ask` duplicate from `runtime.py`
- CLI: `python -m wedge_v1 plugin-registry`


### W6 slice notes

- `wedge_v1/lm/admission.py` — irreducible abstain admission gate (`LM_PROBE_INDICATED` vs `NOT_INDICATED`)
- `wedge_v1/lm/probe.py` — constructive-faithfulness stub backend; allowlist T35/T36/T39 only
- `wedge_v1/lm/marginal.py` — classical vs +stub ΔU diagnostic; `NOT_APPLICABLE` when E-class closed
- CLI: `python -m wedge_v1 lm-admit` · `python -m wedge_v1 lm-probe`
- Execute auth unchanged: `AUTHORIZE_WEDGE_V1_PHASE3_LM_PROBE` (no external LM wired)
- Clean synthetic verdict today: **NOT_INDICATED** / **NOT_APPLICABLE**


### Delta — W5 ingest SLA (2026-07-31)

- `load_corpus(normalize="auto")` applies OCR table only when corruption detected
- Runtime default: auto-normalize at ingest (preprocessing, not intelligence)
- `ingest-sla` pins field recovery ≥90%; recover_gap U ≤0.05 on noisy track
- Adversarial: `ADV_NOISY_OCR_RECOVER` (7-case suite)
