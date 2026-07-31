# Evidence Ledger — Proposed Strengthened Form

*Proposal only until owner approves DIFF E / replacement of `papers/EVIDENCE_LEDGER.md`.
Status vocabulary: PROVEN | SUPPORTED | PLAUSIBLE | SPECULATION | FALSIFIED | VOID | UNRESOLVED.*
*Do not use PROVEN lightly. Prefer SUPPORTED + REPRODUCIBILITY_LIMITATION when primary artifacts are untracked or raw absent from durable store.*
*Last reviewed commit baseline: `0e01d73205e9c35ea32925fd4d6c7e5fceb61137` (+ working-tree freeze overlays).*

| Claim ID | Exact wording | Status | Scope | Protocol | Result artifact | Raw-evidence manifest | Recompute code | Limitations | Withdrawal condition | Last reviewed |
|----------|---------------|--------|-------|----------|-----------------|----------------------|----------------|-------------|----------------------|---------------|
| C_GAP_EXISTS | Held-out exact-value copying failures exist (small LMs, this task) | SUPPORTED | synthetic scribe; exact match | Stage T / anchors PREREGs | `results_anchors_v2_*.json`, `results_arm1_v2_*.json` | N/A (scores in JSON) | rescore scripts | construct ≠ clinic | construct collapse / protocol VOID | freeze HEAD |
| C_FIELD_LOC | Open-vocab fields fail; closed-value fields ≈0 gap | SUPPORTED | same instrument | fieldwise | `results_fieldwise_*.json` | — | — | template-vs-value control only | closed-field nonzero under same protocol | freeze HEAD |
| C_DIVERSITY | Slot training diversity causally lifts held-type recall (+66.7 D5→D80) | SUPPORTED | 10M ALG slot sweep | `PREREG_slot_diversity` | `results_sweep_10m.json` (TRACKED) | `artifacts/slot_diversity/MANIFEST.json` | sweep kernel | behavioral; not circuit ID | protocol VOID | freeze HEAD |
| C_SCALE_FLAT | Parameters alone do not buy copying under unequal token budgets | SUPPORTED (rewritten) | own-stack nano/scale/160M | ownstack factorial | `results_ownstack_v2_*.json` | `artifacts/ownstack_corner/` | — | **Not** a clean parameter-only 50× law; nano=32.8M tok vs scale/160M=200M | equal-token re-run flips | freeze HEAD |
| C_ADAPT_DATA | Adaptation×data interaction (LoRA / Chinchilla / corner) | SUPPORTED | own-stack 160M | PREREG_ownstack | factorial + corner JSONs | same | — | mechanism unidentified (E2 gated) | — | freeze HEAD |
| C_INTERFERENCE | Lexical interference drives residual floor | FALSIFIED | C-1b | PREREG_token_coverage / interference | `results_interference_10m.json` | `artifacts/c1b/` (**primary JSONL TRACKED**) | analyze path | do not revive as leading candidate | new prereg | freeze HEAD |
| C_C3_TB | Transition/boundary binding factors drive residual | FALSIFIED | C-3 primary+replication | PREREG_C3 | `results_c3_10m.json` + replication | `artifacts/c3_primary/` (JSONL TRACKED); `artifacts/c3_replication/` (JSONL IGNORED_LOCAL + local archive) | `trajectory/recompute_c3.py` | morphology exploratory only | new prereg | freeze HEAD |
| C_C3_L | Length factor drives residual | UNRESOLVED | C-3 | same | same | same | same | underpowered / unresolved | decisive length arm | freeze HEAD |
| C_MORPH | Morphology is the causal residual mechanism | SPECULATION | descriptive census only | none causal | C-3 notes / post-hoc | — | — | exploratory; not prereg causal | causal prereg + result | freeze HEAD |
| C_POINTER_P1 | Explicit pointer/copy head closes OOD gap | VOID | Stage P1 | PREREG_pointer_head | `scribe/pointer/result_pointer.json` | `artifacts/pointer_p1/` | — | failed manipulation check | — | freeze HEAD |
| C_POINTER_P2 | Copy-supervised pointer head closes OOD gap (this impl) | FALSIFIED | Stage P2 | PREREG_pointer_head_v2 | `scribe/pointer/result_pointer2.json` | `artifacts/pointer_p2/` | — | this implementation only | new impl + prereg | freeze HEAD |
| C_E1_KILL | Best non-generative (M1) dominates official generative M0 under frozen U | FALSIFIED (generative-substrate product thesis for this regime) / SUPPORTED (KILL measurement) | old-task U + synthetic world | PREREG_E1 | `results_e1_utility.json` (**UNTRACKED_LOCAL**) | `artifacts/e1/` | `trajectory/e1/common.py` | M1 generator-aligned; U-scoped; M2 within δ of M0 not above | new U / new problem | freeze HEAD |
| C_E3_NORM | Normalize-then-match rescues M0 exact failures | FALSIFIED (0/486) | M0 exact-fail pack | PREREG_E3 | `results_e3_normalize_construct.json` (UNTRACKED) | `artifacts/e3/` | `trajectory/e3/run_e3_normalize.py` | thin normalize ≠ synonyms | — | freeze HEAD |
| C_E3_RUBRIC | Sampled exact errors are acceptable semantic equivalents | FALSIFIED on agent-rubric pack (0/100) | n=100 | PREREG_E3 Stage 1 | `results_e3_human.json` (UNTRACKED; filename historical) | `artifacts/e3/` | — | **Not** clinician/human eval; IAA + synonym ontology open | dual-clinician flip | freeze HEAD |
| C_FABRIC_SLICE | Propose→verify→abstain can drive presented-error→0 under decidable R on this task | SUPPORTED (scoped) | fabric grounding.v2 / inst0 | fabric slice | `fabric/results_slice_v1.json` | fabric ledgers IGNORED_LOCAL | `fabric/test_fabric.py` | v2 rules-perfect; not OS; ledger≠append-only DB | open-world claim | freeze HEAD |
| C_NANOSCRIBE_IMPL | NanoScribe architecture is implemented | FALSIFIED / VOID as implementation claim | product/architecture | — | docs only | — | — | Fabric ≠ NanoScribe | evidenced modules | freeze HEAD |
| C_LORA_GEOM | LoRA preserves copy-circuit geometry | SPECULATION | — | E2 GATED/STOP | **none** (`results_e2_*` MISSING) | PARTIAL residue only | — | no RESULT | E2 ID under re-scope | freeze HEAD |
| C_RSTAR_VALUE | Generative proposers add value in regime R★ | PLAUSIBLE (untested) | future E4 | PREREG_E4 (design only) | none | — | — | no builder/data; do not execute without auth | E4 measurement | freeze HEAD |
| C_ZERO_HALLUC_OPEN | Zero hallucination in the open world | SPECULATION / forbidden | open world | — | — | — | — | only “zero accepted violations of R” if ever | — | freeze HEAD |

## Required public wording for E3

> A bounded agent-applied rubric audit of 100 sampled errors found zero acceptable
> semantic equivalents. This does not substitute for independent clinician annotation,
> inter-rater agreement, or validation of a synonym ontology.

> Agent-rubric audit completed; independent dual-clinician validation, IAA, and
> synonym-equivalence validation remain open.
