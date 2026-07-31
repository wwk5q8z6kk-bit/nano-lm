# Evidence ledger

*Single source of truth for claim strength. Docs/evidence layer only.
Aligned with `trajectory/DECISION_P1_program_lock.md` and Paper α remediation.
Machine-readable: `papers/EVIDENCE_LEDGER.json` + `papers/EVIDENCE_MANIFEST.json`.
Amended by scientific remediation 2026-07-31. Last-reviewed commit: `0e01d73205e9c35ea32925fd4d6c7e5fceb61137`.*

**Statuses**

| Status | Meaning |
|--------|---------|
| **PROVEN** | Measured under pre-registered (or locked) protocol; direction stable; would take construct collapse or protocol VOID to withdraw |
| **SUPPORTED** | Strong evidence on this instrument; alternative explanations remain or scope is narrow/descriptive |
| **PLAUSIBLE** | Consistent with data; not uniquely identified; do not build roadmap on it alone |
| **SPECULATION** | Hypothesis / product hope / mechanism story — **not** evidence |
| **FALSIFIED** | Kill-gate or decisive negative; do not revive without new U/problem |
| **VOID** | Protocol invalidated |
| **UNRESOLVED** | Measured but inconclusive |

## Ledger

| Claim ID | Claim | Status | Construct | Protocol | Result artifact | Scope / limitations |
|---|---|---|---|---|---|---|
| C_GAP_EXISTS | Held-out exact-value copying failures exist (small LMs, this task) | **PROVEN** | exact-match held-vs-seen gap | Paper α methods / Stage T | `trajectory/results_* anchors + ladder` | synthetic scribe task; exact≠clinician equivalence |
| C_FIELD_LOCAL | Open-vocabulary fields fail; closed-value fields ≈0 gap | **PROVEN** | fieldwise gap | fieldwise anchors | `anchors/fieldwise JSONs` | this task; task-specific |
| C_DIVERSITY | Slot training diversity causally changes held-type recall (+66.7 D5→D80) | **PROVEN** | held-type recall | PREREG_slot_diversity | `trajectory/results_sweep_10m.json` | allergy slot at scale-10M; one slot |
| C_SCALE_DESCRIPTIVE | Across evaluated own-stack full-FT configs, larger parameter count was not associated with a monotonic reduction in the diluted gap | **SUPPORTED** | diluted gap | PREREG_ownstack_160m + audits | `results_ownstack_v2_*; pretrain/AUDIT.md; scale/AUDIT.md` | descriptive; unequal token budgets (32.8M / ~200M / 3.2B); not parameter-only causal law |
| C_ADAPT_DATA | Adaptation×data interaction (LoRA / Chinchilla data / both) | **SUPPORTED** | diluted gap factorial | ownstack factorial | `results_ownstack_v2_*; results_corner_*` | behavioral; mechanism unidentified; E2 STOP |
| C_E1_M1_KILL | Under frozen E1 U, M1 exceeds best evaluated generative reference (official M0) | **PROVEN** | utility U | PREREG_E1 | `trajectory/results_e1_utility.json` | frozen U; closed-world task; M2 does not dominate M0; new U may re-rank |
| C_E3_AUTO | Normalize-then-match does not rescue M0 exact failures (0/486) | **PROVEN** | normalize-then-match | PREREG_E3 auto | `results_e3_normalize_construct.json` | M0 error mass; one normalize rule |
| C_E3_RUBRIC | Agent-applied rubric audit: 0/100 acceptable semantic equivalents on sampled exact errors | **SUPPORTED** | agent rubric EXACT_SURVIVES | PREREG_E3 Stage1 | `results_e3_human.json` | instrument=agent-rubric-pass-1; not dual-clinician; no IAA |
| C_C1B | C-1b lexical interference REFUTED | **FALSIFIED** | interference hypothesis | C-1b prereg | `results_interference_10m.json` | tested form; raw JSONL local-archived/gitignored |
| C_C3 | C-3 transition/boundary REFUTED; length UNRESOLVED | **FALSIFIED** | mechanistic factors | C-3 | `results_c3_10m.json` | tested factors; L unresolved; raw JSONL local archive |
| C_FABRIC | Fabric demonstrates scoped verification slice under decidable R | **SUPPORTED** | propose→verify→abstain | fabric slice v1 | `fabric/results_slice_v1.json` | synthetic decidable R; not OS; not append-only DB; not open-world |
| C_NANOSCRIBE_SUP | NanoScribe / fabric is superior architecture for this task | **FALSIFIED** | product thesis | E1 KILL | `results_e1_utility.json` | old-task regime; R★ untested |
| C_LORA_GEOM | LoRA preserves copy-circuit geometry | **SPECULATION** | mechanism | PREREG_E2 GATED/STOP | `none (results_e2_* missing)` | banned in prose; no RESULT |
| C_RSTAR | Generative proposers add value in regime R★ | **PLAUSIBLE** | R★ utility | PREREG_E4 (unexecuted) | `none` | untested; E4 not authorized |
| C_ZERO_HALLUC | Zero hallucination (open-world) | **SPECULATION** | open-world | — | `—` | forbidden; only zero accepted violations of R if ever |
| C_CLINICAL | Clinical deployment ready | **SPECULATION** | deployment | — | `—` | forbidden; synthetic only |

## Full row schema

Each row in `EVIDENCE_LEDGER.json` also carries: unit_of_inference, raw_artifact_manifest, recompute_script, withdrawal_condition, last_reviewed_commit.

## How to use

1. Public science may state **PROVEN** and carefully hedged **SUPPORTED** rows.
2. Product/architecture may not treat **SPECULATION** or **FALSIFIED** theses as open.
3. Parameter-only scale claims are **not** Proven; use `C_SCALE_DESCRIPTIVE`.
4. E1 KILL is **M1-specific** under frozen U.
5. E3 exactness is Supported/Proven on automated + agent-rubric instruments — not clinically validated.
