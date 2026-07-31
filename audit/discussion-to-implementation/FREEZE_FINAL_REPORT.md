# Evidence Reconciliation Final Report

## 6. Decision gate

# IDLE_AFTER_FREEZE

(Prior gate was AUDIT_REMEDIATION_REQUIRED; cleared after F0–F3 pass + authorized claim sync.)

Owner must approve lockfile/Paper α diffs (incl. nano **32.8M** token correction),
commit untracked primary E1/E3 artifacts + `artifacts/` manifests, and authorize
durable publication of ignored replication/fabric JSONL before tag
`post-alpha-evidence-freeze-2026-07-31` is meaningful.
After that: **IDLE_AFTER_FREEZE** (or separately **AUTHORIZE_E4_DESIGN_ONLY**).
Do **not** execute E4.

---

## 1. Applied factual corrections (ordinary / verified)

| File | Semantic correction |
|------|---------------------|
| `trajectory/FINDINGS.md` | E3 agent-rubric wording (not human); E2 GATED/STOP after external checks; E1 **M1 dominates** official M0; M2 within δ of M0; ρ = review load; falsifies old regime thesis only |
| `trajectory/results_e1_utility.json` + `results_e1_utility_sensitivity.json` | Conclusion prose: “hallucination weight” → “review-load (ρ) weight”; **U numerics/weights unchanged** |
| `fabric/README.md` | Fabric ≠ OS; ledger = per-run rewritten JSONL; Intent→Control not implemented; expansion STOP |
| `trajectory/PREREG_slot_diversity.md` | EXECUTED + RESULT summary without amending decision rule |
| `trajectory/PREREG_ownstack_160m.md` | Corrected “Not executed” header; RESULT pointer; original vs addenda distinguished |
| `papers/CLAIM_GLOSSARY.md` | Forbid “E3 human evaluation complete”; hedge updated for agent-rubric / IAA open |
| `audit/.../ARCHIVAL_ARTIFACT_INVENTORY.md` + `artifacts/*` | Regenerated inventory/manifests; primary C-1b/C-3 JSONL correctly **TRACKED**; replication JSONL IGNORED_LOCAL + archived |
| `audit/.../METRIC_DEFINITION_CROSSWALK.md` | ρ conflict classified; sensitivity label fix noted |
| `audit/.../EVIDENCE_LEDGER_PROPOSED.md` | Strengthened ledger form (proposal) |
| `audit/.../POST_ALPHA_EVIDENCE_FREEZE.md/.json` + `SHA256SUMS` | Freeze package |

**Applied under 2026-07-31 freeze ruling:** `EMPIRICAL_FOUNDATION.md`, `RESEARCH_PROGRAM.md`, Paper α LaTeX/draft, `REGIME_P1`, `EVIDENCE_LEDGER.md`, NanoScribe banners — see `OWNER_APPROVAL_REQUIRED_DIFFS.md`.

**Already correct before this pass (re-verified):** `PREREG_E2` GATED/STOP; `DECISION_P1` ρ = review load matching `e1/common.py`; no active pods.

---

## 2. Owner approval required

Exact pending diffs: **`audit/discussion-to-implementation/OWNER_APPROVAL_REQUIRED_DIFFS.md`**

Critical pending scientific packaging fixes (do not silent-apply):
- **DIFF H / H′:** Paper α methods — nano pretrain **32.8M** not ~200M; narrow 50× flatness (unequal token budgets).
- **DIFF D / J:** E3 limitation — agent-rubric wording (never “human evaluation”).
- **DIFF I:** E1 public wording — M1 dominates; M2 within δ; does not prove no R★.
- **DIFF A/B/C/C′/E/F:** EMPIRICAL_FOUNDATION, RESEARCH_PROGRAM, REGIME, ledger, NanoScribe banner.

Also proposed: `EVIDENCE_LEDGER_PROPOSED.md`.

---

## 3. Reproducibility status (closed experiments)

| Experiment | Status |
|------------|--------|
| Slot diversity | Result TRACKED; prereg RESULT updated; no raw JSONL expected |
| Stage T / Pythia | Result JSONs TRACKED |
| Own-stack / corner | Result JSONs TRACKED; checkpoints IGNORED_LOCAL |
| C-1b | Result + **raw JSONL TRACKED**; archive copy present |
| C-3 primary | Result + **raw JSONL TRACKED**; recompute verified T/B REFUTED, L UNRESOLVED |
| C-3 replication | Result TRACKED; raw JSONL **IGNORED_LOCAL** + local archive SHA-verified |
| E1 | Result JSON **UNTRACKED_LOCAL**; items JSON present; no JSONL expected |
| E3 | Normalize + agent-rubric JSON **UNTRACKED_LOCAL** |
| Pointer P1/P2 | TRACKED under `scribe/pointer/` (VOID / REFUTED) |
| E2 | **REFERENCED_BUT_MISSING** RESULT; GATED/STOP |

---

## 4. Architecture truth

**Fabric** = measured verification vertical slice: typed claims, source-grounding rules,
contradiction states, abstention, content-addressed IDs, per-run JSONL serialization
(open mode rewrite), 8 unit tests + measured presented-error→0 under rules-strong v2.
**Not** an append-only transactional DB or cognitive OS.

**NanoScribe** = architectural research program. Beyond fabric slice, control plane,
memory, routing, tools, permissions, distributed execution, and UI remain
**unimplemented** unless separately evidenced.

Fabric vertical slice ≠ NanoScribe architecture.

**E1** falsifies generative-substrate product thesis for the frozen old-task utility
and synthetic world (best non-generative **M1** dominates); it does not prove no
generative-value regime R★ exists.

---

## 5. Repository state

| Item | Value |
|------|-------|
| HEAD | `0e01d73205e9c35ea32925fd4d6c7e5fceb61137` |
| origin | `master` matches origin (fetched) |
| Tests | **15 passed** (`pytest`); C3 recompute OK |
| Active pods | **none** (`runpodctl pod list` → `[]`) |
| Working tree | **Dirty** — freeze overlays + many untracked research files |
| Ignored evidence | Replication C-3 JSONL + fabric ledgers IGNORED_LOCAL; archived under `artifacts/local_raw_archive/` |
| Inventory counts | `{"IGNORED_LOCAL": 31, "PARTIAL": 6, "REFERENCED_BUT_MISSING": 1, "TRACKED": 50, "UNTRACKED_LOCAL": 44}` |
| Premature public tag | `post-alpha-evidence-freeze-2026-07-31` → `a9d12cb` — **EXISTS; preserve; do not recreate** |
| Git commit | **not performed** (user did not ask) |

### Proposed commit contents (when authorized)

- `artifacts/` (inventory, manifests, local_raw_archive checksums, freeze JSON)
- `audit/discussion-to-implementation/` freeze/audit package
- E1/E3 result JSON + harness (`trajectory/e1/`, `trajectory/e3/`, `results_e1_*`, `results_e3_*`)
- Ordinary doc corrections listed in §1
- **After approval:** owner lockfile/Paper α diffs from `OWNER_APPROVAL_REQUIRED_DIFFS.md`
- Do **not** recreate `post-alpha-evidence-freeze-2026-07-31`; optional **new distinct** reconciled tag only with authorize_tag

### E2 confirmation (P1–P3)

| Check | Result |
|-------|--------|
| Complete U3 artifact outside repo | **Not found** |
| Hidden external result retrieved | **No** |
| Active pod | **No** |
| Partial residue labeled incomplete | **Yes** |

---

## Branch status snapshot

```
## master...origin/master
 M .gitignore
 M CLAUDE-PROGRESS.md
 M README.md
 M fabric/README.md
 M papers/ASSESSMENT_2026-07-20.md
 M papers/EMPIRICAL_FOUNDATION.md
 M papers/MASTER_PLAN.md
 M papers/NANOSCRIBE_VNEXT.md
 M papers/README.md
 M papers/RESEARCH_PROGRAM.md
 M papers/paper2_draft.md
 M papers/writing_audit.md
 M trajectory/FINDINGS.md
 M trajectory/PREREG_ownstack_160m.md
 M trajectory/PREREG_slot_diversity.md
 M trajectory/PREREG_stageM.md
 M trajectory/REPRODUCIBILITY.md
 M trajectory/kaggle_sweep_10m.py
?? .DS_Store
?? .autonomous/
?? .cursor/
?? .github/
?? AGENTS.md
?? LICENSE
?? artifacts/
?? audit/
?? environment.yml
?? papers/AZ_EXECUTION_PLAN.md
?? papers/CLAIM_GLOSSARY.md
?? papers/EVIDENCE_LEDGER.md
?? papers/EVIDENCE_MANIFEST.json
?? papers/SEQUENTIAL_PIPELINE.md
?? papers/latex/paper1.tex.bak-pre-B
?? pyproject.toml
?? requirements-ml.txt
?? requirements.txt
?? stage_m/_probe.py
?? trajectory/DECISION_P1_program_lock.md
?? trajectory/PIPELINE_GATE_LOG.md
?? trajectory/PREREG_E1_nonlm_baseline.md
?? trajectory/PREREG_E2_lora_universes.md
?? trajectory/PREREG_E3_faithfulness_construct.md
?? trajectory/PREREG_E4_Rstar_killgate.md
?? trajectory/REGIME_P1_where_classical_fails.md
?? trajectory/STAGE1_E3_CONSTRUCT_FIRST_PRINCIPLES.md
?? trajectory/SWEEP_10M_PROVENANCE_NOTE.md
?? trajectory/e1/
?? trajectory/e1_official_m0.log
?? trajectory/e1_official_m0_ownstack.log
?? trajectory/e2/
?? trajectory/e3/
?? trajectory/e3_human_rating_pack.json
?? trajectory/results_e1_construct_validity_note.json
?? trajectory/results_e1_items_M0_ownstack_chinchilla_lora_voff.json
?? trajectory/results_e1_items_M0_ownstack_chinchilla_lora_von.json
?? trajectory/results_e1_items_M0_pythia160m_lora_voff.json
?? trajectory/results_e1_items_M0_pythia160m_lora_von.json
?? trajectory/results_e1_items_M0_scale_voff.json
?? trajectory/results_e1_items_M0_scale_von.json
?? trajectory/results_e1_items_M1_template_voff.json
?? trajectory/results_e1_items_M1_template_von.json
?? trajectory/results_e1_items_M2_dict_span_voff.json
?? trajectory/results_e1_items_M2_dict_span_von.json
?? trajectory/results_e1_items_M3_crf_lite_voff.json
?? trajectory/results_e1_items_M3_crf_lite_von.json
?? trajectory/results_e1_items_M4_constrained_voff.json
?? trajectory/results_e1_items_M4_constrained_von.json
?? trajectory/results_e1_items_M5_span_clf_voff.json
?? trajectory/results_e1_items_M5_span_clf_von.json
?? trajectory/results_e1_nonlm_M0_ownstack_chinchilla_lora_voff.json
?? trajectory/results_e1_nonlm_M0_ownstack_chinchilla_lora_von.json
?? trajectory/results_e1_nonlm_M0_pythia160m_lora_voff.json
?? trajectory/results_e1_nonlm_M0_pythia160m_lora_von.json
?? trajectory/results_e1_nonlm_M0_scale_voff.json
?? trajectory/results_e1_nonlm_M0_scale_von.json
?? trajectory/results_e1_nonlm_M1_template_voff.json
?? trajectory/results_e1_nonlm_M1_template_von.json
?? trajectory/results_e1_nonlm_M2_dict_span_voff.json
?? trajectory/results_e1_nonlm_M2_dict_span_von.json
?? trajectory/results_e1_nonlm_M3_crf_lite_voff.json
?? trajectory/results_e1_nonlm_M3_crf_lite_von.json
?? trajectory/results_e1_nonlm_M4_constrained_voff.json
?? trajectory/results_e1_nonlm_M4_constrained_von.json
?? trajectory/results_e1_nonlm_M5_span_clf_voff.json
?? trajectory/results_e1_nonlm_M5_span_clf_von.json
?? trajectory/results_e1_utility.json
?? trajectory/results_e1_utility_sensitivity.json
?? trajectory/results_e3_human.json
?? trajectory/results_e3_normalize_construct.json
?? trajectory/results_sweep_10m.schema.md
?? trajectory/runpod_partial/
?? trajectory/tokenizer.json

```


## Evidence classes (discovery discipline)

Labels from `REVISED_DISCOVERY_PASS_ACCEPTANCE.md` (retrievability, not confidence):

| Class | Definition | Qualifies | Example | Does **not** imply |
|-------|------------|-----------|---------|---------------------|
| PUBLIC-ANCHORED | Committed/tagged and publicly retrievable | Tagged commit contains the artifact | E1/E3 primaries at `post-alpha-evidence-freeze-2026-07-31` → `a9d12cb` | Later local overlays are public; clinical readiness |
| LOCAL-DOCUMENTARY | Exists locally / in working tree or conversation archive but is not the public archival story for that claim | Untracked or dirty packaging notes | Session `READY_FOR_OWNER.md`, some freeze packaging churn | Public freeze completeness |
| RAW-UNINSPECTED | Bytes exist but have not been human-audited as a claim basis | Large raw dumps pending review | Some local raw archives before inventory | Scientific endorsement |
| ASPIRATIONAL | Protocol/design/product text without a measured RESULT | Roadmaps, ambition, NanoScribe visions | `TECHNOLOGY_ROADMAP.md` NanoScribe modules | Implementation or evidence |
| DECISION-GOVERNANCE | Program lock / kill / product-thesis scoping | Decision records, utility kill rules | E1 product thesis KILL row | New regime unlocks |
| STALE-CONTRADICTORY | Text that conflicts with measured/public state and must not be reused as current status | Old “E2 running”, “E4 untested” as present tense | Pre-remediation narrative residues | Current program authorization |

