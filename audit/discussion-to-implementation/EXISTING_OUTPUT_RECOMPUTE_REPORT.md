# Existing Output Recompute Report (F0–F2)

**Date:** 2026-07-31  
**Mode:** recompute from frozen JSON only — **no model runs**  
**Pre-edit snapshot:** `audit/discussion-to-implementation/PRE_EDIT_SNAPSHOT.md`  
**Branches:** `wip/pre-freeze-snapshot-20260731` @ `f96705e`; dirty WIP `wip/pre-freeze-snapshot-20260731-dirty` @ `6fc9b95`

## Gate verdict

| Gate | Result |
|------|--------|
| F0 Artifact existence | **PASS** (load-bearing paths resolve; pseudo-refs in manifest prose cleaned in claim sync) |
| F1 Status normalization | **PASS** — `CANONICAL_STATUS_TABLE.md` + `.json` |
| F2 Existing-output recomputation | **PASS** |

Overall pre-commit: **F0–F2 PASS** → claim sync (F3) authorized.

---

## F2 checks

| Check | Pass/Fail | Detail |
|-------|-----------|--------|
| Recompute every row \(U\) from \(P,M,\rho,L,C\) | **PASS** | 14/14 rows; \(U=\alpha P-\beta M-\gamma\rho-\lambda L-\kappa C\) with defaults \(1,0.5,0.3,0.02,0.05\) |
| M1 dominates official M0 | **PASS** | M1 `0.9989993963311425` > M0_pythia160m_lora `0.9252173639550433` (margin `0.073782`) |
| M2 within \(\delta=0.05\) of official M0 | **PASS** | M2 `0.8862985693320633`; M0−M2=`0.038919` ≤ 0.05 (M2 below M0 but within non-necessity margin) |
| Decision verdict KILL | **PASS** | `results_e1_utility.json` decision.verdict=KILL; sensitivity_flip=false |
| Sensitivity from frozen table | **PASS** | Embedded OAT cells recompute; `kill_stable_across_all_OAT_cells=true` |
| nonlm summaries vs utility rows | **PASS** | All methods P/M/ρ/L/C/U match |
| Item-level recall vs utility (spot) | **PASS** | M1/M2/official M0 item field-recall matches util M; M0_scale item n_fields=4995 (parse drop) while util uses nonlm aggregate — nonlm↔util still exact |
| E3 normalize rescues | **PASS** | **0/486** on M0_scale\|voff (`norm_rescue_count=0`, `both_fail=486`) |
| E3 pack size | **PASS** | 100 items |
| E3 agent labels | **PASS** | 0/100 faithful; rater `agent-rubric-pass-1`; verdict EXACT_SURVIVES |
| E3 pack_sha256_items | **PASS** | Matches `sha256(json.dumps(items, sort_keys=True))` |
| Pretrain token budget | **PASS** | `pretrain/AUDIT.md`: 4000 steps / **32.8M** tokens; Paper α md+tex already corrected |
| artifacts/SHA256SUMS (content files) | **PASS** | Result JSON digests match; e1/e3 MANIFEST digests refreshed this freeze |

## Precise E1 wording (locked by numbers)

Under frozen old-task \(U\) and \(\delta=0.05\):

- Best non-generative baseline is **M1** (hand-template/rules): \(U\approx 0.999\).
- Best official generative reference is **M0 Pythia-160M LoRA**: \(U\approx 0.925\).
- **M1 strictly dominates** official M0 → **KILL** (generative substrate not preferred for this closed task).
- **M2** (train-dict+span) has \(U\approx 0.886\), below official M0, but **within \(\delta\)** (M0−M2≈0.039≤0.05), so it also satisfies the pre-registered non-necessity margin relative to official M0.
- Do **not** say plural “baselines dominate” without naming M1 as the strict winner.

## REPRODUCIBILITY_LIMITATIONs (non-blocking for F2)

1. Official M0 **adapter weight binaries** / CUDA bit-identical retrain not in archive — U recomputes from frozen components.
2. Raw RunPod logs optional; structured L/C in `trajectory/results_e1_runtime_components.json`.
3. Large Paper-α `*.jsonl` remain gitignored; local hashed archive under `artifacts/local_raw_archive/` with retrieval instructions.
4. E3 dual-clinician IAA **NOT_RUN** — agent-rubric only.
5. Working-tree / lockfile prose is not “immutable” until tagged; this freeze tag archives state, does not retroactively prove preregistration chronology.

## Decision gate (this report)

`IDLE_AFTER_FREEZE` eligible after F3–F5. **Never** `AUTHORIZE_E4_EXECUTE`.
