# Active Frontier — Development Plan

**Branch:** `frontier/active-v1`  
**Mandate:** `frontier/ACTIVE_MANDATE.md`  
**Architecture map:** `frontier/ARCHITECTURE_EVOLUTION.md`  
**Proving ground:** `wedge_v1/` (Wedge A)  
**Rule:** Ship architectural value from failures; never silently rewrite Evidence Core.

---

## Phase map

| Phase | Goal | Status |
|-------|------|--------|
| **P0** Discovery + mandate | Choose wedge A; kill criteria | **DONE** |
| **P1** Verified Ask vertical slice | ask/find/scan/compare/report/ingest | **DONE** |
| **P1b** Architecture feedback loop | gallery → workstream → delta | **ACTIVE** |
| **P2** Owner-corpus loop | Private folder eval + failure gallery | **DONE** (harness; real OWNER_CORPUS pending) |
| **P3** Contact / usefulness / habit | Review labels + habit session + fine gallery | **ACTIVE** |
| **P4** Ingest SLA | OCR/PDF normalize measured recover_gap | **DONE** |
| **P5** Marginal model value | Classical vs +small vs +large under U | **IN PROGRESS** (W6 admission+stub) |
| **P6** Distribution | Minimal UI / install | After retention signal |

---

## Now (architecture, not polish)

Failure-driven registry live: `arch-registry` · traces on `ask` · `adversarial` suite.
`LM_PROBE = NOT_INDICATED` on clean synthetic (E-class closed). W6 admission+stub harness live; real `$OWNER_CORPUS` over_abstention still external-pending.


P3 contact loop: `review`, `habit` session, fine `gallery`, `owner-ready`.
Gate 0 runner: `scripts/gate0_contact.sh` (set `WEDGE_OWNER_CORPUS` first).

**Session 2026-07-31:** Harness green on gitignored `wedge_v1/data/owner_corpus` (6 synthetic stand-ins).
Dogfood 5/5 · draft U≈0.890 · adversarial 6/6 · evolve → **W3**.
**Blocker:** real private folder (≥10 docs) + owner usefulness sentence + review labels.

**Next engineering:** real `$OWNER_CORPUS` contact (only remaining product gate).

**Session 2026-08-22 (eval-arms):** Fixture `eval-arms` CLI — U_classical vs hybrid-stub under ΔU; citation packing in `report`. Clean demo → `KEEP_CLASSICAL`.

**Session 2026-08-22:** `wedge_v1 study check|run` lite path + `questions-v1.json` (10 scoped fixture tasks). Demo study 10/10.

**Session 2026-08-21:** OOS abstain cleans BM25-noise codes; `ask()` emits `epistemic_merge`; evolve no longer re-queues W3 on handled contradictions (recommends `OWNER_CORPUS_CONTACT`).


**Session 2026-08-22 (autonomous):** Continued without private corpus.
- Closed p5 OVER_ABSTENTION class in `ask()`: phrase_span, short compound tokens, majority gate, numeric+keyword windows.
- Guard: long snake_case IDs stay atomic (no false support).
- Stand-in pack `owner_dogfood_tasks_standin.json` O01–O09 green on `data/owner_corpus`.
- Fixture Gate0 script: `scripts/gate0_fixture.sh` (class OWNER_FIXTURE; not private usefulness).
- Remeasure: `python -m wedge_v1 remeasure-oa` → 3/3 on p5 snapshot.
**Still blocked for product gate:** real `$OWNER_CORPUS` / `$WEDGE_OWNER_CORPUS` + owner review labels via `./scripts/gate0_contact.sh`.

**CLI:** `python -m wedge_v1 status` · `lm-admit` · `./scripts/gate0_contact.sh`

1. **W1** BM25 retrieval-margin gating in `ask()` — **DONE**
2. **W2** Evidence-atom hard gate — **DONE**
3. **W3** Multi-doc epistemic merge — **DONE** (compare + ask banners)
4. `python -m wedge_v1 evolve` — recommend next workstream from galleries
5. Re-run `./scripts/gate0_contact.sh` after owner corpus swap

## Next

5. ~~General multi-doc epistemic merge (typed fields, both spans)~~ **DONE** (`field_registry.json` + QPS/error_rate; both spans)
6. ~~Pluggable synonym/OCR/coref modules~~ **DONE** (`plugin-registry`)
7. Real `$OWNER_CORPUS` gallery → feed W1–W4

## Later

8. **W6** admission + stub probe (`lm-admit` / `lm-probe`); external LM only after INDICATED + auth
9. Dense retrieve if BM25 margin saturates

## Never

- NanoScribe OS · OLD_TASK_U gen default · Evidence Core edits · unpaid→paid GPU without owner · governance-doc churn as substitute for architecture


**Session 2026-08-22 (continue autonomous):** Restored `frontier/active-v1` after branch drift. Landed (a) OVER_ABSTENTION relevance/phrase-locate pins, (b) W3 field registry + QPS both-spans merge, (c) W6 `MLXLlamaBackend` span-binding seam via `get_backend`. Smoke/dogfood/adversarial green. LM still `NOT_INDICATED` on clean synthetic.

