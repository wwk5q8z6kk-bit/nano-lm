# Council Hybrid Closeout

*Manual execution of Council-of-Five hybrid (2026-07-31). Autonomous path blocked (Claude credit balance too low).*
*Generated 2026-07-31T18:00Z*

## One-sentence claim (Translator)

> Under the frozen E1 utility, classical/rules methods beat official LoRA-160M on this closed scribe task (**KILL**; M1 U≈0.999 vs official M0 U≈0.925, δ=0.05).

## Verify card (Auditor)

| Check | Result |
|---|---|
| HEAD | `2e03e0df564008cf51c4309e9dbdf01a59c3c7b5` |
| `pytest -q` | PASS (`...........................                                              [100%]`; exit 0) |
| E1 verdict | **KILL** |
| Official M0 | `M0_pythia160m_lora` U=0.9252173639550433 |
| Best non-LM | `M1_template` U=0.9989993963311425 |
| margin / delta | 0.07378203237609926 / 0.05 |
| venue | `runpod-cuda` |
| `results_e1_utility.json` SHA256 | `a5117d2cad25ca53df7d7f3cdb25c563b36cab5cf1ac63be3867b53db76b760f` |
| Active RunPod pods | none expected under freeze |

## Quarantine (Deprecation Prophet) — do not expand

These paths stay **gated / non-authoritative** until a *written* re-scope (new utility or problem) exists:

| Path / track | Status |
|---|---|
| `trajectory/e2/` | GATED_STOP — no LoRA-mechanism claims |
| Fabric V2 / NanoScribe control plane | GATED — harness ≠ product architecture |
| Residual continua / Stage M curiosity | GATED |
| AAEA P2 eng sprint | Optional only with explicit owner authorize — **not** started by this closeout |
| E4 / R★ further runs | No curiosity reopen; revision only under written budget |

## Soft-freeze teeth (Saboteur)

Reopening any gated track requires **all** of:

1. New or amended prereg with explicit utility/problem statement
2. Owner authorization string in-session (`OWNER_RESCOPE_OK` or stronger)
3. No “just one offline CUDA check” without (1)–(2)

## Deletionist constraint

This closeout adds **documentation only**. No AAEA P2 implementation. No new experiments. No tag/push in this step.

## Tag gate

Proposed reconciled freeze tag remains **owner-only**.
**Not created** by hybrid run. Existing immutable tags:

- `paper-alpha-v1` — do not move
- `post-alpha-evidence-freeze-2026-07-31` — do not move


## Phase 3 — Tag decision (2026-07-31T18:09Z)

**Decision: DEFER new freeze tag.**

Reason (Contrarian / Architect): hybrid commit `a61ba41c21b0` sits on HEAD `71004232992d` whose ancestry **includes** E4 (`6af178d8d27d` ancestor=YES). Tagging HEAD as `council-hybrid-freeze-*` would mislabel an E4-containing tree as a freeze.

Honest alternatives left open for owner:
1. **Remain deferred** (default) — this closeout doc is the freeze act.
2. **Clean lineage** — branch from `post-alpha-evidence-freeze-2026-07-31`, cherry-pick `a61ba41c21b0`, tag there (`OWNER_TAG_OK` required).
3. **Non-freeze snapshot tag** on HEAD — allowed only if name does **not** claim freeze (`OWNER_TAG_OK` required).

Protected tags remain immutable: `paper-alpha-v1`, `post-alpha-evidence-freeze-2026-07-31`.

## Phase 4 — IDLE; parked dirty tree (2026-07-31T18:09Z)

Uncommitted paths left in place (not stash-dropped):

```
 M artifacts/POST_ALPHA_EVIDENCE_FREEZE.json
 M artifacts/POST_ALPHA_EVIDENCE_FREEZE.md
 M artifacts/SHA256SUMS
 M audit/discussion-to-implementation/EXECUTIVE_SUMMARY.md
 M audit/discussion-to-implementation/FINAL_FREEZE_READINESS_REPORT.md
 M audit/discussion-to-implementation/POST_ALPHA_EVIDENCE_FREEZE.json
 M audit/discussion-to-implementation/POST_ALPHA_EVIDENCE_FREEZE.md
 M audit/discussion-to-implementation/README.md
 M audit/discussion-to-implementation/SHA256SUMS
 M benchmarks/BENCHMARK_CONSTITUTION.md
 M papers/EMPIRICAL_FOUNDATION.md
?? EVIDENCE_CURRENT.md
?? audit/discussion-to-implementation/STRATIGRAPHY.md
?? audit/discussion-to-implementation/SWARM_QUEEN_SYNTHESIS_2026-07-31.md
?? papers/STRATEGIC_RESET.md
?? papers/WEDGE_V1.md
?? papers/WEDGE_W1.md
?? papers/WEDGE_W1_CORPUS_MANIFEST.json
?? research/decision_records/2026-07-31-owner-accept-A1-design.md
?? research/decision_records/2026-07-31-strategic-reset-choose-A.md
?? research/decision_records/2026-07-31-wedge-v1-lock.md
?? trajectory/PROGRAM_A1_rstar_revision_design.md
?? wedge_v1/README.md
?? wedge_v1/inclusion_predicates.md
```

Gated tracks unchanged: E2 / fabric V2 / NanoScribe / curiosity E4 reopen — **STOP**.

**Program state:** `IDLE_AFTER_HYBRID_COMMIT` — await optional `push` and/or `OWNER_TAG_OK` (clean-lineage cherry-pick). No further hybrid bookkeeping required.
