# External Replication & Reproducibility Registry

This is part of the [November 17 research proposal](FOUNDATION_INTELLIGENCE_PROGRAM.md). Its size limits, experiments and reported scores belong to that proposal. The [current charter](../PROJECT_CHARTER.md) defines NanoLM’s scope.

**Verification standard:** No empirical number enters canonical `docs/` or publication claims unless accompanied by its exact git SHA, immutable artifact hash, reproduction command, and explicit scope boundaries.

---

## 1. Exact Git & Artifact Anchors (Audited 2026-11-17)

| Surface | Branch / Tag | Commit SHA / Digest | Integrity & Access |
|---|---|---|---|
| **Canonical Trunk** | `origin/master` | `acc9b7e896c3b8189d29761d8401e6833c222a28` | Remote GitHub; CI `canonical-integrity` green |
| **Paper α Freeze** | Tag `paper-alpha-v1` | `0e01d73205e9c35ea32925fd4d6c7e5fceb61137` | Immutable tag; do not move/rewrite |
| **Post-α Freeze** | Tag `post-alpha-evidence-freeze-2026-07-31` | `a9d12cb1c456f6c465284e1d469c6326cb14d329` | Immutable tag; do not move/rewrite |
| **H6 Terminal Decision** | `origin/master`: `artifacts/nano_h6/H6_TERMINAL_DECISION.json` | Result `SHA-256 cb612410bd9530062147c1ed36bdcd606f9292a77b750376baf7a9860639e11a` | Reproducible diagnosis: `python3 -m nano_ai.training.diagnose_h6` |
| **H7 Screen & Invalid Eval** | `CROSS_BRANCH` `origin/codex/p1-h7-scribing-path` (PR #117) | Commit `f82d82a76b017c46d0b59b37eb6621ef0901c51c`; Screen `9b18c5ee…`; Eval `9d1b8035…` | Reconciled in `codex/p1-h7-scribing-path:docs/research/H7_EVALUATOR_DIAGNOSIS.md` |
| **C14 Retention Validation** | `CROSS_BRANCH` `origin/codex/p1-h7-scribing-path` | `posthoc_diagnostics.json` `0a560e12…`; `p1_retention_validation_20261026/` | Audit script: `artifacts/p1_retention_validation_20261026/independent_audit_recompute.py` |
| **NanoLM-v3 Gen-3** | `LOCAL_WORKTREE_UNMERGED` `.worktrees/nanolm-v3` (`nanolm/arch-evolution-v3`) | `86371cbb51540dd083d84554b989b48c881b7453` | Local-only commit; 10/10 unit & parity tests pass (`pytest lab/arch/test_gen3.py nanolm/tests/test_nanolm_v3_and_15pillar.py`) |
| **Discovery Lab Track A/B** | `LOCAL_WORKTREE_UNMERGED` `.worktrees/nanolm-lab` (`nanolm/discovery-lab-v1`) | `68291f0af93a5b8d00629490a16a9f65bfa1b31d` | Local-only commit; raw runs under `/Volumes/Express1M2/nanolm-lab/`; active background Phase 2 scripts |
| **Posttrain v1** | `CROSS_BRANCH` `origin/nanolm/posttrain-v1` (PR #171) | `ee3df6b1a52e855d277b1935740f116bd9732152` | Remote branch + `.worktrees/nanolm-posttrain` |

---

## 2. Canonical Trunk Verification Commands

Run on any branch targeting `master`:

```bash
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
python3 fabric/test_fabric.py
python3 trajectory/test_recompute_c3.py
python3 nanoscribe/test_encounter_v0.py
python3 nanoscribe/test_evidence_transport.py
python3 nanoscribe/test_adapt.py
python3 nanoscribe/benchmark.py
python3 -m pytest -q nano nanoscribe nano_ai/tests benchmarks/
```

---

## 3. Known Gaps Preventing Independent External Replication Today

1. **Unpushed `nanolm/arch-evolution-v3` (`86371cb`) & `nanolm/discovery-lab-v1` (`68291f0`):** Neither commit nor `/Volumes/Express1M2/nanolm-lab/runs/` raw weights/JSONs are mirrored on GitHub yet.
2. **`EXP-3` & `Track A` Clean Iso-Param / Iso-FLOP Multi-Seed Parity:** `EXP-3` requires re-running at matched non-embedding parameters (`5.90M` vs `6.73M` mismatch), $\ge 3$ seeds, and sufficient tokens for nonzero copy recall.
3. **Human Clinical / Dual-Rater External Validation:** Zero human dual-clinician IAA evaluations exist for `NanoScribe` (`C12–C14` used AI annotators/adjudicators).
4. **Evaluator Negative-Control Enforcement (`X01`):** Pending PR stacking on `codex/p1-h7-scribing-path` to enforce all six automated negative controls prior to any future state-span run.
