# Fabric Implementation Audit

*Privileges `fabric/*.py`, tests, and `results_slice_v1.json` over aspirational prose.*

## Actual execution path

```
Frozen instrument (inst0 or m0–m4)
  → load checkpoint via trajectory/rescore_anchors.py
  → generate field line (CC|DUR|SEV|MED|ALG)
  → parse regex → Claim packets
  → verify_value / verify_absent (v1 or v2)
  → decide() → PRESENT | ABSTAIN | QUALIFY
  → check_presentation_gate
  → write JSONL ledger row (file opened with "w" each run)
  → aggregate stats → results_slice_v1.json (batch)
```

**Not present:** Intent layer, control kernel, memory write authorization, tool calls.

## Claim reality table

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Typed claim packets | REAL | `schemas.py` frozen dataclasses |
| Immutable proposal preservation | PARTIAL | Claims frozen; no pre-verify proposal object |
| Source-span requirements | REAL | Validators + presentation gate |
| Absence-never-from-silence | REAL | `verify_absence()` + tests |
| Contradiction + counter-evidence | REAL (v2) | CONTRADICTED + spans |
| Lineage hashes | REAL | SHA-256 `_cid` |
| Duplicate detection | ABSENT | — |
| Stale version rejection | ABSENT | — |
| Append-only ledger | DOCUMENTED_ONLY | `"w"` truncates per run |
| Transactional writes | ABSENT | — |
| Schema versioning | ABSENT | — |
| Replay | ABSENT | — |
| Deterministic recomputation | PARTIAL | Unit tests deterministic; generation not |
| Per-type telemetry | DOCUMENTED_ONLY | Aggregate stats; slot on rows |
| Risk controller | MINIMAL | 4-branch `decide()`; label `risk.v1` |
| Abstention | REAL | Measured |
| Review load | NARROW | Unparseable → review counter; Stage A 19% is scribe-era |
| Semantic verifier | ABSENT | v2 rules/templates |
| Compositional verification suite | ABSENT as Stage V | Some unit adversarial cases exist |
| Memory separation | ABSENT | — |
| External tool integration | ABSENT | — |
| OSS baseline adapters | ABSENT | E1 baselines live outside fabric |

## Tests (this audit)

```text
cd fabric && pytest -q
........                                                                 [100%]
```

**8 tests.** Cover: v1 grounding, wrong-speaker reject, v1 blindspots documented, v2 catches three classes, article norm, absence rule, decide policy.

### Adversarial coverage vs required list

| Required adversarial check | Covered now? |
|----------------------------|--------------|
| Unsupported ≠ verified | Yes (policy/tests) |
| Absence not from silence | Yes |
| Wrong-speaker rejected | Yes (v1 test) |
| Wrong-slot rejected | Yes (v2 catches) |
| Contradiction preserves counter-evidence | Partial (state+span); no serialization stress |
| Duplicate idempotent | No |
| Stale state rejected | No |
| Ledger lineage survives serialization | Partial (`to_json` / self-test) |
| Malformed record isolation | No |
| Verifier cannot verify own unsupported content | Partial (presentation gate) |
| Correct claims not lost | Measured historically (`lost_correct=0`); not CI e2e |
| Metrics distinguish raw/verified/abstained/reviewed | Stats fields exist; review path thin |

**Audit decision:** Do not redesign fabric here. Missing adversarial tests are logged as harness hardening — not as license to expand architecture.

## Measured results (anchored)

`fabric/results_slice_v1.json`: under grounding.v2, nano/scale presented_error_rate **0.0** on inst0 and multi-instance cells; provenance_complete true; lost_correct 0.

**Scope (README):** existence proof under rules-strong decidable R; v2 could solve task alone; not open-world product.

## Expansion posture

`EMPIRICAL_FOUNDATION` / fabric README: fabric expansion STOP pending kill-gates / re-scope. E1 already KILL; E2 gated; E4 blocked.
