# Benchmark Result Policy

**Adopted:** 2026-07-31  
**Companion to:** `BENCHMARK_CONSTITUTION.md`

## May enter `leaderboards/` when

- run status is `COMPLETED`;
- manifests, pins, and `SHA256SUMS` are present;
- reproducibility checks pass for the declared suite;
- decision status is at least `LEADERBOARD_ONLY` (Program 0 forbids this);
- owner/queue authorizes publishing that board row.

## May enter `papers/EVIDENCE_LEDGER.md` only when

1. owner-authorized decision gate (laboratory G3+ as applicable);
2. construct review;
3. contamination review;
4. replication;
5. OOD / hidden validation as required by bench gates 4–5;
6. claim scoping in ordinary language.

## Program 0

All smoke decisions:

```text
decision = INFRA_SMOKE_PASS | INFRA_SMOKE_FAIL
promote = false
leaderboard_eligible = false
evidence_ledger_eligible = false
```

## Classical baselines

Extraction-like tasks **require** a classical / deterministic baseline in the same
suite comparison (E1 lesson).

## Failed and partial runs

`FAILED`, `PARTIAL`, and `VOID` runs remain visible under `experiments/**/runs/`
(gitignored generated trees; fixtures may retain golden examples).
